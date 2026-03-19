"""Shared transcription logic — used by both MCP server and CLI.

Core functions: ML dep installation, model loading, transcription,
diarization, formatting. No MCP or CLI-specific code here.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Plugin directory and bundled models
PLUGIN_DIR = Path(__file__).parent
MODELS_DIR = PLUGIN_DIR / "models"

# Lazy-loaded modules
_torch = None
_whisperx = None
_mlx_whisper = None
_DiarizationPipeline = None

# Runtime state
_ml_deps_ready = False
_use_mlx = False
_device = "cpu"
_compute_type = "int8"
_mlx_model = os.environ.get("MLX_MODEL", "mlx-community/whisper-large-v3-mlx")
_model = None
_diarize_pipeline = None


def _find_uv() -> str | None:
    """Find the uv binary, checking PATH and common install locations."""
    found = shutil.which("uv")
    if found:
        return found
    for candidate in [
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
        Path("/usr/local/bin/uv"),
    ]:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _install_ml_deps():
    """Install ML deps using uv (preferred) or pip (fallback).

    Raises RuntimeError with actionable instructions on failure.
    """
    deps = ["torch>=2.8.0", "torchaudio>=2.8.0", "whisperx>=3.8.2"]
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        deps.append("mlx-whisper>=0.4.0")

    # Strategy 1: uv pip install (works in uv-managed venvs)
    uv = _find_uv()
    if uv:
        try:
            print("Installing ML dependencies via uv (first run only, ~2GB download)...")
            subprocess.check_call(
                [uv, "pip", "install", "--quiet"] + deps, timeout=600,
            )
            print("ML dependencies installed.")
            return
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            print(f"uv install failed ({exc}), trying pip fallback...")

    # Strategy 2: pip module (traditional venvs)
    try:
        print("Installing ML dependencies via pip (first run only, ~2GB download)...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet"] + deps, timeout=600,
        )
        print("ML dependencies installed.")
        return
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Both strategies failed — give actionable error
    venv = os.environ.get("VIRTUAL_ENV", "unknown")
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    deps_str = " ".join(f'"{d}"' for d in deps)
    raise RuntimeError(
        f"Failed to install ML dependencies automatically.\n\n"
        f"Environment: Python {py_ver}, venv={venv}\n\n"
        f"Manual fix — run one of:\n"
        f"  uv pip install {deps_str}\n"
        f"  pip install {deps_str}\n"
    )


def ensure_ml_deps():
    """Install and import ML dependencies on first use."""
    global _ml_deps_ready, _torch, _whisperx, _mlx_whisper, _DiarizationPipeline
    global _use_mlx, _device, _compute_type

    if _ml_deps_ready:
        return

    # Check if deps are already importable
    try:
        import torch
        import whisperx
        _torch = torch
        _whisperx = whisperx
    except ImportError:
        _install_ml_deps()
        import torch
        import whisperx
        _torch = torch
        _whisperx = whisperx

    # Detect device
    if _torch.backends.mps.is_available():
        _device = "mps"
        _compute_type = "float16"

    # Try MLX
    try:
        import mlx_whisper
        _mlx_whisper = mlx_whisper
        _use_mlx = True
        print("MLX backend available — using Apple Silicon GPU for ASR")
    except ImportError:
        print("MLX not available — using faster-whisper/CPU backend")

    # Import diarization pipeline
    from whisperx.diarize import DiarizationPipeline
    _DiarizationPipeline = DiarizationPipeline

    _ml_deps_ready = True


def _get_model():
    """Lazy-load the Whisper model (faster-whisper/CPU path only)."""
    global _model
    if _model is None:
        print("Loading Whisper large-v3-turbo model (CPU)...")
        _model = _whisperx.load_model(
            "large-v3-turbo",
            device=_device if _device != "mps" else "cpu",
            compute_type=_compute_type if _device != "mps" else "int8",
            language=None,
        )
        print("Model loaded.")
    return _model


def _transcribe_mlx(
    audio_path: str, language: str | None = None, model_repo: str | None = None,
) -> dict:
    """Transcribe using MLX backend (Apple Silicon GPU)."""
    repo = model_repo or _mlx_model
    print(f"Transcribing with MLX ({repo})...")
    result = _mlx_whisper.transcribe(
        audio_path,
        path_or_hf_repo=repo,
        language=language,
        word_timestamps=True,
    )
    segments = []
    for seg in result.get("segments", []):
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "words": seg.get("words", []),
        })
    detected_lang = result.get("language", language or "unknown")
    print(f"MLX transcription complete. Language: {detected_lang}")
    return {"segments": segments, "language": detected_lang}


def _get_diarize_pipeline():
    """Lazy-load the pyannote diarization pipeline."""
    global _diarize_pipeline
    if _diarize_pipeline is None:
        print("Loading pyannote diarization pipeline...")
        old_cache = os.environ.get("HF_HUB_CACHE")
        old_offline = os.environ.get("HF_HUB_OFFLINE")
        try:
            if MODELS_DIR.is_dir():
                os.environ["HF_HUB_CACHE"] = str(MODELS_DIR)
                os.environ["HF_HUB_OFFLINE"] = "1"
            _diarize_pipeline = _DiarizationPipeline(
                model_name="pyannote/speaker-diarization-3.1",
                device="cpu",
            )
        finally:
            if old_cache is None:
                os.environ.pop("HF_HUB_CACHE", None)
            else:
                os.environ["HF_HUB_CACHE"] = old_cache
            if old_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = old_offline
        print("Diarization pipeline loaded.")
    return _diarize_pipeline


def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_transcript(result: dict, include_timestamps: bool = True) -> str:
    """Format a WhisperX result into readable text."""
    lines = []
    for seg in result.get("segments", []):
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "").strip()
        if not text:
            continue
        if include_timestamps:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            ts = f"[{fmt_time(start)} - {fmt_time(end)}]"
            lines.append(f"{ts} {speaker}: {text}")
        else:
            lines.append(f"{speaker}: {text}")
    return "\n\n".join(lines)


def find_audio_files(directory: str) -> list[Path]:
    audio_extensions = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".wma", ".aac", ".mp4"}
    path = Path(directory)
    if not path.is_dir():
        return []
    return [f for f in sorted(path.iterdir()) if f.suffix.lower() in audio_extensions and f.is_file()]


def transcribe(
    file_path: str,
    language: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    skip_diarization: bool = False,
    model: str | None = None,
) -> dict:
    """Transcribe an audio file. Returns a result dict with segments and metadata.

    This is the core transcription function used by both MCP and CLI.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    ensure_ml_deps()

    start_time = time.time()
    mlx_model = model or _mlx_model

    # Step 1: Load audio
    print(f"Loading audio: {path.name}")
    audio = _whisperx.load_audio(str(path))
    duration_s = len(audio) / 16000
    print(f"Audio loaded: {duration_s:.0f}s ({duration_s/60:.1f} min)")

    # Step 2: Transcribe (MLX GPU or faster-whisper CPU)
    if _use_mlx:
        result = _transcribe_mlx(str(path), language=language, model_repo=mlx_model)
        detected_lang = result["language"]
    else:
        print("Transcribing (CPU)...")
        whisper_model = _get_model()
        result = whisper_model.transcribe(
            audio,
            batch_size=16 if _device != "mps" else 4,
            language=language,
        )
        detected_lang = result.get("language", language or "unknown")
        print(f"Transcription complete. Language: {detected_lang}")

    # Step 3: Diarization (optional, parallel with alignment)
    if not skip_diarization:
        def _align():
            print("Aligning timestamps...")
            align_model, align_metadata = _whisperx.load_align_model(
                language_code=detected_lang,
                device="cpu",
            )
            aligned = _whisperx.align(
                result["segments"],
                align_model,
                align_metadata,
                audio,
                device="cpu",
                return_char_alignments=False,
            )
            print("Alignment complete.")
            return aligned

        def _diarize():
            print("Running speaker diarization...")
            diarize_pipeline = _get_diarize_pipeline()
            diarize_kwargs = {}
            if num_speakers is not None:
                diarize_kwargs["num_speakers"] = num_speakers
            if min_speakers is not None:
                diarize_kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarize_kwargs["max_speakers"] = max_speakers
            diarize_result = diarize_pipeline(str(path), **diarize_kwargs)
            print("Diarization complete.")
            return diarize_result

        with ThreadPoolExecutor(max_workers=2) as executor:
            align_future = executor.submit(_align)
            diarize_future = executor.submit(_diarize)
            aligned_result = align_future.result()
            diarize_segments = diarize_future.result()

        result = _whisperx.assign_word_speakers(diarize_segments, aligned_result)

    elapsed = time.time() - start_time

    speakers = set()
    for seg in result["segments"]:
        if "speaker" in seg:
            speakers.add(seg["speaker"])

    backend = f"MLX ({mlx_model.split('/')[-1]})" if _use_mlx else "CPU"

    return {
        "file": path.name,
        "path": str(path.resolve()),
        "language": detected_lang,
        "duration_s": duration_s,
        "processing_time_s": elapsed,
        "segments": result["segments"],
        "speakers": sorted(speakers),
        "backend": backend,
        "skip_diarization": skip_diarization,
    }


def export_transcript(segments: list, format: str, speaker_names: dict | None = None) -> str:
    """Export segments to a string in the given format (txt, json, srt)."""
    names = speaker_names or {}

    if format == "json":
        return json.dumps(segments, indent=2, ensure_ascii=False)

    elif format == "srt":
        lines = []
        for i, seg in enumerate(segments, 1):
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            speaker = seg.get("speaker", "Unknown")
            speaker = names.get(speaker, speaker)
            text = seg.get("text", "").strip()
            lines.append(str(i))
            lines.append(f"{srt_time(start)} --> {srt_time(end)}")
            lines.append(f"[{speaker}] {text}")
            lines.append("")
        return "\n".join(lines)

    else:  # txt
        applied = []
        for seg in segments:
            s = dict(seg)
            sid = s.get("speaker", "Unknown")
            if sid in names:
                s["speaker"] = names[sid]
            applied.append(s)
        return format_transcript({"segments": applied})
