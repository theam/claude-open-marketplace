"""Shared transcription logic — thin wrapper around the scribe CLI.

Calls `scribe transcribe` for all transcription/diarization work.
No ML dependencies needed — scribe handles everything locally.
"""

import json
import shutil
import subprocess
import time
from pathlib import Path


def _find_scribe() -> str:
    """Find the scribe binary in PATH or common install locations."""
    found = shutil.which("scribe")
    if found:
        return found
    for candidate in [
        Path("/usr/local/bin/scribe"),
        Path("/opt/homebrew/bin/scribe"),
        Path.home() / ".local" / "bin" / "scribe",
    ]:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "scribe CLI not found. Install it with: brew install theam/tap/scribe\n"
        "More info: https://github.com/theam/scribe"
    )


def find_audio_files(directory: str) -> list[Path]:
    """Find audio files in a directory."""
    audio_extensions = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".wma", ".aac", ".mp4"}
    path = Path(directory)
    if not path.is_dir():
        return []
    return [f for f in sorted(path.iterdir()) if f.suffix.lower() in audio_extensions and f.is_file()]


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
    """Format segments into readable text."""
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


def transcribe(
    file_path: str,
    language: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    skip_diarization: bool = False,
    model: str | None = None,
) -> dict:
    """Transcribe an audio file by calling the scribe CLI.

    Returns a result dict with segments and metadata.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    scribe = _find_scribe()
    start_time = time.time()

    # Build scribe command
    cmd = [scribe, "transcribe", str(path), "--format", "json"]
    if not skip_diarization:
        cmd.append("--diarize")
        if num_speakers is not None:
            cmd.extend(["--speakers", str(num_speakers)])
    if language:
        cmd.extend(["--language", language])
    if model:
        cmd.extend(["--model", model])

    # Run scribe
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 min timeout for very long files
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Transcription timed out after 30 minutes for {path.name}")

    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise RuntimeError(f"scribe failed: {stderr or 'unknown error'}")

    # Parse JSON output
    try:
        scribe_output = json.loads(result.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"Failed to parse scribe output: {result.stdout[:200]}")

    elapsed = time.time() - start_time

    # Convert scribe JSON to our segment format
    segments = []
    speakers = set()
    for seg in scribe_output.get("segments", []):
        segment = {
            "start": seg.get("start", 0),
            "end": seg.get("end", 0),
            "text": seg.get("text", ""),
        }
        if "speaker" in seg:
            # Map "Speaker 1" to "SPEAKER_00" format for compatibility
            speaker_name = seg["speaker"]
            segment["speaker"] = speaker_name
            speakers.add(speaker_name)
        if "words" in seg:
            segment["words"] = seg["words"]
        segments.append(segment)

    duration_s = scribe_output.get("metadata", {}).get("duration", 0)

    return {
        "file": path.name,
        "path": str(path.resolve()),
        "language": language or "auto",
        "duration_s": duration_s,
        "processing_time_s": elapsed,
        "segments": segments,
        "speakers": sorted(speakers),
        "backend": "scribe (WhisperKit + SpeakerKit)",
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
