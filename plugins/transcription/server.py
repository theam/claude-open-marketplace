"""Transcription MCP Server — Local-first using scribe CLI.

Thin MCP wrapper around the scribe CLI (https://github.com/theam/scribe).
Requires scribe to be installed: brew install theam/tap/scribe
"""

import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from transcription import (
    export_transcript,
    find_audio_files,
    format_transcript,
    srt_time,
    transcribe,
)

# In-memory transcription store (session-scoped)
_transcriptions: dict[str, dict] = {}

mcp = FastMCP(
    "transcription",
    instructions=(
        "Audio transcription server with speaker diarization. "
        "Use transcribe_audio to transcribe files with speaker identification. "
        "Use list_audios to see available audio files. "
        "Use list_transcriptions to see completed transcriptions. "
        "Use get_transcription to view a specific transcription. "
        "Use set_speaker_name to assign names to identified speakers."
    ),
)


@mcp.tool()
def list_audios(directory: str) -> str:
    """List audio files in a directory with their duration estimates.

    Args:
        directory: Path to directory containing audio files.
    """
    files = find_audio_files(directory)
    if not files:
        return f"No audio files found in {directory}"

    lines = [f"Audio files in {directory}:\n"]
    for i, f in enumerate(files, 1):
        size_mb = f.stat().st_size / (1024 * 1024)
        lines.append(f"{i}. {f.name} ({size_mb:.1f} MB)")

    lines.append(f"\nTotal: {len(files)} files")
    return "\n".join(lines)


@mcp.tool()
def transcribe_audio(
    file_path: str,
    language: str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    skip_diarization: bool = False,
    model: str | None = None,
) -> str:
    """Transcribe an audio file with optional speaker diarization.

    All processing happens locally via the scribe CLI. No data leaves your machine.
    Requires scribe: brew install theam/tap/scribe

    Args:
        file_path: Path to the audio file to transcribe.
        language: Language code (e.g., 'en', 'es'). Auto-detected if not specified.
        num_speakers: Exact number of speakers if known.
        min_speakers: Minimum expected number of speakers (passed to scribe).
        max_speakers: Maximum expected number of speakers (passed to scribe).
        skip_diarization: Skip speaker identification for faster processing.
        model: Whisper model override (e.g., 'large-v3' for best accuracy).
    """
    path = Path(file_path)
    if not path.exists():
        return f"File not found: {file_path}"

    try:
        result = transcribe(
            file_path=file_path,
            language=language,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            skip_diarization=skip_diarization,
            model=model,
        )
    except FileNotFoundError as e:
        return str(e)
    except RuntimeError as e:
        return str(e)

    file_key = result["path"]
    _transcriptions[file_key] = {
        "file": result["file"],
        "path": file_key,
        "language": result["language"],
        "duration_s": result["duration_s"],
        "processing_time_s": result["processing_time_s"],
        "segments": result["segments"],
        "speaker_names": {},
    }

    speakers = result["speakers"]
    diarization_status = "skipped" if skip_diarization else f"{len(speakers)} speakers"

    summary = (
        f"Transcription complete: {result['file']}\n"
        f"Duration: {result['duration_s']/60:.1f} min | "
        f"Processing time: {result['processing_time_s']:.0f}s "
        f"({result['duration_s']/result['processing_time_s']:.1f}x real-time) | "
        f"Backend: {result['backend']}\n"
        f"Language: {result['language']} | "
        f"Diarization: {diarization_status}"
    )
    if speakers:
        summary += f" ({', '.join(speakers)})"

    transcript_text = format_transcript({"segments": result["segments"]})
    summary += (
        f"\nSegments: {len(result['segments'])}\n\n"
        f"--- Transcript ---\n\n{transcript_text}"
    )

    return summary


@mcp.tool()
def list_transcriptions() -> str:
    """List all completed transcriptions in this session."""
    if not _transcriptions:
        return "No transcriptions yet. Use transcribe_audio to transcribe a file."

    lines = ["Completed transcriptions:\n"]
    for key, t in _transcriptions.items():
        speakers = set()
        for seg in t["segments"]:
            if "speaker" in seg:
                speakers.add(seg["speaker"])
        names = t.get("speaker_names", {})
        speaker_info = []
        for s in sorted(speakers):
            name = names.get(s, s)
            speaker_info.append(f"{s}={name}" if s != name else s)

        lines.append(
            f"- {t['file']} ({t['duration_s']/60:.1f} min, "
            f"{t['language']}, {len(speakers)} speakers: {', '.join(speaker_info)})"
        )
    return "\n".join(lines)


@mcp.tool()
def get_transcription(
    file_path: str,
    include_timestamps: bool = True,
    speaker_filter: str | None = None,
) -> str:
    """Get the full transcription text for a previously transcribed file.

    Args:
        file_path: Path to the audio file (as used in transcribe_audio).
        include_timestamps: Whether to include timestamps in output.
        speaker_filter: Only show segments from this speaker (e.g., 'Speaker 1').
    """
    path = Path(file_path).resolve()
    key = str(path)

    if key not in _transcriptions:
        return f"No transcription found for {file_path}. Use transcribe_audio first."

    t = _transcriptions[key]
    result = {"segments": list(t["segments"])}
    names = t.get("speaker_names", {})

    for seg in result["segments"]:
        speaker_id = seg.get("speaker", "Unknown")
        if speaker_id in names:
            seg["speaker"] = names[speaker_id]

    if speaker_filter:
        result["segments"] = [
            s for s in result["segments"]
            if s.get("speaker", "") == speaker_filter
        ]

    return format_transcript(result, include_timestamps=include_timestamps)


@mcp.tool()
def set_speaker_name(file_path: str, speaker_id: str, name: str) -> str:
    """Assign a human-readable name to an identified speaker.

    Args:
        file_path: Path to the transcribed audio file.
        speaker_id: The speaker ID (e.g., 'SPEAKER_00') from the diarization.
        name: The human-readable name to assign (e.g., 'Jaime').
    """
    path = Path(file_path).resolve()
    key = str(path)

    if key not in _transcriptions:
        return f"No transcription found for {file_path}"

    _transcriptions[key]["speaker_names"][speaker_id] = name
    return f"Speaker {speaker_id} renamed to '{name}' in {Path(file_path).name}"


@mcp.tool()
def export_transcription(
    file_path: str,
    output_path: str | None = None,
    format: str = "txt",
) -> str:
    """Export a transcription to a file.

    Args:
        file_path: Path to the transcribed audio file.
        output_path: Where to save the export. Defaults to same directory as audio file.
        format: Export format — 'txt', 'json', or 'srt'.
    """
    path = Path(file_path).resolve()
    key = str(path)

    if key not in _transcriptions:
        return f"No transcription found for {file_path}"

    t = _transcriptions[key]
    names = t.get("speaker_names", {})

    if output_path is None:
        output_path = str(path.with_suffix(f".{format}"))

    out = Path(output_path)

    if format == "json":
        export_data = {
            "file": t["file"],
            "language": t["language"],
            "duration_s": t["duration_s"],
            "speaker_names": names,
            "segments": t["segments"],
        }
        out.write_text(json.dumps(export_data, indent=2, ensure_ascii=False))
    else:
        out.write_text(export_transcript(t["segments"], format=format, speaker_names=names))

    return f"Exported to {out}"


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
