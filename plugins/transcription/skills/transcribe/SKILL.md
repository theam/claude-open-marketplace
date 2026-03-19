---
name: transcribe
description: Transcribe audio files with speaker diarization using local ML models. Works in both Claude Code and Cowork mode.
user_invocable: true
---

# Transcribe Audio

Transcribe an audio file locally with speaker diarization. All processing happens on your machine — no data leaves your device.

## How to use

Run the CLI transcription tool via Bash. The plugin directory is:

```
PLUGIN_DIR=$(dirname "$(dirname "$(which transcribe_cli.py 2>/dev/null || echo "")")")
```

Use the following command pattern:

```bash
uv run --directory {PLUGIN_DIR} python transcribe_cli.py "{audio_file}" [options]
```

Where `{PLUGIN_DIR}` is the absolute path to the transcription plugin directory (the directory containing `transcribe_cli.py`). To find it, look for the transcription plugin in the installed plugins — it will be under `plugins/transcription/` in the monkey-tools plugin directory.

### Arguments

| Argument | Description |
|----------|-------------|
| `audio_file` | **(required)** Path to the audio file (.m4a, .mp3, .wav, .flac, .ogg, .aac, .mp4) |
| `--language LANG` | Language code (e.g., `en`, `es`). Auto-detected if omitted. |
| `--skip-diarization` | Skip speaker identification for faster processing. |
| `--num-speakers N` | Exact number of speakers if known. |
| `--min-speakers N` | Minimum expected number of speakers. |
| `--max-speakers N` | Maximum expected number of speakers. |
| `--model REPO` | MLX model override (e.g., `mlx-community/whisper-large-v3-turbo` for speed). |
| `--export FORMAT` | Output format: `txt` (default), `json`, or `srt`. |

### Examples

Basic transcription:
```bash
uv run --directory /path/to/plugins/transcription python transcribe_cli.py "/path/to/audio.m4a"
```

With language hint and JSON export:
```bash
uv run --directory /path/to/plugins/transcription python transcribe_cli.py "/path/to/audio.m4a" --language es --export json
```

Fast mode (skip diarization):
```bash
uv run --directory /path/to/plugins/transcription python transcribe_cli.py "/path/to/audio.m4a" --skip-diarization
```

## Notes

- **First run** installs ML dependencies (~2GB). This is cached for subsequent uses.
- **Apple Silicon**: Automatically uses MLX for ~5x real-time transcription speed.
- **Output**: Summary stats go to stderr. Clean transcript goes to stdout.
- The transcript text is printed to stdout — capture it or read it directly from the Bash output.

## Platform Notes

- **Claude Code (macOS/Apple Silicon)**: Full support. Uses MLX GPU acceleration. Deps auto-install via `uv` or `pip`.
- **Claude Code (Linux/Intel Mac)**: CPU-only mode (no MLX). Slower but functional.
- **Cowork**: CPU-only. Requires Python 3.11+ in the VM. Network restrictions may prevent auto-install of ML deps — pre-install with `uv pip install -e ".[ml]"` if needed.
- **Pre-install deps** to avoid first-run delay: `uv pip install -e ".[ml-apple]"` (Apple Silicon) or `uv pip install -e ".[ml]"` (other platforms).
