# Transcription MCP Plugin

Local-first audio transcription with speaker diarization for Claude Code and Claude Desktop. Runs entirely on your machine — no data leaves your computer.

Powered by [**scribe**](https://github.com/theam/scribe) — state-of-the-art local transcription CLI using WhisperKit + SpeakerKit on Apple Silicon.

## Prerequisites

1. **Install scribe**:
   ```bash
   brew install theam/tap/scribe
   ```

2. **Python 3.11+** and **uv**:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

No API keys, no accounts, no cloud services needed.

## Install in Claude Code

```bash
claude mcp add transcription -- uv run --directory /path/to/plugins/transcription python server.py
```

## Install in Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "transcription": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/plugins/transcription", "python", "server.py"]
    }
  }
}
```

## Usage

Once installed, ask Claude to:

- **"List audio files in ~/Downloads"** — finds .m4a, .mp3, .wav, .flac, etc.
- **"Transcribe /path/to/recording.m4a"** — transcribes with speaker identification
- **"Transcribe recording.m4a with skip_diarization"** — faster, no speaker labels
- **"Export the transcription as SRT"** — exports to .txt, .json, or .srt
- **"Rename Speaker 1 to Jaime"** — assign names to identified speakers

## Tools

| Tool | Description |
|------|-------------|
| `list_audios` | List audio files in a directory |
| `transcribe_audio` | Transcribe with optional diarization |
| `list_transcriptions` | Show completed transcriptions |
| `get_transcription` | View a transcription |
| `set_speaker_name` | Name a speaker |
| `export_transcription` | Export to txt/json/srt |

## Performance

On Apple Silicon (M-series), scribe processes audio at ~12x real-time with diarization enabled (e.g., a 4-minute recording transcribes in ~20 seconds).

## How it Works

This plugin is a thin MCP wrapper. All ML processing is done by the `scribe` CLI:

```
Claude → MCP Plugin → scribe CLI → WhisperKit (transcription) + SpeakerKit (diarization)
```

The plugin calls `scribe transcribe --diarize --format json`, parses the output, and presents it through MCP tools.
