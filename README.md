# The Agile Monkeys Claude Open Marketplace

AI-powered skills for [Claude](https://claude.ai) by [The Agile Monkeys](https://theagilemonkeys.com).

A plugin marketplace for Claude Code and Claude Desktop. Install once, use everywhere Claude runs.

## Available Plugins

| Plugin | Description | Requirements |
|--------|-------------|--------------|
| **[Transcription](#transcription)** | Local audio transcription with speaker diarization | Apple Silicon Mac, Python 3.11+, [uv](https://docs.astral.sh/uv/) |

## Installation

### Claude Desktop

1. Open **Customize** (left sidebar)
2. Go to **Personal Plugins** and click the **+** button
3. Select **Browse Plugins**
4. Go to the **Personal** tab and click the **+** button
5. Select **Add marketplace**
6. Enter `theam/claude-open-marketplace` and click **Sync**
7. Install the plugins you want from the list

### Claude App — Code Tab

The Code tab in the Claude app uses a separate plugin configuration from the chat interface. To install:

1. Open the **Code** tab
2. Click **Customize** (bottom-left)
3. Follow the same marketplace steps as above (Personal Plugins → Browse → Add marketplace → `theam/claude-open-marketplace`)

> **Note**: Plugins installed in the chat interface are not shared with the Code tab — you need to install them separately.

### Claude Code (CLI)

```bash
claude plugin marketplace add theam/claude-open-marketplace
claude plugin install transcription@claude-open-marketplace
```

---

## Transcription

Local-first audio transcription with speaker diarization. Runs entirely on your machine — no API keys, no cloud services, no data leaves your computer.

Uses Apple Silicon GPU acceleration (MLX) to transcribe audio at ~5x real-time speed. A 75-minute recording processes in ~14 minutes on an M3 Max.

### What it does

- Transcribes audio files (`.m4a`, `.mp3`, `.wav`, `.flac`, and more) with automatic language detection
- Identifies and labels different speakers in multi-speaker recordings
- Exports transcriptions to plain text, JSON, or SRT subtitles
- Lets you assign real names to identified speakers

### How to use it

Use Claude Desktop in **Cowork** mode or **Claude Code** for file-based workflows. The transcription tools work in regular chat mode too — just tell Claude what you need:

- *"List audio files in ~/Downloads"*
- *"Transcribe /path/to/recording.m4a"*
- *"Transcribe it without speaker identification"* (faster, skips diarization)
- *"Export the transcription as SRT subtitles"*
- *"Rename SPEAKER_00 to Sarah"*

### First run

The first transcription takes a few extra minutes to download models:

1. **Python dependencies** (torch, whisperx, mlx-whisper) — ~2GB, cached after first install
2. **Whisper model** (whisper-large-v3-mlx) — ~3GB, cached in `~/.cache/huggingface/`

Speaker diarization models are bundled with the plugin — no HuggingFace account or API keys needed.

### Performance

Tested on M3 Max (36GB RAM):

| Mode | Speed | Use case |
|------|-------|----------|
| Transcription only | ~5.4x real-time | Quick text extraction, no speaker labels |
| With diarization | ~1.1x real-time | Full speaker-labeled transcripts |

### Models

The default model (`whisper-large-v3`) prioritizes accuracy (6.4% WER). For faster processing at slightly lower accuracy, ask Claude to use the turbo model:

*"Transcribe this file using the turbo model"*

---

## Contributing

We welcome contributions. If you'd like to add a new plugin or improve an existing one, open an issue or pull request.

## License

Apache License 2.0 — Copyright 2026 [The Agile Monkeys S.L.](https://theagilemonkeys.com)

Speaker diarization models bundled under MIT license from the [pyannote](https://github.com/pyannote/pyannote-audio) project.
