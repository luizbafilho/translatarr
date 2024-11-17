# Subtitle Translator Service

A service that automatically monitors directories for video files (MKV/MP4), extracts English subtitles, and translates them to Brazilian Portuguese using OpenAI's GPT-4 model. The service maintains the original subtitle formatting and context while translating.

## Features

- 🎬 Monitors multiple directories for new video files
- 📥 Extracts embedded subtitles from MKV and MP4 files
- 🔍 Identifies subtitle language and processes only English subtitles
- 🔄 Maintains subtitle formatting and timing
- 💾 Tracks processed files to avoid duplicate translations
- 🧠 Uses context-aware translation for better consistency
- 🚫 Skips files that already have Portuguese subtitles

## Prerequisites

- Python 3.12+
- FFmpeg
- OpenAI API key
- (Optional) Sentry DSN for error monitoring

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd subtitle_extractor
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Install FFmpeg:
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- macOS: `brew install ffmpeg`
- Windows: Download from the [official FFmpeg website](https://ffmpeg.org/download.html)


## Usage

### Basic Usage

1. Set your OpenAI API key:

```bash
export OPENAI_API_KEY=<your-openai-api-key>
```

2. Run the service:

#### Watch a single directory

```bash
python main.py --watch-dirs /path/to/videos
```


### Watch multiple directories

```bash
python main.py --watch-dirs /path/to/videos1 /path/to/videos2
```


### Test with a single subtitle file

```bash
python main.py --subtitle /path/to/subtitle.srt
```

## Configuration

The service can be configured using environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `WATCH_DIRECTORY`: Default directories to watch, comma-separated (if not specified via CLI)
  Example: `/path/to/videos1,/path/to/videos2,/path/to/videos3`
- `SENTRY_DSN`: Sentry DSN for error monitoring (optional)
- `ENVIRONMENT`: Environment name for Sentry (default: production)

## File Tracking

The service maintains a `.subtitle_tracker.json` file in each watched directory to:
- Track processed files
- Prevent duplicate translations
- Store translation metadata

## Limitations

- Only processes English subtitles to Brazilian Portuguese
- Requires FFmpeg for subtitle extraction
- Depends on OpenAI API availability
- Processes one subtitle at a time