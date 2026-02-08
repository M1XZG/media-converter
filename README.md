# Media Converter

A self-hosted web application for converting video files between formats and extracting audio from video files. Built with Python/Flask and powered by FFmpeg.

## Features

- **Video Format Conversion** — Convert between popular video formats (MP4, AVI, MKV, MOV, WMV, FLV, WebM)
- **Audio Extraction** — Extract audio tracks from video files (MP3, AAC, WAV, FLAC, OGG)
- **No File Size Limits** — Upload files of any size
- **Automatic Cleanup** — All uploaded and converted files are automatically deleted after 24 hours
- **Dark/Light Mode** — Modern UI with dark mode as default and easy toggle
- **Self-Contained** — Runs in a Python virtual environment with minimal dependencies

## Prerequisites

- **Python 3.8+**
- **FFmpeg** — Must be installed and available on your system PATH

### Installing FFmpeg

**Windows (via winget):**
```bash
winget install FFmpeg
```

**Windows (via Chocolatey):**
```bash
choco install ffmpeg
```

**macOS (via Homebrew):**
```bash
brew install ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install ffmpeg
```

## Quick Start

### Windows
```bash
# Clone the repository
git clone https://github.com/yourusername/media-converter.git
cd media-converter

# Run the setup and start script
setup.bat
```

### Linux / macOS
```bash
# Clone the repository
git clone https://github.com/yourusername/media-converter.git
cd media-converter

# Make the script executable and run
chmod +x setup.sh
./setup.sh
```

### Manual Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Then open your browser to **http://localhost:5000**

## Configuration

Environment variables can be set in a `.env` file or exported:

| Variable | Default | Description |
|---|---|---|
| `FLASK_HOST` | `0.0.0.0` | Host to bind to |
| `FLASK_PORT` | `5000` | Port to listen on |
| `MAX_CONTENT_LENGTH` | `0` (unlimited) | Max upload size in bytes (0 = no limit) |
| `CLEANUP_HOURS` | `24` | Hours before files are auto-deleted |

## Supported Formats

### Video Output Formats
| Format | Extension |
|---|---|
| MP4 | `.mp4` |
| AVI | `.avi` |
| MKV | `.mkv` |
| MOV | `.mov` |
| WMV | `.wmv` |
| FLV | `.flv` |
| WebM | `.webm` |

### Audio Output Formats
| Format | Extension |
|---|---|
| MP3 | `.mp3` |
| AAC | `.aac` |
| WAV | `.wav` |
| FLAC | `.flac` |
| OGG | `.ogg` |

## Project Structure

```
media-converter/
├── app.py              # Flask application
├── cleanup.py          # File cleanup utility
├── requirements.txt    # Python dependencies
├── setup.bat           # Windows setup script
├── setup.sh            # Linux/macOS setup script
├── .gitignore
├── README.md
├── templates/
│   └── index.html      # Web UI
├── uploads/            # Temporary upload storage (auto-created)
└── converted/          # Temporary converted file storage (auto-created)
```

## License

MIT License
