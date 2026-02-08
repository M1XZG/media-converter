# Media Converter

A self-hosted web application for converting video files between formats and extracting audio from video files. Built with Python/Flask and powered by FFmpeg.

## Features

- **Video Format Conversion** — Convert between popular video formats (MP4, AVI, MKV, MOV, WMV, FLV, WebM)
- **Audio Extraction** — Extract audio tracks from video files (MP3, AAC, WAV, FLAC, OGG)
- **GPU Acceleration** — Automatically uses hardware encoding (NVIDIA NVENC, AMD AMF, Intel QSV, VA-API) when available, with seamless CPU fallback
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

### Docker (recommended)

```bash
git clone https://github.com/yourusername/media-converter.git
cd media-converter
docker compose up -d
```

Then open **http://localhost:5000**. GPU acceleration works automatically if you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

To run **without GPU** support, remove the `deploy` block from `docker-compose.yml`, or use a simpler run command:

```bash
docker build -t media-converter .
docker run -d -p 5000:5000 --name media-converter media-converter
```

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

## GPU Acceleration

The application automatically detects and uses GPU hardware encoders when available. On startup the console and the web UI will show whether GPU acceleration is active.

| GPU Vendor | Encoder | Requirements |
|---|---|---|
| NVIDIA | NVENC | NVIDIA GPU + driver 470+, FFmpeg built with `--enable-nvenc` |
| AMD | AMF | AMD GPU + Adrenalin driver, FFmpeg built with `--enable-amf` |
| Intel | QSV | Intel iGPU/dGPU + media driver, FFmpeg built with `--enable-libmfx` or `--enable-libvpl` |
| Linux (generic) | VA-API | VA-API capable GPU + `libva`, FFmpeg built with `--enable-vaapi` |

GPU encoding is used for MP4, MKV, and MOV output. Other formats (AVI, WMV, FLV, WebM) and all audio extraction use CPU encoding. If a GPU encoder is detected but fails at runtime, FFmpeg will report an error — the app does not silently fall back mid-conversion.

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
├── Dockerfile          # Docker image definition
├── docker-compose.yml  # Docker Compose with GPU support
├── .dockerignore       # Docker build exclusions
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
