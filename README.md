# Media Converter

A self-hosted web application for converting video files between formats and extracting audio from video files. Built with Python/Flask and powered by FFmpeg.

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Installing FFmpeg](#installing-ffmpeg)
- [GPU Acceleration](#gpu-acceleration)
- [Configuration](#configuration)
- [Supported Formats](#supported-formats)
- [Project Structure](#project-structure)
- [License](#license)

---

## Features

- **Video Format Conversion** — Convert between popular video formats (MP4, AVI, MKV, MOV, WMV, FLV, WebM)
- **Audio Extraction** — Extract audio tracks from video files (MP3, AAC, WAV, FLAC, OGG)
- **Social Media Downloader** — Download media from YouTube, Instagram, TikTok, and X/Twitter with mode selection (video/audio), quality presets, and abort support
- **Media Library Page** — Browse `converted/` and `downloads/` in a simple web file browser with selectable multi-file download
- **Resolution Scaling** — Upscale or downscale video (480p, 720p, 1080p, 1440p, 4K) with high-quality Lanczos filtering
- **GPU Acceleration** — Automatically uses hardware encoding (NVIDIA NVENC, AMD AMF, Intel QSV, VA-API) when available, with seamless CPU fallback
- **Real-time Progress** — Live progress bar with speed and ETA during conversion
- **Abort Support** — Cancel in-progress conversions with automatic cleanup
- **No File Size Limits** — Upload files of any size
- **Automatic Cleanup** — Temporary uploads and converted files are automatically deleted after 24 hours
- **Persistent Downloads** — Downloaded media is stored under `downloads/<service>/` and is never auto-deleted
- **Dark/Light Mode** — Modern UI with dark mode as default and easy toggle
- **Docker Ready** — Run with GPU support via Docker Compose in one command
- **Self-Contained** — Runs in a Python virtual environment with minimal dependencies

---

## Quick Start

<details>
<summary><strong>🐳 Docker (recommended)</strong></summary>

```bash
git clone https://github.com/yourusername/media-converter.git
cd media-converter
docker compose up -d
```

Open **http://localhost:5000**. GPU acceleration works automatically if you have the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

**Without GPU support:**

Remove the `deploy` block from `docker-compose.yml`, or run directly:

```bash
docker build -t media-converter .
docker run -d -p 5000:5000 --name media-converter media-converter
```

</details>

<details>
<summary><strong>🪟 Windows</strong></summary>

**Prerequisites:** Python 3.8+ and [FFmpeg](#installing-ffmpeg) on your PATH.

```bash
git clone https://github.com/yourusername/media-converter.git
cd media-converter
setup.bat
```

</details>

<details>
<summary><strong>🐧 Linux / macOS</strong></summary>

**Prerequisites:** Python 3.8+ and [FFmpeg](#installing-ffmpeg) on your PATH.

```bash
git clone https://github.com/yourusername/media-converter.git
cd media-converter
chmod +x setup.sh
./setup.sh
```

</details>

<details>
<summary><strong>⚙️ Manual Setup</strong></summary>

**Prerequisites:** Python 3.8+ and [FFmpeg](#installing-ffmpeg) on your PATH.

```bash
# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies and run
pip install -r requirements.txt
python app.py
```

</details>

Then open your browser to **http://localhost:5000**

Media Library page: **http://localhost:5000/files**

Downloader support: **YouTube, Instagram, TikTok, X/Twitter**

---

## Installing FFmpeg

FFmpeg must be installed and available on your system PATH (not required for Docker — it's included in the image).
yt-dlp is installed automatically via pip from requirements and is used for YouTube, Instagram, TikTok, and X/Twitter downloads.

Public posts generally work best. Some Instagram or X/Twitter links may require authentication or cookies depending on upstream site restrictions.

<details>
<summary><strong>Windows</strong></summary>

```bash
# Via winget
winget install FFmpeg

# Or via Chocolatey
choco install ffmpeg
```

</details>

<details>
<summary><strong>macOS</strong></summary>

```bash
brew install ffmpeg
```

</details>

<details>
<summary><strong>Ubuntu / Debian</strong></summary>

```bash
sudo apt update && sudo apt install ffmpeg
```

</details>

---

## GPU Acceleration

The application automatically detects and uses GPU hardware encoders when available. On startup, the console and the web UI will show whether GPU acceleration is active.

GPU encoding is used for **MP4, MKV, and MOV** output. Other formats and audio extraction use CPU encoding.

<details>
<summary><strong>Supported GPU encoders</strong></summary>

| GPU Vendor | Encoder | Requirements |
|---|---|---|
| NVIDIA | NVENC | NVIDIA GPU + driver 470+, FFmpeg built with `--enable-nvenc` |
| AMD | AMF | AMD GPU + Adrenalin driver, FFmpeg built with `--enable-amf` |
| Intel | QSV | Intel iGPU/dGPU + media driver, FFmpeg built with `--enable-libmfx` or `--enable-libvpl` |
| Linux (generic) | VA-API | VA-API capable GPU + `libva`, FFmpeg built with `--enable-vaapi` |

When upscaling, quality settings are automatically increased (lower QP/CRF, slower presets) and a post-processing filter chain is applied (sharpening + denoising) for the best possible output.

</details>

---

## Configuration

Environment variables can be set in a `.env` file or exported:

| Variable | Default | Description |
|---|---|---|
| `FLASK_HOST` | `0.0.0.0` | Host to bind to |
| `FLASK_PORT` | `5000` | Port to listen on |
| `MAX_CONTENT_LENGTH` | `0` (unlimited) | Max upload size in bytes (0 = no limit) |
| `CLEANUP_HOURS` | `24` | Hours before files are auto-deleted |

Cleanup applies only to `uploads/` and `converted/`. Files in `downloads/` are preserved.

---

## Supported Formats

<details>
<summary><strong>Video output formats</strong></summary>

| Format | Extension |
|---|---|
| MP4 | `.mp4` |
| AVI | `.avi` |
| MKV | `.mkv` |
| MOV | `.mov` |
| WMV | `.wmv` |
| FLV | `.flv` |
| WebM | `.webm` |

</details>

<details>
<summary><strong>Audio output formats</strong></summary>

| Format | Extension |
|---|---|
| MP3 | `.mp3` |
| AAC | `.aac` |
| WAV | `.wav` |
| FLAC | `.flac` |
| OGG | `.ogg` |

</details>

<details>
<summary><strong>Supported input formats</strong></summary>

MP4, AVI, MKV, MOV, WMV, FLV, WebM, M4V, MPEG, MPG, 3GP, OGV, TS, VOB

</details>

---

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
│   ├── index.html      # Main web UI
│   └── files.html      # Simple media library browser
├── uploads/            # Temporary upload storage (auto-created)
├── converted/          # Temporary converted file storage (auto-created)
└── downloads/
    ├── youtube/         # Persistent YouTube downloads (not auto-cleaned)
    ├── instagram/       # Persistent Instagram downloads (not auto-cleaned)
    ├── tiktok/          # Persistent TikTok downloads (not auto-cleaned)
    └── twitter/         # Persistent X/Twitter downloads (not auto-cleaned)
```

---

## License

MIT License
