"""
Media Converter — Self-hosted video conversion and audio extraction tool.
"""

import os
import uuid
import subprocess
import json
import shutil
import re
import threading
import time
import io
import zipfile
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    jsonify,
    abort,
)
from werkzeug.utils import secure_filename
from apscheduler.schedulers.background import BackgroundScheduler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
CONVERTED_FOLDER = BASE_DIR / "converted"
DOWNLOADS_FOLDER = BASE_DIR / "downloads"
YOUTUBE_DOWNLOADS_FOLDER = DOWNLOADS_FOLDER / "youtube"
INSTAGRAM_DOWNLOADS_FOLDER = DOWNLOADS_FOLDER / "instagram"
TIKTOK_DOWNLOADS_FOLDER = DOWNLOADS_FOLDER / "tiktok"
TWITTER_DOWNLOADS_FOLDER = DOWNLOADS_FOLDER / "twitter"
PORNHUB_DOWNLOADS_FOLDER = DOWNLOADS_FOLDER / "pornhub"
SPOTIFY_DOWNLOADS_FOLDER = DOWNLOADS_FOLDER / "spotify"
CLEANUP_HOURS = int(os.environ.get("CLEANUP_HOURS", 24))


def _env_bool(name: str, default: bool = True) -> bool:
    """Parse a boolean-ish environment variable."""
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on", "enabled")


# Feature flags — control which halves of the app are active.
#
# Out of the box both are enabled, so the app behaves exactly as before
# (video conversion + downloader). Set one to false to "split" the app, e.g.
# a public downloader-only deployment sets ENABLE_CONVERTER=false so nobody can
# run heavyweight video conversions on the server.
ENABLE_CONVERTER = _env_bool("ENABLE_CONVERTER", True)
ENABLE_DOWNLOADER = _env_bool("ENABLE_DOWNLOADER", True)

# Guard against a misconfiguration that would leave the app doing nothing:
# if both are disabled, fall back to full functionality.
if not ENABLE_CONVERTER and not ENABLE_DOWNLOADER:
    ENABLE_CONVERTER = True
    ENABLE_DOWNLOADER = True

APP_TITLE = "Media Converter" if ENABLE_CONVERTER else "Media Downloader"

ALLOWED_INPUT_EXTENSIONS = {
    ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm",
    ".m4v", ".mpeg", ".mpg", ".3gp", ".ogv", ".ts", ".vob",
}

VIDEO_OUTPUT_FORMATS = {
    "mp4": {"ext": ".mp4", "label": "MP4"},
    "avi": {"ext": ".avi", "label": "AVI"},
    "mkv": {"ext": ".mkv", "label": "MKV"},
    "mov": {"ext": ".mov", "label": "MOV"},
    "wmv": {"ext": ".wmv", "label": "WMV"},
    "flv": {"ext": ".flv", "label": "FLV"},
    "webm": {"ext": ".webm", "label": "WebM"},
}

AUDIO_OUTPUT_FORMATS = {
    "mp3": {"ext": ".mp3", "label": "MP3"},
    "aac": {"ext": ".aac", "label": "AAC"},
    "wav": {"ext": ".wav", "label": "WAV"},
    "flac": {"ext": ".flac", "label": "FLAC"},
    "ogg": {"ext": ".ogg", "label": "OGG"},
}

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_CONTENT_LENGTH", 0)) or None  # None = unlimited

UPLOAD_FOLDER.mkdir(exist_ok=True)
CONVERTED_FOLDER.mkdir(exist_ok=True)
YOUTUBE_DOWNLOADS_FOLDER.mkdir(parents=True, exist_ok=True)
INSTAGRAM_DOWNLOADS_FOLDER.mkdir(parents=True, exist_ok=True)
TIKTOK_DOWNLOADS_FOLDER.mkdir(parents=True, exist_ok=True)
TWITTER_DOWNLOADS_FOLDER.mkdir(parents=True, exist_ok=True)
PORNHUB_DOWNLOADS_FOLDER.mkdir(parents=True, exist_ok=True)
SPOTIFY_DOWNLOADS_FOLDER.mkdir(parents=True, exist_ok=True)

# Active conversion jobs: file_id -> job dict
_active_jobs: dict[str, dict] = {}
_youtube_jobs: dict[str, dict] = {}

# Cap the in-memory job maps so long-running instances don't leak memory.
# Finished jobs (complete/error/aborted) are discarded oldest-first once the
# cap is exceeded; in-progress jobs are always kept.
_JOB_RETENTION = 500


def _prune_jobs(jobs: dict) -> None:
    """Bound an in-memory job map by discarding the oldest finished jobs."""
    if len(jobs) <= _JOB_RETENTION:
        return
    finished = [
        (job_id, job)
        for job_id, job in jobs.items()
        if job.get("status") in ("complete", "error", "aborted")
    ]
    finished.sort(key=lambda item: item[1].get("created", 0))
    for job_id, _ in finished:
        if len(jobs) <= _JOB_RETENTION:
            break
        jobs.pop(job_id, None)

YOUTUBE_QUALITY_OPTIONS = {
    "best": "Best",
    "1080": "1080p",
    "720": "720p",
    "480": "480p",
}

YOUTUBE_AUDIO_FORMATS = {"mp3", "aac"}

SUPPORTED_DOWNLOAD_SERVICES = {
    "youtube": {
        "label": "YouTube",
        "folder": YOUTUBE_DOWNLOADS_FOLDER,
        "domains": ("youtube.com", "youtu.be"),
    },
    "instagram": {
        "label": "Instagram",
        "folder": INSTAGRAM_DOWNLOADS_FOLDER,
        "domains": ("instagram.com",),
    },
    "tiktok": {
        "label": "TikTok",
        "folder": TIKTOK_DOWNLOADS_FOLDER,
        "domains": ("tiktok.com", "vt.tiktok.com", "vm.tiktok.com"),
    },
    "twitter": {
        "label": "X/Twitter",
        "folder": TWITTER_DOWNLOADS_FOLDER,
        "domains": ("twitter.com", "x.com"),
    },
    "pornhub": {
        "label": "PornHub",
        "folder": PORNHUB_DOWNLOADS_FOLDER,
        "domains": ("pornhub.com",),
        "hidden": True,
    },
    "spotify": {
        "label": "Spotify",
        "folder": SPOTIFY_DOWNLOADS_FOLDER,
        "domains": ("open.spotify.com", "spotify.com"),
        # Spotify streams are DRM-protected; spotdl fetches the matching
        # audio from YouTube, so downloads are always audio-only.
        "audio_only": True,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    """Check if ffmpeg is accessible on the system PATH."""
    return shutil.which("ffmpeg") is not None


def _ytdlp_available() -> bool:
    """Check if yt-dlp is accessible on the system PATH."""
    return shutil.which("yt-dlp") is not None


def _spotdl_available() -> bool:
    """Check if spotdl is accessible on the system PATH."""
    return shutil.which("spotdl") is not None


def _is_supported_youtube_url(url: str) -> bool:
    """Basic validation for YouTube URLs."""
    pattern = re.compile(r"^(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+$", re.IGNORECASE)
    return bool(pattern.match(url.strip()))


def _detect_download_service(url: str) -> str | None:
    """Map a supported media URL to its download service key."""
    cleaned = url.strip()
    if not cleaned:
        return None

    if not re.match(r"^https?://", cleaned, re.IGNORECASE):
        cleaned = f"https://{cleaned}"

    match = re.match(r"^https?://([^/]+)", cleaned, re.IGNORECASE)
    if not match:
        return None

    host = match.group(1).lower()
    if host.startswith("www."):
        host = host[4:]
    if host.startswith("m."):
        host = host[2:]
    if host.startswith("mobile."):
        host = host[7:]

    for service, info in SUPPORTED_DOWNLOAD_SERVICES.items():
        domains = info["domains"]
        if any(host == domain or host.endswith(f".{domain}") for domain in domains):
            return service

    return None


def _is_supported_download_url(url: str) -> bool:
    """Basic validation for supported yt-dlp-backed services."""
    return _detect_download_service(url) is not None


def _safe_root(root: str) -> Path | None:
    """Map API root key to a filesystem directory."""
    mapping = {
        "converted": CONVERTED_FOLDER,
        "downloads": DOWNLOADS_FOLDER,
    }
    return mapping.get(root)


def _safe_resolve_path(base: Path, rel_path: str) -> Path | None:
    """Resolve a user-supplied relative path safely under base directory."""
    if not rel_path:
        return None
    candidate = (base / rel_path).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    return candidate


def _list_files_recursive(base: Path) -> list[dict]:
    """Return file metadata recursively under base."""
    items: list[dict] = []
    if not base.exists():
        return items

    for file_path in sorted([p for p in base.rglob("*") if p.is_file()], key=lambda p: str(p).lower()):
        rel = file_path.relative_to(base).as_posix()
        items.append(
            {
                "relative_path": rel,
                "name": file_path.name,
                "size": _human_size(file_path.stat().st_size),
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
        )
    return items


# ---------------------------------------------------------------------------
# GPU Encoder Detection
# ---------------------------------------------------------------------------

# Mapping: encoder name -> (test_encoder, friendly label)
_GPU_ENCODERS = {
    "nvenc":  {"h264": "h264_nvenc",  "hevc": "hevc_nvenc",  "label": "NVIDIA NVENC"},
    "amf":    {"h264": "h264_amf",   "hevc": "hevc_amf",   "label": "AMD AMF"},
    "qsv":    {"h264": "h264_qsv",   "hevc": "hevc_qsv",   "label": "Intel QSV"},
    "vaapi":  {"h264": "h264_vaapi", "hevc": "hevc_vaapi", "label": "VA-API"},
}

_detected_gpu: dict | None = None  # cached result


def _detect_gpu_encoder() -> dict:
    """Detect available GPU hardware encoders by probing FFmpeg.

    Returns a dict like:
        {"name": "nvenc", "label": "NVIDIA NVENC", "h264": "h264_nvenc", "hevc": "hevc_nvenc"}
    or an empty dict if no GPU encoder is available.
    """
    global _detected_gpu
    if _detected_gpu is not None:
        return _detected_gpu

    if not _ffmpeg_available():
        _detected_gpu = {}
        return _detected_gpu

    # Query which encoders FFmpeg was compiled with
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        available_encoders = result.stdout if result.returncode == 0 else ""
    except Exception:
        _detected_gpu = {}
        return _detected_gpu

    # Check each GPU family in priority order (NVENC > AMF > QSV > VAAPI)
    for name, info in _GPU_ENCODERS.items():
        h264_enc = info["h264"]
        if h264_enc in available_encoders:
            # Verify the encoder actually works (driver present, device accessible)
            try:
                test = subprocess.run(
                    [
                        "ffmpeg", "-hide_banner", "-loglevel", "error",
                        "-f", "lavfi", "-i", "nullsrc=s=256x256:d=1",
                        "-c:v", h264_enc, "-frames:v", "1",
                        "-f", "null", "-",
                    ],
                    capture_output=True, text=True, timeout=15,
                )
                if test.returncode == 0:
                    _detected_gpu = {
                        "name": name,
                        "label": info["label"],
                        "h264": info["h264"],
                        "hevc": info["hevc"],
                    }
                    return _detected_gpu
            except Exception:
                continue

    _detected_gpu = {}
    return _detected_gpu


def _probe_file(filepath: Path) -> dict:
    """Use ffprobe to get file metadata."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                str(filepath),
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception:
        pass
    return {}


def _human_size(size_bytes: int) -> str:
    """Convert bytes to a human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(size_bytes) < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _human_duration(seconds: float) -> str:
    """Convert seconds to HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _parse_size_to_bytes(size_text: str) -> int | None:
    """Parse a size string like '1.23GiB' or '950MiB' into bytes."""
    if not size_text:
        return None

    match = re.match(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([KMGTPE]?i?B)\s*$", size_text, re.IGNORECASE)
    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2).upper()

    factors = {
        "B": 1,
        "KB": 1000,
        "MB": 1000 ** 2,
        "GB": 1000 ** 3,
        "TB": 1000 ** 4,
        "PB": 1000 ** 5,
        "EB": 1000 ** 6,
        "KIB": 1024,
        "MIB": 1024 ** 2,
        "GIB": 1024 ** 3,
        "TIB": 1024 ** 4,
        "PIB": 1024 ** 5,
        "EIB": 1024 ** 6,
    }

    factor = factors.get(unit)
    if factor is None:
        return None

    return int(value * factor)


def allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_INPUT_EXTENSIONS


def cleanup_old_files():
    """Delete files older than CLEANUP_HOURS from uploads/ and converted/.

    The downloads/ hierarchy is intentionally not cleaned automatically.
    """
    cutoff = datetime.now() - timedelta(hours=CLEANUP_HOURS)
    for folder in (UPLOAD_FOLDER, CONVERTED_FOLDER):
        if not folder.exists():
            continue
        for item in folder.iterdir():
            if item.is_file():
                mtime = datetime.fromtimestamp(item.stat().st_mtime)
                if mtime < cutoff:
                    try:
                        item.unlink()
                        app.logger.info(f"Cleaned up: {item.name}")
                    except OSError:
                        pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _require_feature(is_enabled, message: str):
    """Return a decorator that blocks a route when a feature flag is off."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not is_enabled():
                return jsonify({"error": message}), 403
            return func(*args, **kwargs)
        return wrapper
    return decorator


_require_converter = _require_feature(
    lambda: ENABLE_CONVERTER, "Video conversion is disabled on this server."
)
_require_downloader = _require_feature(
    lambda: ENABLE_DOWNLOADER, "Media downloading is disabled on this server."
)


@app.route("/")
def index():
    gpu = _detect_gpu_encoder()
    return render_template(
        "index.html",
        video_formats=VIDEO_OUTPUT_FORMATS,
        audio_formats=AUDIO_OUTPUT_FORMATS,
        ffmpeg_ok=_ffmpeg_available(),
        ytdlp_ok=_ytdlp_available(),
        youtube_quality_options=YOUTUBE_QUALITY_OPTIONS,
        supported_download_services=[info["label"] for info in SUPPORTED_DOWNLOAD_SERVICES.values() if not info.get("hidden")],
        gpu_info=gpu,
        enable_converter=ENABLE_CONVERTER,
        enable_downloader=ENABLE_DOWNLOADER,
        app_title=APP_TITLE,
    )


@app.route("/upload", methods=["POST"])
@_require_converter
def upload():
    """Handle file upload and return file metadata."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided."}), 400

    file = request.files["file"]
    if file.filename == "" or file.filename is None:
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(sorted(ALLOWED_INPUT_EXTENSIONS))}"}), 400

    # Generate a unique ID for this upload
    file_id = uuid.uuid4().hex
    original_name = secure_filename(file.filename)
    ext = Path(original_name).suffix.lower()
    stored_name = f"{file_id}{ext}"
    filepath = UPLOAD_FOLDER / stored_name

    file.save(str(filepath))

    # Probe the file for metadata
    probe = _probe_file(filepath)
    file_size = filepath.stat().st_size

    duration = None
    video_codec = None
    audio_codec = None
    resolution = None

    if probe:
        fmt = probe.get("format", {})
        duration_str = fmt.get("duration")
        if duration_str:
            try:
                duration = float(duration_str)
            except (ValueError, TypeError):
                pass

        for stream in probe.get("streams", []):
            if stream.get("codec_type") == "video" and not video_codec:
                video_codec = stream.get("codec_name", "unknown")
                w = stream.get("width")
                h = stream.get("height")
                if w and h:
                    resolution = f"{w}x{h}"
            elif stream.get("codec_type") == "audio" and not audio_codec:
                audio_codec = stream.get("codec_name", "unknown")

    return jsonify({
        "file_id": file_id,
        "original_name": original_name,
        "size": _human_size(file_size),
        "duration": _human_duration(duration) if duration else "Unknown",
        "duration_seconds": duration,
        "video_codec": video_codec or "N/A",
        "audio_codec": audio_codec or "N/A",
        "resolution": resolution or "N/A",
        "width": int(resolution.split("x")[0]) if resolution else None,
        "height": int(resolution.split("x")[1]) if resolution else None,
    })


@app.route("/convert", methods=["POST"])
@_require_converter
def convert():
    """Start conversion of an uploaded file (non-blocking)."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request."}), 400

    file_id = data.get("file_id")
    output_format = data.get("format", "").lower()
    mode = data.get("mode", "video")  # "video", "audio", or "gif"
    total_duration = data.get("duration_seconds")  # seconds, from upload probe
    target_resolution = data.get("resolution")  # e.g. "1920x1080" or null for original

    # GIF-specific options (only used when mode == "gif")
    gif_fps = data.get("gif_fps")
    gif_width = data.get("gif_width")  # target width in px, or null for original
    gif_start = data.get("gif_start")  # trim start in seconds, or null
    gif_duration = data.get("gif_duration")  # trim length in seconds, or null

    # Determine if we are upscaling (need higher quality settings)
    is_upscaling = False
    if target_resolution and mode not in ("audio", "gif"):
        try:
            _tw, _th = target_resolution.split("x")
            # Compare target height to source — we don't know source here yet,
            # but the client only sends a resolution when it differs from original
            is_upscaling = True  # will be refined below after finding source
        except (ValueError, AttributeError):
            pass

    if not file_id or (not output_format and mode != "gif"):
        return jsonify({"error": "Missing file_id or format."}), 400

    # Reject if a conversion is already running for this file
    if file_id in _active_jobs and _active_jobs[file_id]["status"] == "converting":
        return jsonify({"error": "A conversion is already in progress for this file."}), 409

    # Find the uploaded file
    source_file = None
    for f in UPLOAD_FOLDER.iterdir():
        if f.stem == file_id:
            source_file = f
            break

    if source_file is None or not source_file.exists():
        return jsonify({"error": "Upload not found. It may have expired."}), 404

    # Refine upscaling detection by probing source resolution
    if target_resolution and mode not in ("audio", "gif"):
        probe_data = _probe_file(source_file)
        src_h = None
        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "video":
                src_h = stream.get("height")
                break
        if src_h:
            try:
                target_h = int(target_resolution.split("x")[1])
                is_upscaling = target_h > src_h
            except (ValueError, IndexError):
                is_upscaling = False
        else:
            is_upscaling = False

    # Determine output settings
    if mode == "audio":
        if output_format not in AUDIO_OUTPUT_FORMATS:
            return jsonify({"error": f"Unsupported audio format: {output_format}"}), 400
        out_ext = AUDIO_OUTPUT_FORMATS[output_format]["ext"]
    elif mode == "gif":
        out_ext = ".gif"
    else:
        if output_format not in VIDEO_OUTPUT_FORMATS:
            return jsonify({"error": f"Unsupported video format: {output_format}"}), 400
        out_ext = VIDEO_OUTPUT_FORMATS[output_format]["ext"]

    output_name = f"{file_id}_converted{out_ext}"
    output_path = CONVERTED_FOLDER / output_name

    # Detect GPU encoder
    gpu = _detect_gpu_encoder()
    hw_accel_used = False

    # Build FFmpeg command
    cmd = ["ffmpeg", "-y"]

    # Add hardware-accelerated decoding if GPU is available.
    # Skipped for audio (no video) and GIF (filter graph runs on CPU frames).
    if gpu and mode not in ("audio", "gif"):
        if gpu["name"] == "nvenc":
            cmd.extend(["-hwaccel", "cuda"])
        elif gpu["name"] == "qsv":
            cmd.extend(["-hwaccel", "qsv"])
        elif gpu["name"] == "vaapi":
            cmd.extend(["-hwaccel", "vaapi",
                        "-hwaccel_device", "/dev/dri/renderD128"])

    # For GIF, an optional trim start is applied before the input for fast seeking.
    gif_trim_length = None
    if mode == "gif":
        try:
            start_val = float(gif_start) if gif_start is not None else 0.0
        except (ValueError, TypeError):
            start_val = 0.0
        if start_val > 0:
            cmd.extend(["-ss", str(start_val)])

    cmd.extend(["-i", str(source_file)])

    if mode == "gif":
        # Optional trim length (applied after input)
        try:
            length_val = float(gif_duration) if gif_duration is not None else 0.0
        except (ValueError, TypeError):
            length_val = 0.0
        if length_val > 0:
            cmd.extend(["-t", str(length_val)])
            gif_trim_length = length_val

        # Frame rate (clamped to a sane range)
        try:
            fps_val = int(gif_fps) if gif_fps is not None else 15
        except (ValueError, TypeError):
            fps_val = 15
        fps_val = max(1, min(fps_val, 50))

        # Target width (height auto to preserve aspect ratio). -1 keeps original.
        try:
            width_val = int(gif_width) if gif_width is not None else -1
        except (ValueError, TypeError):
            width_val = -1
        scale_w = width_val if width_val and width_val > 0 else -1

        # High-quality GIF via per-frame palette generation and dithering.
        vf = (
            f"fps={fps_val},"
            f"scale={scale_w}:-1:flags=lanczos,"
            f"split[s0][s1];[s0]palettegen=stats_mode=diff[p];"
            f"[s1][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
        )
        cmd.extend(["-vf", vf, "-loop", "0"])
    elif mode == "audio":

        # Extract audio only — GPU not used for audio
        cmd.extend(["-vn"])  # no video
        if output_format == "mp3":
            cmd.extend(["-codec:a", "libmp3lame", "-q:a", "2"])
        elif output_format == "aac":
            cmd.extend(["-codec:a", "aac", "-b:a", "192k"])
        elif output_format == "wav":
            cmd.extend(["-codec:a", "pcm_s16le"])
        elif output_format == "flac":
            cmd.extend(["-codec:a", "flac"])
        elif output_format == "ogg":
            cmd.extend(["-codec:a", "libvorbis", "-q:a", "5"])
    else:
        # Video conversion — prefer GPU encoder when available
        # Use higher quality settings when upscaling
        qp_val = "16" if is_upscaling else "20"
        crf_val = "18" if is_upscaling else "23"

        if output_format in ("mp4", "mkv", "mov") and gpu:
            enc = gpu["h264"]
            nvenc_preset = "p7" if is_upscaling else "p5"
            if gpu["name"] == "nvenc":
                cmd.extend(["-codec:v", enc, "-preset", nvenc_preset, "-tune", "hq",
                            "-rc", "constqp", "-qp", qp_val,
                            "-b:v", "0", "-profile:v", "high"])
            elif gpu["name"] == "amf":
                cmd.extend(["-codec:v", enc, "-quality", "quality",
                            "-rc", "cqp", "-qp_i", qp_val, "-qp_p", qp_val,
                            "-qp_b", str(int(qp_val) + 2), "-profile:v", "high"])
            elif gpu["name"] == "qsv":
                cmd.extend(["-codec:v", enc, "-preset", "veryslow" if is_upscaling else "medium",
                            "-global_quality", qp_val, "-profile:v", "high"])
            elif gpu["name"] == "vaapi":
                cmd.extend(["-codec:v", enc, "-qp", qp_val,
                            "-profile:v", "high"])
            cmd.extend(["-codec:a", "aac", "-b:a", "192k"])
            if output_format == "mp4":
                cmd.extend(["-movflags", "+faststart"])
            hw_accel_used = True
        elif output_format == "mp4":
            preset = "slow" if is_upscaling else "medium"
            cmd.extend(["-codec:v", "libx264", "-preset", preset, "-crf", crf_val,
                        "-codec:a", "aac", "-b:a", "192k", "-movflags", "+faststart"])
        elif output_format == "webm":
            webm_crf = "24" if is_upscaling else "30"
            cmd.extend(["-codec:v", "libvpx-vp9", "-crf", webm_crf, "-b:v", "0",
                        "-codec:a", "libopus", "-b:a", "128k"])
        elif output_format == "mkv":
            preset = "slow" if is_upscaling else "medium"
            cmd.extend(["-codec:v", "libx264", "-preset", preset, "-crf", crf_val,
                        "-codec:a", "aac", "-b:a", "192k"])
        elif output_format == "avi":
            cmd.extend(["-codec:v", "mpeg4", "-q:v", "5",
                        "-codec:a", "libmp3lame", "-q:a", "4"])
        elif output_format == "mov":
            cmd.extend(["-codec:v", "libx264", "-preset", "medium", "-crf", "23",
                        "-codec:a", "aac", "-b:a", "192k"])
        elif output_format == "wmv":
            cmd.extend(["-codec:v", "wmv2", "-b:v", "2M",
                        "-codec:a", "wmav2", "-b:a", "192k"])
        elif output_format == "flv":
            cmd.extend(["-codec:v", "flv1", "-b:v", "2M",
                        "-codec:a", "libmp3lame", "-q:a", "4"])

    # Apply resolution scaling if requested (video mode only)
    if mode not in ("audio", "gif") and target_resolution:
        try:
            tw, th = target_resolution.split("x")
            tw, th = int(tw), int(th)
            if is_upscaling:
                # High-quality upscale filter chain:
                # 1. Scale with Lanczos + accurate rounding + full chroma interpolation
                # 2. Adaptive sharpening via unsharp mask to restore detail
                #    (luma: 5x5 kernel, strength 0.5 — adds crispness without ringing)
                # 3. Light deband to reduce color banding from upscale
                vf = (f"scale={tw}:{th}:flags=lanczos+accurate_rnd+full_chroma_int+full_chroma_inp,"
                      f"unsharp=5:5:0.5:5:5:0.3,"
                      f"hqdn3d=1:1:3:3")
            else:
                # Downscale or same — Lanczos is sufficient
                vf = f"scale={tw}:{th}:flags=lanczos+accurate_rnd"
            cmd.extend(["-vf", vf])
        except (ValueError, AttributeError):
            pass  # ignore invalid resolution, use original

    # Add progress output flag (machine-readable to stdout)
    cmd.extend(["-progress", "pipe:1", "-nostats"])
    cmd.append(str(output_path))

    # Initialize job tracking
    job = {
        "status": "converting",
        "percent": 0,
        "speed": "",
        "eta": "",
        "error": None,
        "output_path": str(output_path),
        "output_name": output_name,
        "hw_accel_used": hw_accel_used,
        "gpu_label": gpu.get("label", "") if hw_accel_used else "",
        "process": None,
        "duration": (gif_trim_length if mode == "gif" and gif_trim_length else total_duration) or 0,
        "created": time.time(),
    }
    _active_jobs[file_id] = job
    _prune_jobs(_active_jobs)

    def _run_conversion():
        """Run FFmpeg in the background, parsing progress output."""
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            job["process"] = proc

            # Drain stderr concurrently to avoid a pipe-buffer deadlock. FFmpeg
            # writes progress to stdout (pipe:1) and its logs to stderr. On
            # Windows the anonymous pipe buffer is small (~4 KB); if we only read
            # stdout, ffmpeg can block writing stderr while we block reading
            # stdout, hanging the conversion forever. Collect stderr in a
            # background thread so it never fills up, and keep it for error output.
            stderr_chunks: list[str] = []

            def _drain_stderr(pipe):
                try:
                    for err_line in pipe:
                        stderr_chunks.append(err_line)
                except Exception:
                    pass

            stderr_thread = None
            if proc.stderr is not None:
                stderr_thread = threading.Thread(
                    target=_drain_stderr, args=(proc.stderr,), daemon=True
                )
                stderr_thread.start()

            dur = job["duration"]

            # Read progress from stdout line by line
            for line in proc.stdout:
                line = line.strip()
                if job["status"] == "aborted":
                    break

                if line.startswith("out_time_us="):
                    try:
                        us = int(line.split("=", 1)[1])
                        current_secs = us / 1_000_000
                        if dur and dur > 0:
                            pct = min(int((current_secs / dur) * 100), 99)
                            job["percent"] = pct
                    except (ValueError, ZeroDivisionError):
                        pass
                elif line.startswith("speed="):
                    spd = line.split("=", 1)[1].strip()
                    job["speed"] = spd
                    # Estimate ETA
                    if dur and dur > 0 and spd and spd != "N/A":
                        try:
                            spd_num = float(spd.rstrip("x"))
                            if spd_num > 0:
                                current_secs = (job["percent"] / 100) * dur
                                remaining = (dur - current_secs) / spd_num
                                job["eta"] = _human_duration(remaining)
                        except (ValueError, ZeroDivisionError):
                            pass
                elif line.startswith("progress=end"):
                    break

            proc.wait(timeout=7200)

            if job["status"] == "aborted":
                return  # already handled by abort endpoint

            if stderr_thread is not None:
                stderr_thread.join(timeout=5)

            if proc.returncode != 0:
                stderr_out = "".join(stderr_chunks)
                error_msg = stderr_out.strip().split("\n")[-1] if stderr_out.strip() else "Conversion failed."
                job["status"] = "error"
                job["error"] = f"FFmpeg error: {error_msg}"
            else:
                job["status"] = "complete"
                job["percent"] = 100
                job["output_size"] = _human_size(Path(job["output_path"]).stat().st_size)

        except subprocess.TimeoutExpired:
            job["status"] = "error"
            job["error"] = "Conversion timed out (exceeded 2 hours)."
            try:
                proc.kill()
            except Exception:
                pass
        except FileNotFoundError:
            job["status"] = "error"
            job["error"] = "FFmpeg is not installed or not found on PATH."
        except Exception as e:
            job["status"] = "error"
            job["error"] = f"Unexpected error: {str(e)}"

    thread = threading.Thread(target=_run_conversion, daemon=True)
    thread.start()

    return jsonify({"status": "started", "file_id": file_id})


@app.route("/progress/<file_id>")
@_require_converter
def progress(file_id):
    """Poll conversion progress for a given file."""
    job = _active_jobs.get(file_id)
    if not job:
        return jsonify({"status": "not_found"}), 404

    resp = {
        "status": job["status"],
        "percent": job["percent"],
        "speed": job["speed"],
        "eta": job["eta"],
    }

    if job["status"] == "complete":
        resp["download_id"] = job["output_name"]
        resp["output_size"] = job.get("output_size", "")
        resp["gpu_used"] = job["hw_accel_used"]
        resp["gpu_label"] = job["gpu_label"]
    elif job["status"] == "error":
        resp["error"] = job["error"]
    elif job["status"] == "aborted":
        resp["error"] = "Conversion was aborted."

    return jsonify(resp)


@app.route("/abort/<file_id>", methods=["POST"])
@_require_converter
def abort_conversion(file_id):
    """Abort an in-progress conversion and clean up."""
    job = _active_jobs.get(file_id)
    if not job:
        return jsonify({"error": "No active conversion found."}), 404

    if job["status"] != "converting":
        return jsonify({"error": "Conversion is not in progress."}), 400

    job["status"] = "aborted"

    # Kill the FFmpeg process
    proc = job.get("process")
    if proc:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass

    # Delete partial output file
    try:
        output = Path(job["output_path"])
        if output.exists():
            output.unlink()
    except OSError:
        pass

    return jsonify({"status": "aborted"})


@app.route("/download/<download_id>")
@_require_converter
def download(download_id):
    """Download a converted file."""
    # Sanitize the download_id
    safe_name = secure_filename(download_id)
    filepath = CONVERTED_FOLDER / safe_name

    if not filepath.exists():
        abort(404)

    return send_file(
        str(filepath),
        as_attachment=True,
        download_name=safe_name,
    )


@app.route("/media/download", methods=["POST"])
@app.route("/youtube/download", methods=["POST"])
@_require_downloader
def youtube_download():
    """Start a background media download job for supported services."""
    data = request.get_json() or {}
    url = (data.get("url") or "").strip()
    mode = (data.get("mode") or "video").strip().lower()
    quality = (data.get("quality") or "best").strip().lower()
    audio_format = (data.get("audio_format") or "mp3").strip().lower()

    if not url:
        return jsonify({"error": "Missing media URL."}), 400

    service = _detect_download_service(url)
    if not service:
        supported = ", ".join(info["label"] for info in SUPPORTED_DOWNLOAD_SERVICES.values() if not info.get("hidden"))
        return jsonify({"error": f"Please provide a supported URL. Supported services: {supported}."}), 400

    service_info = SUPPORTED_DOWNLOAD_SERVICES[service]
    output_folder = service_info["folder"]
    is_spotify = service_info.get("audio_only", False)

    if is_spotify:
        if not _spotdl_available():
            return jsonify({"error": "spotdl is not installed or not found on PATH."}), 500
        # Spotify streams are DRM-protected; spotdl matches and fetches the
        # audio from YouTube, so these downloads are always audio-only.
        mode = "audio"
    elif not _ytdlp_available():
        return jsonify({"error": "yt-dlp is not installed or not found on PATH."}), 500

    if mode not in {"video", "audio"}:
        return jsonify({"error": "Unsupported mode. Use 'video' or 'audio'."}), 400

    if quality not in YOUTUBE_QUALITY_OPTIONS:
        return jsonify({"error": "Unsupported quality selection."}), 400

    if mode == "audio" and audio_format not in YOUTUBE_AUDIO_FORMATS:
        return jsonify({"error": "Unsupported audio format. Use mp3 or aac."}), 400

    # spotdl stores AAC audio in an m4a container.
    spotdl_format = "m4a" if audio_format == "aac" else audio_format
    download_ext = "mp4" if mode == "video" else (spotdl_format if is_spotify else audio_format)

    job_id = uuid.uuid4().hex

    job = {
        "status": "downloading",
        "percent": 0,
        "speed": "",
        "eta": "",
        "downloaded_size": "",
        "total_size": "",
        "error": None,
        "output_name": "",
        "output_size": "",
        "process": None,
        "service": service,
        "service_label": service_info["label"],
        "mode": mode,
        "quality": quality,
        "audio_format": audio_format,
        "created": time.time(),
    }
    _youtube_jobs[job_id] = job
    _prune_jobs(_youtube_jobs)

    if is_spotify:
        # spotdl reads Spotify metadata and downloads the matching audio.
        # "{output-ext}" is a spotdl template variable expanded at runtime.
        spotdl_output = str(output_folder / (job_id + ".{output-ext}"))
        cmd = [
            "spotdl",
            "download",
            url,
            "--format",
            spotdl_format,
            "--output",
            spotdl_output,
            "--print-errors",
        ]
    else:
        output_template = str(output_folder / f"{job_id}.%(ext)s")
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--newline",
            "--progress",
            "--restrict-filenames",
            "-o",
            output_template,
        ]

        if mode == "video":
            quality_formats = {
                "best": "bv*+ba/b",
                "1080": "bv*[height<=1080]+ba/b[height<=1080]",
                "720": "bv*[height<=720]+ba/b[height<=720]",
                "480": "bv*[height<=480]+ba/b[height<=480]",
            }
            cmd.extend([
                "-f",
                quality_formats[quality],
                # Prefer H.264 video and AAC audio for broad playback compatibility.
                # Some sites (e.g. TikTok) default to HEVC/H.265, which many players
                # cannot decode correctly and can result in missing audio/video.
                "-S",
                "vcodec:h264,acodec:aac",
                "--merge-output-format",
                "mp4",
            ])
        else:
            cmd.extend([
                "-f",
                "bestaudio/best",
                "-x",
                "--audio-format",
                audio_format,
                "--audio-quality",
                "0",
            ])

        cmd.append(url)

    progress_re = re.compile(r"\[download\]\s+(\d+(?:\.\d+)?)%")
    speed_re = re.compile(r"at\s+([^\s]+)")
    eta_re = re.compile(r"ETA\s+([0-9:]+)")
    total_size_re = re.compile(r"of\s+~?\s*([0-9]+(?:\.[0-9]+)?\s*[KMGTPE]?i?B)", re.IGNORECASE)

    def _run_download():
        """Execute the download tool (yt-dlp or spotdl) and parse progress."""
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            job["process"] = proc
            last_error_line = ""

            if proc.stdout:
                for raw_line in proc.stdout:
                    line = raw_line.strip()
                    if not line:
                        continue

                    if is_spotify:
                        # spotdl does not emit a byte-level percentage, so we
                        # advance a coarse, phase-based progress indicator.
                        low = line.lower()
                        if "processing query" in low:
                            job["percent"] = max(job["percent"], 10)
                        elif line.startswith("Downloaded ") or low.startswith('downloaded "'):
                            job["percent"] = 95
                        elif "no results found" in low:
                            last_error_line = "No matching audio was found for this Spotify track."
                        elif "error" in low:
                            last_error_line = line
                        continue

                    progress_match = progress_re.search(line)
                    if progress_match:
                        try:
                            pct = int(float(progress_match.group(1)))
                            job["percent"] = max(0, min(pct, 99))
                        except (ValueError, TypeError):
                            pass

                    total_size_match = total_size_re.search(line)
                    if total_size_match:
                        total_text = total_size_match.group(1).replace(" ", "")
                        total_bytes = _parse_size_to_bytes(total_text)
                        if total_bytes and total_bytes > 0:
                            job["total_size"] = _human_size(total_bytes)
                            downloaded_bytes = int((job["percent"] / 100) * total_bytes)
                            job["downloaded_size"] = _human_size(downloaded_bytes)

                    speed_match = speed_re.search(line)
                    if speed_match:
                        job["speed"] = speed_match.group(1)

                    eta_match = eta_re.search(line)
                    if eta_match:
                        job["eta"] = eta_match.group(1)

                    if "ERROR:" in line:
                        last_error_line = line

            proc.wait(timeout=7200)

            if job["status"] == "aborted":
                return

            if proc.returncode != 0:
                job["status"] = "error"
                job["error"] = last_error_line or f"{job['service_label']} download failed."
                return

            artifacts = sorted(
                [p for p in output_folder.glob(f"{job_id}*") if p.is_file()],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not artifacts:
                job["status"] = "error"
                job["error"] = last_error_line or "Download completed but output file was not found."
                return

            expected_ext = f".{download_ext}"
            expected_name = f"{job_id}{expected_ext}"

            output_path = next(
                (p for p in artifacts if p.name == expected_name),
                None,
            )

            if output_path is None:
                output_path = next(
                    (p for p in artifacts if "-temp" not in p.name and not p.name.endswith(".part")),
                    None,
                )

            if output_path is None:
                output_path = artifacts[0]

            # Keep only the final artifact for this job and remove temporary side files.
            for artifact in artifacts:
                if artifact == output_path:
                    continue
                try:
                    artifact.unlink()
                except OSError:
                    pass

            job["status"] = "complete"
            job["percent"] = 100
            job["output_name"] = output_path.name
            job["output_size"] = _human_size(output_path.stat().st_size)
            job["downloaded_size"] = job["output_size"]
            job["total_size"] = job["output_size"]

        except subprocess.TimeoutExpired:
            job["status"] = "error"
            job["error"] = "Download timed out (exceeded 2 hours)."
            try:
                proc.kill()
            except Exception:
                pass
        except FileNotFoundError:
            job["status"] = "error"
            tool = "spotdl" if is_spotify else "yt-dlp"
            job["error"] = f"{tool} is not installed or not found on PATH."
        except Exception as e:
            job["status"] = "error"
            job["error"] = f"Unexpected error: {str(e)}"

    thread = threading.Thread(target=_run_download, daemon=True)
    thread.start()

    return jsonify({"status": "started", "job_id": job_id})


@app.route("/media/progress/<job_id>")
@app.route("/youtube/progress/<job_id>")
@_require_downloader
def youtube_progress(job_id):
    """Poll media download progress for a given job."""
    job = _youtube_jobs.get(job_id)
    if not job:
        return jsonify({"status": "not_found"}), 404

    resp = {
        "status": job["status"],
        "percent": job["percent"],
        "speed": job["speed"],
        "eta": job["eta"],
        "downloaded_size": job.get("downloaded_size", ""),
        "total_size": job.get("total_size", ""),
    }

    if job["status"] == "complete":
        resp["download_id"] = job["output_name"]
        resp["download_root"] = "downloads"
        resp["download_path"] = f"{job['service']}/{job['output_name']}"
        resp["output_size"] = job["output_size"]
        resp["service"] = job["service"]
        resp["service_label"] = job["service_label"]
    elif job["status"] == "error":
        resp["error"] = job["error"]
    elif job["status"] == "aborted":
        resp["error"] = f"{job['service_label']} download was aborted."

    return jsonify(resp)


@app.route("/media/abort/<job_id>", methods=["POST"])
@app.route("/youtube/abort/<job_id>", methods=["POST"])
@_require_downloader
def youtube_abort(job_id):
    """Abort an in-progress media download and clean up partial files."""
    job = _youtube_jobs.get(job_id)
    if not job:
        return jsonify({"error": "No active media download found."}), 404

    if job["status"] != "downloading":
        return jsonify({"error": "Media download is not in progress."}), 400

    job["status"] = "aborted"
    proc = job.get("process")
    if proc:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass

    output_folder = SUPPORTED_DOWNLOAD_SERVICES[job["service"]]["folder"]
    for partial in output_folder.glob(f"{job_id}.*"):
        try:
            partial.unlink()
        except OSError:
            pass

    return jsonify({"status": "aborted"})


@app.route("/files")
def files_page():
    """Simple web file browser for converted and downloaded files."""
    converted_files = _list_files_recursive(CONVERTED_FOLDER)
    downloaded_files = _list_files_recursive(DOWNLOADS_FOLDER)
    return render_template(
        "files.html",
        converted_files=converted_files,
        downloaded_files=downloaded_files,
        enable_converter=ENABLE_CONVERTER,
        enable_downloader=ENABLE_DOWNLOADER,
        app_title=APP_TITLE,
    )


@app.route("/files/download")
def files_download_single():
    """Download one file from converted/ or downloads/."""
    root = request.args.get("root", "").strip().lower()
    rel_path = request.args.get("path", "").strip()

    base = _safe_root(root)
    if not base:
        return jsonify({"error": "Invalid root."}), 400

    file_path = _safe_resolve_path(base, rel_path)
    if not file_path or not file_path.exists() or not file_path.is_file():
        return jsonify({"error": "File not found."}), 404

    return send_file(str(file_path), as_attachment=True, download_name=file_path.name)


@app.route("/files/download-selected", methods=["POST"])
def files_download_selected():
    """Download selected files as a zip archive."""
    data = request.get_json() or {}
    selected = data.get("selected") or []
    if not isinstance(selected, list) or not selected:
        return jsonify({"error": "No files selected."}), 400

    file_entries: list[tuple[Path, str]] = []
    for item in selected:
        if not isinstance(item, dict):
            continue
        root = str(item.get("root", "")).strip().lower()
        rel_path = str(item.get("path", "")).strip()
        base = _safe_root(root)
        if not base:
            continue
        resolved = _safe_resolve_path(base, rel_path)
        if not resolved or not resolved.exists() or not resolved.is_file():
            continue
        archive_name = f"{root}/{rel_path}"
        file_entries.append((resolved, archive_name))

    if not file_entries:
        return jsonify({"error": "No valid files selected."}), 400

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, archive_name in file_entries:
            zf.write(src, arcname=archive_name)
    buffer.seek(0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"media_selection_{ts}.zip"
    return send_file(buffer, as_attachment=True, download_name=zip_name, mimetype="application/zip")


@app.route("/files/delete-selected", methods=["POST"])
def files_delete_selected():
    """Delete selected files from converted/ or downloads/."""
    data = request.get_json() or {}
    selected = data.get("selected") or []
    if not isinstance(selected, list) or not selected:
        return jsonify({"error": "No files selected."}), 400

    deleted_count = 0
    missing_count = 0

    for item in selected:
        if not isinstance(item, dict):
            continue

        root = str(item.get("root", "")).strip().lower()
        rel_path = str(item.get("path", "")).strip()
        base = _safe_root(root)
        if not base:
            continue

        resolved = _safe_resolve_path(base, rel_path)
        if not resolved or not resolved.exists() or not resolved.is_file():
            missing_count += 1
            continue

        try:
            resolved.unlink()
            deleted_count += 1
        except OSError:
            continue

    if deleted_count == 0:
        return jsonify({"error": "No valid files could be deleted."}), 400

    return jsonify({
        "status": "deleted",
        "deleted": deleted_count,
        "missing": missing_count,
    })


@app.route("/health")
def health():
    gpu = _detect_gpu_encoder()
    return jsonify({
        "status": "ok",
        "ffmpeg": _ffmpeg_available(),
        "yt_dlp": _ytdlp_available(),
        "downloads": str(DOWNLOADS_FOLDER),
        "gpu": gpu.get("label", "None") if gpu else "None",
        "converter_enabled": ENABLE_CONVERTER,
        "downloader_enabled": ENABLE_DOWNLOADER,
    })


# ---------------------------------------------------------------------------
# Scheduler for automatic cleanup
# ---------------------------------------------------------------------------

scheduler = BackgroundScheduler(daemon=True)
scheduler.add_job(cleanup_old_files, "interval", hours=1, id="cleanup")
scheduler.start()

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", 5000))

    gpu = _detect_gpu_encoder()
    print(f"\n  {APP_TITLE} running at http://localhost:{port}")
    mode = (
        "converter + downloader"
        if ENABLE_CONVERTER and ENABLE_DOWNLOADER
        else ("converter only" if ENABLE_CONVERTER else "downloader only")
    )
    print(f"  Mode: {mode}")
    print(f"  FFmpeg available: {_ffmpeg_available()}")
    print(f"  GPU acceleration: {gpu.get('label', 'Not available') if gpu else 'Not available'}")
    print(f"  Files auto-delete after: {CLEANUP_HOURS} hours\n")

    app.run(host=host, port=port, debug=False)
