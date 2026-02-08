"""
Media Converter — Self-hosted video conversion and audio extraction tool.
"""

import os
import uuid
import subprocess
import json
import shutil
from datetime import datetime, timedelta
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
CLEANUP_HOURS = int(os.environ.get("CLEANUP_HOURS", 24))

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    """Check if ffmpeg is accessible on the system PATH."""
    return shutil.which("ffmpeg") is not None


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


def allowed_file(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_INPUT_EXTENSIONS


def cleanup_old_files():
    """Delete files older than CLEANUP_HOURS from uploads/ and converted/."""
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

@app.route("/")
def index():
    return render_template(
        "index.html",
        video_formats=VIDEO_OUTPUT_FORMATS,
        audio_formats=AUDIO_OUTPUT_FORMATS,
        ffmpeg_ok=_ffmpeg_available(),
    )


@app.route("/upload", methods=["POST"])
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
        "video_codec": video_codec or "N/A",
        "audio_codec": audio_codec or "N/A",
        "resolution": resolution or "N/A",
    })


@app.route("/convert", methods=["POST"])
def convert():
    """Convert an uploaded file to the requested format."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request."}), 400

    file_id = data.get("file_id")
    output_format = data.get("format", "").lower()
    mode = data.get("mode", "video")  # "video" or "audio"

    if not file_id or not output_format:
        return jsonify({"error": "Missing file_id or format."}), 400

    # Find the uploaded file
    source_file = None
    for f in UPLOAD_FOLDER.iterdir():
        if f.stem == file_id:
            source_file = f
            break

    if source_file is None or not source_file.exists():
        return jsonify({"error": "Upload not found. It may have expired."}), 404

    # Determine output settings
    if mode == "audio":
        if output_format not in AUDIO_OUTPUT_FORMATS:
            return jsonify({"error": f"Unsupported audio format: {output_format}"}), 400
        out_ext = AUDIO_OUTPUT_FORMATS[output_format]["ext"]
    else:
        if output_format not in VIDEO_OUTPUT_FORMATS:
            return jsonify({"error": f"Unsupported video format: {output_format}"}), 400
        out_ext = VIDEO_OUTPUT_FORMATS[output_format]["ext"]

    output_name = f"{file_id}_converted{out_ext}"
    output_path = CONVERTED_FOLDER / output_name

    # Build FFmpeg command
    cmd = ["ffmpeg", "-y", "-i", str(source_file)]

    if mode == "audio":
        # Extract audio only
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
        # Video conversion
        if output_format == "mp4":
            cmd.extend(["-codec:v", "libx264", "-preset", "medium", "-crf", "23",
                        "-codec:a", "aac", "-b:a", "192k", "-movflags", "+faststart"])
        elif output_format == "webm":
            cmd.extend(["-codec:v", "libvpx-vp9", "-crf", "30", "-b:v", "0",
                        "-codec:a", "libopus", "-b:a", "128k"])
        elif output_format == "mkv":
            cmd.extend(["-codec:v", "libx264", "-preset", "medium", "-crf", "23",
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

    cmd.append(str(output_path))

    try:
        result = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=7200,  # 2 hour timeout for large files
        )
        if result.returncode != 0:
            error_msg = result.stderr.strip().split("\n")[-1] if result.stderr else "Conversion failed."
            return jsonify({"error": f"FFmpeg error: {error_msg}"}), 500
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Conversion timed out (exceeded 2 hours)."}), 504
    except FileNotFoundError:
        return jsonify({"error": "FFmpeg is not installed or not found on PATH."}), 500

    output_size = output_path.stat().st_size

    return jsonify({
        "download_id": output_name,
        "output_size": _human_size(output_size),
    })


@app.route("/download/<download_id>")
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


@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "ffmpeg": _ffmpeg_available(),
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

    print(f"\n  Media Converter running at http://localhost:{port}")
    print(f"  FFmpeg available: {_ffmpeg_available()}")
    print(f"  Files auto-delete after: {CLEANUP_HOURS} hours\n")

    app.run(host=host, port=port, debug=False)
