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

# Active conversion jobs: file_id -> job dict
_active_jobs: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    """Check if ffmpeg is accessible on the system PATH."""
    return shutil.which("ffmpeg") is not None


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
    gpu = _detect_gpu_encoder()
    return render_template(
        "index.html",
        video_formats=VIDEO_OUTPUT_FORMATS,
        audio_formats=AUDIO_OUTPUT_FORMATS,
        ffmpeg_ok=_ffmpeg_available(),
        gpu_info=gpu,
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
        "duration_seconds": duration,
        "video_codec": video_codec or "N/A",
        "audio_codec": audio_codec or "N/A",
        "resolution": resolution or "N/A",
    })


@app.route("/convert", methods=["POST"])
def convert():
    """Start conversion of an uploaded file (non-blocking)."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Invalid request."}), 400

    file_id = data.get("file_id")
    output_format = data.get("format", "").lower()
    mode = data.get("mode", "video")  # "video" or "audio"
    total_duration = data.get("duration_seconds")  # seconds, from upload probe

    if not file_id or not output_format:
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

    # Detect GPU encoder
    gpu = _detect_gpu_encoder()
    hw_accel_used = False

    # Build FFmpeg command
    cmd = ["ffmpeg", "-y"]

    # Add hardware-accelerated decoding if GPU is available
    if gpu and mode != "audio":
        if gpu["name"] == "nvenc":
            cmd.extend(["-hwaccel", "cuda"])
        elif gpu["name"] == "qsv":
            cmd.extend(["-hwaccel", "qsv"])
        elif gpu["name"] == "vaapi":
            cmd.extend(["-hwaccel", "vaapi",
                        "-hwaccel_device", "/dev/dri/renderD128"])

    cmd.extend(["-i", str(source_file)])

    if mode == "audio":
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
        if output_format in ("mp4", "mkv", "mov") and gpu:
            enc = gpu["h264"]
            if gpu["name"] == "nvenc":
                cmd.extend(["-codec:v", enc, "-preset", "p5", "-tune", "hq",
                            "-rc", "constqp", "-qp", "20",
                            "-b:v", "0", "-profile:v", "high"])
            elif gpu["name"] == "amf":
                cmd.extend(["-codec:v", enc, "-quality", "quality",
                            "-rc", "cqp", "-qp_i", "20", "-qp_p", "20",
                            "-qp_b", "22", "-profile:v", "high"])
            elif gpu["name"] == "qsv":
                cmd.extend(["-codec:v", enc, "-preset", "medium",
                            "-global_quality", "20", "-profile:v", "high"])
            elif gpu["name"] == "vaapi":
                cmd.extend(["-codec:v", enc, "-qp", "20",
                            "-profile:v", "high"])
            cmd.extend(["-codec:a", "aac", "-b:a", "192k"])
            if output_format == "mp4":
                cmd.extend(["-movflags", "+faststart"])
            hw_accel_used = True
        elif output_format == "mp4":
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
        "duration": total_duration or 0,
    }
    _active_jobs[file_id] = job

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

            if proc.returncode != 0:
                stderr_out = proc.stderr.read() if proc.stderr else ""
                error_msg = stderr_out.strip().split("\n")[-1] if stderr_out else "Conversion failed."
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
    gpu = _detect_gpu_encoder()
    return jsonify({
        "status": "ok",
        "ffmpeg": _ffmpeg_available(),
        "gpu": gpu.get("label", "None") if gpu else "None",
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
    print(f"\n  Media Converter running at http://localhost:{port}")
    print(f"  FFmpeg available: {_ffmpeg_available()}")
    print(f"  GPU acceleration: {gpu.get('label', 'Not available') if gpu else 'Not available'}")
    print(f"  Files auto-delete after: {CLEANUP_HOURS} hours\n")

    app.run(host=host, port=port, debug=False)
