# ---------------------------------------------------------------------------
# Media Converter — Docker image with FFmpeg + NVIDIA GPU support
#
# GPU support requires:
#   1. NVIDIA drivers on the host
#   2. NVIDIA Container Toolkit installed
#      (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
#   3. Run with: docker compose up  (uses gpu profile automatically)
#      Or:       docker run --gpus all ...
#
# Without a GPU, the container works fine using CPU encoding.
# ---------------------------------------------------------------------------

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, FFmpeg (with NVENC/VAAPI support), and clean up
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# Copy application code
COPY app.py cleanup.py ./
COPY templates/ templates/

# Create directories for uploads and converted files
RUN mkdir -p uploads converted

# Expose port
EXPOSE 5000

# Environment defaults
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=5000
ENV CLEANUP_HOURS=24

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["python3", "app.py"]
