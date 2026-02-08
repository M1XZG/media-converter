#!/usr/bin/env bash
set -e

echo "============================================"
echo "  Media Converter - Setup"
echo "============================================"
echo

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed or not on PATH."
    echo "Please install Python 3.8+."
    exit 1
fi

# Check for FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "[WARNING] FFmpeg is not installed or not on PATH."
    echo "Conversions will not work without FFmpeg."
    echo "Install it via your package manager (e.g., brew install ffmpeg)."
    echo
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv --without-pip venv
    source venv/bin/activate
    echo "Installing pip..."
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3 > /dev/null 2>&1
    echo
else
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
venv/bin/pip install -r requirements.txt --quiet
echo

# Create directories
mkdir -p uploads converted

# Run the app
echo "============================================"
echo "  Starting Media Converter..."
echo "  Open http://localhost:5000 in your browser"
echo "  Press Ctrl+C to stop"
echo "============================================"
echo
python app.py
