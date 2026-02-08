@echo off
echo ============================================
echo   Media Converter - Setup
echo ============================================
echo.

:: Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not on PATH.
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

:: Check for FFmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo [WARNING] FFmpeg is not installed or not on PATH.
    echo Conversions will not work without FFmpeg.
    echo Install it via: winget install FFmpeg
    echo.
)

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

:: Activate venv
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt --quiet
echo.

:: Create directories
if not exist "uploads" mkdir uploads
if not exist "converted" mkdir converted

:: Run the app
echo ============================================
echo   Starting Media Converter...
echo   Open http://localhost:5000 in your browser
echo   Press Ctrl+C to stop
echo ============================================
echo.
python app.py
