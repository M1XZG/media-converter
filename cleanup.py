"""
Standalone cleanup script — can be run via cron or manually.
Deletes files older than CLEANUP_HOURS from uploads/ and converted/, and
downloaded files older than DOWNLOADS_CLEANUP_MINUTES from downloads/.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
CONVERTED_FOLDER = BASE_DIR / "converted"
DOWNLOADS_FOLDER = BASE_DIR / "downloads"
CLEANUP_HOURS = int(os.environ.get("CLEANUP_HOURS", 24))


def _parse_minutes(name: str, default: int) -> int:
    """Parse a "minutes" env var. 0 / off / disabled / never => disabled (0)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in ("", "0", "off", "false", "no", "disabled", "never"):
        return 0
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else 0


DOWNLOADS_CLEANUP_MINUTES = _parse_minutes("DOWNLOADS_CLEANUP_MINUTES", 30)


def cleanup():
    cutoff = datetime.now() - timedelta(hours=CLEANUP_HOURS)
    removed = 0

    for folder in (UPLOAD_FOLDER, CONVERTED_FOLDER):
        if not folder.exists():
            continue
        for item in folder.iterdir():
            if item.is_file():
                mtime = datetime.fromtimestamp(item.stat().st_mtime)
                if mtime < cutoff:
                    try:
                        item.unlink()
                        print(f"  Removed: {item.name}")
                        removed += 1
                    except OSError as e:
                        print(f"  Error removing {item.name}: {e}")

    return removed


def cleanup_downloads():
    """Delete downloaded files older than DOWNLOADS_CLEANUP_MINUTES."""
    if DOWNLOADS_CLEANUP_MINUTES <= 0 or not DOWNLOADS_FOLDER.exists():
        return 0

    cutoff = datetime.now() - timedelta(minutes=DOWNLOADS_CLEANUP_MINUTES)
    removed = 0
    for item in DOWNLOADS_FOLDER.rglob("*"):
        if not item.is_file():
            continue
        try:
            mtime = datetime.fromtimestamp(item.stat().st_mtime)
        except OSError:
            continue
        if mtime < cutoff:
            try:
                item.unlink()
                print(f"  Removed: {item.relative_to(DOWNLOADS_FOLDER).as_posix()}")
                removed += 1
            except OSError as e:
                print(f"  Error removing {item.name}: {e}")

    return removed


if __name__ == "__main__":
    print(f"Cleaning up files older than {CLEANUP_HOURS} hours...")
    count = cleanup()
    if DOWNLOADS_CLEANUP_MINUTES > 0:
        print(f"Cleaning up downloads older than {DOWNLOADS_CLEANUP_MINUTES} minutes...")
        count += cleanup_downloads()
    print(f"Done. Removed {count} file(s).")
