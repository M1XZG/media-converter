"""
Standalone cleanup script — can be run via cron or manually.
Deletes files older than CLEANUP_HOURS from uploads/ and converted/.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "uploads"
CONVERTED_FOLDER = BASE_DIR / "converted"
CLEANUP_HOURS = int(os.environ.get("CLEANUP_HOURS", 24))


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


if __name__ == "__main__":
    print(f"Cleaning up files older than {CLEANUP_HOURS} hours...")
    count = cleanup()
    print(f"Done. Removed {count} file(s).")
