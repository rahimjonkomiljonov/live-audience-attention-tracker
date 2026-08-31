"""Fetch the MediaPipe Face Landmarker model asset.

The .task bundle is not shipped with the pip package, so it must be downloaded
once before the tracker can run.
"""

from __future__ import annotations

import sys
import urllib.request

import config


def download(force: bool = False) -> bool:
    """Download the model if missing. Returns True if the file is ready."""
    config.ensure_dirs()
    path = config.MODEL_PATH

    if path.exists() and not force:
        size = path.stat().st_size
        if size == config.MODEL_SIZE_BYTES:
            print(f"Model already present: {path} ({size:,} bytes)")
            return True
        print(f"Model at {path} has unexpected size {size:,}; re-downloading.")

    print(f"Downloading {config.MODEL_URL}")
    tmp = path.with_suffix(".task.part")
    try:
        with urllib.request.urlopen(config.MODEL_URL, timeout=60) as response:
            tmp.write_bytes(response.read())
    except Exception as exc:  # noqa: BLE001 - surface any network failure plainly
        print(f"Download failed: {exc}", file=sys.stderr)
        tmp.unlink(missing_ok=True)
        return False

    size = tmp.stat().st_size
    if size != config.MODEL_SIZE_BYTES:
        print(
            f"Warning: downloaded {size:,} bytes, expected {config.MODEL_SIZE_BYTES:,}.",
            file=sys.stderr,
        )
    tmp.replace(path)
    print(f"Saved {path} ({size:,} bytes)")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if download(force="--force" in sys.argv) else 1)
