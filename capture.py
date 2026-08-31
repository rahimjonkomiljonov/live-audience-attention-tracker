"""Threaded frame capture.

OpenCV buffers frames internally, so a slow consumer drifts further and further
behind real time. This reader runs its own thread and keeps only the newest
frame, so the tracker always works on what the camera sees *now*.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2
import numpy as np


class FrameSource:
    """Latest-frame-wins reader for a camera, video file, or still image."""

    def __init__(
        self,
        source: int | str,
        width: int,
        height: int,
    ) -> None:
        self.source = source
        self.width = width
        self.height = height

        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._capture: cv2.VideoCapture | None = None

        self._frame_count = 0
        self._fps = 0.0
        self._fps_window_start = time.monotonic()
        self._fps_window_frames = 0

        # A still image has no stream to pump; it is loaded once and reused.
        self._still: np.ndarray | None = None
        self.is_still = False

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return

        if self._is_image_path(self.source):
            still = cv2.imread(str(self.source))
            if still is None:
                raise RuntimeError(f"Could not read image file: {self.source}")
            self._still = still
            self.is_still = True
            with self._lock:
                self._frame = still
            return

        self._capture = cv2.VideoCapture(self.source)
        if isinstance(self.source, int):
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            # A shallow buffer keeps latency low on cameras that honour it.
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._capture.isOpened():
            raise RuntimeError(
                f"Could not open video source {self.source!r}. "
                "Check the camera is connected and not in use by another app."
            )

        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="capture", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    # -- reading -----------------------------------------------------------

    def read(self) -> np.ndarray | None:
        """Return the most recent frame, or None if nothing has arrived yet."""
        if self.is_still:
            return None if self._still is None else self._still.copy()
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    @property
    def fps(self) -> float:
        return self._fps

    # -- internals ---------------------------------------------------------

    def _run(self) -> None:
        assert self._capture is not None
        while not self._stop.is_set():
            ok, frame = self._capture.read()
            if not ok:
                # Video files end; loop them so test runs keep producing frames.
                if not isinstance(self.source, int):
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.05)
                continue

            with self._lock:
                self._frame = frame
            self._frame_count += 1
            self._tick_fps()

    def _tick_fps(self) -> None:
        self._fps_window_frames += 1
        now = time.monotonic()
        elapsed = now - self._fps_window_start
        if elapsed >= 1.0:
            self._fps = self._fps_window_frames / elapsed
            self._fps_window_frames = 0
            self._fps_window_start = now

    @staticmethod
    def _is_image_path(source: int | str) -> bool:
        if isinstance(source, int):
            return False
        suffix = Path(source).suffix.lower()
        return suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
