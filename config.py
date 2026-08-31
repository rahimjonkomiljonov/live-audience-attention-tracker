"""Central configuration for the audience attention tracker.

Every tunable lives here so thresholds can be adjusted without hunting through
the pipeline. Values are deliberately conservative defaults for a webcam sitting
on top of the presentation screen, facing the audience.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "face_landmarker.task"
LOGS_DIR = BASE_DIR / "logs"
REPORTS_DIR = BASE_DIR / "reports"
STATIC_DIR = BASE_DIR / "static"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_SIZE_BYTES = 3_758_596


@dataclass
class Config:
    """Runtime configuration. A single instance is shared process-wide."""

    # --- Capture -----------------------------------------------------------
    # Either a camera index (int) or a path to a video/image file for testing.
    source: int | str = 0
    capture_width: int = 1280
    capture_height: int = 720
    mirror: bool = True  # selfie-view the display frame; pose math is unaffected

    # --- Inference ---------------------------------------------------------
    # NOTE: MediaPipe's own default is 1. For a crowd this MUST be raised or
    # only a single person is ever tracked.
    max_faces: int = 25
    infer_width: int = 640  # frames are downscaled to this before detection
    infer_height: int = 480
    target_infer_fps: float = 15.0
    min_face_detection_confidence: float = 0.4
    min_face_presence_confidence: float = 0.4
    min_tracking_confidence: float = 0.4

    # --- Attention thresholds ---------------------------------------------
    yaw_threshold_deg: float = 20.0
    pitch_threshold_deg: float = 15.0
    gaze_threshold: float = 0.35
    # PARTIAL state extends each bound by this factor before a face is AWAY.
    partial_factor: float = 1.6
    # Sustained eye closure reads as disengaged regardless of head pose.
    eye_closed_threshold: float = 0.6
    # Weight a PARTIAL face contributes to the engagement score.
    partial_weight: float = 0.5

    # --- Smoothing ---------------------------------------------------------
    ema_alpha: float = 0.2
    history_seconds: int = 300  # rolling window kept for the dashboard sparkline

    # --- Streaming ---------------------------------------------------------
    jpeg_quality: int = 75
    metrics_hz: float = 5.0
    host: str = "127.0.0.1"
    port: int = 8000

    # --- Logging / report --------------------------------------------------
    log_hz: float = 1.0
    dip_threshold_pct: float = 50.0
    dip_min_seconds: int = 15

    calibration: "Calibration" = field(default_factory=lambda: Calibration())


@dataclass
class Calibration:
    """Neutral head orientation, i.e. what 'looking at the screen' means here.

    A camera that is not centred on the audience biases every yaw/pitch reading.
    Calibrating captures the crowd's mean resting orientation and treats that as
    the origin, so the thresholds measure deviation from 'facing the screen'
    rather than deviation from 'facing the lens'.
    """

    yaw_offset: float = 0.0
    pitch_offset: float = 0.0
    calibrated: bool = False

    def reset(self) -> None:
        self.yaw_offset = 0.0
        self.pitch_offset = 0.0
        self.calibrated = False


def parse_args(argv: list[str] | None = None) -> Config:
    """Build a Config from command line overrides."""
    parser = argparse.ArgumentParser(description="Live audience attention tracker")
    parser.add_argument(
        "--source",
        default=None,
        help="Camera index (e.g. 0) or path to a video/image file for testing.",
    )
    parser.add_argument("--host", default=None, help="Bind host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default 8000)")
    parser.add_argument(
        "--max-faces", type=int, default=None, help="Maximum faces to track (default 25)"
    )
    parser.add_argument(
        "--no-mirror", action="store_true", help="Disable selfie-view mirroring."
    )
    args = parser.parse_args(argv)

    cfg = Config()
    if args.source is not None:
        # A bare integer means a camera index; anything else is a file path.
        cfg.source = int(args.source) if args.source.isdigit() else args.source
    if args.host is not None:
        cfg.host = args.host
    if args.port is not None:
        cfg.port = args.port
    if args.max_faces is not None:
        cfg.max_faces = args.max_faces
    if args.no_mirror:
        cfg.mirror = False
    return cfg


def ensure_dirs() -> None:
    for path in (MODEL_DIR, LOGS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
