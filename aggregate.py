"""Crowd-level rollup: per-frame face states -> smoothed engagement metrics.

Metrics here are frame-level aggregates only (see plan's "known limit" note) --
there is no per-person identity tracking, so smoothing is applied to the
aggregate engagement percentage rather than to any individual face.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import config
from attention import FaceAttention, State


@dataclass
class CrowdMetrics:
    timestamp: float
    faces_detected: int
    attentive: int
    partial: int
    away: int
    engagement_raw_pct: float
    engagement_smoothed_pct: float
    mean_yaw: float
    mean_pitch: float
    inference_fps: float


class Aggregator:
    def __init__(self, cfg: config.Config) -> None:
        self.cfg = cfg
        self._ema: float | None = None
        maxlen = int(cfg.history_seconds * cfg.metrics_hz)
        self.history: deque[CrowdMetrics] = deque(maxlen=maxlen)

        self._infer_count = 0
        self._infer_window_start = time.monotonic()
        self._infer_fps = 0.0

        # Calibration sampling buffer, active only while calibrating.
        self._calibrating = False
        self._calib_samples: list[tuple[float, float]] = []
        self._calib_deadline = 0.0
        # Auto-calibrate once, the first time a face is seen after startup --
        # calibration state is in-memory only and resets on every restart, and
        # requiring a manual click every run is an easy step to forget.
        self._auto_calibrate_pending = not cfg.calibration.calibrated

    def tick_inference(self) -> None:
        """Call once per completed detection callback, for inference-FPS tracking."""
        self._infer_count += 1
        now = time.monotonic()
        elapsed = now - self._infer_window_start
        if elapsed >= 1.0:
            self._infer_fps = self._infer_count / elapsed
            self._infer_count = 0
            self._infer_window_start = now

    def start_calibration(self, duration_s: float = 2.0) -> None:
        self._calibrating = True
        self._calib_samples = []
        self._calib_deadline = time.monotonic() + duration_s
        self.cfg.calibration.reset()

    @property
    def is_calibrating(self) -> bool:
        return self._calibrating

    def update(self, faces: list[FaceAttention]) -> CrowdMetrics:
        total = len(faces)
        attentive = sum(1 for f in faces if f.state == State.ATTENTIVE)
        partial = sum(1 for f in faces if f.state == State.PARTIAL)
        away = sum(1 for f in faces if f.state == State.AWAY)

        if total > 0:
            raw_pct = 100.0 * (attentive + self.cfg.partial_weight * partial) / total
            mean_yaw = sum(f.yaw for f in faces) / total
            mean_pitch = sum(f.pitch for f in faces) / total
        else:
            raw_pct = 0.0
            mean_yaw = 0.0
            mean_pitch = 0.0

        if self._auto_calibrate_pending and total > 0:
            self._auto_calibrate_pending = False
            self.start_calibration()

        if self._calibrating and total > 0:
            self._calib_samples.append((mean_yaw, mean_pitch))
            if time.monotonic() >= self._calib_deadline:
                self._finish_calibration()

        if self._ema is None:
            self._ema = raw_pct
        else:
            a = self.cfg.ema_alpha
            self._ema = a * raw_pct + (1 - a) * self._ema

        metrics = CrowdMetrics(
            timestamp=time.time(),
            faces_detected=total,
            attentive=attentive,
            partial=partial,
            away=away,
            engagement_raw_pct=raw_pct,
            engagement_smoothed_pct=self._ema,
            mean_yaw=mean_yaw,
            mean_pitch=mean_pitch,
            inference_fps=self._infer_fps,
        )
        self.history.append(metrics)
        return metrics

    def _finish_calibration(self) -> None:
        self._calibrating = False
        if not self._calib_samples:
            return
        yaws = [s[0] for s in self._calib_samples]
        pitches = [s[1] for s in self._calib_samples]
        self.cfg.calibration.yaw_offset = sum(yaws) / len(yaws)
        self.cfg.calibration.pitch_offset = sum(pitches) / len(pitches)
        self.cfg.calibration.calibrated = True
