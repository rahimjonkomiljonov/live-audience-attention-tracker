"""Session lifecycle: 1 Hz CSV logging of aggregate metrics only.

No frames, crops, or per-person identities are ever written -- only the
crowd-level counts and angles already computed by aggregate.py.
"""

from __future__ import annotations

import csv
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import config
from aggregate import CrowdMetrics
from engine import Engine

CSV_FIELDS = [
    "timestamp_iso",
    "elapsed_s",
    "faces_detected",
    "attentive",
    "partial",
    "away",
    "engagement_pct",
    "mean_yaw",
    "mean_pitch",
    "inference_fps",
]


class SessionRecorder:
    def __init__(self, engine: Engine, cfg: config.Config) -> None:
        self.engine = engine
        self.cfg = cfg
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.active = False
        self.csv_path: Path | None = None
        self.started_at: float | None = None

    def start(self) -> Path:
        if self.active:
            assert self.csv_path is not None
            return self.csv_path

        config.ensure_dirs()
        name = datetime.now().strftime("session_%Y%m%d_%H%M%S.csv")
        self.csv_path = config.LOGS_DIR / name
        self.started_at = time.monotonic()
        self.active = True
        self._stop.clear()

        with self.csv_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(CSV_FIELDS)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self.csv_path

    def stop(self) -> Path | None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.active = False
        return self.csv_path

    def _run(self) -> None:
        interval = 1.0 / self.cfg.log_hz
        assert self.csv_path is not None
        assert self.started_at is not None

        while not self._stop.is_set():
            metrics = self.engine.get_metrics()
            if metrics is not None:
                self._write_row(metrics)
            self._stop.wait(interval)

    def _write_row(self, m: CrowdMetrics) -> None:
        assert self.csv_path is not None
        assert self.started_at is not None
        elapsed = time.monotonic() - self.started_at
        row = [
            datetime.fromtimestamp(m.timestamp, tz=timezone.utc).isoformat(),
            f"{elapsed:.2f}",
            m.faces_detected,
            m.attentive,
            m.partial,
            m.away,
            f"{m.engagement_smoothed_pct:.2f}",
            f"{m.mean_yaw:.2f}",
            f"{m.mean_pitch:.2f}",
            f"{m.inference_fps:.2f}",
        ]
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)
