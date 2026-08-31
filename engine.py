"""Owns the camera thread and the FaceLandmarker, produces annotated frames
and crowd metrics for the server to serve.

LIVE_STREAM mode requires strictly increasing timestamps and delivers results
asynchronously via `result_callback`, which MediaPipe invokes on its own
internal thread -- all shared state below is written under `_lock`.
Submission is serialized (one in-flight detection at a time) so the frame
drawn on in the callback is unambiguously the one that was analyzed.
"""

from __future__ import annotations

import threading
import time

import cv2
import mediapipe as mp
import numpy as np

import config
from aggregate import Aggregator, CrowdMetrics
from attention import State, score_face
from capture import FrameSource

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

_STATE_COLOR = {
    State.ATTENTIVE: (0, 200, 0),
    State.PARTIAL: (0, 180, 255),
    State.AWAY: (0, 0, 220),
}


class Engine:
    def __init__(self, cfg: config.Config) -> None:
        self.cfg = cfg
        self.aggregator = Aggregator(cfg)
        self.source = FrameSource(cfg.source, cfg.capture_width, cfg.capture_height)

        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_metrics: CrowdMetrics | None = None

        self._landmarker: FaceLandmarker | None = None
        self._submit_thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._detecting = threading.Event()  # set while a detection is in flight
        self._pending_frame: np.ndarray | None = None
        self._start_time = time.monotonic()
        self._last_timestamp_ms = -1
        self._error: str | None = None

    # -- lifecycle -----------------------------------------------------

    def start(self) -> None:
        self.source.start()

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(config.MODEL_PATH)),
            running_mode=RunningMode.LIVE_STREAM,
            num_faces=self.cfg.max_faces,
            min_face_detection_confidence=self.cfg.min_face_detection_confidence,
            min_face_presence_confidence=self.cfg.min_face_presence_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            result_callback=self._on_result,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

        self._stop.clear()
        self._submit_thread = threading.Thread(target=self._submit_loop, daemon=True)
        self._submit_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._submit_thread is not None:
            self._submit_thread.join(timeout=2.0)
            self._submit_thread = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        self.source.stop()

    def calibrate(self, duration_s: float = 2.0) -> None:
        self.aggregator.start_calibration(duration_s)

    # -- accessors -------------------------------------------------------

    def get_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def get_metrics(self) -> CrowdMetrics | None:
        with self._lock:
            return self._latest_metrics

    @property
    def error(self) -> str | None:
        return self._error

    # -- submission loop ---------------------------------------------------

    def _submit_loop(self) -> None:
        assert self._landmarker is not None
        min_interval = 1.0 / self.cfg.target_infer_fps
        last_submit = 0.0

        while not self._stop.is_set():
            frame = self.source.read()
            if frame is None:
                time.sleep(0.02)
                continue

            if self._detecting.is_set():
                time.sleep(0.005)
                continue

            now = time.monotonic()
            if now - last_submit < min_interval:
                time.sleep(min_interval - (now - last_submit))
                continue
            last_submit = now

            if self.cfg.mirror:
                frame = cv2.flip(frame, 1)

            resized = cv2.resize(frame, (self.cfg.infer_width, self.cfg.infer_height))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int((time.monotonic() - self._start_time) * 1000)
            if timestamp_ms <= self._last_timestamp_ms:
                timestamp_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = timestamp_ms

            self._pending_frame = resized
            self._detecting.set()
            try:
                self._landmarker.detect_async(mp_image, timestamp_ms)
            except Exception as exc:  # noqa: BLE001
                self._error = str(exc)
                self._detecting.clear()

    # -- result callback (runs on a MediaPipe-owned thread) -----------------

    def _on_result(self, result, mp_image, timestamp_ms: int) -> None:
        try:
            frame = self._pending_frame
            if frame is None:
                return
            height, width = frame.shape[:2]

            faces = []
            n = len(result.face_landmarks)
            for i in range(n):
                landmarks = result.face_landmarks[i]
                matrix = (
                    result.facial_transformation_matrixes[i]
                    if i < len(result.facial_transformation_matrixes)
                    else np.eye(4)
                )
                blendshapes = (
                    result.face_blendshapes[i] if i < len(result.face_blendshapes) else []
                )
                faces.append(
                    score_face(landmarks, matrix, blendshapes, width, height, self.cfg)
                )

            metrics = self.aggregator.update(faces)
            self.aggregator.tick_inference()

            annotated = self._annotate(frame, faces, metrics)
            display = cv2.resize(
                annotated, (self.cfg.capture_width, self.cfg.capture_height)
            )
            ok, buf = cv2.imencode(
                ".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, self.cfg.jpeg_quality]
            )
            if not ok:
                return

            with self._lock:
                self._latest_jpeg = buf.tobytes()
                self._latest_metrics = metrics
        finally:
            self._detecting.clear()

    def _annotate(self, frame: np.ndarray, faces, metrics: CrowdMetrics) -> np.ndarray:
        out = frame.copy()
        for face in faces:
            x, y, w, h = face.bbox
            color = _STATE_COLOR[face.state]
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            label = face.state.value
            cv2.putText(
                out, label, (x, max(0, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
            )

        banner = (
            f"Engagement: {metrics.engagement_smoothed_pct:5.1f}%  "
            f"Faces: {metrics.faces_detected}  "
            f"({metrics.attentive} attentive / {metrics.partial} partial / {metrics.away} away)"
        )
        cv2.rectangle(out, (0, 0), (out.shape[1], 26), (0, 0, 0), -1)
        cv2.putText(
            out, banner, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
        return out
