"""Per-face attention scoring: head pose + gaze -> ATTENTIVE / PARTIAL / AWAY.

This module is intentionally the only place that knows about blendshape names
and rotation-matrix decomposition, so the rest of the pipeline works with a
small, stable `FaceAttention` record.

Sentiment hook
---------------
`face_blendshapes` already contains everything needed for expression-based
sentiment (mouthSmile*, browDown*, jawOpen, etc.) since it's the same output
used for gaze. That was out of scope for this build (attention-only), but a
`score_sentiment(blendshapes: dict[str, float]) -> str` function could be added
here and its result attached to `FaceAttention` without touching `engine.py`
beyond passing the value through.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

import config


class State(str, Enum):
    ATTENTIVE = "attentive"
    PARTIAL = "partial"
    AWAY = "away"


@dataclass
class FaceAttention:
    state: State
    yaw: float
    pitch: float
    roll: float
    gaze_x: float
    gaze_y: float
    eyes_closed: bool
    bbox: tuple[int, int, int, int]  # x, y, w, h in pixel coords of the inference frame


def blendshapes_to_dict(categories) -> dict[str, float]:
    return {c.category_name: c.score for c in categories}


def euler_from_transformation_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    """Decompose a 4x4 facial transformation matrix into yaw/pitch/roll degrees.

    Reuses cv2.RQDecomp3x3 rather than a hand-rolled atan2/asin decomposition,
    which avoids re-deriving gimbal-lock edge cases.
    """
    rotation = matrix[:3, :3]
    euler_angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation)
    pitch, yaw, roll = euler_angles
    return float(yaw), float(pitch), float(roll)


def gaze_offset(blend: dict[str, float]) -> tuple[float, float]:
    """Signed horizontal/vertical eye-direction offset in roughly [-1, 1]."""

    def g(name: str) -> float:
        return blend.get(name, 0.0)

    gaze_x = (
        (g("eyeLookOutLeft") + g("eyeLookInRight")) - (g("eyeLookInLeft") + g("eyeLookOutRight"))
    ) / 2.0
    gaze_y = (
        (g("eyeLookUpLeft") + g("eyeLookUpRight")) - (g("eyeLookDownLeft") + g("eyeLookDownRight"))
    ) / 2.0
    return gaze_x, gaze_y


def eyes_are_closed(blend: dict[str, float], threshold: float) -> bool:
    left = blend.get("eyeBlinkLeft", 0.0)
    right = blend.get("eyeBlinkRight", 0.0)
    return min(left, right) > threshold


def classify(
    yaw: float,
    pitch: float,
    gaze_x: float,
    gaze_y: float,
    eyes_closed: bool,
    cfg: config.Config,
) -> State:
    if eyes_closed:
        return State.AWAY

    calib = cfg.calibration
    d_yaw = abs(yaw - calib.yaw_offset)
    d_pitch = abs(pitch - calib.pitch_offset)
    gaze_mag = (gaze_x**2 + gaze_y**2) ** 0.5

    if (
        d_yaw < cfg.yaw_threshold_deg
        and d_pitch < cfg.pitch_threshold_deg
        and gaze_mag < cfg.gaze_threshold
    ):
        return State.ATTENTIVE

    partial_yaw = cfg.yaw_threshold_deg * cfg.partial_factor
    partial_pitch = cfg.pitch_threshold_deg * cfg.partial_factor
    partial_gaze = cfg.gaze_threshold * cfg.partial_factor

    if d_yaw < partial_yaw and d_pitch < partial_pitch and gaze_mag < partial_gaze:
        return State.PARTIAL

    return State.AWAY


def score_face(
    landmarks,
    transformation_matrix: np.ndarray,
    blendshape_categories,
    frame_width: int,
    frame_height: int,
    cfg: config.Config,
) -> FaceAttention:
    yaw, pitch, roll = euler_from_transformation_matrix(transformation_matrix)
    blend = blendshapes_to_dict(blendshape_categories)
    gaze_x, gaze_y = gaze_offset(blend)
    closed = eyes_are_closed(blend, cfg.eye_closed_threshold)
    state = classify(yaw, pitch, gaze_x, gaze_y, closed, cfg)
    bbox = _bbox_from_landmarks(landmarks, frame_width, frame_height)

    return FaceAttention(
        state=state,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        gaze_x=gaze_x,
        gaze_y=gaze_y,
        eyes_closed=closed,
        bbox=bbox,
    )


def _bbox_from_landmarks(landmarks, width: int, height: int) -> tuple[int, int, int, int]:
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)
