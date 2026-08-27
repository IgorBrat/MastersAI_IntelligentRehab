from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..schema import Joint, PoseSequence, Side


class CameraView(str, Enum):
    """Required framing for an exercise"""

    # Subject faces the camera. Frontal-plane angles (abduction, lateral trunk lean, shoulder hiking) lie in the image plane and are most accurate
    FRONTAL = "frontal"

    # Subject is side-on. Sagittal-plane angles (shoulder flexion, forward trunk flexion, reach distance) lie in the image plane
    SAGITTAL = "sagittal"

    # View-insensitive, e.g. elbow flexion, which is measured from a three-point angle largely robust to yaw
    ANY = "any"


@dataclass(frozen=True)
class ViewCheck:
    """Outcome of validating a recording's framing"""
    ok: bool
    detected: CameraView
    # Median absolute subject yaw: 0 = facing the camera, 90 = fully side-on
    yaw_deg: float
    message_uk: str = ""

    def __bool__(self) -> bool:
        return self.ok


def estimate_yaw(seq: PoseSequence) -> np.ndarray:
    """Per-frame subject yaw in degrees, ``0`` = facing camera, ``90`` = side-on"""
    j = seq.joints.astype(np.float64)
    line = j[:, Joint.L_SHOULDER, :] - j[:, Joint.R_SHOULDER, :]
    # In-image extent (x) versus depth extent (z).
    in_plane = np.abs(line[:, 0])
    depth = np.abs(line[:, 2])
    return np.degrees(np.arctan2(depth, np.maximum(in_plane, 1e-8)))


def check_view(
    seq: PoseSequence,
    required: CameraView,
    *,
    frontal_max_yaw: float = 35.0,
    sagittal_min_yaw: float = 55.0,
) -> ViewCheck:
    """Validating that ``seq`` was recorded from the view ``required``"""
    yaw = float(np.median(estimate_yaw(seq)))

    if yaw <= frontal_max_yaw:
        detected = CameraView.FRONTAL
    elif yaw >= sagittal_min_yaw:
        detected = CameraView.SAGITTAL
    else:
        detected = CameraView.ANY  # ambiguous / oblique

    if required is CameraView.ANY:
        return ViewCheck(ok=True, detected=detected, yaw_deg=yaw)

    if required is CameraView.FRONTAL:
        ok = yaw <= frontal_max_yaw
        msg = "" if ok else (
            f"Повернітесь обличчям до камери - вправу потрібно записувати спереду (зараз корпус повернутий приблизно на {yaw:.0f}°)."
        )
    else:
        ok = yaw >= sagittal_min_yaw
        msg = "" if ok else (
            f"Станьте боком до камери - вправу потрібно записувати з боку (зараз корпус повернутий приблизно на {yaw:.0f}°)."
        )
    return ViewCheck(ok=ok, detected=detected, yaw_deg=yaw, message_uk=msg)


def infer_working_side(seq: PoseSequence, *, min_margin: float = 0.15) -> Side | None:
    """Guess which arm is exercising, from which hand travels further"""
    j = seq.joints.astype(np.float64)
    scale = max(seq.torso_length(), 1e-3)
    excursions = {}
    for side in (Side.LEFT, Side.RIGHT):
        hand = j[:, Joint.L_HAND if side is Side.LEFT else Joint.R_HAND, :]
        path = np.linalg.norm(np.diff(hand, axis=0), axis=-1).sum() / scale
        excursions[side] = float(path)

    left, right = excursions[Side.LEFT], excursions[Side.RIGHT]
    if abs(left - right) < min_margin:
        return None
    return Side.LEFT if left > right else Side.RIGHT
