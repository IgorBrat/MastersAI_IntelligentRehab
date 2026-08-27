"""Shared Kinect v2 skeleton handling"""

from __future__ import annotations

import numpy as np

from ...schema import Joint

# Kinect v2 `JointType` indices.
KINECT_SPINE_BASE = 0
KINECT_SPINE_MID = 1
KINECT_NECK = 2
KINECT_HEAD = 3
KINECT_SHOULDER_LEFT = 4
KINECT_ELBOW_LEFT = 5
KINECT_WRIST_LEFT = 6
KINECT_HAND_LEFT = 7
KINECT_SHOULDER_RIGHT = 8
KINECT_ELBOW_RIGHT = 9
KINECT_WRIST_RIGHT = 10
KINECT_HAND_RIGHT = 11
KINECT_HIP_LEFT = 12
KINECT_KNEE_LEFT = 13
KINECT_ANKLE_LEFT = 14
KINECT_FOOT_LEFT = 15
KINECT_HIP_RIGHT = 16
KINECT_KNEE_RIGHT = 17
KINECT_ANKLE_RIGHT = 18
KINECT_FOOT_RIGHT = 19
KINECT_SPINE_SHOULDER = 20
KINECT_HAND_TIP_LEFT = 21
KINECT_THUMB_LEFT = 22
KINECT_HAND_TIP_RIGHT = 23
KINECT_THUMB_RIGHT = 24

N_KINECT_JOINTS = 25

KINECT_JOINT_MAP: dict[Joint, int | tuple[int, ...]] = {
    Joint.PELVIS: KINECT_SPINE_BASE,
    Joint.SPINE_MID: KINECT_SPINE_MID,
    Joint.NECK: KINECT_SPINE_SHOULDER,
    Joint.HEAD: KINECT_HEAD,
    # canonical LEFT <- source RIGHT
    Joint.L_SHOULDER: KINECT_SHOULDER_RIGHT,
    Joint.L_ELBOW: KINECT_ELBOW_RIGHT,
    Joint.L_WRIST: KINECT_WRIST_RIGHT,
    Joint.L_HAND: KINECT_HAND_RIGHT,
    Joint.R_SHOULDER: KINECT_SHOULDER_LEFT,
    Joint.R_ELBOW: KINECT_ELBOW_LEFT,
    Joint.R_WRIST: KINECT_WRIST_LEFT,
    Joint.R_HAND: KINECT_HAND_LEFT,
    Joint.L_HIP: KINECT_HIP_RIGHT,
    Joint.L_KNEE: KINECT_KNEE_RIGHT,
    Joint.L_ANKLE: KINECT_ANKLE_RIGHT,
    Joint.R_HIP: KINECT_HIP_LEFT,
    Joint.R_KNEE: KINECT_KNEE_LEFT,
    Joint.R_ANKLE: KINECT_ANKLE_LEFT,
}

# Kinect (+y up, +z away from sensor) -> canonical (+z toward camera)
KINECT_AXIS_TRANSFORM = np.diag([1.0, 1.0, -1.0]).astype(np.float64)

KINECT_FPS = 30.0

# Kinect `TrackingState`: 0 NotTracked, 1 Inferred, 2 Tracked
TRACKING_STATE_CONFIDENCE: dict[int, float] = {0: 0.0, 1: 0.4, 2: 1.0}


def tracking_state_to_confidence(states: np.ndarray) -> np.ndarray:
    """Mapping a ``(T, 25)`` tracking-state array to ``(T, 25)`` confidences"""
    states = np.asarray(states)
    out = np.full(states.shape, 1.0, dtype=np.float32)
    for state, conf in TRACKING_STATE_CONFIDENCE.items():
        out[states == state] = conf
    return out


def reshape_flat_skeleton(
    rows: np.ndarray, *, values_per_joint: int, n_joints: int = N_KINECT_JOINTS
) -> tuple[np.ndarray, np.ndarray | None]:
    """Splitting a flat ``(T, n_joints * values_per_joint)`` table into positions and extras"""
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D table, got shape {arr.shape}")
    expected = n_joints * values_per_joint
    if arr.shape[1] != expected:
        raise ValueError(
            f"expected {expected} columns ({n_joints} joints x {values_per_joint} values), got {arr.shape[1]}. Check values_per_joint for this export."
        )
    blocks = arr.reshape(arr.shape[0], n_joints, values_per_joint)
    positions = blocks[:, :, :3]
    extra = blocks[:, :, 3:] if values_per_joint > 3 else None
    return positions, extra


def infer_values_per_joint(n_columns: int, n_joints: int = N_KINECT_JOINTS) -> int:
    """Deduce the per-joint stride from a row width"""
    for stride in (3, 4):
        if n_columns == n_joints * stride:
            return stride
    raise ValueError(
        f"{n_columns} columns is not {n_joints} joints x 3 or x 4. Strip any leading frame-index/timestamp columns before calling."
    )
