from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np

from ...exercises.stroke_upper_limb import (
    SHOULDER_ABDUCTION,
    SHOULDER_EXTERNAL_ROTATION,
    SHOULDER_FLEXION,
)
from ...schema import Joint, N_JOINTS, PoseSequence, Side
from .base import DatasetAdapter, RawTrial

UI_PRMD_EXERCISE_MAP: dict[str, str | None] = {
    "m01": None,  # deep squat
    "m02": None,  # hurdle step
    "m03": None,  # inline lunge
    "m04": None,  # side lunge
    "m05": None,  # sit to stand
    "m06": None,  # standing active straight leg raise
    "m07": SHOULDER_ABDUCTION.id,           # standing shoulder abduction
    "m08": None,                            # standing shoulder extension
    "m09": SHOULDER_EXTERNAL_ROTATION.id,   # standing shoulder internal-external rotation
    "m10": SHOULDER_FLEXION.id,             # standing shoulder scaption
}

# Which arm each subject used for the shoulder movements
UI_PRMD_SIDE_BY_SUBJECT: dict[str, Side] = {
    "s01": Side.RIGHT, "s02": Side.RIGHT, "s03": Side.RIGHT, "s04": Side.RIGHT,
    "s05": Side.RIGHT, "s06": Side.RIGHT, "s07": Side.LEFT,  "s08": Side.RIGHT,
    "s09": Side.RIGHT, "s10": Side.LEFT,
}

# # Vicon

VICON_MARKERS: tuple[str, ...] = (
    "LFHD", "RFHD", "LBHD", "RBHD",           # head cluster
    "C7", "T10", "CLAV", "STRN", "RBAK",      # trunk
    "LSHO", "LUPA", "LELB", "LFRM", "LWRA", "LWRB", "LFIN",   # left arm
    "RSHO", "RUPA", "RELB", "RFRM", "RWRA", "RWRB", "RFIN",   # right arm
    "LASI", "RASI", "LPSI", "RPSI",           # pelvis
    "LTHI", "LKNE", "LTIB", "LANK", "LHEE", "LTOE",           # left leg
    "RTHI", "RKNE", "RTIB", "RANK", "RHEE", "RTOE",           # right leg
)
N_VICON_MARKERS = len(VICON_MARKERS)
_V = {name: i for i, name in enumerate(VICON_MARKERS)}

# Canonical joint -> Vicon marker index, or a tuple to average.
VICON_JOINT_MAP: dict[Joint, int | tuple[int, ...]] = {
    Joint.PELVIS: (_V["LASI"], _V["RASI"], _V["LPSI"], _V["RPSI"]),
    Joint.SPINE_MID: (_V["T10"], _V["STRN"]),
    Joint.NECK: (_V["C7"], _V["CLAV"]),
    Joint.HEAD: (_V["LFHD"], _V["RFHD"], _V["LBHD"], _V["RBHD"]),
    Joint.L_SHOULDER: _V["LSHO"],
    Joint.L_ELBOW: _V["LELB"],
    Joint.L_WRIST: (_V["LWRA"], _V["LWRB"]),
    Joint.L_HAND: _V["LFIN"],
    Joint.R_SHOULDER: _V["RSHO"],
    Joint.R_ELBOW: _V["RELB"],
    Joint.R_WRIST: (_V["RWRA"], _V["RWRB"]),
    Joint.R_HAND: _V["RFIN"],
    Joint.L_HIP: (_V["LASI"], _V["LPSI"]),
    Joint.L_KNEE: _V["LKNE"],
    Joint.L_ANKLE: _V["LANK"],
    Joint.R_HIP: (_V["RASI"], _V["RPSI"]),
    Joint.R_KNEE: _V["RKNE"],
    Joint.R_ANKLE: _V["RANK"],
}

# # Kinect

KINECT_JOINTS: tuple[str, ...] = (
    "waist", "spine", "chest", "neck", "head", "head_tip",
    "left_collar", "left_upper_arm", "left_forearm", "left_hand",
    "right_collar", "right_upper_arm", "right_forearm", "right_hand",
    "left_upper_leg", "left_lower_leg", "left_foot", "left_leg_toes",
    "right_upper_leg", "right_lower_leg", "right_foot", "right_leg_toes",
)
N_KINECT_JOINTS = len(KINECT_JOINTS)
_K = {name: i for i, name in enumerate(KINECT_JOINTS)}

KINECT_HIERARCHY: dict[int, int] = {
    _K["waist"]: -1,
    _K["spine"]: _K["waist"],
    _K["chest"]: _K["spine"],
    _K["neck"]: _K["chest"],
    _K["head"]: _K["neck"],
    _K["head_tip"]: _K["head"],
    _K["left_collar"]: _K["chest"],
    _K["left_upper_arm"]: _K["left_collar"],
    _K["left_forearm"]: _K["left_upper_arm"],
    _K["left_hand"]: _K["left_forearm"],
    _K["right_collar"]: _K["chest"],
    _K["right_upper_arm"]: _K["right_collar"],
    _K["right_forearm"]: _K["right_upper_arm"],
    _K["right_hand"]: _K["right_forearm"],
    _K["left_upper_leg"]: _K["waist"],
    _K["left_lower_leg"]: _K["left_upper_leg"],
    _K["left_foot"]: _K["left_lower_leg"],
    _K["left_leg_toes"]: _K["left_foot"],
    _K["right_upper_leg"]: _K["waist"],
    _K["right_lower_leg"]: _K["right_upper_leg"],
    _K["right_foot"]: _K["right_lower_leg"],
    _K["right_leg_toes"]: _K["right_foot"],
}

# Canonical joint -> Brekel joint index. The source names a joint by the segment
KINECT_JOINT_MAP: dict[Joint, int | tuple[int, ...]] = {
    Joint.PELVIS: _K["waist"],
    Joint.SPINE_MID: _K["spine"],
    Joint.NECK: _K["neck"],
    Joint.HEAD: _K["head"],
    Joint.L_SHOULDER: _K["left_upper_arm"],
    Joint.L_ELBOW: _K["left_forearm"],
    Joint.L_WRIST: _K["left_hand"],
    Joint.L_HAND: _K["left_hand"],
    Joint.R_SHOULDER: _K["right_upper_arm"],
    Joint.R_ELBOW: _K["right_forearm"],
    Joint.R_WRIST: _K["right_hand"],
    Joint.R_HAND: _K["right_hand"],
    Joint.L_HIP: _K["left_upper_leg"],
    Joint.L_KNEE: _K["left_lower_leg"],
    Joint.L_ANKLE: _K["left_foot"],
    Joint.R_HIP: _K["right_upper_leg"],
    Joint.R_KNEE: _K["right_lower_leg"],
    Joint.R_ANKLE: _K["right_foot"],
}

# Vicon is **z-up**: ``(x, y, z) -> (x, z, -y)``.
VICON_AXIS_TRANSFORM = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=np.float64
)

# Kinect is **y-up** (sensor frame: x right, y up, z away from the sensor)
KINECT_AXIS_TRANSFORM = np.diag([1.0, 1.0, -1.0]).astype(np.float64)

UI_PRMD_AXIS_TRANSFORM = VICON_AXIS_TRANSFORM

# Nominal capture rates, from the dataset documentation
VICON_FPS = 100.0
KINECT_FPS = 30.0

# Vicon is millimetres
VICON_UNIT_SCALE = 0.001

#: Kinect is centimetres
KINECT_UNIT_SCALE = 0.01

_FILENAME_RE = re.compile(r"m(?P<movement>\d+)_s(?P<subject>\d+)_e(?P<episode>\d+)", re.I)


def euler_yxz_to_matrix(angles_deg: np.ndarray) -> np.ndarray:
    """YXZ Euler triplets (degrees) -> rotation matrices, shape ``(..., 3, 3)``"""
    a = np.radians(np.asarray(angles_deg, dtype=np.float64))
    y, x, z = a[..., 0], a[..., 1], a[..., 2]
    cy, sy, cx, sx, cz, sz = np.cos(y), np.sin(y), np.cos(x), np.sin(x), np.cos(z), np.sin(z)
    zero, one = np.zeros_like(cy), np.ones_like(cy)
    ry = np.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], -1).reshape(*cy.shape, 3, 3)
    rx = np.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], -1).reshape(*cx.shape, 3, 3)
    rz = np.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], -1).reshape(*cz.shape, 3, 3)
    return ry @ rx @ rz


def kinect_world_positions(offsets: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Forward kinematics: Brekel local offsets + Euler angles -> world positions"""
    offsets = np.asarray(offsets, dtype=np.float64)
    rotations = euler_yxz_to_matrix(angles_deg)
    n_frames, n_joints = offsets.shape[0], offsets.shape[1]

    world_pos = np.zeros((n_frames, n_joints, 3), dtype=np.float64)
    world_rot = np.zeros((n_frames, n_joints, 3, 3), dtype=np.float64)

    order = _topological_order()
    for joint in order:
        parent = KINECT_HIERARCHY[joint]
        if parent < 0:
            world_pos[:, joint] = offsets[:, joint]
            world_rot[:, joint] = rotations[:, joint]
        else:
            world_rot[:, joint] = world_rot[:, parent] @ rotations[:, joint]
            local = np.einsum("tij,tj->ti", world_rot[:, parent], offsets[:, joint])
            world_pos[:, joint] = world_pos[:, parent] + local
    return world_pos


def _topological_order() -> list[int]:
    """Joint indices with every parent before its children"""
    order, placed = [], set()
    remaining = set(KINECT_HIERARCHY)
    while remaining:
        progressed = False
        for joint in sorted(remaining):
            parent = KINECT_HIERARCHY[joint]
            if parent < 0 or parent in placed:
                order.append(joint)
                placed.add(joint)
                remaining.discard(joint)
                progressed = True
        if not progressed:
            raise ValueError("KINECT_HIERARCHY contains a cycle")
    return order


class UiPrmdAdapter(DatasetAdapter):
    """Adapter for UI-PRMD segmented movement files"""

    name = "ui_prmd"

    AXIS_TRANSFORM = VICON_AXIS_TRANSFORM
    UNIT_SCALE = VICON_UNIT_SCALE

    # Below this many frames the filters and smoothness metrics are unusable
    MIN_FRAMES = 20

    def __init__(
        self,
        root: str | Path,
        *,
        sensor: str = "kinect",
        segmented: bool = True,
        unit_scale: float | None = None,
    ) -> None:
        super().__init__(root)
        sensor = sensor.lower()
        if sensor not in ("kinect", "vicon"):
            raise ValueError(f"sensor must be 'kinect' or 'vicon', got {sensor!r}")
        self.sensor = sensor
        self.segmented = segmented
        if sensor == "kinect":
            self.fps = KINECT_FPS
            self.AXIS_TRANSFORM = KINECT_AXIS_TRANSFORM
            self.UNIT_SCALE = KINECT_UNIT_SCALE
        else:
            self.fps = VICON_FPS
            self.AXIS_TRANSFORM = VICON_AXIS_TRANSFORM
            self.UNIT_SCALE = VICON_UNIT_SCALE
        if unit_scale is not None:
            self.UNIT_SCALE = unit_scale

    def _position_files(self) -> list[Path]:
        """Position files for the selected sensor only"""
        out = []
        for path in sorted(self.root.rglob("*_positions*.txt")):
            parts = {p.lower() for p in path.parts}
            if self.sensor in parts and "positions" in parts:
                in_segmented = any(p.lower().endswith("segmented movements") for p in path.parts)
                if in_segmented == self.segmented:
                    out.append(path)
        return out

    def iter_trials(self) -> Iterator[RawTrial]:
        files = self._position_files()
        if not files:
            raise FileNotFoundError(f"{self.name}: no {self.sensor} position files under {self.root}")
        for path in files:
            trial = self._load_trial(path)
            if trial is not None:
                yield trial

    def _load_trial(self, path: Path) -> RawTrial | None:
        match = _FILENAME_RE.search(path.name)
        if match is None:
            return None
        movement = f"m{int(match.group('movement')):02d}"
        exercise_id = UI_PRMD_EXERCISE_MAP.get(movement)
        if exercise_id is None:
            return None

        subject = f"s{int(match.group('subject')):02d}"
        side = UI_PRMD_SIDE_BY_SUBJECT.get(subject)
        if side is None:
            return None

        source = self._read_source(path)
        if source is None or source.shape[0] < self.MIN_FRAMES:
            return None

        joint_map = KINECT_JOINT_MAP if self.sensor == "kinect" else VICON_JOINT_MAP
        canonical = self.to_canonical(source, joint_map)
        confidence = _dropout_confidence(source, joint_map)

        correct = "incorrect" not in str(path).lower()
        sequence = PoseSequence(
            joints=canonical,
            fps=self.fps,
            confidence=confidence,
            meta={
                "dataset": self.name,
                "sensor": self.sensor,
                "movement": movement,
                "episode": int(match.group("episode")),
                "source_path": str(path),
            },
        )
        return RawTrial(
            sequence=sequence,
            subject_id=f"{self.name}:{subject}",
            exercise_id=exercise_id,
            side=side,
            correct=correct,
            pre_segmented=self.segmented,
            meta={
                "movement": movement,
                "sensor": self.sensor,
                "episode": int(match.group("episode")),
                "documented_side": side.name,
            },
        )

    def _read_source(self, path: Path) -> np.ndarray | None:
        """Reading one trial into ``(T, J_src, 3)`` world positions"""
        table = _read_table(path)
        if table is None:
            return None

        if self.sensor == "vicon":
            expected = N_VICON_MARKERS * 3
            if table.shape[1] != expected:
                raise ValueError(
                    f"{path}: expected {expected} columns ({N_VICON_MARKERS} markers x 3), "
                    f"got {table.shape[1]}. Check VICON_MARKERS against the dataset's joint-order table."
                )
            return table.reshape(table.shape[0], N_VICON_MARKERS, 3)

        expected = N_KINECT_JOINTS * 3
        if table.shape[1] != expected:
            raise ValueError(
                f"{path}: expected {expected} columns ({N_KINECT_JOINTS} joints x 3), got {table.shape[1]}."
            )
        offsets = table.reshape(table.shape[0], N_KINECT_JOINTS, 3)

        angles = _read_table(_angles_path(path))
        if angles is None or angles.shape != table.shape:
            raise FileNotFoundError(
                f"{path}: the companion angles file is missing or mis-shaped. Kinect positions in this corpus are"
                "offsets relative to the parent joint, so world coordinates cannot be reconstructed without it."
            )
        return kinect_world_positions(
            offsets, angles.reshape(angles.shape[0], N_KINECT_JOINTS, 3)
        )


def _angles_path(positions_path: Path) -> Path:
    """The ``_angles`` file paired with a ``_positions`` file"""
    name = positions_path.name.replace("positions", "angles")
    return positions_path.parent.parent / "Angles" / name


def _read_table(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        return np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    except (ValueError, OSError):
        return None


def _dropout_confidence(
    source: np.ndarray, joint_map: dict[Joint, int | tuple[int, ...]]
) -> np.ndarray:
    """Zero confidence where a source marker dropped out"""
    missing = np.all(np.abs(source) < 1e-9, axis=-1)  # (T, J_src)
    out = np.ones((source.shape[0], N_JOINTS), dtype=np.float32)
    for canonical, src in joint_map.items():
        idx = list(src) if isinstance(src, tuple) else [int(src)]
        out[:, int(canonical)] = (~missing[:, idx].any(axis=1)).astype(np.float32)
    return out


def verify_side(trial: RawTrial) -> str | None:
    """Re-derive the working side from the data and compare with the documented one"""
    joints = np.asarray(trial.sequence.joints, dtype=np.float64)
    left = np.linalg.norm(joints[:, int(Joint.L_HAND)] - joints[0, int(Joint.L_HAND)], axis=-1).max()
    right = np.linalg.norm(joints[:, int(Joint.R_HAND)] - joints[0, int(Joint.R_HAND)], axis=-1).max()
    observed = Side.LEFT if left > right else Side.RIGHT
    if trial.side is not None and observed is not trial.side:
        return (
            f"documented side {trial.side.name} but the {observed.name} hand moves more "
            f"(L {left:.3f} m vs R {right:.3f} m)"
        )
    return None
