from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Iterable

import numpy as np


class Joint(IntEnum):
    """Canonical joint indices. Order is part of the on-disk data contract"""
    PELVIS = 0
    SPINE_MID = 1
    NECK = 2
    HEAD = 3

    L_SHOULDER = 4
    L_ELBOW = 5
    L_WRIST = 6
    L_HAND = 7

    R_SHOULDER = 8
    R_ELBOW = 9
    R_WRIST = 10
    R_HAND = 11

    L_HIP = 12
    L_KNEE = 13
    L_ANKLE = 14

    R_HIP = 15
    R_KNEE = 16
    R_ANKLE = 17


N_JOINTS = len(Joint)

# parent of each joint, for bone-length constraints and rendering
SKELETON_PARENTS: dict[Joint, Joint] = {
    Joint.PELVIS: Joint.PELVIS,
    Joint.SPINE_MID: Joint.PELVIS,
    Joint.NECK: Joint.SPINE_MID,
    Joint.HEAD: Joint.NECK,
    Joint.L_SHOULDER: Joint.NECK,
    Joint.L_ELBOW: Joint.L_SHOULDER,
    Joint.L_WRIST: Joint.L_ELBOW,
    Joint.L_HAND: Joint.L_WRIST,
    Joint.R_SHOULDER: Joint.NECK,
    Joint.R_ELBOW: Joint.R_SHOULDER,
    Joint.R_WRIST: Joint.R_ELBOW,
    Joint.R_HAND: Joint.R_WRIST,
    Joint.L_HIP: Joint.PELVIS,
    Joint.L_KNEE: Joint.L_HIP,
    Joint.L_ANKLE: Joint.L_KNEE,
    Joint.R_HIP: Joint.PELVIS,
    Joint.R_KNEE: Joint.R_HIP,
    Joint.R_ANKLE: Joint.R_KNEE,
}

BONES: tuple[tuple[Joint, Joint], ...] = tuple(
    (parent, child) for child, parent in SKELETON_PARENTS.items() if child != parent
)


class Side(IntEnum):
    """Which limb an exercise is performed with"""
    LEFT = 0
    RIGHT = 1

    @property
    def other(self) -> "Side":
        return Side.RIGHT if self is Side.LEFT else Side.LEFT

    @property
    def short(self) -> str:
        return "L" if self is Side.LEFT else "R"

    @classmethod
    def parse(cls, value: "str | int | Side") -> "Side":
        if isinstance(value, Side):
            return value
        if isinstance(value, int):
            return cls(value)
        key = value.strip().lower()
        if key in ("l", "left", "ліва", "ліво"):
            return cls.LEFT
        if key in ("r", "right", "права", "право"):
            return cls.RIGHT
        raise ValueError(f"cannot interpret {value!r} as a body side")


def joint_of(side: Side, name: str) -> Joint:
    """Resolving a side-relative joint name, e.g. ``joint_of(Side.LEFT, "SHOULDER")``"""
    return Joint[f"{side.short}_{name.upper()}"]


@dataclass
class PoseSequence:
    """A time-ordered sequence of canonical skeletons"""
    joints: np.ndarray
    fps: float
    confidence: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.joints = np.asarray(self.joints, dtype=np.float32)
        if self.joints.ndim != 3 or self.joints.shape[1:] != (N_JOINTS, 3):
            raise ValueError(
                f"joints must have shape (T, {N_JOINTS}, 3), got {self.joints.shape}"
            )
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.confidence is None:
            self.confidence = np.ones(self.joints.shape[:2], dtype=np.float32)
        else:
            self.confidence = np.asarray(self.confidence, dtype=np.float32)
            if self.confidence.shape != self.joints.shape[:2]:
                raise ValueError(
                    f"confidence shape {self.confidence.shape} does not match "
                    f"joints shape {self.joints.shape[:2]}"
                )

    @property
    def n_frames(self) -> int:
        return int(self.joints.shape[0])

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps

    def __len__(self) -> int:
        return self.n_frames

    def time_axis(self) -> np.ndarray:
        return np.arange(self.n_frames, dtype=np.float32) / self.fps

    def slice(self, start: int, stop: int) -> "PoseSequence":
        """Frame-range slice that preserves ``meta`` (with the offset recorded)"""
        meta = dict(self.meta)
        meta["frame_offset"] = int(meta.get("frame_offset", 0)) + int(start)
        return PoseSequence(
            joints=self.joints[start:stop],
            fps=self.fps,
            confidence=self.confidence[start:stop],
            meta=meta,
        )

    def with_joints(self, joints: np.ndarray) -> "PoseSequence":
        """Copy carrying replacement positions (used by filters / normalisers)"""
        return PoseSequence(
            joints=joints, fps=self.fps, confidence=self.confidence, meta=dict(self.meta)
        )

    def trajectory(self, joint: Joint) -> np.ndarray:
        """``(T, 3)`` path of a single joint"""
        return self.joints[:, int(joint), :]

    def mirrored(self) -> "PoseSequence":
        """Left-right mirror image"""
        swapped = self.joints.copy()
        conf = self.confidence.copy()
        for name in ("SHOULDER", "ELBOW", "WRIST", "HAND", "HIP", "KNEE", "ANKLE"):
            left, right = Joint[f"L_{name}"], Joint[f"R_{name}"]
            swapped[:, [left, right]] = swapped[:, [right, left]]
            conf[:, [left, right]] = conf[:, [right, left]]
        swapped[..., 0] *= -1.0  # flip the left-right axis
        meta = dict(self.meta)
        if "side" in meta:
            meta["side"] = int(Side.parse(meta["side"]).other)
        meta["mirrored"] = not bool(meta.get("mirrored", False))
        return PoseSequence(joints=swapped, fps=self.fps, confidence=conf, meta=meta)

    def low_confidence_joints(self, threshold: float = 0.5) -> list[Joint]:
        """Joints whose median confidence is below ``threshold``"""
        median = np.median(self.confidence, axis=0)
        return [Joint(i) for i in np.flatnonzero(median < threshold)]

    def torso_length(self) -> float:
        """Median pelvis-to-neck distance; the natural scale unit for a subject"""
        d = np.linalg.norm(
            self.joints[:, Joint.NECK, :] - self.joints[:, Joint.PELVIS, :], axis=-1
        )
        return float(np.median(d))


def concat(sequences: Iterable[PoseSequence]) -> PoseSequence:
    """Concatenate sequences along time. All must share an fps"""
    seqs = list(sequences)
    if not seqs:
        raise ValueError("cannot concatenate an empty sequence list")
    fps = seqs[0].fps
    if any(abs(s.fps - fps) > 1e-6 for s in seqs):
        raise ValueError("cannot concatenate sequences with differing fps")
    return PoseSequence(
        joints=np.concatenate([s.joints for s in seqs], axis=0),
        fps=fps,
        confidence=np.concatenate([s.confidence for s in seqs], axis=0),
        meta=dict(seqs[0].meta),
    )
