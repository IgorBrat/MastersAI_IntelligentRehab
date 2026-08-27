from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from ...schema import Joint, N_JOINTS, PoseSequence, Side


@dataclass
class RawTrial:
    """One recorded execution of one exercise by one subject, before windowing"""

    sequence: PoseSequence
    subject_id: str
    exercise_id: str | None
    side: Side | None = None
    correct: bool | None = None
    clinician_score: float | None = None
    errors: tuple[str, ...] = ()
    # True when the file already contains exactly one repetition
    pre_segmented: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def has_supervision(self) -> bool:
        return self.correct is not None or self.clinician_score is not None or bool(self.errors)


class DatasetAdapter(ABC):
    """Base class for corpus adapters"""

    name: str = "unnamed"

    # Axis transform from source to canonical coordinates
    AXIS_TRANSFORM: np.ndarray = np.eye(3, dtype=np.float64)

    # Multiplier converting source units to metres
    UNIT_SCALE: float = 1.0

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        if not self.root.exists():
            raise FileNotFoundError(f"{self.name}: dataset root not found: {self.root}")

    @abstractmethod
    def iter_trials(self) -> Iterator[RawTrial]:
        """Yielding every usable trial in the corpus"""

    def __iter__(self) -> Iterator[RawTrial]:
        return self.iter_trials()

    def to_canonical(
        self, source_joints: np.ndarray, joint_map: dict[Joint, int | tuple[int, ...]]
    ) -> np.ndarray:
        """Remap and reorient a ``(T, J_src, 3)`` array to ``(T, 18, 3)``"""
        src = np.asarray(source_joints, dtype=np.float64)
        if src.ndim != 3 or src.shape[-1] != 3:
            raise ValueError(f"expected (T, J, 3) source joints, got {src.shape}")

        t = src.shape[0]
        out = np.zeros((t, N_JOINTS, 3), dtype=np.float64)
        for canonical, source in joint_map.items():
            if isinstance(source, tuple):
                out[:, int(canonical), :] = src[:, list(source), :].mean(axis=1)
            else:
                out[:, int(canonical), :] = src[:, int(source), :]

        out = out @ np.asarray(self.AXIS_TRANSFORM, dtype=np.float64).T
        return (out * self.UNIT_SCALE).astype(np.float32)


def verify_adapter(trial: RawTrial) -> list[str]:
    """Sanity-check an adapter's output. Returns a list of problems found"""
    problems: list[str] = []
    j = trial.sequence.joints.astype(np.float64)

    if np.median(j[:, Joint.HEAD, 1] - j[:, Joint.PELVIS, 1]) <= 0:
        problems.append("head is not above pelvis: the vertical axis is inverted (check AXIS_TRANSFORM row for y)")

    if np.median(j[:, Joint.L_SHOULDER, 0] - j[:, Joint.R_SHOULDER, 0]) <= 0:
        problems.append("left shoulder is not on the +x side of the right: left and right are swapped (check JOINT_MAP and the x axis sign)")

    torso = float(np.median(np.linalg.norm(j[:, Joint.NECK] - j[:, Joint.PELVIS], axis=-1)))
    if not (0.20 <= torso <= 0.90):
        problems.append(f"implausible torso length {torso:.3f} m (expected 0.20-0.90): check UNIT_SCALE")

    for parent, child in ((Joint.L_SHOULDER, Joint.L_ELBOW), (Joint.L_HIP, Joint.L_KNEE)):
        lengths = np.linalg.norm(j[:, child] - j[:, parent], axis=-1)
        median = float(np.median(lengths))
        if median <= 1e-6:
            problems.append(f"zero-length bone {parent.name}->{child.name}: joint map error")
            continue
        variation = float(np.std(lengths) / median)
        if variation > 0.25:
            problems.append(f"bone {parent.name}->{child.name} varies by {variation:.0%} of its length: joint ordering is probably wrong")

    return problems
