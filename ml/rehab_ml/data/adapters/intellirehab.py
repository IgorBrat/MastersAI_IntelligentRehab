from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import numpy as np

from ...exercises.stroke_upper_limb import (
    ELBOW_FLEXION_EXTENSION,
    SHOULDER_ABDUCTION,
    SHOULDER_FLEXION,
)
from ...schema import PoseSequence, Side
from .base import DatasetAdapter, RawTrial
from .kinect import KINECT_AXIS_TRANSFORM, KINECT_FPS, KINECT_JOINT_MAP, N_KINECT_JOINTS


IRDS_GESTURE_MAP: dict[int, tuple[str | None, Side | None]] = {
    0: (ELBOW_FLEXION_EXTENSION.id, Side.LEFT),   # Elbow Flexion Left
    1: (ELBOW_FLEXION_EXTENSION.id, Side.RIGHT),  # Elbow Flexion Right
    2: (SHOULDER_FLEXION.id, Side.LEFT),          # Shoulder Flexion Left
    3: (SHOULDER_FLEXION.id, Side.RIGHT),         # Shoulder Flexion Right
    4: (SHOULDER_ABDUCTION.id, Side.LEFT),        # Shoulder Abduction Left
    5: (SHOULDER_ABDUCTION.id, Side.RIGHT),       # Shoulder Abduction Right
    6: (None, None),  # Shoulder Forward Elevation -- bilateral, hands clasped
    7: (None, None),  # Side tap Left  -- lower limb
    8: (None, None),  # Side tap Right -- lower limb
}

IRDS_GESTURE_NAMES: dict[int, str] = {
    0: "elbow flexion left", 1: "elbow flexion right",
    2: "shoulder flexion left", 3: "shoulder flexion right",
    4: "shoulder abduction left", 5: "shoulder abduction right",
    6: "shoulder forward elevation", 7: "side tap left", 8: "side tap right",
}

IRDS_CORRECT_LABELS: dict[str, bool | None] = {"1": True, "2": False, "3": None}

IRDS_AFFECTED_SIDE: dict[str, str] = {
    "201": "lower_body", "202": "left", "203": "lower_body", "204": "unknown",
    "205": "left", "206": "right", "207": "right", "209": "right",
    "210": "upper_and_lower_weakness", "211": "left",
    "212": "upper_and_lower_weakness", "213": "right", "214": "unknown",
    "215": "right", "216": "unknown",
}

_FILENAME_RE = re.compile(r"(?P<subject>\d+)_(?P<date>\d+)_(?P<gesture>\d+)_(?P<repetition>\d+)_(?P<correct>\d+)_(?P<position>[A-Za-z-]+)$")


def _cohort(subject: str) -> str:
    if subject.startswith("1"):
        return "physiotherapist"
    if subject.startswith("2"):
        return "patient"
    if subject.startswith("3"):
        return "student"
    return "unknown"


class IntelliRehabAdapter(DatasetAdapter):
    """Adapter for IntelliRehabDS skeleton files"""

    name = "intellirehab"
    AXIS_TRANSFORM = KINECT_AXIS_TRANSFORM
    UNIT_SCALE = 1.0  # Kinect body frames are metres
    MIN_FRAMES = 20

    def __init__(self, root: str | Path, *, variant: str = "Simplified") -> None:
        super().__init__(root)
        self.variant = variant

    def _files(self) -> list[Path]:
        base = self.root / "SkeletonData" / self.variant
        if not base.exists():
            base = self.root / self.variant
        if not base.exists():
            base = self.root
        return sorted(p for p in base.rglob("*.txt") if _FILENAME_RE.match(p.stem))

    def iter_trials(self) -> Iterator[RawTrial]:
        files = self._files()
        if not files:
            raise FileNotFoundError(
                f"{self.name}: no gesture files under {self.root}"
            )
        for path in files:
            trial = self._load_trial(path)
            if trial is not None:
                yield trial

    def _load_trial(self, path: Path) -> RawTrial | None:
        match = _FILENAME_RE.match(path.stem)
        if match is None:
            return None
        subject = match.group("subject")
        gesture = int(match.group("gesture"))
        correct = IRDS_CORRECT_LABELS.get(match.group("correct"))
        if correct is None:
            return None  # CorrectLabel 3: gesture identity itself is unreliable

        exercise_id, side = IRDS_GESTURE_MAP.get(gesture, (None, None))
        if exercise_id is None:
            return None

        positions = self._read_simplified(path)
        if positions is None or positions.shape[0] < self.MIN_FRAMES:
            return None

        sequence = PoseSequence(
            joints=self.to_canonical(positions, KINECT_JOINT_MAP),
            fps=KINECT_FPS,
            meta={
                "dataset": self.name,
                "gesture": gesture,
                "gesture_name": IRDS_GESTURE_NAMES[gesture],
                "position": match.group("position"),
                "repetition": int(match.group("repetition")),
                "source_path": str(path),
            },
        )
        return RawTrial(
            sequence=sequence,
            subject_id=f"{self.name}:{subject}",
            exercise_id=exercise_id,
            side=side,
            correct=correct,
            pre_segmented=True,
            meta={
                "gesture": gesture,
                "cohort": _cohort(subject),
                "affected_side": IRDS_AFFECTED_SIDE.get(subject),
                "position": match.group("position"),
            },
        )

    @staticmethod
    def _read_simplified(path: Path) -> np.ndarray | None:
        """Parsing a Simplified file into ``(T, 25, 3)``"""
        try:
            table = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
        except (ValueError, OSError):
            return None
        if table.ndim != 2 or table.shape[1] != N_KINECT_JOINTS * 3:
            return None
        return table.reshape(table.shape[0], N_KINECT_JOINTS, 3)
