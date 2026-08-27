from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import openpyxl

from ...exercises.stroke_upper_limb import SHOULDER_FLEXION
from ...schema import N_JOINTS, PoseSequence
from .base import DatasetAdapter, RawTrial
from .kinect import (
    KINECT_AXIS_TRANSFORM,
    KINECT_FPS,
    KINECT_JOINT_MAP,
    N_KINECT_JOINTS,
    infer_values_per_joint,
    reshape_flat_skeleton,
    tracking_state_to_confidence,
)

# KIMORE exercise code -> our protocol id. ``None`` means "no counterpart, skip".
KIMORE_EXERCISE_MAP: dict[str, str | None] = {
    "Es1": SHOULDER_FLEXION.id,
    "Es2": None,
    "Es3": None,                   # trunk rotation
    "Es4": None,                   # pelvis rotation
    "Es5": None,                   # squatting
    "Es6": None,
}

_SCORE_MAX = {"TS": 50.0, "PO": 15.0, "CF": 35.0}
# The component used as the headline clinician score
_PRIMARY_SCORE = "TS"


class KimoreAdapter(DatasetAdapter):
    """Adapter for the KIMORE corpus"""

    name = "kimore"
    AXIS_TRANSFORM = KINECT_AXIS_TRANSFORM
    UNIT_SCALE = 1.0  # KIMORE joint positions are already in metres

    def iter_trials(self) -> Iterator[RawTrial]:
        files = sorted(self.root.rglob("JointPosition*.csv"))
        if not files:
            raise FileNotFoundError(f"{self.name}: no JointPosition*.csv under {self.root}")
        for path in files:
            trial = self._load_trial(path)
            if trial is not None:
                yield trial

    def _load_trial(self, path: Path) -> RawTrial | None:
        code = self._exercise_code(path)
        if code is None:
            return None
        mapped = KIMORE_EXERCISE_MAP.get(code)
        if mapped is None:
            return None

        positions, confidence = self._read_joint_positions(path)
        if positions is None or positions.shape[0] < 10:
            return None

        canonical = self.to_canonical(positions, KINECT_JOINT_MAP)
        subject, group = self._subject_and_group(path)
        score, components = self._read_clinical_score(path, code)

        sequence = PoseSequence(
            joints=canonical,
            fps=KINECT_FPS,
            confidence=self._canonical_confidence(confidence, canonical.shape[0]),
            meta={
                "dataset": self.name,
                "group": group,
                "kimore_exercise": code,
                "source_path": str(path),
            },
        )
        return RawTrial(
            sequence=sequence,
            subject_id=f"{self.name}:{subject}",
            exercise_id=mapped,
            side=None,
            clinician_score=score,
            meta={"group": group, "kimore_exercise": code,
                  "clinical_components": components},
        )

    @staticmethod
    def _exercise_code(path: Path) -> str | None:
        for part in reversed(path.parts):
            match = re.fullmatch(r"[Ee]s(\d+)", part)
            if match:
                return f"Es{match.group(1)}"
        match = re.search(r"[Ee]s(\d+)", path.name)
        return f"Es{match.group(1)}" if match else None

    @staticmethod
    def _subject_and_group(path: Path) -> tuple[str, str]:
        """Recover the subject directory and cohort"""
        parts = list(path.parts)
        for i, part in enumerate(parts):
            if re.fullmatch(r"[Ee]s\d+", part) and i >= 1:
                return parts[i - 1], (parts[i - 2] if i >= 2 else "unknown")
        return path.parent.name, path.parent.parent.name

    def _read_joint_positions(
        self, path: Path
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Parsing a JointPosition CSV into ``(T, 25, 3)`` positions"""
        rows: list[list[float]] = []
        with path.open("r", newline="") as handle:
            for raw in csv.reader(handle):
                values: list[float] = []
                for cell in raw:
                    cell = cell.strip()
                    if not cell:
                        continue
                    try:
                        values.append(float(cell))
                    except ValueError:
                        values = []
                        break  # header or comment row
                if values:
                    rows.append(values)
        if not rows:
            return None, None

        widths = np.array([len(r) for r in rows])
        modal = int(np.bincount(widths).argmax())
        table = np.array([r for r in rows if len(r) == modal], dtype=np.float64)
        if table.shape[0] < 10:
            return None, None

        stride, table = self._resolve_stride(table, modal, path)
        positions, extra = reshape_flat_skeleton(
            table, values_per_joint=stride, n_joints=N_KINECT_JOINTS
        )
        return positions, (extra[:, :, 0] if extra is not None else None)

    @staticmethod
    def _resolve_stride(
        table: np.ndarray, modal: int, path: Path
    ) -> tuple[int, np.ndarray]:
        """Determine per-joint stride, stripping leading index columns if needed"""
        try:
            return infer_values_per_joint(modal, N_KINECT_JOINTS), table
        except ValueError:
            pass
        for lead in (1, 2):
            try:
                stride = infer_values_per_joint(modal - lead, N_KINECT_JOINTS)
            except ValueError:
                continue
            return stride, table[:, lead:]
        raise ValueError(
            f"{path}: {modal} columns is not 25 Kinect joints at stride 3 or 4, with or without leading index columns. Inspect the file header."
        )

    @staticmethod
    def _canonical_confidence(
        kinect_conf: np.ndarray | None, n_frames: int
    ) -> np.ndarray | None:
        """Mapping the per-joint fourth value onto canonical confidences"""
        if kinect_conf is None:
            return None
        kinect_conf = tracking_state_to_confidence(kinect_conf)
        out = np.ones((n_frames, N_JOINTS), dtype=np.float32)
        for canonical, source in KINECT_JOINT_MAP.items():
            if isinstance(source, tuple):
                out[:, int(canonical)] = kinect_conf[:, list(source)].min(axis=1)
            else:
                out[:, int(canonical)] = kinect_conf[:, int(source)]
        return np.clip(out, 0.0, 1.0)

    def _read_clinical_score(self, path: Path, code: str) -> tuple[float | None, dict]:
        """Returning ``(TS as 0-100, all components)`` for this trial's exercise"""
        number = code[2:]
        for directory in (path.parent, path.parent.parent, path.parent.parent.parent):
            if not directory.exists():
                continue
            for candidate in sorted(directory.rglob("ClinicalAssessment*.xlsx")):
                components = _read_assessment(candidate, number)
                if components:
                    primary = components.get(_PRIMARY_SCORE)
                    return (
                        None if primary is None else 100.0 * primary / _SCORE_MAX[_PRIMARY_SCORE],
                        components,
                    )
        return None, {}


def _read_assessment(path: Path, exercise_number: str) -> dict[str, float]:
    """Pull the TS / PO / CF scores for one exercise out of a KIMORE workbook"""
    try:
        workbook = openpyxl.load_workbook(path, data_only=True, read_only=True)
    except OSError:
        return {}
    try:
        sheet = workbook.worksheets[0]
        rows = sheet.iter_rows(values_only=True)
        header = next(rows, None)
        if header is None:
            return {}
        wanted = {}
        for index, name in enumerate(header):
            if not isinstance(name, str):
                continue
            match = re.search(r"clinical\s+(TS|PO|CF)\s*Ex#?\s*(\d+)", name, re.I)
            if match and match.group(2) == exercise_number:
                wanted[match.group(1).upper()] = index
        if not wanted:
            return {}
        for row in rows:
            values = {}
            for component, index in wanted.items():
                if index < len(row) and isinstance(row[index], (int, float)):
                    values[component] = float(row[index])
            if values:
                return values
        return {}
    finally:
        workbook.close()
