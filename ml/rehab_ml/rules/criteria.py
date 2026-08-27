from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exercises.base import CompensationLimit, Direction, ExerciseDefinition, Tolerance
from ..features.angles import AngleSeries
from ..taxonomy import ERROR_SPECS, ErrorType, Severity

# Deviation is summarised by this percentile of the gated frames rather than by the maximum, 
# so one bad pose estimate cannot manufacture a SEVERE finding
DEVIATION_PERCENTILE = 90.0

# Frames at the start of a repetition used to establish a baseline-relative zero
BASELINE_FRAMES = 5


@dataclass(frozen=True)
class Finding:
    """One criterion's verdict on one repetition"""

    error: ErrorType
    severity: Severity
    measured: float
    threshold: float
    unit: str

    @property
    def triggered(self) -> bool:
        return self.severity > Severity.NONE

    @property
    def exceedance(self) -> float:
        """How far past the mild threshold the measurement went, ``>= 0``"""
        return max(0.0, self.measured - self.threshold)

    @property
    def _decimals(self) -> int:
        """Digits to print. TL channels are small numbers - a scapularhike tolerance of 0.04 rounds 
        to "0.0" at one decimal, which reads as no tolerance at all in a clinician-facing report
        """
        return 3 if self.unit.strip() == "TL" else 1

    def describe_en(self) -> str:
        spec = ERROR_SPECS[self.error]
        d = self._decimals
        if not self.triggered:
            return f"{spec.name_short}: within tolerance ({self.measured:.{d}f}{self.unit})"
        return (
            f"{spec.name_short}: {self.severity.name} "
            f"({self.measured:.{d}f}{self.unit} vs tolerance {self.threshold:.{d}f}{self.unit})"
        )


def grade(deviation: float, tolerance: Tolerance) -> Severity:
    """Mapping a non-negative deviation onto a severity band"""
    d = float(deviation)
    if d <= tolerance.mild:
        return Severity.NONE
    if d <= tolerance.moderate:
        return Severity.MILD
    if d <= tolerance.severe:
        return Severity.MODERATE
    return Severity.SEVERE


def normalised_penalty(deviation: float, tolerance: Tolerance) -> float:
    """Continuous penalty in ``[0, 1]`` for use by the quality index"""
    span = tolerance.severe - tolerance.mild
    if span <= 1e-9:
        return 1.0 if deviation > tolerance.mild else 0.0
    return float(np.clip((deviation - tolerance.mild) / span, 0.0, 1.0))


# Units for reporting. Torso-length channels are dimensionless
_CHANNEL_UNITS: dict[str, str] = {
    "shoulder_hike": " TL",
    "hand_forward": " TL",
    "hand_up": " TL",
    "hand_lateral": " TL",
    "shoulder_height_working": " TL",
    "shoulder_height_other": " TL",
}


def channel_unit(channel: str) -> str:
    return _CHANNEL_UNITS.get(channel, "°")


def resolve_channel(
    angles: AngleSeries, channel: str, exercise: ExerciseDefinition
) -> np.ndarray:
    """Materialising a channel, including the derived pseudo-channels"""
    if channel == "trunk_lean_contralateral":
        return angles.trunk_lean_contralateral()
    if channel == "shoulder_hike":
        return angles.shoulder_hike(baseline_frames=BASELINE_FRAMES)
    if channel == "plane_deviation":
        if exercise.target_plane_deg is None:
            return np.zeros(angles.n_frames, dtype=np.float64)
        return angles.plane_deviation_from(exercise.target_plane_deg)
    if channel == "elbow_flexion_coupling":
        # The synergy signal is elbow flexion *relative to how the repetition
        # started*; the limit carries baseline_relative=True to subtract it.
        return angles["elbow_flexion"]
    return angles[channel]


def gate_mask(
    angles: AngleSeries, exercise: ExerciseDefinition, gate_frac: float
) -> np.ndarray:
    """Frames where the primary channel has progressed past ``gate_frac`` of target"""
    primary = np.asarray(angles[exercise.primary_channel], dtype=np.float64)
    if gate_frac <= 0.0:
        return np.ones(primary.size, dtype=bool)

    k = max(1, min(BASELINE_FRAMES, primary.size))
    baseline = float(np.median(primary[:k]))
    excursion = abs(exercise.rom.target - baseline)
    if excursion < 1e-9:
        return np.ones(primary.size, dtype=bool)
    return np.abs(primary - baseline) >= gate_frac * excursion


def evaluate_compensation(
    angles: AngleSeries, exercise: ExerciseDefinition, limit: CompensationLimit
) -> Finding:
    """Evaluating one compensation limit over one repetition"""
    series = np.asarray(
        resolve_channel(angles, limit.channel, exercise), dtype=np.float64
    )

    if limit.baseline_relative:
        k = max(1, min(BASELINE_FRAMES, series.size))
        series = series - float(np.median(series[:k]))

    deviation = series if limit.direction is Direction.POSITIVE else np.abs(series)

    mask = gate_mask(angles, exercise, limit.gate_primary_frac)
    gated = deviation[mask]
    if gated.size == 0:
        # movement never got far enough for this limit to apply
        measured = 0.0
    else:
        measured = float(np.percentile(gated, DEVIATION_PERCENTILE))
    measured = max(measured, 0.0)

    return Finding(
        error=limit.error,
        severity=grade(measured, limit.tolerance),
        measured=measured,
        threshold=limit.tolerance.mild,
        unit=channel_unit(limit.channel),
    )


def compensation_penalty(
    angles: AngleSeries, exercise: ExerciseDefinition, limit: CompensationLimit
) -> tuple[Finding, float]:
    """`evaluate_compensation` plus its continuous ``[0, 1]`` penalty"""
    finding = evaluate_compensation(angles, exercise, limit)
    return finding, normalised_penalty(finding.measured, limit.tolerance)
