from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..exercises.base import ExerciseDefinition
from ..taxonomy import ErrorType
from .angles import ANGLE_CHANNELS, AngleSeries
from .kinematics import (
    ldlj,
    n_submovements,
    rom_metrics,
    smooth_signal,
    sparc,
    tempo_metrics,
)

# Fixed sequence length every repetition is resampled to.
DEFAULT_SEQUENCE_LENGTH = 64


def resample(x: np.ndarray, length: int) -> np.ndarray:
    """Linearly resample along axis 0 to exactly ``length`` frames"""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n == length:
        return x.copy()
    if n < 2:
        return np.repeat(x.reshape(1, -1) if x.ndim == 1 else x[:1], length, axis=0)
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, length)
    if x.ndim == 1:
        return np.interp(dst, src, x)
    return np.stack([np.interp(dst, src, x[:, c]) for c in range(x.shape[1])], axis=-1)


@dataclass(frozen=True)
class FeatureConfig:
    """What to include in the extracted representations"""

    sequence_length: int = DEFAULT_SEQUENCE_LENGTH
    channels: tuple[str, ...] = ANGLE_CHANNELS
    # Append per-channel first derivatives (angular velocity).
    include_velocity: bool = True
    # Subtract each channel's value at the start of the repetition
    baseline_centre: bool = True
    smooth_cutoff_hz: float = 6.0

    @property
    def n_sequence_channels(self) -> int:
        return len(self.channels) * (2 if self.include_velocity else 1)


def sequence_features(
    angles: AngleSeries, config: FeatureConfig | None = None
) -> np.ndarray:
    """``(sequence_length, n_sequence_channels)`` model input for one repetition"""
    cfg = config or FeatureConfig()
    raw = np.stack(
        [np.asarray(angles[name], dtype=np.float64) for name in cfg.channels], axis=-1
    )
    if raw.shape[0] >= 5:
        raw = smooth_signal(raw, angles.fps, cutoff_hz=cfg.smooth_cutoff_hz, axis=0)

    blocks = [raw]
    if cfg.include_velocity:
        blocks.append(np.gradient(raw, 1.0 / angles.fps, axis=0))

    stacked = np.concatenate(blocks, axis=-1)
    out = resample(stacked, cfg.sequence_length)

    if cfg.baseline_centre:
        n_pos = len(cfg.channels)
        out[:, :n_pos] -= out[:1, :n_pos]

    return out.astype(np.float32)


SUMMARY_FEATURE_NAMES: tuple[str, ...] = (
    "primary_peak",
    "primary_excursion",
    "primary_rom_ratio",
    "primary_return_error",
    "duration_s",
    "concentric_s",
    "eccentric_s",
    "hold_s",
    "hold_ratio",
    "concentric_eccentric_ratio",
    "sparc",
    "ldlj",
    "n_submovements",
    "peak_angular_velocity",
    "trunk_lean_peak",
    "trunk_flexion_peak",
    "trunk_rotation_peak",
    "shoulder_hike_peak",
    "elbow_coupling_peak",
    "plane_deviation_peak",
    "elbow_flexion_peak",
    "elbow_flexion_mean",
)
N_SUMMARY_FEATURES = len(SUMMARY_FEATURE_NAMES)


def summary_features(
    angles: AngleSeries, exercise: ExerciseDefinition
) -> np.ndarray:
    """``(N_SUMMARY_FEATURES,)`` clinically-named vector for one repetition"""
    primary = np.asarray(angles[exercise.primary_channel], dtype=np.float64)
    rom = rom_metrics(primary)
    tempo = tempo_metrics(primary, angles.fps)

    if primary.size >= 5:
        smoothed = smooth_signal(primary, angles.fps)
    else:
        smoothed = primary
    velocity = np.gradient(smoothed, 1.0 / angles.fps)
    speed = np.abs(velocity)

    target_excursion = abs(exercise.rom.target - rom.start)
    rom_ratio = (
        abs(rom.peak - rom.start) / target_excursion if target_excursion > 1e-9 else 1.0
    )
    hold_ratio = (
        tempo.hold_s / exercise.tempo.hold_s if exercise.tempo.hold_s > 1e-9 else 1.0
    )

    def peak(series: np.ndarray) -> float:
        s = np.asarray(series, dtype=np.float64)
        if s.size == 0:
            return 0.0
        return float(np.percentile(np.abs(s), 90))

    lean = angles.trunk_lean_contralateral()
    hike = angles.shoulder_hike()
    plane_dev = (
        angles.plane_deviation_from(exercise.target_plane_deg)
        if exercise.target_plane_deg is not None
        else np.zeros(angles.n_frames)
    )
    elbow = np.asarray(angles["elbow_flexion"], dtype=np.float64)
    elbow_coupling = elbow - float(np.median(elbow[: max(1, min(5, elbow.size))]))

    values = {
        "primary_peak": float(rom.peak),
        "primary_excursion": float(rom.excursion),
        "primary_rom_ratio": float(np.clip(rom_ratio, 0.0, 3.0)),
        "primary_return_error": float(rom.return_error),
        "duration_s": float(tempo.duration_s),
        "concentric_s": float(tempo.concentric_s),
        "eccentric_s": float(tempo.eccentric_s),
        "hold_s": float(tempo.hold_s),
        "hold_ratio": float(np.clip(hold_ratio, 0.0, 3.0)),
        "concentric_eccentric_ratio": float(
            np.clip(tempo.concentric_eccentric_ratio, 0.0, 10.0)
        ),
        "sparc": sparc(speed, angles.fps),
        "ldlj": ldlj(speed, angles.fps),
        "n_submovements": float(n_submovements(speed, angles.fps)),
        "peak_angular_velocity": float(np.percentile(speed, 95)) if speed.size else 0.0,
        "trunk_lean_peak": peak(lean - float(np.median(lean[:5]))),
        "trunk_flexion_peak": peak(
            angles["trunk_forward_flexion"]
            - float(np.median(angles["trunk_forward_flexion"][:5]))
        ),
        "trunk_rotation_peak": peak(
            angles["trunk_rotation"] - float(np.median(angles["trunk_rotation"][:5]))
        ),
        "shoulder_hike_peak": peak(hike),
        "elbow_coupling_peak": peak(elbow_coupling),
        "plane_deviation_peak": peak(plane_dev),
        "elbow_flexion_peak": float(np.max(elbow)) if elbow.size else 0.0,
        "elbow_flexion_mean": float(np.mean(elbow)) if elbow.size else 0.0,
    }

    out = np.array([values[name] for name in SUMMARY_FEATURE_NAMES], dtype=np.float32)
    # SPARC/LDLJ are NaN for degenerate repetitions. Substitute the neutral sentinels the rule engine also uses, so the classical models never see NaN
    out[SUMMARY_FEATURE_NAMES.index("sparc")] = _finite(
        out[SUMMARY_FEATURE_NAMES.index("sparc")], exercise.sparc_floor
    )
    out[SUMMARY_FEATURE_NAMES.index("ldlj")] = _finite(
        out[SUMMARY_FEATURE_NAMES.index("ldlj")], 0.0
    )
    return out


def _finite(value: float, fallback: float) -> float:
    return float(value) if np.isfinite(value) else float(fallback)


COMPENSATION_FEATURE_INDEX: dict[ErrorType, int] = {
    ErrorType.TRUNK_LATERAL_LEAN: SUMMARY_FEATURE_NAMES.index("trunk_lean_peak"),
    ErrorType.TRUNK_FORWARD_FLEXION: SUMMARY_FEATURE_NAMES.index("trunk_flexion_peak"),
    ErrorType.TRUNK_ROTATION: SUMMARY_FEATURE_NAMES.index("trunk_rotation_peak"),
    ErrorType.SHOULDER_HIKING: SUMMARY_FEATURE_NAMES.index("shoulder_hike_peak"),
    ErrorType.ELBOW_FLEXION_SYNERGY: SUMMARY_FEATURE_NAMES.index("elbow_coupling_peak"),
    ErrorType.OUT_OF_PLANE_DRIFT: SUMMARY_FEATURE_NAMES.index("plane_deviation_peak"),
    ErrorType.INSUFFICIENT_ROM: SUMMARY_FEATURE_NAMES.index("primary_rom_ratio"),
    ErrorType.INCOMPLETE_RETURN: SUMMARY_FEATURE_NAMES.index("primary_return_error"),
    ErrorType.INSUFFICIENT_HOLD: SUMMARY_FEATURE_NAMES.index("hold_ratio"),
    ErrorType.EXCESSIVE_SPEED: SUMMARY_FEATURE_NAMES.index("duration_s"),
    ErrorType.JERKY_MOVEMENT: SUMMARY_FEATURE_NAMES.index("sparc"),
}
