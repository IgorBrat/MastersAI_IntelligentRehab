from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from ..exercises.base import ExerciseDefinition
from ..features.angles import AngleSeries
from ..features.kinematics import smooth_signal


@dataclass(frozen=True)
class Repetition:
    """Frame bounds of one repetition, half-open ``[start, end)``"""
    index: int
    start: int
    end: int
    peak: int
    peak_value: float
    baseline_value: float

    @property
    def n_frames(self) -> int:
        return self.end - self.start

    @property
    def excursion(self) -> float:
        return self.peak_value - self.baseline_value

    def duration_s(self, fps: float) -> float:
        return self.n_frames / fps


@dataclass
class SegmentationConfig:
    """Tunables for repetition detection"""
    # Minimum peak height as a fraction of the prescribed target excursion
    min_excursion_frac: float = 0.35
    # Minimum seconds between successive peaks
    min_peak_distance_s: float = 1.0
    # A valley must fall back to within this fraction of the excursion above baseline for the two peaks around it to count as separate repetitions
    valley_return_frac: float = 0.5
    # Frames of padding kept either side of the detected bounds
    pad_s: float = 0.15
    # Low-pass cutoff applied before peak finding
    cutoff_hz: float = 3.0


def segment_repetitions(
    angles: AngleSeries,
    exercise: ExerciseDefinition,
    config: SegmentationConfig | None = None,
) -> list[Repetition]:
    """Splitting a continuous angle stream into repetitions"""
    cfg = config or SegmentationConfig()
    raw = angles[exercise.primary_channel].astype(np.float64)
    n = raw.size
    if n < 5:
        return []

    signal = smooth_signal(raw, angles.fps, cutoff_hz=cfg.cutoff_hz)

    # Baseline is the resting position: use a low percentile rather than the minimum 
    # so a single bad frame cannot define it
    baseline = float(np.percentile(signal, 10))
    target_excursion = abs(exercise.rom.target - baseline)
    if target_excursion < 1e-6:
        target_excursion = max(float(np.percentile(signal, 90)) - baseline, 1e-6)

    min_height = baseline + cfg.min_excursion_frac * target_excursion
    distance = max(1, int(round(cfg.min_peak_distance_s * angles.fps)))

    peaks, _ = find_peaks(
        signal,
        height=min_height,
        distance=distance,
        prominence=0.25 * cfg.min_excursion_frac * target_excursion,
    )
    if peaks.size == 0:
        return []

    valley_level = baseline + cfg.valley_return_frac * target_excursion
    peaks = _merge_peaks_without_valley(signal, peaks, baseline, cfg.valley_return_frac)
    boundaries = _valleys_between(signal, peaks, valley_level)

    pad = int(round(cfg.pad_s * angles.fps))
    reps: list[Repetition] = []
    for i, peak in enumerate(peaks):
        start = max(0, boundaries[i] - pad)
        end = min(n, boundaries[i + 1] + pad)
        if end - start < 3:
            continue
        reps.append(
            Repetition(
                index=len(reps),
                start=int(start),
                end=int(end),
                peak=int(peak),
                peak_value=float(raw[peak]),
                baseline_value=float(np.median(raw[start : start + max(1, pad)])),
            )
        )
    return reps


def _merge_peaks_without_valley(
    signal: np.ndarray,
    peaks: np.ndarray,
    baseline: float,
    valley_return_frac: float,
) -> np.ndarray:
    """Collapsing peaks that are not separated by a genuine return toward baseline"""
    if peaks.size <= 1:
        return peaks
    kept = [int(peaks[0])]
    for peak in peaks[1:]:
        previous = kept[-1]
        pair_peak = min(float(signal[previous]), float(signal[int(peak)]))
        threshold = baseline + valley_return_frac * (pair_peak - baseline)
        valley = float(np.min(signal[previous : int(peak) + 1]))
        if valley <= threshold:
            kept.append(int(peak))
        elif signal[int(peak)] > signal[previous]:
            kept[-1] = int(peak)  # same repetition; keep its highest point
    return np.asarray(kept, dtype=int)


def _valleys_between(
    signal: np.ndarray, peaks: np.ndarray, valley_level: float
) -> list[int]:
    """Boundary indices around each peak: ``len(peaks) + 1`` values"""
    bounds: list[int] = []

    first = int(peaks[0])
    idx = first
    while idx > 0 and (signal[idx - 1] < signal[idx] or signal[idx] > valley_level):
        idx -= 1
    bounds.append(idx)

    for left, right in zip(peaks[:-1], peaks[1:]):
        segment = signal[left : right + 1]
        bounds.append(int(left + np.argmin(segment)))

    last = int(peaks[-1])
    idx = last
    n = signal.size
    while idx < n - 1 and (signal[idx + 1] < signal[idx] or signal[idx] > valley_level):
        idx += 1
    bounds.append(idx + 1)

    return bounds


class SlidingRepetitionDetector:
    """Repetition detector for the live-feedback path"""

    def __init__(
        self,
        exercise: ExerciseDefinition,
        fps: float,
        *,
        rise_frac: float = 0.55,
        fall_frac: float = 0.30,
        min_duration_s: float = 0.8,
    ) -> None:
        self.exercise = exercise
        self.fps = fps
        self.rise_frac = rise_frac
        self.fall_frac = fall_frac
        self.min_frames = max(2, int(round(min_duration_s * fps)))

        self._baseline: float | None = None
        self._n_seen = 0
        self._in_rep = False
        self._rep_start = 0
        self._peak_value = -np.inf
        self._peak_index = 0
        self._completed = 0

    @property
    def completed_reps(self) -> int:
        return self._completed

    @property
    def in_repetition(self) -> bool:
        return self._in_rep

    def reset(self) -> None:
        self.__init__(self.exercise, self.fps, rise_frac=self.rise_frac,
                      fall_frac=self.fall_frac,
                      min_duration_s=self.min_frames / self.fps)

    def update(self, primary_value: float) -> Repetition | None:
        """Feed one frame's primary-channel value"""
        value = float(primary_value)
        index = self._n_seen
        self._n_seen += 1

        # Establish the resting baseline from the first half-second, then hold it
        if self._baseline is None or index < int(0.5 * self.fps):
            self._baseline = value if self._baseline is None else min(self._baseline, value)
            return None

        excursion = abs(self.exercise.rom.target - self._baseline)
        if excursion < 1e-6:
            return None
        rise = self._baseline + self.rise_frac * excursion
        fall = self._baseline + self.fall_frac * excursion

        if not self._in_rep:
            if value >= rise:
                self._in_rep = True
                self._rep_start = index
                self._peak_value = value
                self._peak_index = index
            return None

        if value > self._peak_value:
            self._peak_value = value
            self._peak_index = index

        if value <= fall and (index - self._rep_start) >= self.min_frames:
            rep = Repetition(
                index=self._completed,
                start=self._rep_start,
                end=index + 1,
                peak=self._peak_index,
                peak_value=self._peak_value,
                baseline_value=self._baseline,
            )
            self._in_rep = False
            self._peak_value = -np.inf
            self._completed += 1
            return rep
        return None
