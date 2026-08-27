from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, savgol_filter


EPS = 1e-12

_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def derivative(x: np.ndarray, fps: float, axis: int = 0) -> np.ndarray:
    """Central-difference time derivative with one-sided ends"""
    return np.gradient(np.asarray(x, dtype=np.float64), 1.0 / fps, axis=axis)


def smooth_signal(x: np.ndarray, fps: float, cutoff_hz: float = 6.0, axis: int = 0) -> np.ndarray:
    """Zero-phase low-pass appropriate for human limb motion"""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[axis]
    nyquist = 0.5 * fps
    normalised = cutoff_hz / nyquist

    if n >= 18 and 0.0 < normalised < 0.95:
        b, a = butter(4, normalised, btype="low")
        padlen = min(3 * max(len(a), len(b)), n - 1)
        return filtfilt(b, a, x, axis=axis, padlen=padlen)

    window = max(5, int(round(fps / max(cutoff_hz, 0.5))) | 1)
    if window >= n:
        window = n if n % 2 == 1 else n - 1
    if window < 5:
        return x  # too short to filter meaningfully
    return savgol_filter(x, window_length=window, polyorder=2, axis=axis, mode="interp")


def speed_profile(
    trajectory: np.ndarray, fps: float, *, smooth: bool = True, cutoff_hz: float = 4.0
) -> np.ndarray:
    """Tangential speed ``(T,)`` of a ``(T, 3)`` joint path, in units/second"""
    path = np.asarray(trajectory, dtype=np.float64)
    if smooth and path.shape[0] >= 5:
        path = smooth_signal(path, fps, cutoff_hz=cutoff_hz, axis=0)
    vel = derivative(path, fps, axis=0)
    return np.linalg.norm(vel, axis=-1)


# Smoothness

def sparc(speed: np.ndarray, fps: float, *, fc_max: float = 10.0, amp_threshold: float = 0.05) -> float:
    """Spectral arc length of a speed profile"""
    v = np.asarray(speed, dtype=np.float64)
    if v.size < 8 or not np.any(np.abs(v) > EPS):
        return float("nan")

    n_fft = int(2 ** np.ceil(np.log2(max(v.size * 4, 64))))
    spectrum = np.abs(np.fft.rfft(v, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fps)
    peak = spectrum.max()
    if peak <= EPS:
        return float("nan")
    spectrum = spectrum / peak

    in_band = freqs <= fc_max
    if in_band.sum() < 4:
        return float("nan")
    freqs, spectrum = freqs[in_band], spectrum[in_band]

    above = np.flatnonzero(spectrum >= amp_threshold)
    cutoff_idx = int(above[-1]) if above.size else len(spectrum) - 1
    cutoff_idx = max(cutoff_idx, 3)
    freqs, spectrum = freqs[: cutoff_idx + 1], spectrum[: cutoff_idx + 1]

    df = np.diff(freqs / freqs[-1])
    dv = np.diff(spectrum)
    return float(-np.sum(np.sqrt(df**2 + dv**2)))


def ldlj(speed: np.ndarray, fps: float) -> float:
    """Log dimensionless jerk of a speed profile. Larger (nearer 0) = smoother"""
    v = np.asarray(speed, dtype=np.float64)
    if v.size < 9:
        return float("nan")
    v = smooth_signal(v, fps)
    v_peak = float(np.max(np.abs(v)))
    duration = v.size / fps
    if v_peak <= EPS or duration <= EPS:
        return float("nan")

    jerk = derivative(derivative(v, fps), fps)
    integral = float(_trapezoid(jerk**2, dx=1.0 / fps))
    dlj = (duration**3 / v_peak**2) * integral
    if dlj <= EPS:
        return float("nan")
    return float(-np.log(dlj))


def n_submovements(speed: np.ndarray, fps: float, *, prominence_ratio: float = 0.10) -> int:
    """Counting velocity peaks, i.e. corrective submovements"""
    v = np.asarray(speed, dtype=np.float64)
    if v.size < 5:
        return 0
    v = smooth_signal(v, fps)
    peak = float(np.max(v))
    if peak <= EPS:
        return 0
    peaks, _ = find_peaks(
        v, prominence=prominence_ratio * peak, distance=max(1, int(round(0.15 * fps)))
    )
    return int(peaks.size)


# Tempo

@dataclass(frozen=True)
class TempoMetrics:
    """Timing decomposition of a single repetition"""
    duration_s: float
    concentric_s: float
    eccentric_s: float
    # Bias-corrected end-range hold; the value clinical criteria are scored on
    hold_s: float
    # Frame index of peak excursion within the repetition
    peak_index: int
    peak_value: float

    hold_raw_s: float = 0.0

    @property
    def concentric_eccentric_ratio(self) -> float:
        """>1 means the lowering phase is faster than the lifting phase"""
        return self.concentric_s / max(self.eccentric_s, EPS)


def tempo_metrics(
    signal: np.ndarray,
    fps: float,
    *,
    hold_band_frac: float = 0.08,
    hold_speed_frac: float = 0.30,
) -> TempoMetrics:
    """Splitting a single-repetition excursion signal into phases"""
    x = np.asarray(signal, dtype=np.float64)
    n = x.size
    if n == 0:
        return TempoMetrics(0.0, 0.0, 0.0, 0.0, 0, float("nan"))

    peak_idx = int(np.argmax(x))
    peak_val = float(x[peak_idx])
    baseline = float(np.min(x))
    excursion = max(peak_val - baseline, EPS)

    amplitude_ok = x >= peak_val - hold_band_frac * excursion
    if n >= 3:
        speed = np.abs(np.gradient(x, 1.0 / fps))
        peak_speed = float(np.max(speed))
        speed_ok = speed <= max(hold_speed_frac * peak_speed, EPS)
    else:
        speed_ok = np.ones(n, dtype=bool)
    stationary = amplitude_ok & speed_ok

    lo = hi = peak_idx
    if stationary[peak_idx]:
        while lo > 0 and stationary[lo - 1]:
            lo -= 1
        while hi < n - 1 and stationary[hi + 1]:
            hi += 1
        hold_frames = hi - lo + 1
    else:
        # The peak was passed through, not held
        hold_frames = 0

    hold_raw_s = hold_frames / fps
    moving_s = max(n / fps - hold_raw_s, 0.0)
    hold_s = max(hold_raw_s - _approach_bias_fraction(hold_speed_frac) * moving_s, 0.0)

    return TempoMetrics(
        duration_s=n / fps,
        concentric_s=lo / fps,
        eccentric_s=(n - 1 - hi) / fps,
        hold_s=hold_s,
        peak_index=peak_idx,
        peak_value=peak_val,
        hold_raw_s=hold_raw_s,
    )


def _approach_bias_fraction(speed_frac: float) -> float:
    """Fraction of the moving phases that any velocity threshold misclassifies as hold"""
    f = float(np.clip(speed_frac, 0.0, 1.0))
    return float((1.0 - np.sqrt(1.0 - np.sqrt(f))) / 2.0)


# RoM

@dataclass(frozen=True)
class RomMetrics:
    """Range-of-motion summary of one repetition's primary angle"""
    start: float
    peak: float
    end: float
    excursion: float

    @property
    def return_error(self) -> float:
        """How far short of the starting position the limb was left, in degrees"""
        return abs(self.end - self.start)


def rom_metrics(signal: np.ndarray, *, edge_frames: int = 3) -> RomMetrics:
    """Extracting start / peak / end of a repetition's primary angle"""
    x = np.asarray(signal, dtype=np.float64)
    if x.size == 0:
        return RomMetrics(float("nan"), float("nan"), float("nan"), 0.0)
    k = max(1, min(edge_frames, x.size))
    start = float(np.median(x[:k]))
    end = float(np.median(x[-k:]))
    peak = float(np.max(x))
    return RomMetrics(start=start, peak=peak, end=end, excursion=peak - start)
