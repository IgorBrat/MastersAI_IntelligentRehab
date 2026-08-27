from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features.angles import compute_angles
from ..pose.normalize import preprocess

# Channels compared. Trunk channels are graded on magnitude, so they are compared on
# peak absolute value; the rest on peak signed value.
AGREEMENT_CHANNELS: tuple[str, ...] = (
    "shoulder_elevation",
    "shoulder_abduction",
    "shoulder_flexion",
    "plane_of_elevation",
    "shoulder_rotation",
    "elbow_flexion",
    "trunk_forward_flexion",
    "trunk_lateral_tilt",
    "trunk_rotation",
)

# ICC below this is "poor" agreement by the usual convention
ICC_POOR = 0.50

# ICC above this is "good"; 0.90+ is "excellent".
ICC_GOOD = 0.75


@dataclass(frozen=True)
class ChannelAgreement:
    """Agreement statistics for one angle channel"""
    channel: str
    n: int
    bias: float
    sd: float
    loa_low: float
    loa_high: float
    icc: float
    pearson: float
    reference_mean: float

    @property
    def verdict(self) -> str:
        if not np.isfinite(self.icc):
            return "undefined"
        if self.icc >= 0.90:
            return "excellent"
        if self.icc >= ICC_GOOD:
            return "good"
        if self.icc >= ICC_POOR:
            return "moderate"
        return "POOR - do not rely on this channel across sensors"

    def as_dict(self) -> dict:
        return {
            "channel": self.channel,
            "n": self.n,
            "bias": round(self.bias, 3),
            "sd": round(self.sd, 3),
            "loa_95": [round(self.loa_low, 3), round(self.loa_high, 3)],
            "icc_2_1": round(self.icc, 4),
            "pearson_r": round(self.pearson, 4),
            "reference_mean": round(self.reference_mean, 3),
            "verdict": self.verdict,
        }


def icc_2_1(reference: np.ndarray, comparison: np.ndarray) -> float:
    """ICC(2,1): two-way random effects, single measurement, absolute agreement"""
    y = np.stack([np.asarray(reference, float), np.asarray(comparison, float)], axis=1)
    n, k = y.shape
    if n < 3:
        return float("nan")
    grand = y.mean()
    ms_rows = k * ((y.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((y.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    residual = y - y.mean(axis=1, keepdims=True) - y.mean(axis=0, keepdims=True) + grand
    ms_err = (residual ** 2).sum() / ((n - 1) * (k - 1))
    denominator = ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n
    if abs(denominator) < 1e-12:
        return float("nan")
    return float((ms_rows - ms_err) / denominator)


def trial_peaks(angles, channels: tuple[str, ...] = AGREEMENT_CHANNELS) -> dict[str, float]:
    """One scalar per channel for a trial: the value the criteria actually grade on"""
    out: dict[str, float] = {}
    for channel in channels:
        series = np.asarray(angles[channel], dtype=np.float64)
        if not np.isfinite(series).any():
            out[channel] = float("nan")
            continue
        # Trunk deviations are graded on magnitude in either direction
        out[channel] = float(
            np.nanmax(np.abs(series)) if channel.startswith("trunk") else np.nanmax(series)
        )
    return out


def compare(
    reference: dict[tuple, dict[str, float]],
    comparison: dict[tuple, dict[str, float]],
    channels: tuple[str, ...] = AGREEMENT_CHANNELS,
) -> list[ChannelAgreement]:
    """Agreement per channel over trials present in both sensors"""
    keys = sorted(set(reference) & set(comparison))
    out: list[ChannelAgreement] = []
    for channel in channels:
        ref = np.array([reference[k].get(channel, np.nan) for k in keys], dtype=np.float64)
        cmp_ = np.array([comparison[k].get(channel, np.nan) for k in keys], dtype=np.float64)
        usable = np.isfinite(ref) & np.isfinite(cmp_)
        ref, cmp_ = ref[usable], cmp_[usable]
        if ref.size < 3:
            out.append(ChannelAgreement(channel, int(ref.size), *([float("nan")] * 6)))
        else:
            diff = cmp_ - ref
            bias = float(diff.mean())
            sd = float(diff.std(ddof=1))
            r = float(np.corrcoef(ref, cmp_)[0, 1]) if ref.size > 2 else float("nan")
            out.append(
                ChannelAgreement(
                    channel=channel,
                    n=int(ref.size),
                    bias=bias,
                    sd=sd,
                    loa_low=bias - 1.96 * sd,
                    loa_high=bias + 1.96 * sd,
                    icc=icc_2_1(ref, cmp_),
                    pearson=r,
                    reference_mean=float(ref.mean()),
                )
            )
    return out


def collect_peaks(adapter, channels: tuple[str, ...] = AGREEMENT_CHANNELS) -> dict[tuple, dict]:
    """Running an adapter and return ``{trial_key: {channel: peak}}``"""
    out: dict[tuple, dict] = {}
    for trial in adapter.iter_trials():
        angles = compute_angles(preprocess(trial.sequence), trial.side)
        key = (
            trial.meta.get("movement"),
            trial.subject_id,
            trial.meta.get("episode"),
            trial.correct,
        )
        out[key] = trial_peaks(angles, channels)
    return out


# EQI's performance components, in weight order
EQI_COMPONENTS: tuple[str, ...] = ("rom", "smoothness", "tempo", "hold", "return_to_start")


def collect_index_parts(dataset) -> dict[tuple, dict[str, float]]:
    """``{trial_key: {part: value}}`` for every EQI ingredient plus the index itself"""
    out: dict[tuple, dict[str, float]] = {}
    for sample in dataset.samples:
        if sample.meta.get("mirrored"):
            continue
        key = (
            sample.meta.get("movement"),
            sample.subject_id,
            sample.meta.get("episode"),
            sample.correct,
        )
        parts: dict[str, float] = {}
        components = sample.meta.get("eqi_components") or {}
        for name in EQI_COMPONENTS:
            parts[f"component: {name}"] = float(components.get(name, np.nan))
        for name, value in (sample.meta.get("eqi_penalties") or {}).items():
            parts[f"penalty: {name}"] = float(value)
        parts["EQI"] = float(sample.eqi)
        out[key] = parts
    return out


def index_report(reference_dataset, comparison_dataset) -> dict:
    """Cross-sensor agreement at the level of the index and its ingredients"""
    ref = collect_index_parts(reference_dataset)
    cmp_ = collect_index_parts(comparison_dataset)
    keys = sorted(set(ref) & set(cmp_))
    if not keys:
        return {"error": "no trials paired between the two datasets"}

    parts = sorted({p for k in keys for p in ref[k]} & {p for k in keys for p in cmp_[k]})
    # report the index last, after the ingredients that produce it
    parts = [p for p in parts if p != "EQI"] + ["EQI"]
    results = compare(ref, cmp_, tuple(parts))
    poor = [r.channel for r in results if np.isfinite(r.icc) and r.icc < ICC_POOR]
    eqi = next((r for r in results if r.channel == "EQI"), None)
    return {
        "n_paired": len(keys),
        "parts": [r.as_dict() for r in results],
        "poor_agreement": poor,
        "index_icc": None if eqi is None else round(eqi.icc, 4),
        "note": (
            "Agreement on what the index consumes, not on the raw angles. A part that two "
            "simultaneous sensors disagree on makes the score depend on the camera rather "
            "than on the patient, which matters more than discrimination for tracking progress at home."
        ),
    }


def report(reference_adapter, comparison_adapter, *, channels=AGREEMENT_CHANNELS) -> dict:
    """Full agreement report between two adapters over the same corpus"""
    ref = collect_peaks(reference_adapter, channels)
    cmp_ = collect_peaks(comparison_adapter, channels)
    results = compare(ref, cmp_, channels)
    paired = len(set(ref) & set(cmp_))
    poor = [r.channel for r in results if np.isfinite(r.icc) and r.icc < ICC_POOR]
    return {
        "reference": getattr(reference_adapter, "sensor", reference_adapter.name),
        "comparison": getattr(comparison_adapter, "sensor", comparison_adapter.name),
        "n_reference_trials": len(ref),
        "n_comparison_trials": len(cmp_),
        "n_paired": paired,
        "channels": [r.as_dict() for r in results],
        "poor_agreement": poor,
        "note": (
            "Peak-per-trial comparison of two simultaneous recordings. Bias is usually a "
            "skeleton-model offset (skin markers vs modelled joint centres), not sensor noise."
        ),
    }
