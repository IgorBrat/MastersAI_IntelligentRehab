"""Is the obligatory-rise coefficient justified, and does the correction depend on its value"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rehab_ml.data.adapters.intellirehab import IntelliRehabAdapter  
from rehab_ml.data.adapters.ui_prmd import UiPrmdAdapter  
from rehab_ml.data.dataset import RepetitionDataset, build_dataset  
from rehab_ml.features import angles as angles_mod  
from rehab_ml.features.angles import compute_angles  
from rehab_ml.pose.normalize import preprocess  
from rehab_ml.pose.view import infer_working_side  
from rehab_ml.schema import Side  
from rehab_ml.training.eqi_validation import discriminate  


SHIPPED_SLOPE = angles_mod.SCAPULOHUMERAL_RISE_PER_DEG  # 0.00071
ONSET = angles_mod.SCAPULOHUMERAL_ONSET_DEG            # 30.0


SWEEP = [0.0, 0.0002, 0.0004, 0.00055, 0.00071, 0.00085, 0.001, 0.00119, 0.0014, 0.0018]

ARMS = [
    ("IRDS patients", lambda: IntelliRehabAdapter("../data/IntelliRehabDS"), "patient"),
    ("UI-PRMD vicon", lambda: UiPrmdAdapter("../data/UIPRMD", sensor="vicon"), None),
    ("UI-PRMD kinect", lambda: UiPrmdAdapter("../data/UIPRMD", sensor="kinect"), None),
]


@contextmanager
def slope(value: float):
    """Forcing the obligatory-rise coefficient for the duration of the block
    """
    original = angles_mod.SCAPULOHUMERAL_RISE_PER_DEG
    angles_mod.SCAPULOHUMERAL_RISE_PER_DEG = value
    try:
        yield
    finally:
        angles_mod.SCAPULOHUMERAL_RISE_PER_DEG = original


# Region - A. the regression


def correct_frames(factory, cohort: str | None) -> list[dict]:
    """Per-trial elevation and shoulder-height frames, correct repetitions only

    Mirrors `build_dataset`'s preprocessing and side inference so the angles are the same
    ones the criterion sees.
    """
    rows: list[dict] = []
    for trial in factory():
        if trial.exercise_id is None or trial.correct is None:
            continue
        if float(trial.correct) <= 0.5:  # only correct repetitions define the obligatory rise
            continue
        if cohort is not None and trial.meta.get("cohort") != cohort:
            continue
        sequence = (
            trial.sequence
            if trial.sequence.meta.get("preprocessed")
            else preprocess(trial.sequence)
        )
        side = trial.side if trial.side is not None else infer_working_side(sequence)
        sides = (side,) if side is not None else (Side.LEFT, Side.RIGHT)
        for working in sides:
            a = compute_angles(sequence, working)
            elevation = np.asarray(a.shoulder_elevation, dtype=float)
            height = np.asarray(a.shoulder_height_working, dtype=float)
            finite = np.isfinite(elevation) & np.isfinite(height)
            if finite.sum() < 10:
                continue
            rows.append(
                {
                    "subject": trial.subject_id,
                    "exercise": trial.exercise_id,
                    "elevation": elevation[finite],
                    "height": height[finite],
                }
            )
    return rows


def fit(rows: list[dict], *, above_onset: bool) -> dict | None:
    """Least-squares slope of shoulder height on elevation, in torso lengths per degree"""
    xs, ys = [], []
    for row in rows:
        x, y = row["elevation"], row["height"]
        if above_onset:
            keep = x > ONSET
            x, y = x[keep], y[keep]
        if x.size < 10 or np.ptp(x) < 5.0:
            continue
        xs.append(x - x.mean())
        ys.append(y - y.mean())
    if not xs:
        return None
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if x.size < 20 or np.allclose(x.var(), 0.0):
        return None
    slope_value = float(x @ y / (x @ x))
    denom = float(np.sqrt((x @ x) * (y @ y)))
    r = float(x @ y / denom) if denom > 1e-12 else float("nan")
    return {
        "slope": round(slope_value, 6),
        "r": round(r, 3),
        "n_frames": int(x.size),
        "n_repetitions": len(xs),
        "obligatory_at_90deg": round(slope_value * (90.0 - ONSET), 4),
    }


def regression_report(rows: list[dict]) -> dict:
    """Pooled, per-subject and leave-one-subject-out slopes"""
    subjects = sorted({row["subject"] for row in rows})
    per_subject = {}
    for subject in subjects:
        mine = [r for r in rows if r["subject"] == subject]
        got = fit(mine, above_onset=True)
        if got:
            per_subject[subject] = got

    loso = {}
    for subject in subjects:
        others = [r for r in rows if r["subject"] != subject]
        got = fit(others, above_onset=True)
        if got:
            loso[subject] = got["slope"]

    values = [v["slope"] for v in per_subject.values()]
    loso_values = list(loso.values())
    return {
        "pooled_all_frames": fit(rows, above_onset=False),
        "pooled_above_onset": fit(rows, above_onset=True),
        "n_subjects_fitted": len(per_subject),
        "per_subject_slope": {k: v["slope"] for k, v in per_subject.items()},
        "per_subject_r": {k: v["r"] for k, v in per_subject.items()},
        "per_subject_spread": (
            {
                "median": round(float(np.median(values)), 6),
                "iqr": [
                    round(float(np.percentile(values, 25)), 6),
                    round(float(np.percentile(values, 75)), 6),
                ],
                "min": round(float(np.min(values)), 6),
                "max": round(float(np.max(values)), 6),
                "cv": round(float(np.std(values) / np.mean(values)), 3)
                if np.mean(values) else None,
            }
            if values
            else None
        ),
        "loso_slope": {k: round(v, 6) for k, v in loso.items()},
        "loso_spread": (
            {
                "min": round(float(np.min(loso_values)), 6),
                "max": round(float(np.max(loso_values)), 6),
                "range_frac_of_mean": round(
                    float(np.ptp(loso_values) / np.mean(loso_values)), 4
                )
                if np.mean(loso_values) else None,
            }
            if loso_values
            else None
        ),
        "note": (
            "Repetitions are mean-centred before pooling."
        ),
    }


# Region - B and C. scoring


def score(dataset: RepetitionDataset, cohort: str | None, n_resamples: int) -> dict:
    """AUC and the subject-disjoint thresholded accuracy for the gated EQI"""
    result = discriminate(dataset, cohort=cohort, n_resamples=n_resamples)
    if "formulations" not in result:
        return {"error": result.get("error", "no formulations")}
    gated = result["formulations"]["gated"]
    return {
        "n": result["n"],
        "auc": gated["auc"],
        "auc_ci": gated["auc_ci"],
        "bal_loso": gated["balanced_accuracy_loso"],
        "bal_loso_ci": gated["balanced_accuracy_loso_ci"],
    }


def sweep(factory, cohort: str | None, n_resamples: int) -> dict:
    """Scoring EQI at each candidate coefficient"""
    out = {}
    for value in SWEEP:
        with slope(value):
            dataset = build_dataset(factory(), mirror_augment=False, self_paced=True)
            out[f"{value:.5f}"] = score(dataset, cohort, n_resamples)
        got = out[f"{value:.5f}"]
        print(
            f"    slope {value:.5f}  AUC {got.get('auc', float('nan')):.4f}"
            f"  bal {got.get('bal_loso', float('nan')):.4f}"
        )
    return out


def out_of_fold(factory, cohort: str | None, loso: dict[str, float], n_resamples: int) -> dict:
    """Building each subject with the slope fitted on the OTHER subjects, then scoring"""
    by_subject: dict[str, list] = {}
    for trial in factory():
        by_subject.setdefault(trial.subject_id, []).append(trial)

    merged: list = []
    applied: dict[str, float] = {}
    fallback = float(np.median(list(loso.values()))) if loso else SHIPPED_SLOPE
    for subject, trials in by_subject.items():
        value = loso.get(subject, fallback)
        applied[subject] = round(value, 6)
        with slope(value):
            built = build_dataset(trials, mirror_augment=False, self_paced=True)
        merged.extend(built.samples)

    dataset = RepetitionDataset(samples=merged)
    result = score(dataset, cohort, n_resamples)
    result["slope_applied_per_subject"] = applied
    result["n_subjects_with_own_fold"] = len(
        [s for s in by_subject if s in loso]
    )
    return result


def main() -> None:
    out: dict = {"shipped_slope": SHIPPED_SLOPE, "onset_deg": ONSET, "arms": {}}
    target = Path("reports/scapular_slope.json")

    for label, factory, cohort in ARMS:
        print(f"\n=== {label} ===")
        arm: dict = {}

        print("  A. regression on correct repetitions")
        rows = correct_frames(factory, cohort)
        if not rows:
            arm["error"] = "no clinician-labelled correct repetitions"
            out["arms"][label] = arm
            continue
        arm["regression"] = regression_report(rows)
        pooled = arm["regression"]["pooled_all_frames"]
        onset_fit = arm["regression"]["pooled_above_onset"]
        print(
            f"    pooled all frames  slope {pooled['slope']:.5f}  r {pooled['r']:.2f}"
            f"  n {pooled['n_frames']}"
        )
        print(
            f"    pooled above {ONSET:.0f}deg  slope {onset_fit['slope']:.5f}"
            f"  r {onset_fit['r']:.2f}  n {onset_fit['n_frames']}"
        )
        spread = arm["regression"]["per_subject_spread"]
        if spread:
            print(
                f"    per-subject  median {spread['median']:.5f}"
                f"  range {spread['min']:.5f}-{spread['max']:.5f}  cv {spread['cv']}"
            )

        print("  B. sensitivity sweep")
        arm["sweep"] = sweep(factory, cohort, n_resamples=400)

        print("  C. out-of-fold slope")
        arm["out_of_fold"] = out_of_fold(
            factory, cohort, arm["regression"]["loso_slope"], n_resamples=2000
        )
        print(
            f"    AUC {arm['out_of_fold'].get('auc')}"
            f"  bal {arm['out_of_fold'].get('bal_loso')}"
        )

        print("  D. shipped constant, full resamples, for reference")
        with slope(SHIPPED_SLOPE):
            dataset = build_dataset(factory(), mirror_augment=False, self_paced=True)
            arm["shipped"] = score(dataset, cohort, n_resamples=2000)
        with slope(0.0):
            dataset = build_dataset(factory(), mirror_augment=False, self_paced=True)
            arm["uncorrected"] = score(dataset, cohort, n_resamples=2000)
        print(
            f"    shipped AUC {arm['shipped'].get('auc')}"
            f"  uncorrected AUC {arm['uncorrected'].get('auc')}"
        )

        out["arms"][label] = arm
        target.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"  written to {target}")

    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten to {target}")


if __name__ == "__main__":
    main()
