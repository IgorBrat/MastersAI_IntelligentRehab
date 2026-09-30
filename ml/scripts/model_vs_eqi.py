"""Paired comparison of the learned model against EQI on one cohort"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from rehab_ml.training.eqi_validation import (  
    _balanced_at_predictions,
    _balanced_loso,
    _paired_ci,
)

REPORTS = pathlib.Path(__file__).resolve().parents[1] / "reports"

COHORT_PREFIX = {"physiotherapist": "1", "patient": "2", "student": "3"}


def load(path: pathlib.Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    predictions = report.get("predictions")
    if not predictions or not predictions.get("subject"):
        raise SystemExit(f"{path.name} carries no per-repetition predictions")
    return predictions


def cohort_mask(subjects: list[str], cohort: str | None) -> np.ndarray:
    if cohort is None:
        return np.ones(len(subjects), dtype=bool)
    prefix = COHORT_PREFIX[cohort]
    return np.array([s.split(":")[-1].startswith(prefix) for s in subjects])


def compare(predictions: dict, cohort: str | None, n_resamples: int) -> dict:
    subjects = np.asarray(predictions["subject"])
    keep = cohort_mask(list(subjects), cohort)

    def column(name):
        raw = predictions.get(name)
        if raw is None:
            return None
        return np.array([np.nan if v is None else float(v) for v in raw])[keep]

    subjects = subjects[keep]
    correct = column("correct_true")
    model = column("correct_pred")
    eqi = column("eqi_rule")
    rule = column("rule_calibrated_pred")

    finite = np.isfinite(correct) & np.isfinite(model) & np.isfinite(eqi)
    subjects, correct, model, eqi = subjects[finite], correct[finite], model[finite], eqi[finite]
    rule = rule[finite] if rule is not None else None

    def model_ba(rows):
        return _balanced_at_predictions(model[rows] > 0.5, correct[rows])

    def eqi_ba(rows):
        return _balanced_loso(eqi[rows], correct[rows], subjects[rows])

    def delta(rows):
        return model_ba(rows) - eqi_ba(rows)

    everything = np.arange(len(correct))
    out = {
        "n": int(len(correct)),
        "n_subjects": int(np.unique(subjects).size),
        "cohort": cohort or "all",
        "positive_rate": round(float(np.mean(correct > 0.5)), 4),
        "arms": {
            "model": {
                "balanced_accuracy": round(float(model_ba(everything)), 4),
                "ci": _paired_ci(model_ba, list(subjects), n_resamples),
                "operating_point": "fixed 0.5 on the model's correctness probability",
            },
            "eqi_graded": {
                "balanced_accuracy": round(float(eqi_ba(everything)), 4),
                "ci": _paired_ci(eqi_ba, list(subjects), n_resamples),
                "operating_point": "median EQI of the OTHER subjects, per subject",
            },
        },
        "delta_balanced_accuracy": round(float(delta(everything)), 4),
        "delta_ci": _paired_ci(delta, list(subjects), n_resamples),
    }
    if rule is not None and np.isfinite(rule).any():
        out["arms"]["rule_calibrated"] = {
            "balanced_accuracy": round(
                float(_balanced_at_predictions(rule > 0.5, correct)), 4
            ),
            "operating_point": "per-fold calibrated tolerances, gate on anything",
        }

    ci = out["delta_ci"]
    out["resolved"] = bool(ci and (ci[0] > 0 or ci[1] < 0))
    out["reading"] = (
        "delta = model - EQI. A positive interval excluding zero means the model wins; "
        "an interval containing zero means this sample cannot separate them, which is "
        "NOT the same as showing they are equivalent."
    )
    # The asymmetry matters and must travel with the number.
    out["caveat"] = (
        "EQI's cut is derived from held-out subjects while the model sits at a fixed "
        "0.5, because the run stored only the model's thresholded decision. That "
        "asymmetry favours EQI, so a model win is conservative and an EQI win is not."
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="training report under reports/ carrying predictions")
    parser.add_argument("--cohort", default="patient", choices=[*COHORT_PREFIX, "all"])
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--out", default="model_vs_eqi.json")
    args = parser.parse_args()

    path = pathlib.Path(args.report)
    if not path.is_file():
        path = REPORTS / args.report

    cohort = None if args.cohort == "all" else args.cohort
    result = compare(load(path), cohort, args.resamples)
    result["source_report"] = path.name

    destination = REPORTS / args.out
    destination.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    print(f"\nwritten to {destination}")


if __name__ == "__main__":
    main()
