"""Paired confidence intervals on the scapulohumeral correction's effect"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rehab_ml.data.adapters.intellirehab import IntelliRehabAdapter  
from rehab_ml.data.adapters.ui_prmd import UiPrmdAdapter  
from rehab_ml.data.dataset import build_dataset  
from rehab_ml.features.angles import AngleSeries  
from rehab_ml.training.eqi_validation import (  
    _auc,
    _balanced_loso,
    _paired_ci,
    _rep_rows,
)

N_RESAMPLES = 4000

ARMS = [
    ("IRDS patients", lambda: IntelliRehabAdapter("../data/IntelliRehabDS"), "patient"),
    ("IRDS all", lambda: IntelliRehabAdapter("../data/IntelliRehabDS"), None),
    ("UI-PRMD vicon", lambda: UiPrmdAdapter("../data/UIPRMD", sensor="vicon"), None),
    ("UI-PRMD kinect", lambda: UiPrmdAdapter("../data/UIPRMD", sensor="kinect"), None),
]


@contextmanager
def correction(enabled: bool):
    """Forcing `shoulder_hike` on or off for the duration of the block"""
    original = AngleSeries.shoulder_hike

    def patched(self, baseline_frames=5, *, scapular_correction=True):
        return original(self, baseline_frames, scapular_correction=enabled)

    AngleSeries.shoulder_hike = patched
    try:
        yield
    finally:
        AngleSeries.shoulder_hike = original


def rows_for(factory, cohort: str | None, *, enabled: bool) -> list[dict]:
    with correction(enabled):
        dataset = build_dataset(factory(), mirror_augment=False, self_paced=True)
        return _rep_rows(dataset, None, cohort)


def paired(label: str, factory, cohort: str | None) -> dict:
    """Bootstrapping the on-minus-off difference over the same repetitions"""
    off = rows_for(factory, cohort, enabled=False)
    on = rows_for(factory, cohort, enabled=True)
    if len(off) != len(on) or len(off) < 20:
        return {"error": f"row mismatch: {len(off)} off vs {len(on)} on"}

    correct = np.array([r["correct"] for r in off])
    subjects = [r["subject"] for r in off]
    # The builds are deterministic over the same trials, but pairing is the whole point
    if not np.array_equal(correct, np.array([r["correct"] for r in on])):
        return {"error": "label vectors differ between arms - rows are not aligned"}
    if subjects != [r["subject"] for r in on]:
        return {"error": "subject vectors differ between arms - rows are not aligned"}

    s_off = np.array([r["eqi"]["gated"] for r in off])
    s_on = np.array([r["eqi"]["gated"] for r in on])
    subject_array = np.asarray(subjects)

    d_auc = _auc(s_on, correct) - _auc(s_off, correct)
    d_bal = _balanced_loso(s_on, correct, subject_array) - _balanced_loso(
        s_off, correct, subject_array
    )

    auc_ci = _paired_ci(
        lambda idx: _auc(s_on[idx], correct[idx]) - _auc(s_off[idx], correct[idx]),
        subjects,
        N_RESAMPLES,
    )
    bal_ci = _paired_ci(
        lambda idx: (
            _balanced_loso(s_on[idx], correct[idx], subject_array[idx])
            - _balanced_loso(s_off[idx], correct[idx], subject_array[idx])
        ),
        subjects,
        N_RESAMPLES,
    )

    def excludes_zero(ci) -> bool | None:
        return None if ci is None else bool(ci[0] > 0 or ci[1] < 0)

    return {
        "n": len(off),
        "n_subjects": len(set(subjects)),
        "delta_auc": round(float(d_auc), 4),
        "delta_auc_ci": auc_ci,
        "delta_auc_excludes_zero": excludes_zero(auc_ci),
        "delta_bal_loso": round(float(d_bal), 4),
        "delta_bal_loso_ci": bal_ci,
        "delta_bal_loso_excludes_zero": excludes_zero(bal_ci),
        "note": (
            "Paired subject-level bootstrap over the SAME repetitions scored under both "
            "criterion definitions. Per-arm intervals in scapular_ab.json are not a "
            "substitute: they ignore the pairing and overlap heavily."
        ),
    }


def main() -> None:
    out: dict = {}
    header = f"{'arm':16}{'n':>6}{'dAUC':>9}{'AUC 95% CI':>22}{'dbal':>9}{'bal 95% CI':>22}"
    print(header)
    print("-" * len(header))
    for label, factory, cohort in ARMS:
        result = paired(label, factory, cohort)
        out[label] = result
        if "error" in result:
            print(f"{label:16}  {result['error']}")
            continue

        def fmt(ci) -> str:
            return "n/a" if ci is None else f"[{ci[0]:+.4f}, {ci[1]:+.4f}]"

        print(
            f"{label:16}{result['n']:>6}{result['delta_auc']:>+9.4f}"
            f"{fmt(result['delta_auc_ci']):>22}"
            f"{result['delta_bal_loso']:>+9.4f}{fmt(result['delta_bal_loso_ci']):>22}"
        )

    target = Path("reports/scapular_paired_ci.json")
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten to {target}")


if __name__ == "__main__":
    main()
