"""Testing whether a conjunctive gate's sensitivity is p^k"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rehab_ml.data.adapters.intellirehab import IntelliRehabAdapter  
from rehab_ml.data.dataset import build_dataset  
from rehab_ml.rules.calibration import fit as calibrate_rules  
from rehab_ml.training.aggregation import _resolve  
from rehab_ml.training.eqi_validation import DEFAULT_LABEL_SOURCES  

ROOT = "../data/IntelliRehabDS"
COHORT = "patient"
SELF_PACED = True


def _usable(dataset, cohort):
    """The same admission rule aggregation.compare uses, so the arms stay comparable"""
    admitted = {s.upper() for s in DEFAULT_LABEL_SOURCES}
    return [
        s
        for s in dataset.samples
        if np.isfinite(s.correct)
        and s.label_source.name.upper() in admitted
        and (cohort is None or s.meta.get("cohort") == cohort)
        and s.meta.get("rule_measured")
        and s.meta.get("eqi_components")
    ]


def collect(dataset, *, cohort=COHORT, self_paced=SELF_PACED) -> list[dict]:
    """Walking LOSO and recording, per held-out CORRECT repetition, which criteria it cleared

    Only correct repetitions are collected: sensitivity is defined over them, and p is fitted
    on them, so they are the population the p^k prediction is about.
    """
    usable = _usable(dataset, cohort)
    by_subject: dict[str, list] = defaultdict(list)
    for sample in usable:
        by_subject[sample.subject_id].append(sample)

    if len(by_subject) < 3:
        return []

    rows: list[dict] = []
    for held_out in sorted(by_subject):
        train = [s for subj, rows_ in by_subject.items() if subj != held_out for s in rows_]
        test = by_subject[held_out]
        exercise_ids = {s.exercise_id for s in train} & {s.exercise_id for s in test}
        calibrations = {
            ex_id: calibrate_rules(
                [s for s in train if s.exercise_id == ex_id], _resolve(ex_id, self_paced)
            )
            for ex_id in exercise_ids
        }

        for sample in test:
            if sample.correct <= 0.5:
                continue
            calibration = calibrations.get(sample.exercise_id)
            if calibration is None:
                continue
            measured = sample.meta.get("rule_measured") or {}
            # Exactly the criteria calibration.verdict() gates on: a slug with no fitted
            # tolerance is skipped there, so it must be skipped here too.
            graded = {
                slug: float(value) <= calibration.tolerances[slug].mild
                for slug, value in measured.items()
                if slug in calibration.tolerances
            }
            if not graded:
                continue
            rows.append(
                {
                    "subject": sample.subject_id,
                    "exercise": sample.exercise_id,
                    "k": len(graded),
                    "passed_all": all(graded.values()),
                    "per_criterion": graded,
                    "percentile": calibration.percentiles[0],
                    "n_skipped_criteria": len(calibration.skipped),
                }
            )
    return rows


def summarise(rows: list[dict], label: str) -> dict:
    """Comparing observed all-clear rate against p^k under both nominal and observed p
    """
    if not rows:
        return {"scope": label, "error": "no repetitions"}

    ks = [r["k"] for r in rows]
    k_mode = max(set(ks), key=ks.count)
    p_nominal = float(np.mean([r["percentile"] for r in rows])) / 100.0

    # p observed: pooled per-criterion pass rate over held-out correct repetitions
    flags = [passed for r in rows for passed in r["per_criterion"].values()]
    p_observed = float(np.mean(flags))

    observed = float(np.mean([r["passed_all"] for r in rows]))
    pred_nominal = float(np.mean([p_nominal**k for k in ks]))
    pred_observed = float(np.mean([p_observed**k for k in ks]))

    per_criterion = defaultdict(list)
    for r in rows:
        for slug, passed in r["per_criterion"].items():
            per_criterion[slug].append(passed)

    return {
        "scope": label,
        "n_correct_repetitions": len(rows),
        "k_mode": k_mode,
        "k_range": [min(ks), max(ks)],
        "p_nominal": round(p_nominal, 4),
        "p_observed": round(p_observed, 4),
        "predicted_nominal": round(pred_nominal, 4),
        "predicted_observed": round(pred_observed, 4),
        "observed_sensitivity": round(observed, 4),
        "ratio_vs_nominal": round(observed / pred_nominal, 3) if pred_nominal else None,
        "ratio_vs_observed": round(observed / pred_observed, 3) if pred_observed else None,
        "operating_point_note": (
            "Multiplicative decay in k is structural; its SEVERITY depends on p. At p=0.85 a "
            "7-criterion gate retains 0.321 of correct repetitions; at p=0.95 it retains 0.698. "
            "Any quotation of this result must state the operating point."
        ),
        "p_k_grid": {
            str(pp): {f"k={kk}": round(pp**kk, 4) for kk in (4, 5, 6, 7, 8)}
            for pp in (0.85, 0.90, 0.95, 0.99)
        },
        "criteria_skipped_mean": round(
            float(np.mean([r["n_skipped_criteria"] for r in rows])), 2
        ),
        "per_criterion_pass_rate": {
            slug: round(float(np.mean(v)), 4) for slug, v in sorted(per_criterion.items())
        },
    }


def main() -> None:
    dataset = build_dataset(IntelliRehabAdapter(ROOT), mirror_augment=False, self_paced=SELF_PACED)
    rows = collect(dataset)

    scopes = {"POOLED": rows}
    for exercise_id in sorted({r["exercise"] for r in rows}):
        scopes[exercise_id] = [r for r in rows if r["exercise"] == exercise_id]

    out = {name: summarise(rs, name) for name, rs in scopes.items()}


    print("A. Against NOMINAL p - does the calibration percentile predict the gate's sensitivity?")
    header = f"{'scope':28}{'n':>6}{'k':>4}{'p_nom':>8}{'pred':>9}{'observed':>10}{'ratio':>8}"
    print(header)
    print("-" * len(header))
    for name, s in out.items():
        if "error" in s:
            print(f"{name:28}  {s['error']}")
            continue
        print(
            f"{name:28}{s['n_correct_repetitions']:>6}{s['k_mode']:>4}"
            f"{s['p_nominal']:>8.3f}{s['predicted_nominal']:>9.3f}"
            f"{s['observed_sensitivity']:>10.3f}{s['ratio_vs_nominal']:>8.3f}"
        )

    print()
    print("B. Against OBSERVED p - with the out-of-fold shift in p removed, what is left is")
    print("   DEPENDENCE between criteria. Only this block supports a dependence reading.")
    header = f"{'scope':28}{'n':>6}{'k':>4}{'p_obs':>8}{'pred':>9}{'observed':>10}{'ratio':>8}"
    print(header)
    print("-" * len(header))
    for name, s in out.items():
        if "error" in s:
            continue
        print(
            f"{name:28}{s['n_correct_repetitions']:>6}{s['k_mode']:>4}"
            f"{s['p_observed']:>8.3f}{s['predicted_observed']:>9.3f}"
            f"{s['observed_sensitivity']:>10.3f}{s['ratio_vs_observed']:>8.3f}"
        )
    print()
    print("   ratio 1.00 = criteria independent; >1.00 = failures co-occur; <1.00 = disjoint.")


    print()
    print("C. Operating point. p^k at the percentiles this project fits (85/95/99):")
    print(f"{'p':>8}" + "".join(f"{'k=' + str(k):>9}" for k in (4, 5, 6, 7, 8)))
    for p in (0.85, 0.90, 0.95, 0.99):
        print(f"{p:>8.2f}" + "".join(f"{p**k:>9.3f}" for k in (4, 5, 6, 7, 8)))
    print("   Multiplicative decay in k is structural. 'Retains only a third' is true at")
    print("   p=0.85 ONLY - at 0.95 a 7-criterion gate retains 0.698. State the p.")

    target = Path("reports/conjunctive_sensitivity.json")
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten to {target}")


if __name__ == "__main__":
    main()
