"""Measuring whether the conjunctive progression layer is doing any work"""

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
from rehab_ml.rules.criteria import grade  
from rehab_ml.rules.quality_index import EqiConfig  
from rehab_ml.taxonomy import ERROR_SPECS, Severity, safety_errors  
from rehab_ml.training.aggregation import _eqi_under, _resolve  
from rehab_ml.training.eqi_validation import DEFAULT_LABEL_SOURCES  

ROOT = "../data/IntelliRehabDS"
COHORT = "patient"
SELF_PACED = True
SAFETY_FIRING_RATE = 0.5  # mirrors adaptation.SAFETY_FIRING_RATE

SAFETY_SLUGS = frozenset(ERROR_SPECS[e].name_short for e in safety_errors())


def _usable(dataset, cohort):
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
    """Recording, per held-out repetition, whether EQI passed it and whether it is unsafe"""
    cfg = EqiConfig()
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

        train_scores = [
            _eqi_under(s, calibrations[s.exercise_id], cfg)
            for s in train
            if s.exercise_id in calibrations
        ]
        train_scores = [v for v in train_scores if np.isfinite(v)]
        if not train_scores:
            continue
        cut = float(np.median(train_scores))

        for sample in test:
            calibration = calibrations.get(sample.exercise_id)
            if calibration is None:
                continue
            score = _eqi_under(sample, calibration, cfg)
            if not np.isfinite(score):
                continue
            measured = sample.meta.get("rule_measured") or {}

            worst = Severity.NONE
            offenders = {}
            for slug in SAFETY_SLUGS:
                if slug not in measured or slug not in calibration.tolerances:
                    continue
                severity = grade(float(measured[slug]), calibration.tolerances[slug])
                offenders[slug] = int(severity)
                if severity > worst:
                    worst = severity

            rows.append(
                {
                    "subject": sample.subject_id,
                    "exercise": sample.exercise_id,
                    "correct": bool(sample.correct > 0.5),
                    "eqi": float(score),
                    "eqi_passes": bool(score >= cut),
                    # engine.safety_flags uses severity >= MODERATE
                    "unsafe": bool(worst >= Severity.MODERATE),
                    "worst_safety_severity": int(worst),
                    "safety_severities": offenders,
                }
            )
    return rows


def summarise(rows: list[dict], label: str) -> dict:
    """The 2x2 of scoring-layer verdict against the safety flag"""
    if not rows:
        return {"scope": label, "error": "no repetitions"}

    n = len(rows)
    passed = [r for r in rows if r["eqi_passes"]]
    caught = [r for r in passed if r["unsafe"]]

    evaluated = sorted({s for r in rows for s in r["safety_severities"]})

    sessions: dict[tuple, list] = defaultdict(list)
    for r in rows:
        sessions[(r["subject"], r["exercise"])].append(r)
    held = [
        key
        for key, reps in sessions.items()
        if np.mean([x["unsafe"] for x in reps]) > SAFETY_FIRING_RATE
        and np.mean([x["eqi_passes"] for x in reps]) >= 0.5
    ]

    return {
        "scope": label,
        "n": n,
        "n_subjects": len({r["subject"] for r in rows}),
        "safety_criteria_evaluated": evaluated,
        "scoring_layer_passes": len(passed),
        "unsafe_repetitions": sum(r["unsafe"] for r in rows),
        "passed_but_unsafe": len(caught),
        "passed_but_unsafe_rate": round(len(caught) / n, 4),
        "share_of_passes_that_are_unsafe": round(len(caught) / len(passed), 4) if passed else None,
        # If the layer only ever fires on repetitions the clinician also called incorrect,
        # it is adding genuine information the scoring layer missed
        "passed_but_unsafe_and_labelled_incorrect": sum(1 for r in caught if not r["correct"]),
        # Control for the line above: if the flag is uninformative about correctness across
        # ALL flagged repetitions, then 0-of-12 is the criterion being inert, not small n
        "unsafe_and_labelled_incorrect": sum(1 for r in rows if r["unsafe"] and not r["correct"]),
        # The right reference for 'passed_but_unsafe_and_labelled_incorrect' is NOT the overall
        # incorrect rate: repetitions that clear the scoring layer are enriched for correct ones
        # by construction
        "incorrect_rate_among_passes": (
            round(float(np.mean([not r["correct"] for r in passed])), 4) if passed else None
        ),
        "incorrect_rate_when_unsafe": (
            round(float(np.mean([not r["correct"] for r in rows if r["unsafe"]])), 4)
            if any(r["unsafe"] for r in rows)
            else None
        ),
        "incorrect_rate_overall": round(float(np.mean([not r["correct"] for r in rows])), 4),
        "n_sessions": len(sessions),
        "sessions_held_by_safety_layer": len(held),
        "note": (
            "'passed_but_unsafe' is what the second layer exists to catch: the scoring layer "
            "admitted the repetition while a safety-relevant criterion sat at MODERATE or worse. "
            "Session figures use subject-exercise as a session proxy; the corpus has no session ids."
        ),
    }


def main() -> None:
    dataset = build_dataset(IntelliRehabAdapter(ROOT), mirror_augment=False, self_paced=SELF_PACED)
    rows = collect(dataset)

    scopes = {"POOLED": rows}
    for exercise_id in sorted({r["exercise"] for r in rows}):
        scopes[exercise_id] = [r for r in rows if r["exercise"] == exercise_id]
    out = {name: summarise(rs, name) for name, rs in scopes.items()}

    header = (
        f"{'scope':28}{'n':>6}{'passes':>8}{'unsafe':>8}"
        f"{'pass&unsafe':>13}{'of those wrong':>16}{'sessions held':>15}"
    )
    print(header)
    print("-" * len(header))
    for name, s in out.items():
        if "error" in s:
            print(f"{name:28}  {s['error']}")
            continue
        print(
            f"{name:28}{s['n']:>6}{s['scoring_layer_passes']:>8}{s['unsafe_repetitions']:>8}"
            f"{s['passed_but_unsafe']:>13}"
            f"{s['passed_but_unsafe_and_labelled_incorrect']:>16}"
            f"{s['sessions_held_by_safety_layer']:>15}"
        )

    pooled = out.get("POOLED", {})
    print()
    print(f"safety criteria actually evaluated: {pooled.get('safety_criteria_evaluated')}")
    ref = pooled.get("incorrect_rate_among_passes")
    caught = pooled.get("passed_but_unsafe", 0)
    if ref is not None and caught:
        print(
            f"incorrect rate among scoring-layer passes: {ref:.3f} "
            f"-> P(0 of {caught} incorrect by chance) = {(1 - ref) ** caught:.3f}"
        )

    target = Path("reports/safety_layer.json")
    target.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwritten to {target}")


if __name__ == "__main__":
    main()
