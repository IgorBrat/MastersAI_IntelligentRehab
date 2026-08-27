from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..rules.calibration import Calibration
from ..rules.calibration import fit as calibrate_rules
from ..rules.calibration import verdict as rule_verdict
from ..rules.criteria import normalised_penalty
from ..rules.quality_index import EqiConfig
from .eqi_validation import (
    DEFAULT_LABEL_SOURCES,
    _balanced_at_predictions,
    _paired_ci,
    _weighted_penalty,
)
from ..exercises.registry import as_self_paced, get_exercise


ARMS = ("rule_binary", "eqi_graded")


def _eqi_under(sample, calibration: Calibration, config: EqiConfig) -> float:
    """Recomputing one repetition's EQI with its penalties re-graded against a calibration"""
    components = sample.meta.get("eqi_components") or {}
    measured = sample.meta.get("rule_measured") or {}
    protocol_penalties = sample.meta.get("eqi_penalties") or {}
    if not components:
        return float("nan")

    penalties = {}
    for slug, stored in protocol_penalties.items():
        tolerance = calibration.tolerances.get(slug)
        if tolerance is None:
            penalties[slug] = float(stored)
            continue
        if slug not in measured:
            penalties[slug] = float(stored)
            continue
        penalties[slug] = normalised_penalty(float(measured[slug]), tolerance)

    weights = config.weights.normalised()
    performance = sum(
        float(components.get(name, 0.0)) * getattr(weights, name)
        for name in ("rom", "smoothness", "tempo", "hold", "return_to_start")
    )
    span = 1.0 - config.gate_floor
    return 100.0 * performance * (1.0 - span * _weighted_penalty(penalties, config))


def compare(
    dataset,
    *,
    cohort: str | None = "patient",
    self_paced: bool = True,
    config: EqiConfig | None = None,
    n_resamples: int = 2000,
    label_sources: frozenset[str] | tuple[str, ...] = DEFAULT_LABEL_SOURCES,
) -> dict:
    """Comparing the two aggregations leave-one-subject-out

    Returns balanced accuracy per arm, the paired difference, and bootstrap intervals that resample subjects
    """
    cfg = config or EqiConfig()
    admitted = {s.upper() for s in label_sources}

    usable = [
        s
        for s in dataset.samples
        if np.isfinite(s.correct)
        and s.label_source.name.upper() in admitted
        and (cohort is None or s.meta.get("cohort") == cohort)
        and s.meta.get("rule_measured")
        and s.meta.get("eqi_components")
    ]
    by_subject: dict[str, list] = defaultdict(list)
    for sample in usable:
        by_subject[sample.subject_id].append(sample)

    if len(by_subject) < 3:
        return {"n": len(usable), "error": "need at least 3 subjects for LOSO"}

    truth, subjects = [], []
    predictions: dict[str, list[float]] = {arm: [] for arm in ARMS}
    n_skipped = 0

    for held_out in sorted(by_subject):
        train = [s for subj, rows in by_subject.items() if subj != held_out for s in rows]
        test = by_subject[held_out]

        exercise_ids = {s.exercise_id for s in train} & {s.exercise_id for s in test}
        calibrations = {
            exercise_id: calibrate_rules(
                [s for s in train if s.exercise_id == exercise_id],
                _resolve(exercise_id, self_paced),
            )
            for exercise_id in exercise_ids
        }

        # The cut comes from the TRAINING set, which is what keeps the two arms symmetric:
        # the rule arm takes its thresholds from there too.
        train_scores = [
            _eqi_under(s, calibrations[s.exercise_id], cfg)
            for s in train
            if s.exercise_id in calibrations
        ]
        train_scores = [v for v in train_scores if np.isfinite(v)]
        if not train_scores:
            n_skipped += len(test)
            continue
        cut = float(np.median(train_scores))

        for sample in test:
            calibration = calibrations.get(sample.exercise_id)
            if calibration is None:
                n_skipped += 1
                continue
            score = _eqi_under(sample, calibration, cfg)
            if not np.isfinite(score):
                n_skipped += 1
                continue
            truth.append(float(sample.correct))
            subjects.append(sample.subject_id)
            predictions["rule_binary"].append(float(rule_verdict(sample, calibration)))
            predictions["eqi_graded"].append(float(score >= cut))

    if len(truth) < 20:
        return {"n": len(truth), "error": "too few evaluable repetitions"}

    correct = np.asarray(truth)
    preds = {arm: np.asarray(v, dtype=bool) for arm, v in predictions.items()}

    out = {
        arm: {
            "balanced_accuracy": round(_balanced_at_predictions(preds[arm], correct), 4),
            "ci": _paired_ci(
                lambda idx, a=arm: _balanced_at_predictions(preds[a][idx], correct[idx]),
                subjects,
                n_resamples,
            ),
            "sensitivity": round(float(preds[arm][correct > 0.5].mean()), 4),
            "specificity": round(float((~preds[arm][correct <= 0.5]).mean()), 4),
            "predicted_correct_rate": round(float(preds[arm].mean()), 4),
        }
        for arm in ARMS
    }

    def _delta(idx: np.ndarray) -> float:
        return _balanced_at_predictions(
            preds["eqi_graded"][idx], correct[idx]
        ) - _balanced_at_predictions(preds["rule_binary"][idx], correct[idx])

    delta = out["eqi_graded"]["balanced_accuracy"] - out["rule_binary"]["balanced_accuracy"]
    # Paired on the difference: both arms score the same repetitions, so marginal
    # intervals would understate how well the contrast resolves.
    delta_ci = _paired_ci(_delta, subjects, n_resamples)

    return {
        "n": len(truth),
        "n_subjects": len(set(subjects)),
        "n_skipped": n_skipped,
        "cohort": cohort,
        "positive_rate": round(float(correct.mean()), 4),
        "arms": out,
        "delta_balanced_accuracy": round(float(delta), 4),
        "delta_ci": delta_ci,
        "resolved": bool(delta_ci and (delta_ci[0] > 0 or delta_ci[1] < 0)),
        "note": (
            "Both arms use the SAME per-fold calibrated tolerances applied to the SAME "
            "stored measurements, and both are leave-one-subject-out. The difference is "
            "the aggregation: graded-and-weighted versus gate-on-anything."
        ),
    }


def _resolve(exercise_id: str, self_paced: bool):
    """Looking up an exercise definition, optionally dropping the tempo prescription"""
    exercise = get_exercise(exercise_id)
    return as_self_paced(exercise) if self_paced else exercise


def report(dataset, **kwargs) -> dict:
    """Running the comparison pooled and then once per exercise"""
    out = {"POOLED": compare(dataset, **kwargs)}
    for exercise_id in sorted(dataset.exercise_ids):
        out[exercise_id] = compare(dataset.filter(exercise_id=exercise_id), **kwargs)
    return out
