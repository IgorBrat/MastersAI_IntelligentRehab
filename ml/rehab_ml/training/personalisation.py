"""Personalised versus population thresholds"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from ..exercises.base import Tolerance
from ..exercises.registry import get_exercise
from ..rules.calibration import DEFAULT_PERCENTILES, Calibration, fit, verdict
from .metrics import bootstrap_ci


# Shrinkage constant. At ``k = k0`` the hybrid weights personal and population equally;
# below it leans population, above it leans personal
SHRINKAGE_K0 = 8
# Subject/exercise needs at least this many held-out repetitions to be scored
MIN_EVAL = 8


def blend(personal: Calibration, population: Calibration, weight: float) -> Calibration:
    """Convex blend of two threshold sets, ``weight`` toward ``personal``"""
    w = float(np.clip(weight, 0.0, 1.0))
    merged: dict[str, Tolerance] = dict(population.tolerances)
    for slug, personal_tolerance in personal.tolerances.items():
        base = population.tolerances.get(slug)
        if base is None:
            merged[slug] = personal_tolerance
            continue
        merged[slug] = Tolerance(
            w * personal_tolerance.mild + (1 - w) * base.mild,
            w * personal_tolerance.moderate + (1 - w) * base.moderate,
            w * personal_tolerance.severe + (1 - w) * base.severe,
        )
    return Calibration(
        tolerances=merged,
        n_reference=personal.n_reference,
        percentiles=personal.percentiles,
        skipped=personal.skipped,
    )


@dataclass
class Outcome:
    """Pooled held-out predictions for one condition at one ``k``"""
    y_true: list[float]
    predictions: dict[str, list[float]]
    subjects: list[str]
    n_subjects_personalised: int
    n_subjects_unable: int


def _run_one_k(dataset, k: int, percentiles=DEFAULT_PERCENTILES) -> Outcome:
    by_subject_exercise: dict[tuple[str, str], list] = defaultdict(list)
    for sample in dataset.samples:
        by_subject_exercise[(sample.subject_id, sample.exercise_id)].append(sample)

    conditions = ("population", "personalised", "hybrid")
    out = Outcome([], {c: [] for c in conditions}, [], 0, 0)
    personalised_subjects, unable_subjects = set(), set()

    for (subject, exercise_id), samples in sorted(by_subject_exercise.items()):
        exercise = get_exercise(exercise_id)

        # Population thresholds: every OTHER subject, same exercise
        others = [
            s
            for s in dataset.samples
            if s.exercise_id == exercise_id and s.subject_id != subject
        ]
        population = fit(others, exercise, percentiles=percentiles) if others else None
        if population is None:
            continue

        correct = [s for s in samples if s.correct > 0.5]
        if len(correct) < k:
            unable_subjects.add(subject)
            continue

        calibration_ids = {id(s) for s in correct[:k]}
        held_out = [s for s in samples if id(s) not in calibration_ids]
        if len(held_out) < MIN_EVAL:
            continue

        personal = fit(correct[:k], exercise, percentiles=percentiles, min_samples=k)
        hybrid = blend(personal, population, k / (k + SHRINKAGE_K0))
        personalised_subjects.add(subject)

        for sample in held_out:
            out.y_true.append(float(sample.correct > 0.5))
            out.subjects.append(subject)
            out.predictions["population"].append(float(verdict(sample, population)))
            out.predictions["personalised"].append(float(verdict(sample, personal)))
            out.predictions["hybrid"].append(float(verdict(sample, hybrid)))

    out.n_subjects_personalised = len(personalised_subjects)
    out.n_subjects_unable = len(unable_subjects - personalised_subjects)
    return out


def run(
    dataset,
    k_values: tuple[int, ...] = (3, 5, 8, 12, 20),
    *,
    n_resamples: int = 1000,
    cohort: str | None = None,
) -> dict:
    """Sweeping the number of supervised repetitions and compare the three conditions"""
    results: dict[str, dict] = {}
    for k in k_values:
        outcome = _run_one_k(dataset, k)
        if not outcome.y_true:
            continue
        y = np.array(outcome.y_true)
        subjects = np.array(outcome.subjects)

        mask = np.ones(len(y), dtype=bool)
        if cohort is not None:
            lookup = {s.subject_id: s.meta.get("cohort") for s in dataset.samples}
            mask = np.array([lookup.get(s) == cohort for s in subjects])
        labels = y[mask] > 0.5
        if mask.sum() < 20 or labels.all() or (~labels).all():
            continue  # a subset with only one class cannot support a metric

        entry: dict = {
            "n_evaluated": int(mask.sum()),
            "n_subjects_personalised": outcome.n_subjects_personalised,
            "n_subjects_unable": outcome.n_subjects_unable,
        }
        for condition, values in outcome.predictions.items():
            pred = np.array(values)[mask]
            entry[condition] = bootstrap_ci(
                y[mask], pred, list(subjects[mask]), n_resamples=n_resamples
            ).as_dict()
        results[str(k)] = entry

    return {
        "cohort": cohort or "all",
        "shrinkage_k0": SHRINKAGE_K0,
        "by_k": results,
        "verdict": _verdict(results),
    }


def _verdict(results: dict) -> str:
    if not results:
        return "no usable subject/exercise combination"
    best_k, best_gain = None, -np.inf
    for k, entry in results.items():
        gain = entry["hybrid"]["estimate"] - entry["population"]["estimate"]
        if gain > best_gain:
            best_k, best_gain = k, gain
    entry = results[best_k]
    separated = entry["hybrid"]["ci_low"] > entry["population"]["ci_high"]
    if best_gain <= 0.005:
        return (
            "personalisation gives no measurable benefit at any k tested; population thresholds are sufficient"
        )
    if separated:
        return (
            f"personalisation helps: best at k={best_k} (+{best_gain:.3f} balanced accuracy, intervals do not overlap)"
        )
    return (
        f"personalisation trends positive (best k={best_k}, +{best_gain:.3f}) but the intervals overlap; this sample does not resolve it"
    )
