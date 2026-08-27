from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

from ..rules.quality_index import EqiConfig
from ..taxonomy import ErrorType
from .metrics import AgreementMetrics, agreement


FORMULATIONS = ("gated", "additive", "rom_only")
MEANINGFUL_DELTA_AUC = 0.01


def _weighted_penalty(penalties: dict[str, float], config: EqiConfig) -> float:
    """Clinically-weighted mean penalty, matching `quality_index.compute_eqi`"""
    if not penalties:
        return 0.0
    total = weight_sum = 0.0
    for name, value in penalties.items():
        try:
            error = ErrorType[name.upper()]
        except KeyError:
            continue
        w = config.weight_for(error)
        total += w * float(np.clip(value, 0.0, 1.0))
        weight_sum += w
    return total / weight_sum if weight_sum > 1e-9 else 0.0


def eqi_variants(
    components: dict[str, float],
    penalties: dict[str, float],
    config: EqiConfig | None = None,
) -> dict[str, float]:
    """Recomputing EQI under each formulation from a stored decomposition"""
    cfg = config or EqiConfig()
    weights = cfg.weights.normalised()
    performance = sum(
        float(components.get(name, 0.0)) * getattr(weights, name)
        for name in ("rom", "smoothness", "tempo", "hold", "return_to_start")
    )
    penalty = _weighted_penalty(penalties, cfg)
    span = 1.0 - cfg.gate_floor  # the gate's maximum effect

    return {
        "gated": 100.0 * performance * (1.0 - span * penalty),
        # Same maximum influence as the gate, applied additively.
        "additive": float(np.clip(100.0 * (performance - span * penalty), 0.0, 100.0)),
        "rom_only": 100.0 * float(components.get("rom", 0.0)),
    }


@dataclass
class RecordingScore:
    """One recording: mean EQI per formulation, against one therapist score"""
    recording_id: str
    subject_id: str
    exercise_id: str
    n_repetitions: int
    clinician: float
    eqi: dict[str, float]


def aggregate_per_recording(dataset, config: EqiConfig | None = None) -> list[RecordingScore]:
    """Collapsing repetitions to recordings, which is the granularity of the label"""
    groups: dict[str, list] = defaultdict(list)
    for sample in dataset.samples:
        if not np.isfinite(sample.clinician_score):
            continue
        groups[sample.meta.get("recording_id", sample.subject_id)].append(sample)

    out: list[RecordingScore] = []
    for recording_id, samples in sorted(groups.items()):
        variants = [
            eqi_variants(
                s.meta.get("eqi_components", {}), s.meta.get("eqi_penalties", {}), config
            )
            for s in samples
        ]
        out.append(
            RecordingScore(
                recording_id=recording_id,
                subject_id=samples[0].subject_id,
                exercise_id=samples[0].exercise_id,
                n_repetitions=len(samples),
                clinician=float(samples[0].clinician_score),
                eqi={
                    name: float(np.mean([v[name] for v in variants]))
                    for name in FORMULATIONS
                },
            )
        )
    return out


def validate(dataset, config: EqiConfig | None = None, *, n_resamples: int = 2000) -> dict:
    """Running both analyses. Returns a JSON-ready report"""
    recordings = aggregate_per_recording(dataset, config)
    if len(recordings) < 3:
        return {
            "n_recordings": len(recordings),
            "error": "too few recordings with a clinician score to correlate",
        }

    clinician = np.array([r.clinician for r in recordings])
    subjects = [r.subject_id for r in recordings]
    results: dict[str, dict] = {}

    for name in FORMULATIONS:
        predicted = np.array([r.eqi[name] for r in recordings])
        stats: AgreementMetrics = agreement(predicted, clinician)
        results[name] = {
            **stats.as_dict(),
            "spearman_ci": _spearman_ci(predicted, clinician, subjects, n_resamples),
            "mean_eqi": round(float(np.mean(predicted)), 2),
        }

    ranked = sorted(
        (n for n in FORMULATIONS if np.isfinite(results[n]["spearman"])),
        key=lambda n: results[n]["spearman"],
        reverse=True,
    )
    return {
        "n_recordings": len(recordings),
        "n_subjects": len(set(subjects)),
        "repetitions_per_recording": round(
            float(np.mean([r.n_repetitions for r in recordings])), 2
        ),
        "clinician_score_range": [
            round(float(clinician.min()), 1),
            round(float(clinician.max()), 1),
        ],
        "formulations": results,
        "ranking": ranked,
        "verdict": _verdict(results, ranked),
    }


def _auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """Probability that a randomly chosen correct rep outranks an incorrect one"""
    from scipy.stats import rankdata

    pos = positive.astype(bool)
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


DEFAULT_LABEL_SOURCES = frozenset({"HUMAN"})


def _rep_rows(
    dataset,
    config: EqiConfig | None,
    cohort: str | None,
    label_sources: frozenset[str] | tuple[str, ...] = DEFAULT_LABEL_SOURCES,
):
    """Per-repetition rows carrying both EQI formulations and their divergence"""
    cfg = config or EqiConfig()
    admitted = {s.upper() for s in label_sources}
    rows = []
    for sample in dataset.samples:
        if not np.isfinite(sample.correct):
            continue
        if sample.label_source.name.upper() not in admitted:
            continue
        if cohort is not None and sample.meta.get("cohort") != cohort:
            continue
        components = sample.meta.get("eqi_components", {})
        penalties = sample.meta.get("eqi_penalties", {})
        if not components:
            continue
        weights = cfg.weights.normalised()
        performance = sum(
            float(components.get(name, 0.0)) * getattr(weights, name)
            for name in ("rom", "smoothness", "tempo", "hold", "return_to_start")
        )
        penalty = _weighted_penalty(penalties, cfg)
        variants = eqi_variants(components, penalties, cfg)
        rows.append(
            {
                "eqi": variants,
                "penalty": penalty,
                "performance": performance,
                "divergence": abs(variants["gated"] - variants["additive"]),
                "correct": float(sample.correct),
                "subject": sample.subject_id,
            }
        )
    return rows


def discriminate(
    dataset,
    config: EqiConfig | None = None,
    *,
    cohort: str | None = None,
    n_resamples: int = 2000,
    label_sources: frozenset[str] | tuple[str, ...] = DEFAULT_LABEL_SOURCES,
) -> dict:
    """Does EQI separate clinician-labelled correct from incorrect repetitions"""
    rows = _rep_rows(dataset, config, cohort, label_sources)
    if len(rows) < 20:
        return {"n": len(rows), "error": "too few human-labelled repetitions"}

    correct = np.array([r["correct"] for r in rows])
    subjects = [r["subject"] for r in rows]
    subject_array = np.asarray(subjects)
    out: dict[str, dict] = {}

    for name in FORMULATIONS:
        scores = np.array([r["eqi"][name] for r in rows])
        median = float(np.median(scores))
        out[name] = {
            "auc": round(_auc(scores, correct), 4),
            "auc_ci": _paired_ci(
                lambda idx, s=scores: _auc(s[idx], correct[idx]), subjects, n_resamples
            ),
            "mean_correct": round(float(np.mean(scores[correct > 0.5])), 2),
            "mean_incorrect": round(float(np.mean(scores[correct <= 0.5])), 2),
            "balanced_accuracy_at_median": round(
                _balanced_at(scores, correct, median), 4
            ),
            # against a LOSO model
            "balanced_accuracy_loso": round(
                _balanced_loso(scores, correct, subject_array), 4
            ),
            "balanced_accuracy_loso_ci": _paired_ci(
                lambda idx, s=scores: _balanced_loso(
                    s[idx], correct[idx], subject_array[idx]
                ),
                subjects,
                n_resamples,
            ),
        }

    return {
        "n": len(rows),
        "n_subjects": len(set(subjects)),
        "cohort": cohort,
        "positive_rate": round(float(np.mean(correct)), 4),
        "formulations": out,
        "note": (
            "AUC against physiotherapist labels. Higher is better; 0.5 is chance. "
            "EQI fits no parameters, so its AUC needs no fold structure and is "
            "directly comparable with a LOSO model AUC. The THRESHOLDED figures are "
            "not equivalent: quote balanced_accuracy_loso against a LOSO model, not "
            "balanced_accuracy_at_median, whose cut is set partly by the graded subject's own scores."
        ),
    }


def _balanced_at(scores: np.ndarray, correct: np.ndarray, threshold: float) -> float:
    predicted = scores >= threshold
    pos, neg = correct > 0.5, correct <= 0.5
    if not pos.any() or not neg.any():
        return float("nan")
    sensitivity = float(predicted[pos].mean())
    specificity = float((~predicted[neg]).mean())
    return 0.5 * (sensitivity + specificity)


def _balanced_loso(
    scores: np.ndarray, correct: np.ndarray, subjects: np.ndarray
) -> float:
    """Scoring balanced accuracy with each subject cut at the median of the others"""
    subj = np.asarray(subjects)
    predicted = np.zeros(scores.shape, dtype=bool)
    graded = np.zeros(scores.shape, dtype=bool)
    for subject in np.unique(subj):
        held = subj == subject
        others = ~held
        if others.any():
            threshold = float(np.median(scores[others]))
            predicted[held] = scores[held] >= threshold
            graded[held] = True
    if not graded.any():
        return float("nan")
    return _balanced_at_predictions(predicted[graded], correct[graded])


def _balanced_at_predictions(predicted: np.ndarray, correct: np.ndarray) -> float:
    """Scoring balanced accuracy from predictions that were already made"""
    pos, neg = correct > 0.5, correct <= 0.5
    if not pos.any() or not neg.any():
        return float("nan")
    return 0.5 * (
        float(predicted[pos].mean()) + float((~predicted[neg]).mean())
    )


def gate_stratified(
    dataset,
    config: EqiConfig | None = None,
    *,
    cohort: str | None = "patient",
    n_resamples: int = 2000,
    label_sources: frozenset[str] | tuple[str, ...] = DEFAULT_LABEL_SOURCES,
) -> dict:
    """Testing the multiplicative gate where it can possibly matter"""
    rows = _rep_rows(dataset, config, cohort, label_sources)
    if len(rows) < 30:
        return {"n": len(rows), "error": "too few human-labelled repetitions"}

    divergence = np.array([r["divergence"] for r in rows])
    correct = np.array([r["correct"] for r in rows])
    gated = np.array([r["eqi"]["gated"] for r in rows])
    additive = np.array([r["eqi"]["additive"] for r in rows])
    subjects = np.array([r["subject"] for r in rows])

    identical = divergence <= 1e-9
    rest = ~identical
    cut = float(np.median(divergence[rest])) if rest.any() else 0.0

    masks = {
        "zero_divergence": identical,
        "moderate": rest & (divergence <= cut),
        "high": rest & (divergence > cut),
    }
    strata = {
        name: _gate_contrast(
            gated[mask], additive[mask], correct[mask], divergence[mask],
            list(subjects[mask]), n_resamples,
        )
        for name, mask in masks.items()
    }

    pooled = _gate_contrast(
        gated, additive, correct, divergence, list(subjects), n_resamples
    )

    return {
        "n": len(rows),
        "n_subjects": len(set(subjects.tolist())),
        "cohort": cohort,
        "stratified_on": (
            "|gated - additive| per repetition, measured directly. The closed form "
            "span * penalty * (1 - performance) identifies this as the right axis but "
            "breaks where the additive score clips at zero."
        ),
        "median_divergence_of_nonzero": round(cut, 6),
        "pooled": pooled,
        "strata": strata,
        "hypothesis": (
            "Compensation substitutes for lost excursion, so the gate should beat the "
            "additive form "
            "where compensation coincides with a performance deficit (HIGH divergence), and "
            "be exactly equivalent where the divergence term is zero. The "
            "zero_divergence stratum is an algebraic control - any non-zero effect "
            "there is an analysis error."
        ),
        "verdict": _gate_verdict(strata),
    }


def _gate_contrast(
    gated: np.ndarray,
    additive: np.ndarray,
    correct: np.ndarray,
    penalty: np.ndarray,
    subjects: list[str],
    n_resamples: int,
) -> dict:
    """Paired gated-minus-additive AUC difference for one stratum"""
    auc_gated = _auc(gated, correct)
    auc_additive = _auc(additive, correct)
    delta = auc_gated - auc_additive

    def statistic(idx: np.ndarray) -> float:
        return _auc(gated[idx], correct[idx]) - _auc(additive[idx], correct[idx])

    return {
        "n": int(gated.size),
        "positive_rate": round(float(np.mean(correct)), 4) if gated.size else None,
        "mean_penalty": round(float(np.mean(penalty)), 4) if penalty.size else None,
        "auc_gated": None if np.isnan(auc_gated) else round(auc_gated, 4),
        "auc_additive": None if np.isnan(auc_additive) else round(auc_additive, 4),
        "delta_auc": None if np.isnan(delta) else round(delta, 4),
        "delta_ci": _paired_ci(statistic, subjects, n_resamples),
        # How often the two formulations disagree about which rep is better. Near
        # zero means the comparison had no room to detect anything at all.
        "discordance": round(_discordance(gated, additive), 4),
    }


def _discordance(a: np.ndarray, b: np.ndarray, max_pairs: int = 20000) -> float:
    """Fraction of repetition pairs the two formulations rank in opposite order"""
    n = a.size
    if n < 2:
        return float("nan")
    rng = np.random.default_rng(0)
    total = n * (n - 1) // 2
    if total <= max_pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        keep = i != j
        i, j = i[keep], j[keep]
    da, db = a[i] - a[j], b[i] - b[j]
    comparable = (da != 0) & (db != 0)
    if not comparable.any():
        return float("nan")
    return float((np.sign(da[comparable]) != np.sign(db[comparable])).mean())


def _paired_ci(
    statistic, subjects: list[str], n_resamples: int, alpha: float = 0.05
) -> list[float] | None:
    """Bootstrap a statistic by resampling subjects"""
    subj = np.asarray(subjects)
    unique = np.unique(subj)
    if unique.size < 3:
        return None
    index_of = {s: np.flatnonzero(subj == s) for s in unique}
    rng = np.random.default_rng(0)
    values = []
    for _ in range(n_resamples):
        rows = np.concatenate(
            [index_of[s] for s in rng.choice(unique, size=unique.size, replace=True)]
        )
        value = statistic(rows)
        if np.isfinite(value):
            values.append(float(value))
    if len(values) < 20:
        return None
    lo, hi = np.percentile(values, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [round(float(lo), 4), round(float(hi), 4)]


def _gate_verdict(strata: dict) -> str:
    high = strata.get("high", {})
    control = strata.get("zero_divergence", {})
    high_ci = high.get("delta_ci")
    discordance = high.get("discordance")

    control_delta = control.get("delta_auc")
    if control_delta not in (None, 0.0) and abs(control_delta) > 1e-9:
        return (
            "ANALYSIS ERROR: the zero-divergence control shows a non-zero difference, "
            "but the two formulations are algebraically equal there."
        )

    if not high_ci:
        return "high-divergence stratum did not yield a usable interval"

    delta = high.get("delta_auc") or 0.0
    meaningful = abs(delta) >= MEANINGFUL_DELTA_AUC
    size_note = (
        f"Effect size is {delta:+.4f} AUC"
        + ("." if meaningful else ", which is negligible in practice.")
        + (
            f" The two forms rank only {discordance:.1%} of repetition pairs "
            "differently, which bounds how large any difference could have been."
            if discordance is not None and np.isfinite(discordance)
            else ""
        )
    )

    if high_ci[0] > 0.0:
        headline = (
            "DIRECTION AS PREDICTED: the gate beats the additive form where "
            "compensation coincides with a performance deficit, is exactly equivalent "
            "in the algebraic control, and the interval excludes zero."
        )
        return f"{headline} {size_note}" if meaningful else (
            f"{headline} BUT: {size_note} Statistically resolvable, practically negligible"
        )
    if high_ci[1] < 0.0:
        return f"The additive form is significantly better at high divergence. {size_note}"
    return (
        f"Not resolved: the high-divergence interval includes zero. The gate is not supported by this sample. {size_note}"
    )


def _spearman_ci(
    predicted: np.ndarray, clinician: np.ndarray, subjects: list[str], n_resamples: int
) -> list[float] | None:
    """Bootstrap interval on Spearman rho, resampling subjects"""
    subj = np.array(subjects)
    unique = np.unique(subj)
    if unique.size < 3:
        return None
    index_of = {s: np.flatnonzero(subj == s) for s in unique}
    rng = np.random.default_rng(0)
    values = []
    for _ in range(n_resamples):
        rows = np.concatenate(
            [index_of[s] for s in rng.choice(unique, size=unique.size, replace=True)]
        )
        if np.unique(predicted[rows]).size < 3 or np.unique(clinician[rows]).size < 3:
            continue
        rho, _ = spearmanr(predicted[rows], clinician[rows])
        if np.isfinite(rho):
            values.append(float(rho))
    if len(values) < 20:
        return None
    return [round(float(v), 4) for v in np.percentile(values, [2.5, 97.5])]


def _verdict(results: dict, ranked: list[str]) -> str:
    if not ranked:
        return "no formulation produced a usable correlation"
    best = ranked[0]
    gated = results.get("gated", {}).get("spearman", float("nan"))
    additive = results.get("additive", {}).get("spearman", float("nan"))
    if not (np.isfinite(gated) and np.isfinite(additive)):
        return f"{best} ranks highest; the gated-vs-additive comparison is incomplete"

    gated_ci = results["gated"].get("spearman_ci")
    additive_ci = results["additive"].get("spearman_ci")
    if gated_ci and additive_ci:
        if gated_ci[0] > additive_ci[1]:
            return "gated agrees with clinicians significantly better than additive"
        if additive_ci[0] > gated_ci[1]:
            return "additive agrees significantly better than gated"
        return (
            "gated and additive intervals overlap: this sample does not resolve which formulation agrees better."
        )
    return f"{best} ranks highest, without a resolvable interval"
