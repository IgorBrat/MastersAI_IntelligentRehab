from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from ..rules.quality_index import EqiConfig
from .eqi_validation import DEFAULT_LABEL_SOURCES, _auc, _paired_ci, _rep_rows, _weighted_penalty


# the five performance components, in the order used throughout
COMPONENTS: tuple[str, ...] = ("rom", "smoothness", "tempo", "hold", "return_to_start")

# below this standard deviation a component is treated as carrying no information
DEAD_SD = 0.01


@dataclass(frozen=True)
class ComponentStat:
    """Descriptive statistics for one component, against its assigned weight"""
    name: str
    weight: float
    mean: float
    sd: float

    @property
    def is_dead(self) -> bool:
        return self.sd < DEAD_SD

    def as_dict(self) -> dict:
        return {
            "component": self.name,
            "weight": round(self.weight, 4),
            "mean": round(self.mean, 4),
            "sd": round(self.sd, 4),
            "dead": self.is_dead,
        }


def audit(dataset, config: EqiConfig | None = None, *, cohort: str | None = "patient",
          label_sources=DEFAULT_LABEL_SOURCES) -> dict:
    """Descriptive audit: assigned weight versus measured variance, per component"""
    cfg = config or EqiConfig()
    weights = cfg.weights.normalised()
    rows = _rep_rows(dataset, cfg, cohort, label_sources)
    if not rows:
        return {"error": "no repetitions matched the cohort and label-source filter"}

    values = _component_matrix(dataset, cfg, cohort, label_sources)
    stats = [
        ComponentStat(name, float(getattr(weights, name)),
                      float(values[:, i].mean()), float(values[:, i].std()))
        for i, name in enumerate(COMPONENTS)
    ]
    live = [s for s in stats if not s.is_dead]
    dead_weight = sum(s.weight for s in stats if s.is_dead)
    live_total = sum(s.weight for s in live) or 1.0

    return {
        "n": int(values.shape[0]),
        "cohort": cohort,
        "components": [s.as_dict() for s in stats],
        "dead_components": [s.name for s in stats if s.is_dead],
        "dead_weight_fraction": round(dead_weight, 4),
        "effective_weights": {s.name: round(s.weight / live_total, 4) for s in live},
        "note": (
            "A dead component is not excluded from the index - it is added as a constant, "
            "which inflates every score and compresses the dynamic range of the rest. "
            "Effective weights are what the index behaves as once they are removed."
        ),
    }


def _component_matrix(dataset, cfg, cohort, label_sources) -> np.ndarray:
    """``(n, 5)`` component values for the filtered repetitions"""
    out = []
    admitted = {s.upper() for s in label_sources}
    for sample in dataset.samples:
        if not np.isfinite(sample.correct):
            continue
        if sample.label_source.name.upper() not in admitted:
            continue
        if cohort is not None and sample.meta.get("cohort") != cohort:
            continue
        components = sample.meta.get("eqi_components") or {}
        if not components:
            continue
        out.append([float(components.get(name, 0.0)) for name in COMPONENTS])
    return np.asarray(out, dtype=np.float64)


def _rows_with_components(dataset, cfg, cohort, label_sources):
    """Repetition rows carrying components, penalty, label and subject together"""
    admitted = {s.upper() for s in label_sources}
    rows = []

    for sample in dataset.samples:
        if not np.isfinite(sample.correct):
            continue
        if sample.label_source.name.upper() not in admitted:
            continue
        if cohort is not None and sample.meta.get("cohort") != cohort:
            continue
        components = sample.meta.get("eqi_components") or {}
        if not components:
            continue
        rows.append({
            "x": np.array([float(components.get(k, 0.0)) for k in COMPONENTS]),
            "penalty": _weighted_penalty(sample.meta.get("eqi_penalties") or {}, cfg),
            "correct": float(sample.correct),
            "subject": sample.subject_id,
        })
    return rows


def compare_weightings(
    dataset,
    config: EqiConfig | None = None,
    *,
    cohort: str | None = "patient",
    label_sources=DEFAULT_LABEL_SOURCES,
    n_resamples: int = 2000,
) -> dict:
    """Clinical vs renormalised vs fold-fitted weights, scored by AUC"""
    cfg = config or EqiConfig()
    rows = _rows_with_components(dataset, cfg, cohort, label_sources)
    if len(rows) < 30:
        return {"n": len(rows), "error": "too few labelled repetitions"}

    x = np.stack([r["x"] for r in rows])
    penalty = np.array([r["penalty"] for r in rows])
    y = np.array([r["correct"] for r in rows])
    subjects = np.array([r["subject"] for r in rows])
    span = 1.0 - cfg.gate_floor
    gate = 1.0 - span * penalty

    weights = cfg.weights.normalised()
    clinical = np.array([float(getattr(weights, name)) for name in COMPONENTS])

    alive = x.std(axis=0) >= DEAD_SD
    renormalised = np.where(alive, clinical, 0.0)
    total = renormalised.sum()
    renormalised = renormalised / total if total > 0 else clinical

    fitted_scores, fitted_weights = _fold_fitted(x, y, subjects, gate)

    arms = {
        "clinical": 100.0 * (x @ clinical) * gate,
        "renormalised": 100.0 * (x @ renormalised) * gate,
        "fitted": fitted_scores,
    }
    results = {}
    for name, score in arms.items():
        usable = np.isfinite(score)
        results[name] = {
            "auc": round(_auc(score[usable], y[usable]), 4),
            "auc_ci": _paired_ci(
                lambda idx, s=score: _auc(s[idx], y[idx]), list(subjects), n_resamples
            ),
            "n_scored": int(usable.sum()),
        }

    # paired difference against the clinical arm
    reference = arms["clinical"]
    for name, score in arms.items():
        if name == "clinical":
            results[name]["delta_auc"] = 0.0
            results[name]["delta_ci"] = None
            continue
        usable = np.isfinite(score) & np.isfinite(reference)

        def statistic(idx, s=score, r=reference):
            keep = idx[np.isfinite(s[idx]) & np.isfinite(r[idx])]
            if keep.size < 3:
                return float("nan")
            return _auc(s[keep], y[keep]) - _auc(r[keep], y[keep])

        results[name]["delta_auc"] = round(
            _auc(score[usable], y[usable]) - _auc(reference[usable], y[usable]), 4
        )
        results[name]["delta_ci"] = _paired_ci(statistic, list(subjects), n_resamples)

    return {
        "n": len(rows),
        "n_subjects": int(np.unique(subjects).size),
        "cohort": cohort,
        "clinical_weights": {k: round(float(v), 4) for k, v in zip(COMPONENTS, clinical)},
        "renormalised_weights": {k: round(float(v), 4) for k, v in zip(COMPONENTS, renormalised)},
        "fitted_weights_mean": {k: round(float(v), 4) for k, v in zip(COMPONENTS, fitted_weights)},
        "arms": results,
        "verdict": _verdict(results),
    }


def _fold_fitted(x, y, subjects, gate, *, max_iter: int = 400):
    """Leave-one-subject-out fitted weights, plus their mean across folds"""
    scores = np.full(x.shape[0], np.nan)
    collected = []
    for subject in np.unique(subjects):
        test = subjects == subject
        train = ~test
        if train.sum() < 10 or np.unique(y[train]).size < 2:
            continue
        model = LogisticRegression(max_iter=max_iter)
        model.fit(x[train], y[train])
        coef = np.clip(model.coef_.ravel(), 0.0, None)
        total = coef.sum()
        coef = coef / total if total > 1e-9 else np.full(len(COMPONENTS), 1 / len(COMPONENTS))
        collected.append(coef)
        scores[test] = 100.0 * (x[test] @ coef) * gate[test]
    mean_weights = (
        np.mean(collected, axis=0) if collected else np.full(len(COMPONENTS), np.nan)
    )
    return scores, mean_weights


def _verdict(results: dict) -> str:
    """Reading the paired delta interval"""
    best = max(results, key=lambda k: results[k]["auc"])
    if best == "clinical":
        return (
            "The clinical weights are not improved on. The variance mismatch is real, but reweighting does not exploit it"
        )
    delta = results[best].get("delta_auc") or 0.0
    ci = results[best].get("delta_ci")
    if not ci:
        return f"'{best}' leads by {delta:+.4f} AUC; no usable interval on the difference."
    if ci[0] > 0.0:
        return (
            f"'{best}' beats the clinical weights by {delta:+.4f} AUC, 95% CI "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] excluding zero. Consistent within resamples, "
            "so the reweighting is supported even though the two separate AUC intervals "
            "overlap - the arms score the same repetitions"
        )
    return (
        f"'{best}' leads by {delta:+.4f} AUC but the paired interval "
        f"[{ci[0]:+.4f}, {ci[1]:+.4f}] includes zero. Not resolved by this sample."
    )
