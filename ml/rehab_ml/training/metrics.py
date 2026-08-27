from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.stats import kendalltau, spearmanr, binomtest

from ..taxonomy import ERROR_SLUGS, ErrorType, N_ERROR_TYPES, EQI_GRADE_BOUNDARIES


EPS = 1e-12

@dataclass(frozen=True)
class BinaryMetrics:
    """Confusion-matrix-derived metrics for one binary task"""

    n: int
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total else float("nan")

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else float("nan")

    @property
    def recall(self) -> float:
        """Sensitivity - the metric that matters most clinically"""
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else float("nan")

    @property
    def specificity(self) -> float:
        return self.tn / (self.tn + self.fp) if (self.tn + self.fp) else float("nan")

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if not (np.isfinite(p) and np.isfinite(r)) or (p + r) <= EPS:
            return float("nan")
        return 2 * p * r / (p + r)

    @property
    def balanced_accuracy(self) -> float:
        r, s = self.recall, self.specificity
        values = [v for v in (r, s) if np.isfinite(v)]
        return float(np.mean(values)) if values else float("nan")

    @property
    def support_positive(self) -> int:
        return self.tp + self.fn

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "accuracy": _r(self.accuracy),
            "balanced_accuracy": _r(self.balanced_accuracy),
            "precision": _r(self.precision),
            "recall": _r(self.recall),
            "specificity": _r(self.specificity),
            "f1": _r(self.f1),
            "support_positive": self.support_positive,
            "confusion": {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn},
        }


def binary_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, *, mask: np.ndarray | None = None
) -> BinaryMetrics:
    """Confusion metrics, ignoring non-finite targets and masked entries"""
    t = np.asarray(y_true, dtype=np.float64).ravel()
    p = np.asarray(y_pred, dtype=np.float64).ravel()
    valid = np.isfinite(t) & np.isfinite(p)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool).ravel()
    t, p = t[valid] > 0.5, p[valid] > 0.5
    return BinaryMetrics(
        n=int(t.size),
        tp=int(np.sum(t & p)),
        fp=int(np.sum(~t & p)),
        tn=int(np.sum(~t & ~p)),
        fn=int(np.sum(t & ~p)),
    )


@dataclass
class MultiLabelReport:
    """Per-error metrics plus macro and micro aggregates"""
    per_error: dict[str, BinaryMetrics] = field(default_factory=dict)

    def macro(self, attribute: str) -> float:
        """Unweighted mean over errors that actually occur in the evaluation set"""
        values = [
            getattr(m, attribute)
            for m in self.per_error.values()
            if m.support_positive > 0 and np.isfinite(getattr(m, attribute))
        ]
        return float(np.mean(values)) if values else float("nan")

    def micro(self) -> BinaryMetrics:
        return BinaryMetrics(
            n=sum(m.n for m in self.per_error.values()),
            tp=sum(m.tp for m in self.per_error.values()),
            fp=sum(m.fp for m in self.per_error.values()),
            tn=sum(m.tn for m in self.per_error.values()),
            fn=sum(m.fn for m in self.per_error.values()),
        )

    def as_dict(self) -> dict:
        return {
            "macro_f1": _r(self.macro("f1")),
            "macro_recall": _r(self.macro("recall")),
            "macro_precision": _r(self.macro("precision")),
            "macro_balanced_accuracy": _r(self.macro("balanced_accuracy")),
            "micro": self.micro().as_dict(),
            "per_error": {k: v.as_dict() for k, v in self.per_error.items()},
        }


def multilabel_report(
    y_true: np.ndarray, y_pred: np.ndarray, *, slugs: tuple[str, ...] = ERROR_SLUGS
) -> MultiLabelReport:
    """Per-error metrics from ``(N, N_ERROR_TYPES)`` arrays with NaN masking"""
    t = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    if t.ndim != 2 or t.shape[1] != N_ERROR_TYPES:
        raise ValueError(f"expected (N, {N_ERROR_TYPES}) targets, got {t.shape}")
    return MultiLabelReport(
        per_error={slugs[i]: binary_metrics(t[:, i], p[:, i]) for i in range(t.shape[1])}
    )


@dataclass(frozen=True)
class AgreementMetrics:
    """Agreement between a continuous score and a therapist's rating"""
    n: int
    spearman: float
    spearman_p: float
    kendall: float
    pearson: float
    mae: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "spearman": _r(self.spearman),
            "spearman_p": _r(self.spearman_p, 4),
            "kendall_tau": _r(self.kendall),
            "pearson": _r(self.pearson),
            "mae": _r(self.mae),
        }


def agreement(predicted: np.ndarray, reference: np.ndarray) -> AgreementMetrics:
    """Correlate a predicted score against a clinician reference"""
    a = np.asarray(predicted, dtype=np.float64).ravel()
    b = np.asarray(reference, dtype=np.float64).ravel()
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]

    nan = float("nan")
    if a.size < 3 or np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return AgreementMetrics(int(a.size), nan, nan, nan, nan, nan)

    rho, p_value = spearmanr(a, b)
    tau, _ = kendalltau(a, b)
    pearson = float(np.corrcoef(a, b)[0, 1])
    return AgreementMetrics(
        n=int(a.size),
        spearman=float(rho),
        spearman_p=float(p_value),
        kendall=float(tau),
        pearson=pearson,
        mae=float(np.mean(np.abs(a - b))),
    )


def calibrate_eqi_to_grade(
    eqi: np.ndarray, grades: np.ndarray, n_grades: int = 5
) -> tuple[float, ...]:
    """Fitting EQI cut-points that best reproduce observed therapist grades"""
    e = np.asarray(eqi, dtype=np.float64).ravel()
    g = np.asarray(grades, dtype=np.float64).ravel()
    valid = np.isfinite(e) & np.isfinite(g)
    e, g = e[valid], g[valid]
    if e.size < n_grades:
        return EQI_GRADE_BOUNDARIES

    cumulative = []
    for grade in range(n_grades - 1):
        share = float(np.mean(g <= grade))
        cumulative.append(min(max(share, 0.0), 1.0))
    boundaries = [float(np.quantile(e, q)) for q in cumulative]

    for i in range(1, len(boundaries)):
        boundaries[i] = max(boundaries[i], boundaries[i - 1] + 1e-6)
    return tuple(boundaries)


@dataclass
class ComparisonReport:
    """Comparison of the rule baseline against a learned model"""

    rule: dict
    model: dict
    n_samples: int
    label_source: str

    def as_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "label_source": self.label_source,
            "rule_based": self.rule,
            "model": self.model,
            "delta": _delta(self.rule, self.model),
        }


def _delta(rule: dict, model: dict) -> dict:
    out: dict = {}
    for key, rule_value in rule.items():
        model_value = model.get(key)
        if isinstance(rule_value, (int, float)) and isinstance(model_value, (int, float)):
            out[key] = _r(model_value - rule_value)
    return out


def mcnemar(rule_correct: np.ndarray, model_correct: np.ndarray) -> dict:
    """Exact McNemar test on paired correct/incorrect decisions"""
    a = np.asarray(rule_correct, dtype=bool).ravel()
    b = np.asarray(model_correct, dtype=bool).ravel()
    if a.size != b.size:
        raise ValueError("paired arrays must be the same length")

    model_only = int(np.sum(~a & b))
    rule_only = int(np.sum(a & ~b))
    discordant = model_only + rule_only
    if discordant == 0:
        return {
            "model_only_correct": 0,
            "rule_only_correct": 0,
            "p_value": 1.0,
            "note": "identical decisions on every sample",
        }
    result = binomtest(model_only, discordant, 0.5)
    return {
        "model_only_correct": model_only,
        "rule_only_correct": rule_only,
        "n_discordant": discordant,
        "p_value": float(result.pvalue),
    }


@dataclass(frozen=True)
class Interval:
    """A point estimate with a bootstrap confidence interval"""
    estimate: float
    low: float
    high: float
    n_subjects: int
    n_resamples: int

    @property
    def width(self) -> float:
        return self.high - self.low

    def as_dict(self) -> dict:
        return {
            "estimate": _r(self.estimate),
            "ci_low": _r(self.low),
            "ci_high": _r(self.high),
            "ci_width": _r(self.width),
            "n_subjects": self.n_subjects,
            "n_resamples": self.n_resamples,
        }

    def __str__(self) -> str:
        return f"{self.estimate:.3f} [{self.low:.3f}, {self.high:.3f}]"


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subjects: Sequence[str],
    *,
    metric: str = "balanced_accuracy",
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap confidence interval, resampling **subjects**"""
    true = np.asarray(y_true, dtype=np.float64).ravel()
    pred = np.asarray(y_pred, dtype=np.float64).ravel()
    subj = np.asarray(list(subjects))
    if not (len(true) == len(pred) == len(subj)):
        raise ValueError(
            f"length mismatch: y_true={len(true)}, y_pred={len(pred)}, subjects={len(subj)}"
        )

    unique = np.unique(subj)
    index_of = {s: np.flatnonzero(subj == s) for s in unique}
    point = getattr(binary_metrics(true, pred), metric)

    rng = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        drawn = rng.choice(unique, size=unique.size, replace=True)
        rows = np.concatenate([index_of[s] for s in drawn])
        value = getattr(binary_metrics(true[rows], pred[rows]), metric)
        if np.isfinite(value):
            samples.append(float(value))

    if len(samples) < 20:
        nan = float("nan")
        return Interval(point, nan, nan, int(unique.size), len(samples))
    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return Interval(float(point), float(low), float(high), int(unique.size), len(samples))


def _r(value: float, digits: int = 4) -> float:
    return float(round(value, digits)) if np.isfinite(value) else float("nan")


__all__ = [
    "AgreementMetrics",
    "Interval",
    "bootstrap_ci",
    "BinaryMetrics",
    "ComparisonReport",
    "ErrorType",
    "MultiLabelReport",
    "agreement",
    "binary_metrics",
    "calibrate_eqi_to_grade",
    "mcnemar",
    "multilabel_report",
]
