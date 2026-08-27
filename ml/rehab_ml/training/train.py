from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..data.dataset import LabelSource, RepetitionDataset, leave_one_subject_out
from ..exercises.registry import get_exercise
from ..rules.calibration import fit as calibrate_rules
from ..rules.calibration import verdict as rule_verdict
from ..models.clinical_hybrid import ClinicalHybridModel, HybridConfig
from ..models.heads import LossWeights, compute_loss, error_pos_weight
from ..models.sequence import LSTMClassifier, SequenceModelConfig, TCNClassifier
from .metrics import (
    ComparisonReport,
    bootstrap_ci,
    agreement,
    binary_metrics,
    mcnemar,
    multilabel_report,
)
from ..features.angles import (
    PLANE_OF_ELEVATION_MAX_ELEVATION_DEG,
    PLANE_OF_ELEVATION_MIN_ELEVATION_DEG,
    SCAPULOHUMERAL_ONSET_DEG,
    SCAPULOHUMERAL_RISE_PER_DEG,
)


MODEL_REGISTRY = {
    "lstm": LSTMClassifier,
    "tcn": TCNClassifier,
    "hybrid": ClinicalHybridModel,
}


@dataclass
class TrainConfig:
    """Optimisation settings"""

    model: str = "hybrid"
    epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    # Fraction of training subjects held out for early stopping
    val_fraction: float = 0.2
    patience: int = 12
    seed: int = 0
    device: str = "cpu"
    loss_weights: LossWeights = field(default_factory=LossWeights)
    # Balance the per-error loss by observed class frequency
    use_pos_weight: bool = True

    def build_model(self) -> torch.nn.Module:
        if self.model not in MODEL_REGISTRY:
            raise KeyError(
                f"unknown model {self.model!r}; known: {', '.join(MODEL_REGISTRY)}"
            )
        if self.model == "hybrid":
            return ClinicalHybridModel(HybridConfig())
        return MODEL_REGISTRY[self.model](SequenceModelConfig())


def _make_loader(
    dataset: RepetitionDataset, batch_size: int, *, shuffle: bool
) -> DataLoader:
    stacked = dataset.stack()
    tensors = TensorDataset(
        torch.from_numpy(stacked["sequence"]),
        torch.from_numpy(stacked["summary"]),
        torch.from_numpy(stacked["errors"]),
        torch.from_numpy(stacked["correct"]),
        torch.from_numpy(stacked["eqi"]),
    )
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _to_batch(raw: tuple[torch.Tensor, ...], device: str) -> dict[str, torch.Tensor]:
    sequence, summary, errors, correct, eqi = (t.to(device) for t in raw)
    return {
        "sequence": sequence,
        "summary": summary,
        "errors": errors,
        "correct": correct,
        "eqi": eqi,
    }


def train_model(
    train_set: RepetitionDataset,
    config: TrainConfig | None = None,
    *,
    val_set: RepetitionDataset | None = None,
    verbose: bool = False,
) -> tuple[torch.nn.Module, dict]:
    """Training one model. Returns ``(model, history)``"""
    cfg = config or TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if val_set is None:
        subjects = list(train_set.subjects)
        n_val = max(1, int(round(cfg.val_fraction * len(subjects)))) if len(subjects) > 2 else 0
        if n_val:
            rng = np.random.default_rng(cfg.seed)
            shuffled = list(subjects)
            rng.shuffle(shuffled)
            val_subjects = shuffled[:n_val]
            val_set = train_set.filter(subjects=val_subjects)
            train_set = train_set.filter(
                subjects=[s for s in subjects if s not in val_subjects]
            )

    model = cfg.build_model().to(cfg.device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.epochs)

    pos_weight = None
    if cfg.use_pos_weight:
        pos_weight = error_pos_weight(
            torch.from_numpy(train_set.stack()["errors"])
        ).to(cfg.device)

    train_loader = _make_loader(train_set, cfg.batch_size, shuffle=True)
    history: dict = {"train_loss": [], "val_loss": [], "epochs_run": 0}
    best_state, best_val, stale = None, float("inf"), 0

    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for raw in train_loader:
            batch = _to_batch(raw, cfg.device)
            outputs = model(
                batch["sequence"], summary=batch["summary"], baseline_eqi=batch["eqi"]
            )
            loss, _ = compute_loss(
                outputs, batch, cfg.loss_weights, error_pos_weight=pos_weight
            )
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        history["train_loss"].append(float(np.mean(losses)) if losses else float("nan"))

        if val_set is not None and len(val_set):
            val_loss = _evaluate_loss(model, val_set, cfg, pos_weight)
            history["val_loss"].append(val_loss)
            if val_loss < best_val - 1e-4:
                best_val, stale = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= cfg.patience:
                    if verbose:
                        print(f"  early stop at epoch {epoch + 1}")
                    break
        history["epochs_run"] = epoch + 1

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_val if np.isfinite(best_val) else None
    return model, history


@torch.no_grad()
def _evaluate_loss(
    model: torch.nn.Module,
    dataset: RepetitionDataset,
    cfg: TrainConfig,
    pos_weight: torch.Tensor | None,
) -> float:
    model.eval()
    loader = _make_loader(dataset, cfg.batch_size, shuffle=False)
    losses = []
    for raw in loader:
        batch = _to_batch(raw, cfg.device)
        outputs = model(
            batch["sequence"], summary=batch["summary"], baseline_eqi=batch["eqi"]
        )
        loss, _ = compute_loss(outputs, batch, cfg.loss_weights, error_pos_weight=pos_weight)
        losses.append(float(loss))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict(
    model: torch.nn.Module, dataset: RepetitionDataset, *, device: str = "cpu"
) -> dict[str, np.ndarray]:
    """Running inference over a dataset, returning numpy arrays"""
    model.eval().to(device)
    loader = _make_loader(dataset, 64, shuffle=False)
    correct, errors, eqi = [], [], []
    for raw in loader:
        batch = _to_batch(raw, device)
        outputs = model(
            batch["sequence"], summary=batch["summary"], baseline_eqi=batch["eqi"]
        )
        p_correct, p_errors = outputs.probabilities()
        correct.append(p_correct.cpu().numpy())
        errors.append(p_errors.cpu().numpy())
        eqi.append(outputs.eqi.cpu().numpy())
    return {
        "correct_prob": np.concatenate(correct),
        "error_prob": np.concatenate(errors),
        "eqi": np.concatenate(eqi),
    }


def cross_validate(
    dataset: RepetitionDataset,
    config: TrainConfig | None = None,
    *,
    threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Leave-one-subject-out evaluation of a model against the rule baseline"""
    cfg = config or TrainConfig()
    arrays, fold_reports = _run_folds(dataset, cfg, threshold, verbose)
    return _build_report(dataset, cfg, arrays, fold_reports)


def _run_folds(
    dataset: RepetitionDataset,
    cfg: TrainConfig,
    threshold: float,
    verbose: bool,
    *,
    label: str = "",
) -> tuple[dict, list[dict]]:
    """Training one model per held-out subject and pool the predictions"""
    folds = leave_one_subject_out(dataset)
    if not folds:
        raise ValueError("dataset contains no subjects")

    pooled: dict[str, list] = {
        "correct_true": [], "correct_pred": [],
        "errors_true": [], "errors_pred": [],
        "eqi_model": [], "eqi_rule": [], "clinician": [],
        "rule_correct_pred": [], "rule_calibrated_pred": [], "subject": [],
    }
    calibrations: list[dict] = []
    fold_reports = []

    for index, (train_set, test_set) in enumerate(folds):
        # Mirror-augmented copies may train but must never be evaluated
        test_set = RepetitionDataset(
            samples=[s for s in test_set.samples if not s.meta.get("mirrored")],
            feature_config=test_set.feature_config,
        )
        if not len(train_set) or not len(test_set):
            continue
        # Fit rule thresholds on this fold's training subjects only
        fold_calibration = {
            ex: calibrate_rules(
                train_set.filter(exercise_id=ex).samples, get_exercise(ex)
            )
            for ex in train_set.exercise_ids
        }
        calibrations.append(
            {"fold": index, **{ex: c.as_dict() for ex, c in fold_calibration.items()}}
        )

        model, history = train_model(train_set, cfg)
        predictions = predict(model, test_set, device=cfg.device)
        stacked = test_set.stack()

        pooled["correct_true"].append(stacked["correct"])
        pooled["correct_pred"].append((predictions["correct_prob"] >= threshold).astype(float))
        pooled["errors_true"].append(stacked["errors"])
        pooled["errors_pred"].append((predictions["error_prob"] >= threshold).astype(float))
        pooled["eqi_model"].append(predictions["eqi"])
        pooled["eqi_rule"].append(stacked["eqi"])
        pooled["clinician"].append(stacked["clinician"])
        # The rule engine's own correct/incorrect decision on the same samples
        pooled["rule_correct_pred"].append(
            np.array([float(s.meta.get("rule_correct", np.nan)) for s in test_set.samples])
        )
        pooled["rule_calibrated_pred"].append(
            np.array([
                float(rule_verdict(s, fold_calibration[s.exercise_id]))
                if s.exercise_id in fold_calibration
                else float(s.meta.get("rule_correct", np.nan))
                for s in test_set.samples
            ])
        )
        pooled["subject"].extend(s.subject_id for s in test_set.samples)

        fold_reports.append(
            {
                "fold": index,
                "held_out_subject": test_set.subjects[0] if test_set.subjects else None,
                "n_train": len(train_set),
                "n_test": len(test_set),
                "epochs_run": history["epochs_run"],
                "best_val_loss": history["best_val_loss"],
                "train_loss": [round(v, 5) for v in history["train_loss"]],
                "val_loss": [round(v, 5) for v in history["val_loss"]],
            }
        )
        if verbose:
            held = test_set.subjects[0] if test_set.subjects else "?"
            prefix = f"[{label}] " if label else ""
            print(f"  {prefix}fold {index + 1}/{len(folds)}  held out {held}  n_test={len(test_set)}")

    if not fold_reports:
        raise ValueError("no usable folds; every fold had an empty train or test split")

    arrays = {k: (np.concatenate(v) if v and k != "subject" else v) for k, v in pooled.items()}
    if calibrations:
        arrays["_calibrations"] = calibrations
    return arrays, fold_reports


def _build_report(
    dataset: RepetitionDataset, cfg: TrainConfig, arrays: dict, fold_reports: list[dict]
) -> dict:
    """Turn pooled fold predictions into the JSON report"""
    model_binary = binary_metrics(arrays["correct_true"], arrays["correct_pred"])
    rule_binary = binary_metrics(arrays["correct_true"], arrays["rule_correct_pred"])
    calibrated_binary = binary_metrics(
        arrays["correct_true"], arrays.get("rule_calibrated_pred", arrays["rule_correct_pred"])
    )
    model_errors = multilabel_report(arrays["errors_true"], arrays["errors_pred"])

    comparison = ComparisonReport(
        rule={
            "accuracy": rule_binary.accuracy,
            "balanced_accuracy": rule_binary.balanced_accuracy,
            "precision": rule_binary.precision,
            "recall": rule_binary.recall,
            "f1": rule_binary.f1,
        },
        model={
            "accuracy": model_binary.accuracy,
            "balanced_accuracy": model_binary.balanced_accuracy,
            "precision": model_binary.precision,
            "recall": model_binary.recall,
            "f1": model_binary.f1,
        },
        n_samples=int(arrays["correct_true"].size),
        label_source=_dominant_label_source(dataset),
    )

    return {
        "config": {**asdict(cfg), "loss_weights": asdict(cfg.loss_weights)},
        "dataset": dataset.summary(),
        "n_folds": len(fold_reports),
        "folds": fold_reports,
        "binary_correct": model_binary.as_dict(),
        "errors": model_errors.as_dict(),
        "rule_vs_model": comparison.as_dict(),
        "rule_calibrated": calibrated_binary.as_dict(),
        "calibration": _first_calibration(arrays),
        # provenance of the measurement, not of the model
        "angle_config": _angle_provenance(),
        "mcnemar": mcnemar(
            np.asarray(arrays["rule_correct_pred"]) == np.asarray(arrays["correct_true"]),
            np.asarray(arrays["correct_pred"]) == np.asarray(arrays["correct_true"]),
        ),
        "eqi_vs_clinician": agreement(arrays["eqi_rule"], arrays["clinician"]).as_dict(),
        "model_eqi_vs_clinician": agreement(arrays["eqi_model"], arrays["clinician"]).as_dict(),
        "intervals": _intervals(arrays),
        "predictions": _predictions(arrays),
        "by_cohort": _stratified(dataset, arrays),
        "warnings": _warnings(dataset),
    }


N_RESAMPLES = 2000


def _intervals(arrays: dict) -> dict:
    """Subject-level bootstrap intervals for the headline metrics"""
    subjects = list(arrays.get("subject") or [])
    true = arrays.get("correct_true")
    if true is None or len(subjects) != len(true):
        return {}
    out: dict[str, dict] = {}
    for name, key in (
        ("model", "correct_pred"),
        ("rule_protocol", "rule_correct_pred"),
        ("rule_calibrated", "rule_calibrated_pred"),
    ):
        pred = arrays.get(key)
        if pred is None:
            continue
        for metric in ("balanced_accuracy", "recall"):
            out[f"{name}.{metric}"] = bootstrap_ci(
                true, pred, subjects, metric=metric, n_resamples=N_RESAMPLES
            ).as_dict()
    return out


def _predictions(arrays: dict) -> dict:
    """Per-sample predictions"""
    subjects = list(arrays.get("subject") or [])
    if not subjects:
        return {}
    keys = ("correct_true", "correct_pred", "rule_correct_pred",
            "rule_calibrated_pred", "eqi_model", "eqi_rule", "clinician")
    out: dict[str, list] = {"subject": subjects}
    for key in keys:
        value = arrays.get(key)
        if value is not None and len(value) == len(subjects):
            out[key] = [None if not np.isfinite(v) else round(float(v), 4) for v in value]
    return out


def _angle_provenance() -> dict:
    """Measurement constants that change the rule-derived training targets"""
    return {
        "scapulohumeral_rise_per_deg": SCAPULOHUMERAL_RISE_PER_DEG,
        "scapulohumeral_onset_deg": SCAPULOHUMERAL_ONSET_DEG,
        "plane_gate_deg": [
            PLANE_OF_ELEVATION_MIN_ELEVATION_DEG,
            PLANE_OF_ELEVATION_MAX_ELEVATION_DEG,
        ],
    }


def _first_calibration(arrays: dict) -> dict | None:
    """The first fold's fitted thresholds"""
    calibrations = arrays.get("_calibrations")
    if calibrations is None:
        return None
    calibrations = list(calibrations)
    return calibrations[0] if calibrations else None


def _stratified(dataset: RepetitionDataset, arrays: dict) -> dict:
    """Metrics split by subject cohort, when the corpus records one"""
    lookup = {s.subject_id: s.meta.get("cohort") for s in dataset.samples}
    subjects = arrays.get("subject") or []
    if not lookup or len(subjects) != len(arrays.get("correct_true", [])):
        return {}
    cohorts = np.array([lookup.get(sid) or "unknown" for sid in subjects])
    if set(cohorts) == {"unknown"}:
        return {}
    out: dict[str, dict] = {}
    for name in sorted(set(cohorts)):
        mask = cohorts == name
        model = binary_metrics(
            arrays["correct_true"][mask], arrays["correct_pred"][mask]
        )
        rule = binary_metrics(
            arrays["correct_true"][mask], arrays["rule_correct_pred"][mask]
        )
        calibrated = binary_metrics(
            arrays["correct_true"][mask],
            arrays.get("rule_calibrated_pred", arrays["rule_correct_pred"])[mask],
        )
        out[name] = {
            "n": int(mask.sum()),
            "positive_rate": float(np.mean(arrays["correct_true"][mask] > 0.5)),
            "model": model.as_dict(),
            "rule": rule.as_dict(),
            "rule_calibrated": calibrated.as_dict(),
            # A cohort with almost no negatives cannot support a balanced-accuracy estimate
            "degenerate": bool(
                min(
                    float(np.mean(arrays["correct_true"][mask] > 0.5)),
                    1.0 - float(np.mean(arrays["correct_true"][mask] > 0.5)),
                )
                < 0.05
            ),
        }
    return out


def cross_validate_per_exercise(
    dataset: RepetitionDataset,
    config: TrainConfig | None = None,
    *,
    threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Training one **specialist model per exercise**, each with its own LOSO run"""
    cfg = config or TrainConfig()
    per_exercise: dict[str, dict] = {}
    merged: dict[str, list] = {}
    all_folds: list[dict] = []

    for exercise_id in dataset.exercise_ids:
        subset = dataset.filter(exercise_id=exercise_id)
        if len(subset.subjects) < 3:
            if verbose:
                print(f"  skipping {exercise_id}: only {len(subset.subjects)} subjects")
            continue
        if verbose:
            print(f"\n[{exercise_id}] {len(subset)} repetitions, "
                  f"{len(subset.subjects)} subjects")
        arrays, folds = _run_folds(subset, cfg, threshold, verbose, label=exercise_id)
        per_exercise[exercise_id] = _build_report(subset, cfg, arrays, folds)
        all_folds.extend({**f, "exercise_id": exercise_id} for f in folds)
        for key, value in arrays.items():
            merged.setdefault(key, []).append(value)

    if not per_exercise:
        raise ValueError("no exercise had enough subjects for a leave-one-subject-out run")

    list_valued = {"subject", "_calibrations"}
    pooled = {
        key: (sum((list(v) for v in values), []) if key in list_valued
              else np.concatenate(values))
        for key, values in merged.items()
    }
    return {
        "mode": "per_exercise",
        "per_exercise": per_exercise,
        "overall": _build_report(dataset, cfg, pooled, all_folds),
        "n_models": len(per_exercise) * max(1, len(dataset.subjects)),
    }


def _dominant_label_source(dataset: RepetitionDataset) -> str:
    counts: dict[str, int] = {}
    for sample in dataset.samples:
        counts[sample.label_source.value] = counts.get(sample.label_source.value, 0) + 1
    return max(counts, key=counts.get) if counts else "unknown"


def _warnings(dataset: RepetitionDataset) -> list[str]:
    """Conditions that would make the reported numbers misleading"""
    out: list[str] = []
    sources = {s.label_source for s in dataset.samples}

    if LabelSource.SYNTHETIC in sources:
        out.append(
            "Dataset contains SYNTHETIC samples. These are generated from the same "
            "parameters the system measures, so any accuracy computed on them is circular."
        )
    if sources == {LabelSource.RULE}:
        out.append(
            "All labels are rule-derived. Model-versus-rule comparison on these is "
            "circular: it measures how well the model imitates the rule engine, not "
            "which is clinically more accurate."
        )
    if len(dataset.subjects) < 5:
        out.append(
            f"Only {len(dataset.subjects)} subjects. Leave-one-subject-out estimates "
            "from so few subjects carry very wide confidence intervals."
        )
    return out


def save_report(report: dict, path: str | Path) -> Path:
    """Writing a cross-validation report as JSON"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str))
    return destination
