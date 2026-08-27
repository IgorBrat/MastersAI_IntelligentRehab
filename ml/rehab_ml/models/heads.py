from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..taxonomy import N_ERROR_TYPES


@dataclass
class HeadOutputs:
    """What every model returns"""

    # (B,) logit for "this repetition was performed correctly"
    correct_logit: torch.Tensor

    # (B, N_ERROR_TYPES) per-error logits
    error_logits: torch.Tensor

    # (B,) predicted Exercise Quality Index, in [0, 100]
    eqi: torch.Tensor

    # (B, F) auxiliary prediction of the interpretable summary features
    # Only produced by `clinical_hybrid.ClinicalHybridModel`
    clinical_features: torch.Tensor | None = None

    # (B, F) the normalised target the above is trained agains
    clinical_target: torch.Tensor | None = None

    # (B, D) pooled representation, kept for t-SNE plots
    embedding: torch.Tensor | None = None

    def probabilities(self) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.sigmoid(self.correct_logit), torch.sigmoid(self.error_logits)


class PredictionHeads(nn.Module):
    """The three output heads applied to a pooled sequence embedding"""

    def __init__(self, in_features: int, *, n_errors: int = N_ERROR_TYPES) -> None:
        super().__init__()
        self.correct = nn.Linear(in_features, 1)
        self.errors = nn.Linear(in_features, n_errors)
        self.eqi = nn.Linear(in_features, 1)

    def forward(self, embedding: torch.Tensor) -> HeadOutputs:
        return HeadOutputs(
            correct_logit=self.correct(embedding).squeeze(-1),
            error_logits=self.errors(embedding),
            eqi=100.0 * torch.sigmoid(self.eqi(embedding).squeeze(-1)),
            embedding=embedding,
        )


@dataclass
class LossWeights:
    """Relative weights of the multi-task objective"""
    correct: float = 1.0
    errors: float = 1.5
    eqi: float = 0.5
    clinical_features: float = 0.3


def masked_bce(
    logits: torch.Tensor, targets: torch.Tensor, *, pos_weight: torch.Tensor | None = None
) -> torch.Tensor:
    """Binary cross-entropy that ignores ``NaN`` targets"""
    mask = torch.isfinite(targets)
    if not mask.any():
        return logits.sum() * 0.0

    safe_targets = torch.where(mask, targets, torch.zeros_like(targets))
    per_element = F.binary_cross_entropy_with_logits(
        logits, safe_targets, reduction="none", pos_weight=pos_weight
    )
    return (per_element * mask).sum() / mask.sum()


def compute_loss(
    outputs: HeadOutputs,
    batch: dict[str, torch.Tensor],
    weights: LossWeights | None = None,
    *,
    error_pos_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-task loss. Returns ``(total, per_term_values)`` for logging"""
    w = weights or LossWeights()
    parts: dict[str, float] = {}

    correct_target = batch["correct"].to(outputs.correct_logit.dtype)
    loss_correct = masked_bce(outputs.correct_logit, correct_target)
    parts["correct"] = float(loss_correct.detach())

    loss_errors = masked_bce(
        outputs.error_logits, batch["errors"].to(outputs.error_logits.dtype),
        pos_weight=error_pos_weight,
    )
    parts["errors"] = float(loss_errors.detach())

    eqi_target = batch["eqi"].to(outputs.eqi.dtype)
    eqi_mask = torch.isfinite(eqi_target)
    if eqi_mask.any():
        loss_eqi = F.smooth_l1_loss(outputs.eqi[eqi_mask] / 100.0, eqi_target[eqi_mask] / 100.0)
    else:
        loss_eqi = outputs.eqi.sum() * 0.0
    parts["eqi"] = float(loss_eqi.detach())

    total = w.correct * loss_correct + w.errors * loss_errors + w.eqi * loss_eqi

    if outputs.clinical_features is not None and (
        outputs.clinical_target is not None or "summary" in batch
    ):
        target = (
            outputs.clinical_target
            if outputs.clinical_target is not None
            else batch["summary"]
        ).to(outputs.clinical_features.dtype)
        mask = torch.isfinite(target)
        if mask.any():
            loss_clinical = F.smooth_l1_loss(
                outputs.clinical_features[mask], target[mask]
            )
            parts["clinical_features"] = float(loss_clinical.detach())
            total = total + w.clinical_features * loss_clinical

    parts["total"] = float(total.detach())
    return total, parts


def error_pos_weight(targets: torch.Tensor, *, cap: float = 20.0) -> torch.Tensor:
    """Per-error positive-class weight from observed label frequencies"""
    mask = torch.isfinite(targets)
    safe = torch.where(mask, targets, torch.zeros_like(targets))
    positives = (safe * mask).sum(dim=0)
    watched = mask.sum(dim=0).clamp(min=1)
    negatives = watched - positives
    weight = negatives / positives.clamp(min=1.0)
    return weight.clamp(min=0.1, max=cap)
