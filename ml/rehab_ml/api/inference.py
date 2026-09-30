"""Running an exported model artifact beside the rule engine"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..exercises.base import ExerciseDefinition
from ..features.angles import AngleSeries
from ..features.extractor import FeatureConfig, sequence_features, summary_features
from ..taxonomy import ERROR_SLUGS
from ..training.export import load


class ModelRunner:
    def __init__(self, model: torch.jit.ScriptModule, meta: dict, *, path: Path) -> None:
        self._model = model
        self.meta = meta
        self.path = path

        config = dict(meta["feature_config"])
        # JSON has no tuples, and `n_sequence_channels` counts the channel list
        config["channels"] = tuple(config["channels"])
        self.feature_config = FeatureConfig(**config)

        self.exercise_id: str | None = meta.get("exercise_id")
        self.self_paced: bool = bool(meta.get("self_paced", False))

    @classmethod
    def from_path(cls, path: str | Path) -> "ModelRunner":
        target = Path(path)
        model, meta = load(target)
        return cls(model, meta, path=target)

    def startup_warnings(self, *, self_paced: bool) -> list[str]:
        out = []
        if self.self_paced != self_paced:
            out.append(
                f"{self.path.name} was fitted with self_paced={self.self_paced} but "
                f"the service is running with self_paced={self_paced}. The hold_ratio "
                "summary feature does not mean the same thing in both. Re-export or "
                "restart the service to match."
            )
        if self.exercise_id is None:
            out.append(
                f"{self.path.name} is a POOLED artifact (no exercise_id). The "
                "per-exercise specialist models score better on every exercise."
            )
        reference = self.meta.get("loso_reference")
        if not reference:
            out.append(
                f"{self.path.name} carries no --loso-report, so nothing records an "
                "honest accuracy figure for it."
            )
        return out

    def applies_to(self, exercise_id: str) -> bool:
        """Whether this artifact may score the given exercise"""
        return self.exercise_id is None or self.exercise_id == exercise_id

    def info(self) -> dict:
        """What `/health` publishes, so a stored report stays interpretable"""
        reference = self.meta.get("loso_reference") or {}
        return {
            "path": str(self.path),
            "architecture": self.meta.get("architecture"),
            "exercise_id": self.exercise_id,
            "self_paced": self.self_paced,
            "corpus": self.meta.get("corpus"),
            "created_utc": self.meta.get("created_utc"),
            "git_sha": self.meta.get("git_sha"),
            "fitted_on_subjects": self.meta.get("dataset", {}).get("n_subjects"),
            "fitted_on_repetitions": self.meta.get("dataset", {}).get("n_samples"),
            "loso_balanced_accuracy_patients": reference.get(
                "balanced_accuracy_patients"
            ),
            "loso_report": reference.get("path"),
            "accuracy_note": self.meta.get("accuracy"),
            "validated_on_phone": False,
        }

    @torch.no_grad()
    def predict(
        self,
        window: AngleSeries,
        exercise: ExerciseDefinition,
        *,
        rule_eqi: float,
    ) -> dict:
        """Scoring one repetition window whose rule EQI is already known"""
        sequence = sequence_features(window, self.feature_config)
        summary = summary_features(window, exercise)

        outputs = self._model(
            torch.from_numpy(np.ascontiguousarray(sequence[None], dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(summary[None], dtype=np.float32)),
            torch.tensor([float(rule_eqi)], dtype=torch.float32),
        )
        correct_prob, error_prob, eqi = (t.detach().cpu().numpy() for t in outputs)

        model_eqi = float(eqi[0])
        return {
            "correct_prob": float(correct_prob[0]),
            "eqi": model_eqi,
            "eqi_rule": float(rule_eqi),
            "eqi_correction": model_eqi - float(rule_eqi),
            "errors": {
                slug: float(p) for slug, p in zip(ERROR_SLUGS, error_prob[0])
            },
            "note": (
                "Model output. The EQI here is the rule engine's EQI plus a bounded "
                "learned correction, never a score from scratch. Not validated on "
                "phone-captured pose."
            ),
        }


__all__ = ["ModelRunner"]
