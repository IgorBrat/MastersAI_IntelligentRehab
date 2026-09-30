"""Turning a trained model into a single deployable artifact"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from ..features.extractor import FeatureConfig, SUMMARY_FEATURE_NAMES
from ..taxonomy import ERROR_SLUGS

#: Sidecar schema version. Bump when a field the service reads changes meaning
ARTIFACT_FORMAT = "rehab-ml/model@1"


class DeployableModel(torch.nn.Module):
    """Wrapping a trained model so it returns plain tensors"""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        sequence: torch.Tensor,
        summary: torch.Tensor,
        baseline_eqi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.model(sequence, summary=summary, baseline_eqi=baseline_eqi)
        correct_prob, error_prob = outputs.probabilities()
        return correct_prob, error_prob, outputs.eqi


def _git_sha() -> str | None:
    """The commit the artifact was built from, when there is one"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = result.stdout.strip()
    return sha if result.returncode == 0 and sha else None


def example_inputs(
    feature_config: FeatureConfig, *, batch: int = 1
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Zero tensors of the exact shapes the artifact expects, for tracing"""
    return (
        torch.zeros(batch, feature_config.sequence_length, feature_config.n_sequence_channels),
        torch.zeros(batch, len(SUMMARY_FEATURE_NAMES)),
        torch.zeros(batch),
    )


def trace(model: torch.nn.Module, feature_config: FeatureConfig) -> torch.jit.ScriptModule:
    """Tracing a trained model into a frozen TorchScript module"""
    wrapped = DeployableModel(model).eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapped, example_inputs(feature_config), strict=False)
    return torch.jit.freeze(traced)


def sidecar_path(path: str | Path) -> Path:
    """The metadata file that sits beside a `.pt`"""
    return Path(path).with_suffix(".json")


def save(
    model: torch.nn.Module,
    path: str | Path,
    *,
    feature_config: FeatureConfig,
    provenance: dict,
) -> tuple[Path, Path]:
    """Writing the TorchScript artifact and its sidecar. Returns both paths"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    scripted = trace(model, feature_config)
    scripted.save(str(target))

    meta = {
        "format": ARTIFACT_FORMAT,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(),
        "inputs": {
            "sequence": [feature_config.sequence_length, feature_config.n_sequence_channels],
            "summary": [len(SUMMARY_FEATURE_NAMES)],
            "baseline_eqi": [],
        },
        "outputs": ["correct_prob", "error_prob", "eqi"],
        "feature_config": asdict(feature_config),
        "summary_feature_names": list(SUMMARY_FEATURE_NAMES),
        "error_slugs": list(ERROR_SLUGS),
        **provenance,
    }
    meta_path = sidecar_path(target)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return target, meta_path


def load(path: str | Path) -> tuple[torch.jit.ScriptModule, dict]:
    """Loading an artifact and its sidecar, refusing a mismatched pair"""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"no model artifact at {target}")

    meta_path = sidecar_path(target)
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"model artifact {target.name} has no sidecar at {meta_path.name}; "
            "it cannot be used without one -- re-export it"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    recorded = meta.get("format")
    if recorded != ARTIFACT_FORMAT:
        raise ValueError(
            f"{meta_path.name} is {recorded!r}, this build reads {ARTIFACT_FORMAT!r}"
        )


    if list(meta.get("error_slugs", [])) != list(ERROR_SLUGS):
        raise ValueError(
            f"{meta_path.name} was built against a different error taxonomy "
            f"({len(meta.get('error_slugs', []))} types, this build has {len(ERROR_SLUGS)}); "
            "re-export it"
        )

    model = torch.jit.load(str(target), map_location="cpu")
    model.eval()
    return model, meta


__all__ = [
    "ARTIFACT_FORMAT",
    "DeployableModel",
    "example_inputs",
    "load",
    "save",
    "sidecar_path",
    "trace",
]
