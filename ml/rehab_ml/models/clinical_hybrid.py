from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..features.extractor import N_SUMMARY_FEATURES
from .heads import HeadOutputs, PredictionHeads
from .sequence import ChannelNorm, MaskedMeanMaxPool, SequenceModelConfig, _TemporalBlock


@dataclass
class HybridConfig(SequenceModelConfig):
    """Hybrid-specific settings on top of the shared encoder config"""
    n_summary_features: int = N_SUMMARY_FEATURES
    clinical_hidden: int = 48
    # Largest EQI correction the residual head may apply, in EQI points
    max_eqi_correction: float = 25.0
    # False is to ablate the auxiliary clinical-feature head
    use_clinical_supervision: bool = True
    # False is to ablate the residual formulation and predict EQI directly
    use_residual_eqi: bool = True


class ClinicalHybridModel(nn.Module):
    """Sequence encoder fused with clinical summary features"""

    def __init__(self, config: HybridConfig | None = None) -> None:
        super().__init__()
        self.config = config or HybridConfig()
        cfg = self.config

        # Sequence branch
        self.sequence_norm = ChannelNorm(cfg.in_channels)
        self.input_proj = nn.Conv1d(cfg.in_channels, cfg.hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                _TemporalBlock(cfg.hidden, cfg.kernel_size, 2**i, cfg.dropout)
                for i in range(cfg.n_layers + 1)
            ]
        )
        self.pool = MaskedMeanMaxPool()
        sequence_dim = 2 * cfg.hidden

        # Clinical branch
        self.clinical_norm = nn.LayerNorm(cfg.n_summary_features, elementwise_affine=False)
        self.clinical_encoder = nn.Sequential(
            nn.Linear(cfg.n_summary_features, cfg.clinical_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.clinical_hidden, cfg.clinical_hidden),
            nn.GELU(),
        )

        # gated fusion
        fused_dim = sequence_dim + cfg.clinical_hidden
        self.gate = nn.Sequential(nn.Linear(fused_dim, fused_dim), nn.Sigmoid())
        self.dropout = nn.Dropout(cfg.dropout)
        self.heads = PredictionHeads(fused_dim)

        # auxiliary heads
        self.clinical_predictor = (
            nn.Linear(sequence_dim, cfg.n_summary_features)
            if cfg.use_clinical_supervision
            else None
        )
        self.eqi_residual = nn.Linear(fused_dim, 1) if cfg.use_residual_eqi else None

    def encode_sequence(self, sequence: torch.Tensor) -> torch.Tensor:
        x = self.sequence_norm(sequence).transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.pool(x.transpose(1, 2))

    def forward(
        self,
        sequence: torch.Tensor,
        summary: torch.Tensor | None = None,
        baseline_eqi: torch.Tensor | None = None,
        **_: object,
    ) -> HeadOutputs:
        sequence_embedding = self.encode_sequence(sequence)

        if summary is None:
            summary = sequence.new_zeros(
                (sequence.shape[0], self.config.n_summary_features)
            )
        clinical_embedding = self.clinical_encoder(self.clinical_norm(summary))

        fused = torch.cat([sequence_embedding, clinical_embedding], dim=-1)
        fused = fused * self.gate(fused)
        fused = self.dropout(fused)

        outputs = self.heads(fused)

        if self.clinical_predictor is not None:
            outputs.clinical_features = self.clinical_predictor(sequence_embedding)
            outputs.clinical_target = self.clinical_norm(summary).detach()

        if self.eqi_residual is not None and baseline_eqi is not None:
            correction = self.config.max_eqi_correction * torch.tanh(
                self.eqi_residual(fused).squeeze(-1)
            )
            outputs.eqi = (baseline_eqi.to(correction.dtype) + correction).clamp(0.0, 100.0)

        outputs.embedding = fused
        return outputs

    @torch.no_grad()
    def gate_activation(self, sequence: torch.Tensor, summary: torch.Tensor) -> dict[str, float]:
        """Mean gate value for the sequence and clinical halves"""
        sequence_embedding = self.encode_sequence(sequence)
        clinical_embedding = self.clinical_encoder(self.clinical_norm(summary))
        fused = torch.cat([sequence_embedding, clinical_embedding], dim=-1)
        gate = self.gate(fused)
        split = sequence_embedding.shape[-1]
        return {
            "sequence": float(gate[:, :split].mean()),
            "clinical": float(gate[:, split:].mean()),
        }
