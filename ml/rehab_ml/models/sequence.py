from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..features.angles import N_ANGLE_CHANNELS
from .heads import HeadOutputs, PredictionHeads


@dataclass
class SequenceModelConfig:
    """Shared architecture settings"""
    in_channels: int = 2 * N_ANGLE_CHANNELS
    hidden: int = 64
    n_layers: int = 2
    dropout: float = 0.3
    # TCN only
    kernel_size: int = 5


class ChannelNorm(nn.Module):
    """Per-channel standardisation of the input sequence"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class MaskedMeanMaxPool(nn.Module):
    """Concatenated mean and max pooling over time"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return torch.cat([x.mean(dim=1), x.amax(dim=1)], dim=-1)


class LSTMClassifier(nn.Module):
    """Bidirectional LSTM encoder"""
    def __init__(self, config: SequenceModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or SequenceModelConfig()
        cfg = self.config
        self.norm = ChannelNorm(cfg.in_channels)
        self.lstm = nn.LSTM(
            input_size=cfg.in_channels,
            hidden_size=cfg.hidden,
            num_layers=cfg.n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=cfg.dropout if cfg.n_layers > 1 else 0.0,
        )
        self.pool = MaskedMeanMaxPool()
        self.dropout = nn.Dropout(cfg.dropout)
        self.heads = PredictionHeads(4 * cfg.hidden)

    def forward(self, sequence: torch.Tensor, **_: object) -> HeadOutputs:
        x = self.norm(sequence)
        x, _ = self.lstm(x)
        return self.heads(self.dropout(self.pool(x)))


class _TemporalBlock(nn.Module):
    """Dilated causal-padded residual block"""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dropout(self.act(self.norm1(self.conv1(x))))
        x = self.dropout(self.act(self.norm2(self.conv2(x))))
        return x + residual


class TCNClassifier(nn.Module):
    """Temporal convolutional encoder with exponentially dilated blocks"""

    def __init__(self, config: SequenceModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or SequenceModelConfig()
        cfg = self.config
        self.norm = ChannelNorm(cfg.in_channels)
        self.input_proj = nn.Conv1d(cfg.in_channels, cfg.hidden, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                _TemporalBlock(cfg.hidden, cfg.kernel_size, 2**i, cfg.dropout)
                for i in range(cfg.n_layers + 1)
            ]
        )
        self.pool = MaskedMeanMaxPool()
        self.dropout = nn.Dropout(cfg.dropout)
        self.heads = PredictionHeads(2 * cfg.hidden)

    def forward(self, sequence: torch.Tensor, **_: object) -> HeadOutputs:
        x = self.norm(sequence).transpose(1, 2)  # (B, C, T)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)  # (B, T, D)
        return self.heads(self.dropout(self.pool(x)))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
