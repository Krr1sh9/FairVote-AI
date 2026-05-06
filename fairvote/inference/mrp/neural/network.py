"""Network architecture for neural RR-MRP."""
from __future__ import annotations

from typing import Tuple

from .dependencies import nn, torch

class _MLP(nn.Module):
    """Small configurable MLP that outputs unnormalised true-category logits."""

    def __init__(self, input_dim: int, output_dim: int, hidden_layers: Tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(input_dim)
        for width in hidden_layers:
            layers.append(nn.Linear(prev, int(width)))
            layers.append(nn.ReLU())
            if dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = int(width)
        layers.append(nn.Linear(prev, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

__all__ = ["_MLP"]
