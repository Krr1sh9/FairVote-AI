"""Data types and fit diagnostics for neural RR-MRP."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

ArrayLike = np.ndarray | Sequence[Sequence[float]]


def _array_to_list_or_none(values: np.ndarray | None) -> list[float] | None:
    if values is None:
        return None
    return [float(v) for v in np.asarray(values, dtype=float).reshape(-1)]


@dataclass(frozen=True)
class RRNeuralMRPFitInfo:
    """Diagnostics returned by :meth:`RRNeuralMRPModel.fit`.

    ``steps`` is the number of optimisation steps actually completed. It equals
    the requested number unless early stopping is triggered. ``history`` is kept
    as the backwards-compatible name for full training loss history; it is only
    populated when ``keep_history=True``. Validation history is recorded whenever
    a validation set or validation split is provided.
    """

    steps: int
    final_loss: float
    history: np.ndarray | None = None
    validation_loss: float | None = None
    validation_history: np.ndarray | None = None
    best_validation_loss: float | None = None
    best_step: int | None = None
    early_stopped: bool = False
    runtime_sec: float = 0.0
    device: str = "cpu"
    checkpoint_path: str | None = None

    @property
    def train_history(self) -> np.ndarray | None:
        """Alias for the training loss history."""
        return self.history

    def to_dict(self, *, include_history: bool = False) -> dict[str, Any]:
        """Return JSON-serialisable diagnostics for reports or manifests."""
        out: dict[str, Any] = {
            "steps": int(self.steps),
            "final_loss": float(self.final_loss),
            "validation_loss": None if self.validation_loss is None else float(self.validation_loss),
            "best_validation_loss": None if self.best_validation_loss is None else float(self.best_validation_loss),
            "best_step": None if self.best_step is None else int(self.best_step),
            "early_stopped": bool(self.early_stopped),
            "runtime_sec": float(self.runtime_sec),
            "device": self.device,
            "checkpoint_path": self.checkpoint_path,
        }
        if include_history:
            out["history"] = _array_to_list_or_none(self.history)
            out["validation_history"] = _array_to_list_or_none(self.validation_history)
        else:
            out["history_length"] = 0 if self.history is None else int(np.asarray(self.history).size)
            out["validation_history_length"] = (
                0 if self.validation_history is None else int(np.asarray(self.validation_history).size)
            )
        return out


__all__ = ["ArrayLike", "RRNeuralMRPFitInfo"]
