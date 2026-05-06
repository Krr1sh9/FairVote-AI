"""Diagnostics containers for MRP-style iterative fits."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class FitDiagnostics:
    """Convergence summary for an iterative MRP fit."""

    steps: int
    final_loss: float
    runtime_sec: float
    history: np.ndarray | None = None

    def to_jsonable(self) -> dict:
        data = asdict(self)
        if self.history is not None:
            data["history"] = [float(x) for x in np.asarray(self.history, dtype=float).tolist()]
        return data


# Backwards-compatible name used by earlier modules/tests.
FitInfo = FitDiagnostics
