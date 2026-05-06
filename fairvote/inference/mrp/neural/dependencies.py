"""Optional PyTorch dependency loader for neural MRP."""
from __future__ import annotations

try:  # Keep the import error explicit for installations missing the ML dependency.
    import torch
    from torch import nn
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only when torch is absent
    raise ModuleNotFoundError(
        'RRNeuralMRPModel requires PyTorch. Install with `pip install -e ".[neural]"` '
        "or another compatible torch installation before using "
        "fairvote.inference.mrp.rr_neural_mrp."
    ) from exc

__all__ = ["torch", "nn"]
