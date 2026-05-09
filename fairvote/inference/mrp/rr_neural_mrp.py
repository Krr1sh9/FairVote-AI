"""Backwards-compatible facade for neural RR-aware MRP.

Implementation is split under :mod:`fairvote.inference.mrp.neural` so the old
single 800-line module no longer carries training, diagnostics and ensemble
logic in one place. Existing imports continue to work.
"""
from __future__ import annotations

from .neural import (
    NeuralRRMRPFitInfo,
    NeuralRRMRPModel,
    RRNeuralMRPEnsemble,
    RRNeuralMRPFitInfo,
    RRNeuralMRPModel,
    fit_rr_neural_mrp_ensemble,
)

__all__ = [
    "RRNeuralMRPFitInfo",
    "RRNeuralMRPModel",
    "RRNeuralMRPEnsemble",
    "fit_rr_neural_mrp_ensemble",
    "NeuralRRMRPFitInfo",
    "NeuralRRMRPModel",
]
