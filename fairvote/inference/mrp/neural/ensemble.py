"""Seed ensembles for neural RR-MRP uncertainty checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from .api import RRNeuralMRPModel
from .types import ArrayLike, RRNeuralMRPFitInfo

@dataclass(frozen=True)
class RRNeuralMRPEnsemble:
    """Small multi-seed ensemble wrapper for neural uncertainty checks."""

    models: tuple[RRNeuralMRPModel, ...]
    fit_infos: tuple[RRNeuralMRPFitInfo, ...]

    def predict_true_proba(self, X: ArrayLike) -> np.ndarray:
        preds = np.stack([model.predict_true_proba(X) for model in self.models], axis=0)
        return np.mean(preds, axis=0)

    def predict_true_proba_std(self, X: ArrayLike) -> np.ndarray:
        preds = np.stack([model.predict_true_proba(X) for model in self.models], axis=0)
        return np.std(preds, axis=0)

    def poststratify(self, X_pop: ArrayLike, weights: Sequence[float]) -> np.ndarray:
        preds = np.stack([model.poststratify(X_pop, weights) for model in self.models], axis=0)
        return RRNeuralMRPModel._validate_probability_vector(np.mean(preds, axis=0), name="ensemble_poststratify")

    def uncertainty_summary(self, X: ArrayLike) -> dict[str, float]:
        std = self.predict_true_proba_std(X)
        return {
            "mean_seed_std": float(np.mean(std)),
            "max_seed_std": float(np.max(std)),
        }


def fit_rr_neural_mrp_ensemble(
    X: ArrayLike,
    y_reported: Sequence[int],
    *,
    k: int,
    epsilon: float,
    seeds: Sequence[int],
    model_kwargs: Optional[dict[str, Any]] = None,
    fit_kwargs: Optional[dict[str, Any]] = None,
) -> RRNeuralMRPEnsemble:
    """Fit a small multi-seed ensemble for uncertainty sensitivity checks."""
    if not seeds:
        raise ValueError("seeds must contain at least one seed")
    model_kwargs = dict(model_kwargs or {})
    fit_kwargs = dict(fit_kwargs or {})
    models: list[RRNeuralMRPModel] = []
    infos: list[RRNeuralMRPFitInfo] = []
    for seed in seeds:
        model = RRNeuralMRPModel(k=k, epsilon=epsilon, seed=int(seed), **model_kwargs)
        info = model.fit(X, y_reported, **fit_kwargs)
        models.append(model)
        infos.append(info)
    return RRNeuralMRPEnsemble(tuple(models), tuple(infos))


__all__ = ["RRNeuralMRPEnsemble", "fit_rr_neural_mrp_ensemble"]
