"""Common data structures and helpers for experiment estimator runners."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np

from fairvote.inference.mrp import RRMultinomialModel, build_design_matrix
from ..config import ExperimentConfig, TrialConfig
from ..context import ExperimentContext
from ..perturbation import PerturbedLabels

@dataclass(frozen=True)
class TrialData:
    """Data derived from one sampled respondent set and one epsilon."""

    sample: Any
    perturbation: PerturbedLabels
    X_train: np.ndarray
    region_values: np.ndarray
    age_values: np.ndarray
    region_levels: list[str]
    age_levels: list[str]
    feature_values: dict[str, np.ndarray]
    n_effective: int


def build_trial_data(
    *,
    config: ExperimentConfig,
    context: ExperimentContext,
    sample: Any,
    perturbation: PerturbedLabels,
) -> TrialData:
    """Encode sampled features once and reuse them across estimator runners."""
    train_features = {f: sample.features[f].astype(int) for f in config.feature_order}
    X_train, _design_info_train = build_design_matrix(
        train_features,
        context.pop.feature_levels,
        feature_order=config.feature_order,
        intercept=True,
    )
    return TrialData(
        sample=sample,
        perturbation=perturbation,
        X_train=X_train,
        region_values=sample.features["region"].astype(int),
        age_values=sample.features["age_group"].astype(int),
        region_levels=list(context.pop.feature_levels["region"]),
        age_levels=list(context.pop.feature_levels["age_group"]),
        feature_values={f: sample.features[f].astype(int) for f in config.feature_order},
        n_effective=int(sample.idx.size),
    )


def estimate_subgroup_distribution(
    labels: np.ndarray,
    *,
    values: np.ndarray,
    levels: list[str],
    k: int,
    estimator: Callable[[np.ndarray], np.ndarray],
) -> Dict[str, np.ndarray]:
    """Apply an estimator to each observed level of a demographic feature."""
    out: Dict[str, np.ndarray] = {}
    for level_idx, level_name in enumerate(levels):
        mask = values == level_idx
        out[level_name] = estimator(labels[mask]) if np.any(mask) else np.full(k, 1.0 / k, dtype=float)
    return out


def estimate_subgroup_mean_proba(
    probs: np.ndarray,
    *,
    values: np.ndarray,
    levels: list[str],
    k: int,
) -> Dict[str, np.ndarray]:
    """Average row-level probability predictions within observed sample subgroups."""
    out: Dict[str, np.ndarray] = {}
    for level_idx, level_name in enumerate(levels):
        mask = values == level_idx
        if np.any(mask):
            pred = np.mean(probs[mask], axis=0)
            total = float(np.sum(pred))
            out[level_name] = pred / total if total > 0 else np.full(k, 1.0 / k, dtype=float)
        else:
            out[level_name] = np.full(k, 1.0 / k, dtype=float)
    return out


def _fit_linear_model(
    *,
    config: ExperimentConfig,
    trial: TrialConfig,
    X_train: np.ndarray,
    y: np.ndarray,
    epsilon: float,
    seed_offset: int,
) -> RRMultinomialModel:
    model = RRMultinomialModel(config.k, l2=config.mrp_l2, seed=config.seed + seed_offset + trial.trial)
    model.fit(
        X_train,
        y,
        epsilon,
        lr=config.mrp_lr,
        steps=config.mrp_steps,
        batch_size=config.mrp_batch_size,
        verbose_every=config.verbose_every,
    )
    return model


def _near_identity_epsilon() -> float:
    # Large enough to make the RR matrix effectively the identity in double
    # precision while still being finite. Used only for oracle/naive ablations.
    return 50.0
