"""Hierarchical/partial-pooling RR-aware MRP estimator runner."""

from __future__ import annotations

import time

import numpy as np

from fairvote.inference.mrp import HierarchicalRRMRPModel

from ..config import ExperimentConfig, MethodResult, TrialConfig
from ..context import ExperimentContext
from .common import TrialData
from .linear_mrp import _poststratify_model_predictions


def _cell_feature_values(context: ExperimentContext, feature_order: list[str]) -> dict[str, np.ndarray]:
    """Return post-stratification cell columns keyed by feature name."""
    return {feature: context.cells[:, idx].astype(int) for idx, feature in enumerate(feature_order)}


def hierarchical_rr_mrp_poststrat(
    config: ExperimentConfig,
    context: ExperimentContext,
    trial: TrialConfig,
    data: TrialData,
) -> MethodResult:
    """Hierarchical/partial-pooling RR-aware MRP followed by post-stratification."""
    start = time.perf_counter()
    model = HierarchicalRRMRPModel(
        config.k,
        global_l2=config.hierarchical_global_l2,
        effect_l2=config.hierarchical_effect_l2,
        seed=config.seed + 1799 + trial.trial,
    )
    fit_info = model.fit(
        data.feature_values,
        data.perturbation.reported_categories,
        context.pop.feature_levels,
        trial.epsilon,
        feature_order=config.feature_order,
        lr=config.mrp_lr,
        steps=config.mrp_steps,
        batch_size=config.mrp_batch_size,
        verbose_every=config.verbose_every,
    )
    cell_features = _cell_feature_values(context, config.feature_order)
    overall, by_feature = _poststratify_model_predictions(
        context=context,
        cell_theta=model.predict_theta_from_features(cell_features),
    )
    return MethodResult(
        "hierarchical_rr_mrp_poststrat",
        overall,
        by_feature,
        time.perf_counter() - start,
        {
            "poststratified": 1,
            "partial_pooling": 1,
            "hierarchical_final_loss": fit_info.final_loss,
            "hierarchical_effect_l2": config.hierarchical_effect_l2,
        },
    )
