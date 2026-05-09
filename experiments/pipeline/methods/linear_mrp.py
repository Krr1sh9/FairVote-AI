"""Linear RR-aware MRP estimator runners."""
from __future__ import annotations

import time
import warnings
from typing import Dict

import numpy as np

from ..config import ExperimentConfig, MethodResult, TrialConfig
from ..context import ExperimentContext
from ..metrics import poststrat_from_cell_theta
from .common import TrialData, _fit_linear_model, _near_identity_epsilon, estimate_subgroup_mean_proba

def linear_rr_no_poststrat(
    config: ExperimentConfig,
    context: ExperimentContext,
    trial: TrialConfig,
    data: TrialData,
) -> MethodResult:
    """RR-aware linear model averaged over the respondent sample, without poststratification."""
    del context
    start = time.perf_counter()
    model = _fit_linear_model(
        config=config,
        trial=trial,
        X_train=data.X_train,
        y=data.perturbation.reported_categories,
        epsilon=trial.epsilon,
        seed_offset=899,
    )
    probs = model.predict_theta(data.X_train)
    overall = np.mean(probs, axis=0)
    overall = overall / max(float(np.sum(overall)), 1e-12)
    by_feature = {
        "region": estimate_subgroup_mean_proba(probs, values=data.region_values, levels=data.region_levels, k=config.k),
        "age_group": estimate_subgroup_mean_proba(probs, values=data.age_values, levels=data.age_levels, k=config.k),
    }
    return MethodResult(
        "linear_rr_no_poststrat",
        overall,
        by_feature,
        time.perf_counter() - start,
        {"poststratified": 0, "linear_final_loss": model.fit_diagnostics.final_loss if model.fit_diagnostics else None},
    )


def _poststratify_model_predictions(
    *,
    context: ExperimentContext,
    cell_theta: np.ndarray,
) -> tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    return poststrat_from_cell_theta(
        cell_theta,
        context.cells,
        context.cell_counts.astype(float),
        by=context.by,
        feature_levels=context.pop.feature_levels,
        include_features=context.include_features,
    )


def mrp_rr_poststrat(
    config: ExperimentConfig,
    context: ExperimentContext,
    trial: TrialConfig,
    data: TrialData,
) -> MethodResult:
    """Linear RR-aware MRP followed by poststratification."""
    start = time.perf_counter()
    model = _fit_linear_model(
        config=config,
        trial=trial,
        X_train=data.X_train,
        y=data.perturbation.reported_categories,
        epsilon=trial.epsilon,
        seed_offset=999,
    )
    overall, by_feature = _poststratify_model_predictions(context=context, cell_theta=model.predict_theta(context.X_cells))
    return MethodResult(
        "mrp_rr_poststrat",
        overall,
        by_feature,
        time.perf_counter() - start,
        {"poststratified": 1, "linear_final_loss": model.fit_diagnostics.final_loss if model.fit_diagnostics else None},
    )

def oracle_true_linear_mrp_poststrat(
    config: ExperimentConfig,
    context: ExperimentContext,
    trial: TrialConfig,
    data: TrialData,
) -> MethodResult:
    """Oracle linear poststratification fitted to synthetic true labels, not RR reports."""
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = _fit_linear_model(
            config=config,
            trial=trial,
            X_train=data.X_train,
            y=data.perturbation.true_categories,
            epsilon=_near_identity_epsilon(),
            seed_offset=1499,
        )
    overall, by_feature = _poststratify_model_predictions(context=context, cell_theta=model.predict_theta(context.X_cells))
    return MethodResult(
        "oracle_true_linear_mrp_poststrat",
        overall,
        by_feature,
        time.perf_counter() - start,
        {
            "oracle_uses_true_labels": 1,
            "poststratified": 1,
            "linear_final_loss": model.fit_diagnostics.final_loss if model.fit_diagnostics else None,
        },
    )
