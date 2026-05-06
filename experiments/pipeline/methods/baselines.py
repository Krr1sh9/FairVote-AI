"""Simple oracle, raw, and analytical RR-debias estimator runners."""
from __future__ import annotations

import time
import numpy as np

from fairvote.privacy import estimate_distribution
from ..config import ExperimentConfig, MethodResult, TrialConfig
from ..context import ExperimentContext
from ..metrics import distribution_from_labels
from .common import TrialData, estimate_subgroup_distribution

def oracle_true_sample_distribution(
    config: ExperimentConfig,
    context: ExperimentContext,
    trial: TrialConfig,
    data: TrialData,
) -> MethodResult:
    """Oracle sample aggregate using synthetic true labels before RR/noise."""
    del context, trial
    start = time.perf_counter()
    y_true = data.perturbation.true_categories
    overall = distribution_from_labels(y_true, config.k)
    by_feature = {
        "region": estimate_subgroup_distribution(
            y_true, values=data.region_values, levels=data.region_levels, k=config.k, estimator=lambda y: distribution_from_labels(y, config.k)
        ),
        "age_group": estimate_subgroup_distribution(
            y_true, values=data.age_values, levels=data.age_levels, k=config.k, estimator=lambda y: distribution_from_labels(y, config.k)
        ),
    }
    return MethodResult(
        "oracle_true_sample_distribution",
        overall,
        by_feature,
        time.perf_counter() - start,
        {"oracle_uses_true_labels": 1, "poststratified": 0},
    )


def raw_reported_distribution(
    config: ExperimentConfig,
    context: ExperimentContext,
    trial: TrialConfig,
    data: TrialData,
) -> MethodResult:
    """Naive reported-label distribution with no RR correction."""
    del context, trial
    start = time.perf_counter()
    reported = data.perturbation.reported_categories
    overall = distribution_from_labels(reported, config.k)
    by_feature = {
        "region": estimate_subgroup_distribution(
            reported,
            values=data.region_values,
            levels=data.region_levels,
            k=config.k,
            estimator=lambda y: distribution_from_labels(y, config.k),
        ),
        "age_group": estimate_subgroup_distribution(
            reported,
            values=data.age_values,
            levels=data.age_levels,
            k=config.k,
            estimator=lambda y: distribution_from_labels(y, config.k),
        ),
    }
    return MethodResult("raw_reported_distribution", overall, by_feature, time.perf_counter() - start)


def baseline_rr_debias(
    config: ExperimentConfig,
    context: ExperimentContext,
    trial: TrialConfig,
    data: TrialData,
) -> MethodResult:
    """Analytical RR debiasing overall and within respondent sample slices."""
    del context
    start = time.perf_counter()
    reported = data.perturbation.reported_categories
    estimator = lambda y: estimate_distribution(y, trial.epsilon, config.k)
    overall = estimator(reported)
    by_feature = {
        "region": estimate_subgroup_distribution(
            reported, values=data.region_values, levels=data.region_levels, k=config.k, estimator=estimator
        ),
        "age_group": estimate_subgroup_distribution(
            reported, values=data.age_values, levels=data.age_levels, k=config.k, estimator=estimator
        ),
    }
    return MethodResult("baseline_rr_debias", overall, by_feature, time.perf_counter() - start)
