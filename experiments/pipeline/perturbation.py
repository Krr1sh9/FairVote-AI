"""Misreporting and Randomized Response perturbation helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fairvote.privacy import privatize_many
from fairvote.simulation.bias_models import (
    apply_misreporting,
    build_shy_model_from_epsilon,
    make_shy_supporter_model,
)
from .config import ExperimentConfig
from .scenarios import EPSILON_DEPENDENT_MISREPORT_SCENARIOS, SHY_MISREPORT_SCENARIOS, VALID_SCENARIOS


@dataclass(frozen=True)
class PerturbedLabels:
    """Labels for one epsilon after pre-LDP misreporting and RR perturbation."""

    true_categories: np.ndarray
    stated_categories: np.ndarray
    reported_categories: np.ndarray
    misreport_model: Any | None


def apply_misreport_and_rr(
    *,
    config: ExperimentConfig,
    scenario: str,
    true_categories: np.ndarray,
    epsilon: float,
    rng: np.random.Generator,
) -> PerturbedLabels:
    """Apply scenario-specific misreporting and then k-ary RR."""
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    y_true = np.asarray(true_categories, dtype=int)
    mis = None
    if scenario in SHY_MISREPORT_SCENARIOS:
        if scenario in EPSILON_DEPENDENT_MISREPORT_SCENARIOS:
            mis = build_shy_model_from_epsilon(config.k, config.shy_category, epsilon)
        else:
            mis = make_shy_supporter_model(config.k, config.shy_category, honesty=config.shy_honesty)
        stated = apply_misreporting(y_true, mis, rng=rng)
    else:
        stated = y_true
    reported = privatize_many(stated, epsilon, config.k, rng=rng)
    return PerturbedLabels(
        true_categories=y_true,
        stated_categories=np.asarray(stated, dtype=int),
        reported_categories=np.asarray(reported, dtype=int),
        misreport_model=mis,
    )
