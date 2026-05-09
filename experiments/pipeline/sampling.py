"""Sampling-stage helpers for the MRP-vs-baselines pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np

from fairvote.simulation.bias_models import apply_nonresponse, make_default_feature_nonresponse_profile
from fairvote.simulation.sampling import biased_frame_sample, simple_random_sample, stratified_sample

from .config import ExperimentConfig
from .scenarios import NONRESPONSE_SCENARIOS, VALID_SCENARIOS


def draw_sample(config: ExperimentConfig, pop: Any, *, rng: np.random.Generator) -> Any:
    """Draw the requested sample frame before scenario-specific nonresponse."""
    if config.sampling == "srs":
        return simple_random_sample(pop, config.n_sample, rng=rng, replace=False)
    if config.sampling == "stratified":
        return stratified_sample(
            pop,
            config.n_sample,
            strata=config.strata,
            rng=rng,
            allocation=config.allocation,
            min_per_stratum=config.min_per_stratum,
            replace_within=False,
        )
    if config.sampling == "biased":
        return biased_frame_sample(
            pop,
            config.n_sample,
            rng=rng,
            feature=config.biased_feature,
            level_multipliers=config.biased_multipliers if config.biased_multipliers else None,
            replace=False,
        )
    raise ValueError("sampling must be one of: srs, stratified, biased")


def apply_scenario_nonresponse(sample: Any, pop: Any, scenario: str, *, rng: np.random.Generator) -> Any:
    """Apply the nonresponse part of the selected scenario."""
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if scenario in NONRESPONSE_SCENARIOS:
        return apply_nonresponse(sample, pop, rng=rng, feature_profile=make_default_feature_nonresponse_profile())
    return sample
