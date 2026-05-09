"""Population and poststratification context for experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from fairvote.inference.mrp import build_design_matrix
from fairvote.simulation.population import (
    make_realistic_uk_like_population,
    poststrat_table,
    subgroup_true_distribution,
)

from .config import ExperimentConfig
from .metrics import feature_masses_from_cells, truth_overall
from .scenarios import ScenarioInfo, apply_truth_scenario, scenario_info


@dataclass(frozen=True)
class ExperimentContext:
    """Precomputed population-level objects shared by all trials."""

    pop: Any
    truth_overall: np.ndarray
    truth_region: dict[str, np.ndarray]
    truth_age: dict[str, np.ndarray]
    by: list[str]
    cells: np.ndarray
    cell_counts: np.ndarray
    X_cells: np.ndarray
    region_masses: dict[str, float]
    age_masses: dict[str, float]
    include_features: tuple[str, str] = ("region", "age_group")
    scenario_info: ScenarioInfo | None = None


def build_context(config: ExperimentConfig, scenario: str | None = None) -> ExperimentContext:
    """Generate the synthetic population and all static poststratification structures."""
    pop = make_realistic_uk_like_population(config.population_n, config.k, seed=config.seed)
    info = scenario_info(scenario) if scenario is not None else None
    if scenario is not None:
        pop = apply_truth_scenario(pop, scenario, k=config.k, seed=config.seed)
    by = list(config.feature_order)
    cells, cell_counts, _level_names = poststrat_table(pop, by=by)
    cell_features = {feature: cells[:, j].astype(int) for j, feature in enumerate(by)}
    X_cells, _design_info_cells = build_design_matrix(
        cell_features,
        pop.feature_levels,
        feature_order=config.feature_order,
        intercept=True,
    )
    return ExperimentContext(
        pop=pop,
        truth_overall=truth_overall(pop, config.k),
        truth_region=subgroup_true_distribution(pop, "region"),
        truth_age=subgroup_true_distribution(pop, "age_group"),
        by=by,
        cells=cells,
        cell_counts=cell_counts.astype(float),
        X_cells=X_cells,
        region_masses=feature_masses_from_cells(
            cells=cells,
            counts=cell_counts.astype(float),
            by=by,
            feature="region",
            feature_levels=pop.feature_levels,
        ),
        age_masses=feature_masses_from_cells(
            cells=cells,
            counts=cell_counts.astype(float),
            by=by,
            feature="age_group",
            feature_levels=pop.feature_levels,
        ),
        scenario_info=info,
    )
