from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairvote.experiment_grid import BIAS_CONDITIONS, FULL_EPSILONS, FULL_SAMPLE_SIZES
from fairvote.metrics import l1_error, max_absolute_error
from fairvote.poststratification import poststratified_estimate
from fairvote.privacy.estimators import debiased_estimate, raw_frequencies
from fairvote.simulation.population import Population
from fairvote.simulation.sampling import run_synthetic_poll

SUPPORTED_EPSILONS: tuple[float, ...] = FULL_EPSILONS
SUPPORTED_SAMPLE_SIZES: tuple[int, ...] = FULL_SAMPLE_SIZES
SUPPORTED_BIAS_LEVELS: tuple[str, ...] = BIAS_CONDITIONS

METHODS: tuple[str, ...] = ("raw_frequencies", "rr_debiased", "poststratified")
METHOD_LABELS: dict[str, str] = {
    "raw_frequencies": "Raw privatised reports",
    "rr_debiased": "Overall RR correction",
    "poststratified": "Poststratified RR estimate",
}


@dataclass(frozen=True)
class StudyResult:
    truth: np.ndarray
    estimates: dict[str, np.ndarray]
    l1_errors: dict[str, float]
    max_abs_errors: dict[str, float]
    sample_cell_counts: np.ndarray
    sample_cell_proportions: np.ndarray
    demographic_imbalance: float
    fallback_cells: int


def evaluate_poll(
    population: Population,
    n_respondents: int,
    epsilon: float,
    bias: str,
    rng: np.random.Generator,
) -> StudyResult:
    _validate_supported_settings(epsilon, n_respondents, bias)

    sample = run_synthetic_poll(population, n_respondents, epsilon, bias, rng)
    k = population.n_categories
    poststratified = poststratified_estimate(
        sample.cell_indices,
        sample.reported_categories,
        population.weights,
        epsilon,
        k,
    )

    truth = population.true_distribution()
    estimates = {
        "raw_frequencies": raw_frequencies(sample.reported_categories, k),
        "rr_debiased": debiased_estimate(sample.reported_categories, epsilon, k),
        "poststratified": poststratified.estimate,
    }
    l1_errors = {method: l1_error(estimate, truth) for method, estimate in estimates.items()}
    max_abs_errors = {method: max_absolute_error(estimate, truth) for method, estimate in estimates.items()}

    cell_counts = np.bincount(sample.cell_indices, minlength=population.n_cells).astype(int)
    cell_proportions = cell_counts.astype(float) / float(n_respondents)
    demographic_imbalance = float(np.abs(cell_proportions - population.weights).sum())

    return StudyResult(
        truth=truth,
        estimates=estimates,
        l1_errors=l1_errors,
        max_abs_errors=max_abs_errors,
        sample_cell_counts=cell_counts,
        sample_cell_proportions=cell_proportions,
        demographic_imbalance=demographic_imbalance,
        fallback_cells=poststratified.fallback_cells,
    )


def _validate_supported_settings(epsilon: float, n_respondents: int, bias: str) -> None:
    if epsilon not in SUPPORTED_EPSILONS:
        raise ValueError(f"epsilon must be one of {SUPPORTED_EPSILONS}, got {epsilon!r}.")
    if n_respondents not in SUPPORTED_SAMPLE_SIZES:
        raise ValueError(f"n_respondents must be one of {SUPPORTED_SAMPLE_SIZES}, got {n_respondents!r}.")
    if bias not in SUPPORTED_BIAS_LEVELS:
        raise ValueError(f"bias must be one of {SUPPORTED_BIAS_LEVELS}, got {bias!r}.")
