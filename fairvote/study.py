from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairvote.experiment_grid import BIAS_CONDITIONS, FULL_EPSILONS, FULL_SAMPLE_SIZES
from fairvote.metrics import l1_error, max_absolute_error
from fairvote.poststratification import poststratified_estimate
from fairvote.privacy.estimators import debiased_estimate, raw_frequencies
from fairvote.simulation.population import Population
from fairvote.simulation.sampling import run_synthetic_poll

# The shared study workflow accepts the same epsilon, sample-size and bias values defined by the full experiment grid.
SUPPORTED_EPSILONS: tuple[float, ...] = FULL_EPSILONS
SUPPORTED_SAMPLE_SIZES: tuple[int, ...] = FULL_SAMPLE_SIZES
SUPPORTED_BIAS_LEVELS: tuple[str, ...] = BIAS_CONDITIONS

# These identifiers define the three estimators compared throughout the study.
METHODS: tuple[str, ...] = ("raw_frequencies", "rr_debiased", "poststratified")

# Human-readable labels are kept separately from the method identifiers used in results and model code.
METHOD_LABELS: dict[str, str] = {
    "raw_frequencies": "Raw privatised reports",
    "rr_debiased": "Overall RR correction",
    "poststratified": "Poststratified RR estimate",
}


@dataclass(frozen=True)
class StudyResult:
    # The known synthetic population distribution is used as the common truth for estimator evaluation.
    truth: np.ndarray

    # estimates stores the three population-level estimates produced from the same synthetic poll.
    estimates: dict[str, np.ndarray]

    # These dictionaries store the two error measures for each estimator against the known truth.
    l1_errors: dict[str, float]
    max_abs_errors: dict[str, float]

    # These arrays describe the realised demographic composition of the sampled respondents.
    sample_cell_counts: np.ndarray
    sample_cell_proportions: np.ndarray

    # Demographic imbalance is the L1 distance between realised sample proportions and population weights.
    demographic_imbalance: float

    # fallback_cells records how many demographic cells required the poststratification fallback.
    fallback_cells: int


def evaluate_poll(
    population: Population,
    n_respondents: int,
    epsilon: float,
    bias: str,
    rng: np.random.Generator,
) -> StudyResult:
    # The shared workflow is restricted to settings that belong to the full experiment design.
    _validate_supported_settings(epsilon, n_respondents, bias)

    # One synthetic poll is sampled and its responses are passed through the Randomised Response mechanism.
    sample = run_synthetic_poll(population, n_respondents, epsilon, bias, rng)
    k = population.n_categories

    # The poststratified estimator uses the realised cell membership, privatised reports and known population cell weights.
    poststratified = poststratified_estimate(
        sample.cell_indices,
        sample.reported_categories,
        population.weights,
        epsilon,
        k,
    )

    # The population specification provides the exact synthetic truth used to score all three estimators.
    truth = population.true_distribution()

    # All three estimates are calculated from the same privatised synthetic poll.
    estimates = {
        "raw_frequencies": raw_frequencies(sample.reported_categories, k),
        "rr_debiased": debiased_estimate(sample.reported_categories, epsilon, k),
        "poststratified": poststratified.estimate,
    }

    # Each estimator is compared with the same truth using L1 error and maximum absolute category error.
    l1_errors = {method: l1_error(estimate, truth) for method, estimate in estimates.items()}
    max_abs_errors = {method: max_absolute_error(estimate, truth) for method, estimate in estimates.items()}

    # Cell counts and proportions record the realised demographic composition of this particular sample.
    cell_counts = np.bincount(sample.cell_indices, minlength=population.n_cells).astype(int)
    cell_proportions = cell_counts.astype(float) / float(n_respondents)

    # Demographic imbalance measures the total absolute difference between sample cell proportions and the known population weights.
    demographic_imbalance = float(np.abs(cell_proportions - population.weights).sum())

    # The returned result contains the estimator outputs, errors and demographic diagnostics needed by experiments, the app and the selector.
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
    # Each setting is checked against the values defined by the full experiment grid.
    if epsilon not in SUPPORTED_EPSILONS:
        raise ValueError(f"epsilon must be one of {SUPPORTED_EPSILONS}, got {epsilon!r}.")
    if n_respondents not in SUPPORTED_SAMPLE_SIZES:
        raise ValueError(f"n_respondents must be one of {SUPPORTED_SAMPLE_SIZES}, got {n_respondents!r}.")
    if bias not in SUPPORTED_BIAS_LEVELS:
        raise ValueError(f"bias must be one of {SUPPORTED_BIAS_LEVELS}, got {bias!r}.")
