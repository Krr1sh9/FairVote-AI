from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairvote.experiment_grid import BIAS_CONDITIONS
from fairvote.privacy.mechanisms.kary_rr import privatize_many
from fairvote.simulation.population import Population

# These age-group multipliers modify the population cell weights before sampling and therefore control the simulated demographic sampling bias.
BIAS_MULTIPLIERS: dict[str, dict[str, float]] = {
    "none": {"18-34": 1.0, "35-54": 1.0, "55+": 1.0},
    "moderate": {"18-34": 1.8, "35-54": 1.0, "55+": 0.6},
    "strong": {"18-34": 3.5, "35-54": 1.0, "55+": 0.3},
}

# The supported bias labels are shared with the experiment-grid definition.
BIAS_LEVELS: tuple[str, ...] = BIAS_CONDITIONS


@dataclass(frozen=True)
class PollSample:
    # cell_indices records the demographic cell assigned to each sampled respondent.
    cell_indices: np.ndarray

    # reported_categories contains the privatised categories produced by Randomised Response rather than the original sampled categories.
    reported_categories: np.ndarray

    def __len__(self) -> int:
        # The sample length is the number of respondent-level cell assignments.
        return int(self.cell_indices.size)


def sampling_probabilities(population: Population, bias: str) -> np.ndarray:
    # Only the bias conditions with defined age-group multipliers can be used to construct sampling probabilities.
    if bias not in BIAS_MULTIPLIERS:
        raise ValueError(f"bias must be one of {BIAS_LEVELS}, got {bias!r}.")

    # Each demographic cell receives the multiplier associated with its age-group label.
    multipliers = np.array(
        [BIAS_MULTIPLIERS[bias][age] for age in population.age_groups],
        dtype=float,
    )

    # Multiplying the population weights by these factors shifts the sampling distribution while retaining the population weights as its baseline.
    unnormalized = population.weights * multipliers
    total = float(unnormalized.sum())
    if total <= 0.0:
        raise ValueError("Sampling probabilities are all zero for this bias level.")

    # Normalisation converts the adjusted cell weights into probabilities that sum to one.
    return unnormalized / total


def run_synthetic_poll(
    population: Population,
    n_respondents: int,
    epsilon: float,
    bias: str,
    rng: np.random.Generator,
) -> PollSample:
    # The poll requires a positive integer respondent count and explicitly rejects booleans.
    if not isinstance(n_respondents, int) or isinstance(n_respondents, bool):
        raise TypeError("n_respondents must be an int.")
    if n_respondents <= 0:
        raise ValueError("n_respondents must be > 0.")

    # Respondent cells are sampled from the bias-adjusted demographic distribution.
    probabilities = sampling_probabilities(population, bias)
    cell_indices = rng.choice(population.n_cells, size=n_respondents, p=probabilities)

    # Each respondent's original category is drawn from the preference distribution of the demographic cell that was sampled for them.
    uniform_draws = rng.random(n_respondents)
    cumulative_by_respondent = np.cumsum(population.preferences, axis=1)[cell_indices]
    true_categories = (uniform_draws[:, None] > cumulative_by_respondent).sum(axis=1)

    # The upper bound keeps every generated category index within the valid range for the population.
    true_categories = np.minimum(true_categories, population.n_categories - 1).astype(int)

    # Randomised Response is applied to the sampled original categories before the poll result is returned.
    reported_categories = privatize_many(
        true_categories,
        epsilon,
        population.n_categories,
        rng,
    )

    # The returned sample exposes demographic cell indices and privatised reports, but not the original respondent categories.
    return PollSample(
        cell_indices=np.asarray(cell_indices, dtype=int),
        reported_categories=reported_categories,
    )
