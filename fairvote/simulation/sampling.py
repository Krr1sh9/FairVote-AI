from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairvote.experiment_grid import BIAS_CONDITIONS
from fairvote.privacy.mechanisms.kary_rr import privatize_many
from fairvote.simulation.population import Population

BIAS_MULTIPLIERS: dict[str, dict[str, float]] = {
    "none": {"18-34": 1.0, "35-54": 1.0, "55+": 1.0},
    "moderate": {"18-34": 1.8, "35-54": 1.0, "55+": 0.6},
    "strong": {"18-34": 3.5, "35-54": 1.0, "55+": 0.3},
}

BIAS_LEVELS: tuple[str, ...] = BIAS_CONDITIONS


@dataclass(frozen=True)
class PollSample:
    cell_indices: np.ndarray
    reported_categories: np.ndarray

    def __len__(self) -> int:
        return int(self.cell_indices.size)


def sampling_probabilities(population: Population, bias: str) -> np.ndarray:
    if bias not in BIAS_MULTIPLIERS:
        raise ValueError(f"bias must be one of {BIAS_LEVELS}, got {bias!r}.")

    multipliers = np.array(
        [BIAS_MULTIPLIERS[bias][age] for age in population.age_groups],
        dtype=float,
    )
    unnormalized = population.weights * multipliers
    total = float(unnormalized.sum())
    if total <= 0.0:
        raise ValueError("Sampling probabilities are all zero for this bias level.")
    return unnormalized / total


def run_synthetic_poll(
    population: Population,
    n_respondents: int,
    epsilon: float,
    bias: str,
    rng: np.random.Generator,
) -> PollSample:
    if not isinstance(n_respondents, int) or isinstance(n_respondents, bool):
        raise TypeError("n_respondents must be an int.")
    if n_respondents <= 0:
        raise ValueError("n_respondents must be > 0.")

    probabilities = sampling_probabilities(population, bias)
    cell_indices = rng.choice(population.n_cells, size=n_respondents, p=probabilities)

    uniform_draws = rng.random(n_respondents)
    cumulative_by_respondent = np.cumsum(population.preferences, axis=1)[cell_indices]
    true_categories = (uniform_draws[:, None] > cumulative_by_respondent).sum(axis=1)
    true_categories = np.minimum(true_categories, population.n_categories - 1).astype(int)

    reported_categories = privatize_many(
        true_categories,
        epsilon,
        population.n_categories,
        rng,
    )

    return PollSample(
        cell_indices=np.asarray(cell_indices, dtype=int),
        reported_categories=reported_categories,
    )
