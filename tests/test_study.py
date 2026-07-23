from __future__ import annotations

import numpy as np
import pytest

from fairvote.simulation.population import default_population
from fairvote.study import METHODS, evaluate_poll


def test_evaluate_poll_returns_all_estimates_and_diagnostics() -> None:
    population = default_population()
    result = evaluate_poll(
        population,
        1000,
        1.0,
        "moderate",
        np.random.default_rng(5),
    )

    assert set(result.estimates) == set(METHODS)
    assert set(result.l1_errors) == set(METHODS)
    assert set(result.max_abs_errors) == set(METHODS)
    for estimate in result.estimates.values():
        assert estimate.shape == (population.n_categories,)
        assert np.all(np.isfinite(estimate))
        assert estimate.min() >= 0.0
        assert estimate.sum() == pytest.approx(1.0)

    assert result.sample_cell_counts.sum() == 1000
    assert result.sample_cell_proportions.sum() == pytest.approx(1.0)
    assert result.demographic_imbalance >= 0.0
    assert result.fallback_cells >= 0


def test_evaluate_poll_is_deterministic_for_the_same_seed() -> None:
    population = default_population()
    first = evaluate_poll(population, 500, 0.5, "strong", np.random.default_rng(22))
    second = evaluate_poll(population, 500, 0.5, "strong", np.random.default_rng(22))

    for method in METHODS:
        np.testing.assert_array_equal(first.estimates[method], second.estimates[method])
    np.testing.assert_array_equal(first.sample_cell_counts, second.sample_cell_counts)


@pytest.mark.parametrize(
    ("epsilon", "n_respondents", "bias"),
    [
        (4.0, 1000, "none"),
        (1.0, 5000, "none"),
        (1.0, 1000, "extreme"),
    ],
)
def test_evaluate_poll_rejects_unevaluated_settings(
    epsilon: float,
    n_respondents: int,
    bias: str,
) -> None:
    with pytest.raises(ValueError):
        evaluate_poll(
            default_population(),
            n_respondents,
            epsilon,
            bias,
            np.random.default_rng(0),
        )
