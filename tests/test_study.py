from __future__ import annotations

import numpy as np
import pytest

from fairvote.metrics import l1_error, max_absolute_error
from fairvote.simulation.population import default_population
from fairvote.study import METHODS, evaluate_poll


def test_evaluate_poll_integrates_all_three_estimators_and_metrics() -> None:
    population = default_population()
    result = evaluate_poll(
        population,
        1000,
        1.0,
        "moderate",
        np.random.default_rng(5),
    )

    np.testing.assert_allclose(result.truth, population.true_distribution())
    assert set(result.estimates) == set(METHODS)
    assert set(result.l1_errors) == set(METHODS)
    assert set(result.max_abs_errors) == set(METHODS)

    for method in METHODS:
        estimate = result.estimates[method]
        assert estimate.shape == (population.n_categories,)
        assert np.all(np.isfinite(estimate))
        assert estimate.min() >= 0.0
        assert estimate.sum() == pytest.approx(1.0)
        assert result.l1_errors[method] == pytest.approx(l1_error(estimate, result.truth))
        assert result.max_abs_errors[method] == pytest.approx(max_absolute_error(estimate, result.truth))

    assert result.sample_cell_counts.sum() == 1000
    assert result.sample_cell_proportions.sum() == pytest.approx(1.0)
    assert result.demographic_imbalance >= 0.0
    assert result.fallback_cells >= 0


def test_evaluate_poll_is_deterministic_for_a_fixed_seed() -> None:
    population = default_population()

    first = evaluate_poll(
        population,
        500,
        0.5,
        "strong",
        np.random.default_rng(22),
    )
    second = evaluate_poll(
        population,
        500,
        0.5,
        "strong",
        np.random.default_rng(22),
    )

    np.testing.assert_array_equal(first.truth, second.truth)
    np.testing.assert_array_equal(
        first.sample_cell_counts,
        second.sample_cell_counts,
    )
    np.testing.assert_array_equal(
        first.sample_cell_proportions,
        second.sample_cell_proportions,
    )

    for method in METHODS:
        np.testing.assert_array_equal(
            first.estimates[method],
            second.estimates[method],
        )
        assert first.l1_errors[method] == second.l1_errors[method]
        assert first.max_abs_errors[method] == second.max_abs_errors[method]

    assert first.demographic_imbalance == second.demographic_imbalance
    assert first.fallback_cells == second.fallback_cells
