from __future__ import annotations

import numpy as np
import pytest

from fairvote.metrics import l1_error, max_absolute_error
from fairvote.simulation.population import default_population
from fairvote.study import METHODS, evaluate_poll


def test_evaluate_poll_integrates_all_three_estimators_and_metrics() -> None:
    # This integration test runs one complete synthetic poll through the shared study workflow.
    population = default_population()
    result = evaluate_poll(
        population,
        1000,
        1.0,
        "moderate",
        np.random.default_rng(5),
    )

    # The study result must retain the known synthetic population truth and provide outputs for every configured estimator.
    np.testing.assert_allclose(result.truth, population.true_distribution())
    assert set(result.estimates) == set(METHODS)
    assert set(result.l1_errors) == set(METHODS)
    assert set(result.max_abs_errors) == set(METHODS)

    # Every estimator output must be a finite three-category probability distribution with metrics consistent with the shared metric functions.
    for method in METHODS:
        estimate = result.estimates[method]
        assert estimate.shape == (population.n_categories,)
        assert np.all(np.isfinite(estimate))
        assert estimate.min() >= 0.0
        assert estimate.sum() == pytest.approx(1.0)
        assert result.l1_errors[method] == pytest.approx(l1_error(estimate, result.truth))
        assert result.max_abs_errors[method] == pytest.approx(max_absolute_error(estimate, result.truth))

    # Respondent counts and proportions must describe the full sample, while the two demographic diagnostics must remain non-negative counts or distances.
    assert result.sample_cell_counts.sum() == 1000
    assert result.sample_cell_proportions.sum() == pytest.approx(1.0)
    assert result.demographic_imbalance >= 0.0
    assert result.fallback_cells >= 0


def test_evaluate_poll_is_deterministic_for_a_fixed_seed() -> None:
    # Two independently created generators with the same seed are evaluated under identical poll settings.
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

    # The fixed synthetic truth and the sampled demographic diagnostics must be reproduced exactly.
    np.testing.assert_array_equal(first.truth, second.truth)
    np.testing.assert_array_equal(
        first.sample_cell_counts,
        second.sample_cell_counts,
    )
    np.testing.assert_array_equal(
        first.sample_cell_proportions,
        second.sample_cell_proportions,
    )

    # Each estimator's returned distribution and both stored error metrics must also be identical across the two runs.
    for method in METHODS:
        np.testing.assert_array_equal(
            first.estimates[method],
            second.estimates[method],
        )
        assert first.l1_errors[method] == second.l1_errors[method]
        assert first.max_abs_errors[method] == second.max_abs_errors[method]

    # The remaining scalar diagnostics must match exactly for the repeated fixed-seed evaluation.
    assert first.demographic_imbalance == second.demographic_imbalance
    assert first.fallback_cells == second.fallback_cells
