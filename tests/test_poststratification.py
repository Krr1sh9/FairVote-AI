from __future__ import annotations

import math

import numpy as np
import pytest

from fairvote.poststratification import poststratified_estimate
from fairvote.privacy.estimators import debiased_estimate, project_distribution
from fairvote.simulation.population import default_population
from fairvote.simulation.sampling import run_synthetic_poll

EPSILON_LN4 = math.log(4.0)
CELL_ZERO_REPORTS = [0, 0, 0, 0, 1, 2]
CELL_ONE_REPORTS = [1, 1, 1, 1, 0, 2]


def test_hand_calculated_result() -> None:
    cells = np.array([0] * 6 + [1] * 6)
    reports = np.array(CELL_ZERO_REPORTS + CELL_ONE_REPORTS)
    weights = np.array([0.25, 0.75])

    result = poststratified_estimate(cells, reports, weights, EPSILON_LN4, 3)

    np.testing.assert_allclose(result.cell_estimates[0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.cell_estimates[1], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(result.estimate, [0.25, 0.75, 0.0], atol=1e-12)
    assert result.fallback_cells == 0


def test_projection_is_applied_only_after_weighting() -> None:
    cells = np.array([0] * 6 + [1] * 6)
    reports = np.array([0] * 6 + [1] * 6)
    weights = np.array([0.25, 0.75])

    result = poststratified_estimate(cells, reports, weights, EPSILON_LN4, 3)

    np.testing.assert_allclose(result.cell_estimates[0], [5 / 3, -1 / 3, -1 / 3])
    np.testing.assert_allclose(result.cell_estimates[1], [-1 / 3, 5 / 3, -1 / 3])
    weighted_raw = weights @ result.cell_estimates
    np.testing.assert_allclose(result.estimate, project_distribution(weighted_raw))
    assert not np.allclose(result.estimate, [0.25, 0.75, 0.0])


def test_unnormalised_weights_are_renormalised() -> None:
    cells = np.array([0] * 6 + [1] * 6)
    reports = np.array(CELL_ZERO_REPORTS + CELL_ONE_REPORTS)

    result = poststratified_estimate(cells, reports, np.array([1.0, 3.0]), EPSILON_LN4, 3)
    np.testing.assert_allclose(result.estimate, [0.25, 0.75, 0.0], atol=1e-12)


def test_estimate_is_a_valid_distribution() -> None:
    population = default_population()
    sample = run_synthetic_poll(population, 800, 0.5, "strong", np.random.default_rng(31))

    result = poststratified_estimate(
        sample.cell_indices,
        sample.reported_categories,
        population.weights,
        0.5,
        3,
    )

    assert np.all(np.isfinite(result.estimate))
    assert result.estimate.min() >= 0.0
    assert result.estimate.sum() == pytest.approx(1.0)
    assert result.cell_estimates.shape == (population.n_cells, 3)


def test_empty_cells_use_the_raw_overall_estimate() -> None:
    cells = np.array([0] * 6 + [1] * 6)
    reports = np.array(CELL_ZERO_REPORTS + CELL_ONE_REPORTS)
    weights = np.array([0.2, 0.3, 0.5])

    result = poststratified_estimate(cells, reports, weights, EPSILON_LN4, 3)
    raw_overall = debiased_estimate(
        reports,
        EPSILON_LN4,
        3,
        clip=False,
        renormalize=False,
    )

    assert result.fallback_cells == 1
    np.testing.assert_allclose(result.cell_estimates[2], raw_overall, atol=1e-12)
    expected = project_distribution(weights @ result.cell_estimates)
    np.testing.assert_allclose(result.estimate, expected, atol=1e-12)


def test_fallback_count_matches_unsampled_cells() -> None:
    cells = np.array([0] * 6)
    reports = np.array(CELL_ZERO_REPORTS)
    result = poststratified_estimate(cells, reports, np.full(4, 0.25), EPSILON_LN4, 3)
    assert result.fallback_cells == 3


@pytest.mark.parametrize(
    ("cells", "reports", "weights"),
    [
        (np.array([0, 1]), np.array([0]), np.array([0.5, 0.5])),
        (np.array([]), np.array([]), np.array([0.5, 0.5])),
        (np.array([0, 5]), np.array([0, 1]), np.array([0.5, 0.5])),
        (np.array([0, 1]), np.array([0, 1]), np.array([0.5, -0.5])),
        (np.array([0, 1]), np.array([0, 1]), np.array([0.0, 0.0])),
        (np.array([0, 1]), np.array([0, 1]), np.array([np.nan, 1.0])),
        (np.array([0, 1]), np.array([0, 1]), np.array([])),
    ],
)
def test_malformed_inputs_are_rejected(
    cells: np.ndarray,
    reports: np.ndarray,
    weights: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        poststratified_estimate(cells, reports, weights, 1.0, 3)
