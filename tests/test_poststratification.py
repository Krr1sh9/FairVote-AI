from __future__ import annotations

import math

import numpy as np

from fairvote.poststratification import poststratified_estimate
from fairvote.privacy.estimators import debiased_estimate, project_distribution

EPSILON_LN4 = math.log(4.0)
CELL_ZERO_REPORTS = [0, 0, 0, 0, 1, 2]
CELL_ONE_REPORTS = [1, 1, 1, 1, 0, 2]


def test_hand_calculated_poststratified_estimate() -> None:
    cells = np.array([0] * 6 + [1] * 6)
    reports = np.array(CELL_ZERO_REPORTS + CELL_ONE_REPORTS)
    weights = np.array([0.25, 0.75])

    result = poststratified_estimate(cells, reports, weights, EPSILON_LN4, 3)

    np.testing.assert_allclose(
        result.cell_estimates[0],
        [1.0, 0.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result.cell_estimates[1],
        [0.0, 1.0, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(result.estimate, [0.25, 0.75, 0.0], atol=1e-12)
    assert result.fallback_cells == 0


def test_projection_is_applied_after_cell_weighting() -> None:
    cells = np.array([0] * 6 + [1] * 6)
    reports = np.array([0] * 6 + [1] * 6)
    weights = np.array([0.25, 0.75])

    result = poststratified_estimate(cells, reports, weights, EPSILON_LN4, 3)

    np.testing.assert_allclose(
        result.cell_estimates[0],
        [5 / 3, -1 / 3, -1 / 3],
    )
    np.testing.assert_allclose(
        result.cell_estimates[1],
        [-1 / 3, 5 / 3, -1 / 3],
    )
    weighted_raw = weights @ result.cell_estimates
    np.testing.assert_allclose(result.estimate, project_distribution(weighted_raw))


def test_empty_cells_use_the_overall_rr_correction_as_fallback() -> None:
    cells = np.array([0] * 6 + [1] * 6)
    reports = np.array(CELL_ZERO_REPORTS + CELL_ONE_REPORTS)
    weights = np.array([0.2, 0.3, 0.5])

    result = poststratified_estimate(cells, reports, weights, EPSILON_LN4, 3)
    overall = debiased_estimate(
        reports,
        EPSILON_LN4,
        3,
        clip=False,
        renormalize=False,
    )

    assert result.fallback_cells == 1
    np.testing.assert_allclose(result.cell_estimates[2], overall, atol=1e-12)
    np.testing.assert_allclose(
        result.estimate,
        project_distribution(weights @ result.cell_estimates),
        atol=1e-12,
    )
