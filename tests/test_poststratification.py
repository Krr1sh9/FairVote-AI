from __future__ import annotations

import math

import numpy as np

from fairvote.poststratification import poststratified_estimate
from fairvote.privacy.estimators import debiased_estimate, project_distribution

# This epsilon value makes exp(epsilon) equal to four and gives convenient hand-calculable Randomised Response probabilities.
EPSILON_LN4 = math.log(4.0)

# These fixed report patterns are chosen so the unconstrained RR inversions for the two populated cells are easy to verify analytically.
CELL_ZERO_REPORTS = [0, 0, 0, 0, 1, 2]
CELL_ONE_REPORTS = [1, 1, 1, 1, 0, 2]


def test_hand_calculated_poststratified_estimate() -> None:
    # The sample contains two equally sized observed cells, while the target population weights them as one quarter and three quarters.
    cells = np.array([0] * 6 + [1] * 6)
    reports = np.array(CELL_ZERO_REPORTS + CELL_ONE_REPORTS)
    weights = np.array([0.25, 0.75])

    result = poststratified_estimate(cells, reports, weights, EPSILON_LN4, 3)

    # The first report pattern analytically inverts to a point mass on category zero.
    np.testing.assert_allclose(
        result.cell_estimates[0],
        [1.0, 0.0, 0.0],
        atol=1e-12,
    )

    # The second report pattern analytically inverts to a point mass on category one.
    np.testing.assert_allclose(
        result.cell_estimates[1],
        [0.0, 1.0, 0.0],
        atol=1e-12,
    )

    # Weighting those two cell estimates by the supplied population shares gives the expected final distribution.
    np.testing.assert_allclose(result.estimate, [0.25, 0.75, 0.0], atol=1e-12)

    # Both demographic cells contain sampled respondents, so no fallback is required.
    assert result.fallback_cells == 0


def test_projection_is_applied_after_cell_weighting() -> None:
    # Each populated cell reports only one category, producing unconstrained RR inversions with values outside the probability range.
    cells = np.array([0] * 6 + [1] * 6)
    reports = np.array([0] * 6 + [1] * 6)
    weights = np.array([0.25, 0.75])

    result = poststratified_estimate(cells, reports, weights, EPSILON_LN4, 3)

    # The stored cell estimates are the raw analytical inversions, confirming that they were not clipped or renormalised individually.
    np.testing.assert_allclose(
        result.cell_estimates[0],
        [5 / 3, -1 / 3, -1 / 3],
    )
    np.testing.assert_allclose(
        result.cell_estimates[1],
        [-1 / 3, 5 / 3, -1 / 3],
    )

    # The implementation first combines the raw cell estimates with the population weights and only then applies clipping and renormalisation.
    weighted_raw = weights @ result.cell_estimates
    np.testing.assert_allclose(result.estimate, project_distribution(weighted_raw))


def test_empty_cells_use_the_overall_rr_correction_as_fallback() -> None:
    # The weight vector defines three demographic cells, but the sample contains respondents only from cells zero and one.
    cells = np.array([0] * 6 + [1] * 6)
    reports = np.array(CELL_ZERO_REPORTS + CELL_ONE_REPORTS)
    weights = np.array([0.2, 0.3, 0.5])

    result = poststratified_estimate(cells, reports, weights, EPSILON_LN4, 3)

    # The expected fallback is the unconstrained RR inversion calculated from all reports in the sample.
    overall = debiased_estimate(
        reports,
        EPSILON_LN4,
        3,
        clip=False,
        renormalize=False,
    )

    # Exactly one cell is empty, and its stored cell estimate must equal the whole-sample fallback.
    assert result.fallback_cells == 1
    np.testing.assert_allclose(result.cell_estimates[2], overall, atol=1e-12)

    # The final estimate is obtained by weighting all three stored cell estimates and then applying the shared clipping and renormalisation step.
    np.testing.assert_allclose(
        result.estimate,
        project_distribution(weights @ result.cell_estimates),
        atol=1e-12,
    )
