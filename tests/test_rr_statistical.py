from __future__ import annotations

import numpy as np
import pytest

from fairvote.privacy.estimators import debiased_estimate
from fairvote.privacy.mechanisms.kary_rr import (
    counts_from_reports,
    privatize_many,
    rr_transition_matrix,
)

# Both tests in this module are marked as statistical because their assertions depend on fixed-seed Monte Carlo simulations.
pytestmark = pytest.mark.statistical


def test_reported_frequencies_match_the_theoretical_channel_output() -> None:
    # A fixed random seed makes the large-sample simulation reproducible.
    rng = np.random.default_rng(101)
    epsilon, k, n = 1.0, 3, 200_000
    truth = np.array([0.5, 0.3, 0.2])

    # Original categories are sampled from the known distribution and then passed through k-ary Randomised Response.
    categories = rng.choice(k, size=n, p=truth)
    reports = privatize_many(categories, epsilon, k, rng)
    observed = counts_from_reports(reports, k) / n

    # Multiplying the true distribution by the transition matrix gives the theoretical distribution of reported categories.
    expected = truth @ rr_transition_matrix(epsilon, k)

    # The observed report frequencies are required to be within 0.01 of the theoretical channel output in every category.
    np.testing.assert_allclose(observed, expected, atol=0.01)


def test_lower_epsilon_gives_a_more_variable_debiased_estimator() -> None:
    # The comparison uses the same three-category truth, sample size and number of Monte Carlo repetitions at each epsilon.
    k, n, repetitions = 3, 2_000, 60
    truth = np.array([0.5, 0.3, 0.2])
    variances = []

    for epsilon in (0.25, 1.0, 3.0):
        # The generator is reset to the same seed for each epsilon so the comparison starts from the same deterministic random stream.
        rng = np.random.default_rng(202)
        estimates = []
        for _ in range(repetitions):
            categories = rng.choice(k, size=n, p=truth)
            reports = privatize_many(categories, epsilon, k, rng)

            # The unconstrained analytical inversion is used so clipping or renormalisation does not alter the variance being compared.
            estimates.append(debiased_estimate(reports, epsilon, k, clip=False, renormalize=False))

        # Variance is calculated across repetitions for each category and then averaged across the three categories.
        variances.append(float(np.var(np.asarray(estimates), axis=0).mean()))

    # For this fixed-seed simulation, the average estimator variance decreases as epsilon increases across the three tested values.
    assert variances[0] > variances[1] > variances[2]
