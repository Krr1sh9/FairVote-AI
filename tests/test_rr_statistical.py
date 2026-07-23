from __future__ import annotations

import numpy as np
import pytest

from fairvote.privacy.estimators import debiased_estimate
from fairvote.privacy.mechanisms.kary_rr import (
    counts_from_reports,
    privatize_many,
    rr_transition_matrix,
)

pytestmark = pytest.mark.statistical


def test_reported_frequencies_match_the_theoretical_channel_output() -> None:
    rng = np.random.default_rng(101)
    epsilon, k, n = 1.0, 3, 200_000
    truth = np.array([0.5, 0.3, 0.2])

    categories = rng.choice(k, size=n, p=truth)
    reports = privatize_many(categories, epsilon, k, rng)
    observed = counts_from_reports(reports, k) / n

    expected = truth @ rr_transition_matrix(epsilon, k)
    np.testing.assert_allclose(observed, expected, atol=0.01)


def test_lower_epsilon_gives_a_more_variable_debiased_estimator() -> None:
    k, n, repetitions = 3, 2_000, 60
    truth = np.array([0.5, 0.3, 0.2])
    variances = []

    for epsilon in (0.25, 1.0, 3.0):
        rng = np.random.default_rng(202)
        estimates = []
        for _ in range(repetitions):
            categories = rng.choice(k, size=n, p=truth)
            reports = privatize_many(categories, epsilon, k, rng)
            estimates.append(debiased_estimate(reports, epsilon, k, clip=False, renormalize=False))
        variances.append(float(np.var(np.asarray(estimates), axis=0).mean()))

    assert variances[0] > variances[1] > variances[2]
