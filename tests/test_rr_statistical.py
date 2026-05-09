"""Statistical tests for the k-ary Randomized Response mechanism."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from fairvote.privacy.mechanisms.kary_rr import privatize_many, rr_params, rr_transition_matrix


def test_rr_output_distribution_matches_theory_at_large_n() -> None:
    rng = np.random.default_rng(4242)
    k = 4
    epsilon = 0.8
    n = 140_000
    true_dist = np.array([0.08, 0.22, 0.35, 0.35], dtype=float)
    truth = rng.choice(k, size=n, p=true_dist)
    reports = privatize_many(truth, epsilon=epsilon, k=k, rng=rng)

    observed = np.bincount(reports, minlength=k) / n
    expected = true_dist @ rr_transition_matrix(epsilon, k)
    assert np.allclose(observed, expected, atol=0.006)


@pytest.mark.statistical
def test_p_keep_increases_monotonically_with_epsilon_grid() -> None:
    eps_grid = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0])
    for k in [2, 3, 5, 9]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            p_keep = np.array([rr_params(float(eps), k).p_keep for eps in eps_grid])
        assert np.all(np.diff(p_keep) > 0.0)


def _debiased_category_variance(theta: float, *, epsilon: float, k: int, n: int) -> float:
    params = rr_params(epsilon, k)
    reported_prob = params.q + (params.p - params.q) * theta
    return reported_prob * (1.0 - reported_prob) / (n * (params.p - params.q) ** 2)


@pytest.mark.statistical
def test_debiased_estimator_variance_increases_as_epsilon_decreases() -> None:
    # Analytical variance for one category of the canonical RR inverse. This is
    # less flaky than trying to assert the ordering from a small Monte Carlo run.
    k = 4
    theta = 0.30
    n = 2_000
    eps_grid = [0.2, 0.5, 1.0, 2.0]
    variances = np.array([_debiased_category_variance(theta, epsilon=eps, k=k, n=n) for eps in eps_grid])
    assert np.all(np.isfinite(variances))
    # Stronger privacy (smaller epsilon) must have larger debiased variance.
    assert np.all(np.diff(variances) < 0.0)
    assert variances[0] > 20.0 * variances[-1]
