# tests/test_central_dp.py
"""
Tests for the central DP (Laplace mechanism) module.
"""
from __future__ import annotations

import numpy as np
import pytest


def test_laplace_mechanism_shape_and_normalisation():
    """Output is length k, sums to ~1, all entries >= 0."""
    from fairvote.privacy.mechanisms.laplace_mechanism import laplace_mechanism

    k = 5
    counts = np.array([100, 200, 150, 50, 500], dtype=float)
    epsilon = 1.0

    result = laplace_mechanism(counts, epsilon, k, rng=np.random.default_rng(42))

    assert result.shape == (k,)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)
    assert abs(float(result.sum()) - 1.0) < 1e-10


def test_estimate_distribution_central_dp_shape():
    """Convenience wrapper produces correct shape and normalisation."""
    from fairvote.privacy.mechanisms.laplace_mechanism import estimate_distribution_central_dp

    rng = np.random.default_rng(7)
    k = 5
    n = 500
    cats = rng.integers(0, k, size=n, dtype=int)

    result = estimate_distribution_central_dp(cats, epsilon=1.0, k=k, rng=rng)

    assert result.shape == (k,)
    assert np.all(np.isfinite(result))
    assert np.all(result >= 0.0)
    assert abs(float(result.sum()) - 1.0) < 1e-10


def test_central_dp_unbiased_over_many_trials():
    """Mean estimate converges to true distribution over many samples."""
    from fairvote.privacy.mechanisms.laplace_mechanism import estimate_distribution_central_dp

    k = 4
    n = 2000
    epsilon = 1.0
    n_trials = 500

    # Fixed true distribution
    true_dist = np.array([0.4, 0.3, 0.2, 0.1])

    rng = np.random.default_rng(123)
    estimates = np.zeros((n_trials, k), dtype=float)

    for t in range(n_trials):
        # Draw true categories from the known distribution
        cats = rng.choice(k, size=n, p=true_dist)
        rng_dp = np.random.default_rng(123 + t)
        estimates[t] = estimate_distribution_central_dp(cats, epsilon, k, rng=rng_dp)

    mean_est = estimates.mean(axis=0)

    # Mean estimate should be close to true distribution
    # (tolerance accounts for both sampling and DP noise)
    assert np.allclose(mean_est, true_dist, atol=0.03), (
        f"Mean estimate {mean_est} too far from truth {true_dist}"
    )


def test_central_dp_beats_ldp_at_same_epsilon():
    """
    For the same epsilon and large n, central DP should have
    lower expected L1 error than LDP (k-ary RR).
    """
    from fairvote.privacy import (
        estimate_distribution,
        estimate_distribution_central_dp,
        privatize_many,
    )

    k = 5
    n = 3000
    epsilon = 1.0
    n_trials = 100

    rng = np.random.default_rng(42)
    true_dist = np.array([0.35, 0.25, 0.20, 0.12, 0.08])

    ldp_errors = []
    cdp_errors = []

    for t in range(n_trials):
        cats = rng.choice(k, size=n, p=true_dist)

        # LDP: privatize then estimate
        rng_ldp = np.random.default_rng(42 + t)
        reported = privatize_many(cats, epsilon, k, rng=rng_ldp)
        est_ldp = estimate_distribution(reported, epsilon, k)
        ldp_errors.append(float(np.sum(np.abs(est_ldp - true_dist))))

        # Central DP: add Laplace to true counts
        rng_cdp = np.random.default_rng(42 + t + 10000)
        est_cdp = estimate_distribution_central_dp(cats, epsilon, k, rng=rng_cdp)
        cdp_errors.append(float(np.sum(np.abs(est_cdp - true_dist))))

    mean_ldp = float(np.mean(ldp_errors))
    mean_cdp = float(np.mean(cdp_errors))

    # Central DP should be substantially better (lower error)
    assert mean_cdp < mean_ldp, (
        f"Central DP (mean L1={mean_cdp:.4f}) should beat "
        f"LDP (mean L1={mean_ldp:.4f}) at same epsilon"
    )


def test_higher_epsilon_means_less_noise():
    """Higher epsilon (less privacy) should give lower error on average."""
    from fairvote.privacy.mechanisms.laplace_mechanism import estimate_distribution_central_dp

    k = 5
    n = 1000
    n_trials = 200

    rng = np.random.default_rng(99)
    true_dist = np.array([0.3, 0.25, 0.2, 0.15, 0.1])

    errors_by_eps = {}
    for eps in [0.5, 2.0]:
        errs = []
        for t in range(n_trials):
            cats = rng.choice(k, size=n, p=true_dist)
            rng_dp = np.random.default_rng(99 + t + int(eps * 1000))
            est = estimate_distribution_central_dp(cats, eps, k, rng=rng_dp)
            errs.append(float(np.sum(np.abs(est - true_dist))))
        errors_by_eps[eps] = float(np.mean(errs))

    # Higher epsilon should give lower error
    assert errors_by_eps[2.0] < errors_by_eps[0.5], (
        f"eps=2.0 error ({errors_by_eps[2.0]:.4f}) should be < "
        f"eps=0.5 error ({errors_by_eps[0.5]:.4f})"
    )


def test_invalid_epsilon_raises():
    from fairvote.privacy.mechanisms.laplace_mechanism import laplace_mechanism

    counts = np.array([10, 20, 30], dtype=float)
    with pytest.raises(ValueError):
        laplace_mechanism(counts, epsilon=-1.0, k=3)
    with pytest.raises(ValueError):
        laplace_mechanism(counts, epsilon=0.0, k=3)


def test_invalid_k_raises():
    from fairvote.privacy.mechanisms.laplace_mechanism import estimate_distribution_central_dp

    with pytest.raises(ValueError):
        estimate_distribution_central_dp([0, 1], epsilon=1.0, k=1)


def test_wrong_count_length_raises():
    from fairvote.privacy.mechanisms.laplace_mechanism import laplace_mechanism

    counts = np.array([10, 20, 30], dtype=float)
    with pytest.raises(ValueError):
        laplace_mechanism(counts, epsilon=1.0, k=5)  # k=5 but only 3 counts
