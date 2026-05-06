from __future__ import annotations

import numpy as np

from experiments.theory_validation import analytic_debiased_variance, monte_carlo_unbiasedness, privacy_ratio


def test_privacy_ratio_matches_exp_epsilon():
    epsilon = 1.3
    assert np.isclose(privacy_ratio(epsilon, 5), np.exp(epsilon), rtol=1e-12)


def test_unclipped_inverse_is_approximately_unbiased():
    theta = np.array([0.50, 0.30, 0.20])
    result = monte_carlo_unbiasedness(theta, epsilon=1.5, n=1200, reps=80, seed=99)
    assert result["max_abs_bias"] < 0.05


def test_analytic_variance_is_positive_and_finite():
    theta = np.array([0.40, 0.35, 0.25])
    var = analytic_debiased_variance(theta, epsilon=1.0, n=500)
    assert var.shape == theta.shape
    assert np.all(np.isfinite(var))
    assert np.all(var > 0)
