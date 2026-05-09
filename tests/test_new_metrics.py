# tests/test_new_metrics.py
"""
Tests for newly added metrics:
  - correct_winner
  - rmse_per_candidate / overall_rmse
  - error_ratio
"""

import numpy as np
import pytest

from fairvote.metrics.group_metrics import (
    correct_winner,
    error_ratio,
    overall_rmse,
    rmse_per_candidate,
)

# ============================================================================
# correct_winner
# ============================================================================


class TestCorrectWinner:
    def test_same_winner(self):
        est = np.array([0.1, 0.6, 0.3])
        truth = np.array([0.2, 0.5, 0.3])
        assert correct_winner(est, truth) is True

    def test_different_winner(self):
        est = np.array([0.6, 0.1, 0.3])
        truth = np.array([0.2, 0.5, 0.3])
        assert correct_winner(est, truth) is False

    def test_tie_uses_first_argmax(self):
        est = np.array([0.5, 0.5])
        truth = np.array([0.5, 0.5])
        # argmax returns first occurrence for ties — both return 0
        assert correct_winner(est, truth) is True

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError):
            correct_winner(np.array([0.5, 0.5]), np.array([0.3, 0.3, 0.4]))

    def test_2d_raises(self):
        with pytest.raises(ValueError):
            correct_winner(np.array([[0.5, 0.5]]), np.array([[0.3, 0.7]]))


# ============================================================================
# rmse_per_candidate / overall_rmse
# ============================================================================


class TestRMSE:
    def test_perfect_estimates(self):
        truth = np.array([0.5, 0.3, 0.2])
        trials = [truth.copy() for _ in range(10)]
        rmse = rmse_per_candidate(trials, truth)
        np.testing.assert_allclose(rmse, [0.0, 0.0, 0.0], atol=1e-10)

    def test_known_rmse(self):
        truth = np.array([0.5, 0.5])
        # Two trials with known errors
        trials = [
            np.array([0.6, 0.4]),  # error: [0.1, -0.1]
            np.array([0.4, 0.6]),  # error: [-0.1, 0.1]
        ]
        rmse = rmse_per_candidate(trials, truth)
        # RMSE per candidate = sqrt(mean([0.01, 0.01])) = 0.1
        np.testing.assert_allclose(rmse, [0.1, 0.1], atol=1e-10)

    def test_overall_rmse_is_mean_of_per_candidate(self):
        truth = np.array([0.4, 0.3, 0.3])
        trials = [np.array([0.5, 0.25, 0.25]), np.array([0.3, 0.35, 0.35])]
        per_cand = rmse_per_candidate(trials, truth)
        expected = float(np.mean(per_cand))
        assert abs(overall_rmse(trials, truth) - expected) < 1e-10

    def test_empty_trials_returns_nan(self):
        truth = np.array([0.5, 0.5])
        rmse = rmse_per_candidate([], truth)
        assert all(np.isnan(rmse))

    def test_single_trial(self):
        truth = np.array([0.5, 0.5])
        trials = [np.array([0.6, 0.4])]
        rmse = rmse_per_candidate(trials, truth)
        np.testing.assert_allclose(rmse, [0.1, 0.1], atol=1e-10)


# ============================================================================
# error_ratio
# ============================================================================


class TestErrorRatio:
    def test_equal_errors_ratio_one(self):
        est = {"A": np.array([0.6, 0.4]), "B": np.array([0.55, 0.45])}
        truth = {"A": np.array([0.5, 0.5]), "B": np.array([0.45, 0.55])}
        # Both have L1 error = 0.2
        ratio = error_ratio(est, truth)
        assert abs(ratio - 1.0) < 1e-10

    def test_unequal_errors(self):
        est = {"A": np.array([0.8, 0.2]), "B": np.array([0.55, 0.45])}
        truth = {"A": np.array([0.5, 0.5]), "B": np.array([0.5, 0.5])}
        # A: L1 = 0.6, B: L1 = 0.1 → ratio = 6.0
        ratio = error_ratio(est, truth)
        assert abs(ratio - 6.0) < 1e-10

    def test_single_group_returns_nan(self):
        est = {"A": np.array([0.5, 0.5])}
        truth = {"A": np.array([0.5, 0.5])}
        ratio = error_ratio(est, truth)
        assert np.isnan(ratio)

    def test_empty_returns_nan(self):
        ratio = error_ratio({}, {})
        assert np.isnan(ratio)

    def test_min_mass_filter(self):
        est = {"A": np.array([0.8, 0.2]), "B": np.array([0.55, 0.45])}
        truth = {"A": np.array([0.5, 0.5]), "B": np.array([0.5, 0.5])}
        masses = {"A": 0.9, "B": 0.01}
        # B is below min_mass threshold → only one group → NaN
        ratio = error_ratio(
            est,
            truth,
            group_masses=masses,
            min_mass=0.05,
            normalise_masses=False,
        )
        assert np.isnan(ratio)

    def test_zero_min_error_returns_inf(self):
        est = {"A": np.array([0.5, 0.5]), "B": np.array([0.8, 0.2])}
        truth = {"A": np.array([0.5, 0.5]), "B": np.array([0.5, 0.5])}
        # A has 0 error, B has 0.6 → ratio = inf
        ratio = error_ratio(est, truth)
        assert ratio == float("inf")
