# tests/test_debias.py
import numpy as np
import pytest


def test_rr_estimate_distribution_recovers_truth_reasonably():
    """
    End-to-end sanity check:
      true -> privatize_many -> estimate_distribution
    should recover the true distribution reasonably well (large n).
    """
    try:
        from fairvote.privacy import estimate_distribution, privatize_many
    except Exception:
        pytest.skip("fairvote.privacy.privatize_many / estimate_distribution not available")

    rng = np.random.default_rng(123)
    k = 5
    eps = 1.0
    n = 8000

    true_p = np.array([0.10, 0.20, 0.30, 0.15, 0.25], dtype=float)
    true_p = true_p / true_p.sum()

    true = rng.choice(k, size=n, p=true_p)

    # Try to make the test deterministic even if your code uses global RNG
    np.random.seed(123)

    reported = privatize_many(true, epsilon=eps, k=k)
    est = estimate_distribution(reported, epsilon=eps, k=k)

    est = np.asarray(est, dtype=float)
    assert est.shape == (k,)
    assert np.all(np.isfinite(est))
    assert np.all(est >= -1e-6)
    assert np.isclose(est.sum(), 1.0, atol=1e-3)

    l1 = float(np.sum(np.abs(est - true_p)))
    # This threshold is intentionally loose to avoid flaky tests.
    assert l1 < 0.20
