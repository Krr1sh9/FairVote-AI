from __future__ import annotations

import numpy as np
import pytest

try:
    from ._helpers import call_with_supported_kwargs
except ModuleNotFoundError:
    from ._helpers import call_with_supported_kwargs


def _ci_low_high(ci):
    # Support direct tuple return
    if isinstance(ci, tuple) and len(ci) == 2:
        return ci[0], ci[1]

    # Support dict-like or attribute-like return types.
    if isinstance(ci, dict):
        lo = ci.get("low") or ci.get("lo") or ci.get("lower")
        hi = ci.get("high") or ci.get("hi") or ci.get("upper")
        return lo, hi
    for lo_name in ("low", "lo", "lower"):
        for hi_name in ("high", "hi", "upper"):
            if hasattr(ci, lo_name) and hasattr(ci, hi_name):
                return getattr(ci, lo_name), getattr(ci, hi_name)
    raise TypeError(f"Unsupported CI type: {type(ci)}")


def test_bootstrap_ci_returns_bounds():
    privacy = pytest.importorskip("fairvote.privacy")
    bootstrap_ci = privacy.bootstrap_ci
    privatize_many = privacy.privatize_many
    estimate_distribution = privacy.estimate_distribution

    rng = np.random.default_rng(7)
    k = 5
    n = 400
    epsilon = 1.0

    truth = rng.integers(0, k, size=n, dtype=int)
    reported = call_with_supported_kwargs(privatize_many, truth=truth, categories=truth, epsilon=epsilon, k=k, seed=7)
    if reported is None:
        reported = privatize_many(truth, epsilon=epsilon, k=k, seed=7)

    # Run bootstrap with small n_boot to keep tests quick
    ci = call_with_supported_kwargs(
        bootstrap_ci,
        reported=reported,
        epsilon=epsilon,
        k=k,
        n_boot=200,
        alpha=0.05,
        seed=7,
        estimator=estimate_distribution,
    )
    if ci is None:
        ci = bootstrap_ci(reported, epsilon=epsilon, k=k, n_boot=200, alpha=0.05, seed=7)

    lo, hi = _ci_low_high(ci)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)

    assert lo.shape == (k,)
    assert hi.shape == (k,)
    assert np.all(np.isfinite(lo))
    assert np.all(np.isfinite(hi))
    assert np.all(lo <= hi + 1e-12)
    # bounds should lie in [0,1] loosely (allow tiny numerical drift)
    assert float(lo.min()) >= -1e-6
    assert float(hi.max()) <= 1.0 + 1e-6
