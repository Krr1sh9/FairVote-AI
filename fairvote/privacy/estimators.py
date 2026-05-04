"""Estimators for recovering aggregate distributions from privatized reports.

The functions in this module use the known k-ary Randomized Response
observation model to debias reported category counts. They operate on
privatized reports and do not require individual true answers.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np

from fairvote.privacy.mechanisms.kary_rr import counts_from_reports, rr_params


ArrayLike = Union[np.ndarray, list]


def estimate_distribution_from_counts(
    counts: ArrayLike,
    epsilon: float,
    k: int,
    *,
    clip: bool = True,
    renormalize: bool = True,
) -> np.ndarray:
    """
    Unbiased (in expectation) estimator of the true category distribution under k-ary RR.

    Observation model:
      Let theta_j = P(true=j)
      Let y_j = P(report=j)

      Under k-ary RR:
        y_j = q + (p - q)*theta_j
      where:
        p = exp(eps) / (exp(eps) + k - 1)
        q = (1 - p) / (k - 1)

      Solve:
        theta_j = (y_j - q) / (p - q)

    Implementation uses y_hat = counts / n.

    Notes:
      - For finite samples, the raw estimator can go negative or >1.
        clip=True clamps to [0,1]. renormalize=True forces sum to 1.
    """
    # Retrieve the k-ary RR channel probabilities (p for truth retention,
    # q for uniform flipping) that define the inverse observation model.
    params = rr_params(epsilon, k)

    c = np.asarray(counts, dtype=float)
    if c.ndim != 1 or c.size != k:
        raise ValueError(f"counts must be a 1D array of length {k}.")
    if np.any(c < 0):
        raise ValueError("counts must be non-negative.")

    n = float(np.sum(c))
    if n <= 0:
        raise ValueError("counts sum must be > 0.")

    # The observed reported distribution is debiased through the inverse of the
    # known RR channel. No individual true answers are needed for this step.
    y_hat = c / n
    # p - q must be strictly positive; otherwise the RR channel is non-invertible
    # and no debiasing is possible.
    denom = (params.p - params.q)
    if denom <= 0:
        raise RuntimeError("Invalid RR parameters: p - q must be positive.")

    # Apply the analytic inverse:  theta_j = (y_hat_j - q) / (p - q).
    # This is the core estimator; it requires only the reported distribution.
    theta_hat = (y_hat - params.q) / denom

    if clip:
        theta_hat = np.clip(theta_hat, 0.0, 1.0)

    if renormalize:
        # Clipping can disturb the probability simplex, so the final vector is
        # normalised for downstream plotting and metric calculations.
        s = float(np.sum(theta_hat))
        if s <= 0:
            # Extreme case: everything clipped to 0 due to heavy noise / tiny n.
            # Fall back to uniform.
            theta_hat = np.full(k, 1.0 / k, dtype=float)
        else:
            theta_hat = theta_hat / s

    return theta_hat


def estimate_distribution(
    reported_categories: ArrayLike,
    epsilon: float,
    k: int,
    *,
    clip: bool = True,
    renormalize: bool = True,
) -> np.ndarray:
    """
    Convenience wrapper: estimate true distribution from raw reported category labels.
    """
    # Convert raw reported labels into histogram counts, then call the
    # count-based estimator.  This two-step design avoids duplicating the
    # debiasing algebra.
    counts = counts_from_reports(reported_categories, k)
    return estimate_distribution_from_counts(
        counts, epsilon, k, clip=clip, renormalize=renormalize
    )


def bootstrap_ci(
    reported_categories: ArrayLike,
    epsilon: float,
    k: int,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: Optional[np.random.Generator] = None,
    clip: bool = True,
    renormalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Percentile bootstrap confidence intervals for each category proportion.

    Returns:
      (lower, upper) arrays of shape (k,)

    Method:
      - resample reported categories with replacement
      - compute theta_hat each resample
      - take empirical quantiles at alpha/2 and 1 - alpha/2

    Notes:
      - This gives uncertainty under the observed reported data distribution.
      - For report-ready results, you can also compute interval coverage in simulation experiments.
    """
    reps = np.asarray(reported_categories, dtype=int)
    if reps.ndim != 1 or reps.size == 0:
        raise ValueError("reported_categories must be a non-empty 1D array-like.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if not isinstance(n_boot, int) or n_boot < 200:
        raise ValueError("n_boot must be an int >= 200 for a stable CI.")

    if rng is None:
        rng = np.random.default_rng()

    # Pre-allocate the bootstrap matrix once rather than appending rows.
    n = reps.size
    boot = np.empty((n_boot, k), dtype=float)

    for b in range(n_boot):
        # Resample reported answers, not true answers. The bootstrap therefore
        # reflects uncertainty in the observed privatized dataset.
        idx = rng.integers(0, n, size=n, dtype=int)
        sample = reps[idx]
        boot[b] = estimate_distribution(
            sample, epsilon, k, clip=clip, renormalize=renormalize
        )

    # Take the alpha/2 and 1-alpha/2 empirical quantiles to form a
    # two-sided percentile confidence interval per category.
    lo_q = alpha / 2.0
    hi_q = 1.0 - alpha / 2.0
    lower = np.quantile(boot, lo_q, axis=0)
    upper = np.quantile(boot, hi_q, axis=0)
    return lower, upper
