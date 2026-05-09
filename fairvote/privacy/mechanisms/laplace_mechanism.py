# fairvote/privacy/mechanisms/laplace_mechanism.py
"""
Central Differential Privacy via the Laplace mechanism for aggregate category counts.

Trust model:
  The collector receives raw (true/stated) responses, computes exact counts,
  then adds calibrated Laplace noise before publishing.  This gives epsilon-DP
  for the published histogram but requires trust in the collector.

Sensitivity:
  For a histogram query over k categories (each person contributes +1 to exactly
  one bin), the L1 sensitivity is 2: adding or removing one person changes two
  bins by 1 each.  This is configurable via the `sensitivity` parameter.

Contrast with LDP (k-ary RR):
  In LDP each respondent randomises their own answer, so the collector receives
  randomized reports rather than an explicit raw-answer field.  Central DP achieves
  better utility at the same epsilon because noise is injected once (on the aggregate)
  rather than per-person.
"""

from __future__ import annotations

import math

import numpy as np

ArrayLike = np.ndarray | list


def laplace_mechanism(
    true_counts: ArrayLike,
    epsilon: float,
    k: int,
    *,
    sensitivity: float = 2.0,
    rng: np.random.Generator | None = None,
    clip: bool = True,
    renormalize: bool = True,
) -> np.ndarray:
    """
    Apply the Laplace mechanism to a vector of true category counts.

    Args:
      true_counts: length-k array of non-negative integer counts.
      epsilon: privacy parameter (> 0).  Smaller = more privacy, more noise.
      k: number of categories.
      sensitivity: L1 sensitivity of the histogram query (default 2).
      rng: numpy random Generator (optional; created if None).
      clip: if True, clip noisy counts to >= 0.
      renormalize: if True, normalise the result to sum to 1.

    Returns:
      Estimated distribution as a length-k float array summing to 1.
    """
    # Input validation precedes any noise addition so that invalid epsilon
    # or k cannot silently produce unreliable noisy counts.
    _validate_inputs(epsilon, k, sensitivity)

    counts = np.asarray(true_counts, dtype=float)
    if counts.ndim != 1 or counts.size != k:
        raise ValueError(f"true_counts must be a 1D array of length {k}.")
    if np.any(counts < 0):
        raise ValueError("true_counts must be non-negative.")

    n = float(np.sum(counts))
    if n <= 0:
        raise ValueError("true_counts must sum to > 0.")

    if rng is None:
        rng = np.random.default_rng()

    # Laplace scale = sensitivity / epsilon.  Larger sensitivity or smaller
    # epsilon widens the noise, exactly implementing the epsilon-DP guarantee
    # for the aggregate histogram query.
    scale = float(sensitivity) / float(epsilon)

    # Each bin receives independent noise.  After adding noise the counts
    # may be negative or exceed n; clipping and renormalisation handle this.
    noise = rng.laplace(loc=0.0, scale=scale, size=k)
    noisy_counts = counts + noise

    # Convert absolute counts to proportions before optional clipping.
    theta_hat = noisy_counts / n

    if clip:
        theta_hat = np.clip(theta_hat, 0.0, None)

    if renormalize:
        s = float(np.sum(theta_hat))
        theta_hat = np.full(k, 1.0 / k, dtype=float) if s <= 0 else theta_hat / s

    # The cast ensures a consistent float dtype for downstream comparisons.
    return theta_hat.astype(float)


def estimate_distribution_central_dp(
    true_categories: ArrayLike,
    epsilon: float,
    k: int,
    *,
    sensitivity: float = 2.0,
    rng: np.random.Generator | None = None,
    clip: bool = True,
    renormalize: bool = True,
) -> np.ndarray:
    """
    Convenience wrapper: estimate the category distribution under central DP.

    This simulates a trusted collector who:
      1. Receives raw category labels from all respondents.
      2. Computes exact counts per category.
      3. Adds Laplace noise (calibrated to epsilon) before publishing.

    Args:
      true_categories: 1D array of integer category labels in [0, k-1].
      epsilon, k, sensitivity, rng, clip, renormalize: see laplace_mechanism().

    Returns:
      Estimated distribution as a length-k float array summing to 1.
    """
    cats = np.asarray(true_categories, dtype=int)
    if cats.ndim != 1:
        raise ValueError("true_categories must be a 1D array of ints.")
    if cats.size == 0:
        raise ValueError("true_categories must be non-empty.")

    if not isinstance(k, int) or k < 2:
        raise ValueError("k must be an int >= 2.")
    if np.any((cats < 0) | (cats >= k)):
        raise ValueError(f"All categories must be in [0, {k - 1}].")

    # Aggregate raw category labels into per-bin counts before applying
    # Laplace noise.  This is the step where a trusted collector sees the
    # exact counts, unlike LDP which randomises per respondent.
    counts = np.bincount(cats, minlength=k).astype(float)
    return laplace_mechanism(
        counts,
        epsilon,
        k,
        sensitivity=sensitivity,
        rng=rng,
        clip=clip,
        renormalize=renormalize,
    )


def _validate_inputs(epsilon: float, k: int, sensitivity: float) -> None:
    if not isinstance(k, int) or k < 2:
        raise ValueError("k must be an int >= 2.")
    if not isinstance(epsilon, (int, float)) or not math.isfinite(float(epsilon)):
        raise ValueError("epsilon must be a finite number.")
    if float(epsilon) <= 0.0:
        raise ValueError("epsilon must be > 0.")
    if not isinstance(sensitivity, (int, float)) or float(sensitivity) <= 0.0:
        raise ValueError("sensitivity must be > 0.")
