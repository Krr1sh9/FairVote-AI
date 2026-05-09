"""Estimators for recovering aggregate distributions from privatized reports.

This module is intentionally thin: the Randomized Response inverse lives in the
canonical channel module, :mod:`fairvote.privacy.mechanisms.kary_rr`.  These
wrappers preserve the public API used by older scripts while avoiding duplicate
RR algebra.
"""

from __future__ import annotations

import numpy as np

from fairvote.privacy.mechanisms.kary_rr import debias_distribution, invert_rr_counts

ArrayLike = np.ndarray | list


def estimate_distribution_from_counts(
    counts: ArrayLike,
    epsilon: float,
    k: int,
    *,
    clip: bool = True,
    renormalize: bool = True,
) -> np.ndarray:
    """Estimate the latent true distribution from RR reported counts."""

    return invert_rr_counts(counts, epsilon, k, clip=clip, renormalize=renormalize)


def estimate_distribution(
    reported_categories: ArrayLike,
    epsilon: float,
    k: int,
    *,
    clip: bool = True,
    renormalize: bool = True,
) -> np.ndarray:
    """Estimate the latent true distribution from raw RR reported labels."""

    return debias_distribution(
        reported_categories,
        epsilon,
        k,
        clip=clip,
        renormalize=renormalize,
    )


def bootstrap_ci(
    reported_categories: ArrayLike,
    epsilon: float,
    k: int,
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    clip: bool = True,
    renormalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile bootstrap confidence intervals for each category proportion."""

    reps = np.asarray(reported_categories, dtype=int)
    if reps.ndim != 1 or reps.size == 0:
        raise ValueError("reported_categories must be a non-empty 1D array-like.")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if not isinstance(n_boot, int) or n_boot < 200:
        raise ValueError("n_boot must be an int >= 200 for a stable CI.")

    generator = rng if rng is not None else np.random.default_rng()
    n = int(reps.size)
    boot = np.empty((n_boot, int(k)), dtype=float)

    for b in range(n_boot):
        idx = generator.integers(0, n, size=n, dtype=int)
        sample = reps[idx]
        boot[b] = estimate_distribution(
            sample,
            epsilon,
            k,
            clip=clip,
            renormalize=renormalize,
        )

    lo_q = alpha / 2.0
    hi_q = 1.0 - alpha / 2.0
    lower = np.quantile(boot, lo_q, axis=0)
    upper = np.quantile(boot, hi_q, axis=0)
    return lower, upper
