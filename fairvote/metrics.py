from __future__ import annotations

from collections.abc import Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


def l1_error(estimate: ArrayLike, truth: ArrayLike) -> float:
    return float(np.abs(_error_vector(estimate, truth)).sum())


def max_absolute_error(estimate: ArrayLike, truth: ArrayLike) -> float:
    return float(np.abs(_error_vector(estimate, truth)).max())


def _error_vector(estimate: ArrayLike, truth: ArrayLike) -> np.ndarray:
    estimate_array = np.asarray(estimate, dtype=float)
    truth_array = np.asarray(truth, dtype=float)

    if estimate_array.ndim != 1 or truth_array.ndim != 1:
        raise ValueError("estimate and truth must be 1D arrays.")
    if estimate_array.shape != truth_array.shape:
        raise ValueError(
            f"estimate and truth must have the same shape, got {estimate_array.shape} and {truth_array.shape}."
        )
    if estimate_array.size == 0:
        raise ValueError("estimate and truth must be non-empty.")
    if not np.all(np.isfinite(estimate_array)) or not np.all(np.isfinite(truth_array)):
        raise ValueError("estimate and truth must be finite.")

    return estimate_array - truth_array
