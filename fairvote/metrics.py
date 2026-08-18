from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# ArrayLike allows the metric functions to accept either ordinary numeric sequences or NumPy arrays.
ArrayLike = Sequence[float] | np.ndarray


def l1_error(estimate: ArrayLike, truth: ArrayLike) -> float:
    # L1 error is the sum of the absolute category-wise differences between the estimate and the truth.
    return float(np.abs(_error_vector(estimate, truth)).sum())


def max_absolute_error(estimate: ArrayLike, truth: ArrayLike) -> float:
    # Maximum absolute error is the largest absolute difference in any single category.
    return float(np.abs(_error_vector(estimate, truth)).max())


def _error_vector(estimate: ArrayLike, truth: ArrayLike) -> np.ndarray:
    # Both inputs are converted to floating-point arrays before their shapes and values are validated.
    estimate_array = np.asarray(estimate, dtype=float)
    truth_array = np.asarray(truth, dtype=float)

    # The metrics compare one-dimensional category vectors only.
    if estimate_array.ndim != 1 or truth_array.ndim != 1:
        raise ValueError("estimate and truth must be 1D arrays.")

    # The implementation requires the estimate and truth to have the same shape.
    if estimate_array.shape != truth_array.shape:
        raise ValueError(
            f"estimate and truth must have the same shape, got {estimate_array.shape} and {truth_array.shape}."
        )

    # Empty vectors are rejected because the maximum absolute error is undefined without any categories.
    if estimate_array.size == 0:
        raise ValueError("estimate and truth must be non-empty.")

    # Non-finite input values are rejected before the error vector is calculated.
    if not np.all(np.isfinite(estimate_array)) or not np.all(np.isfinite(truth_array)):
        raise ValueError("estimate and truth must be finite.")

    # The signed category-wise differences are shared by both public metric functions.
    return estimate_array - truth_array
