"""Shared likelihood utilities for RR-aware MRP estimators.

The linear, hierarchical, misreport-aware, and learned-misreport models all use
one mathematical core: latent class probabilities are transformed through an
observation channel and optimised by the marginal likelihood of the reported
labels.  Centralising this code reduces duplicated gradient logic and makes the
privacy/noise likelihood auditable.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObservationLikelihood:
    """Mini-batch likelihood and gradient through the softmax logits."""

    nll: float
    grad_logits: np.ndarray
    reported_probs: np.ndarray
    observed_probs: np.ndarray


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    z = np.asarray(logits, dtype=float)
    if z.ndim != 2:
        raise ValueError("logits must be a 2D array")
    if not np.all(np.isfinite(z)):
        raise ValueError("logits contains NaN or infinite values")
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    denom = np.sum(exp_z, axis=1, keepdims=True)
    return exp_z / denom


def validate_observation_channel(channel: np.ndarray, *, k: int | None = None) -> np.ndarray:
    """Validate a non-negative row-stochastic observation channel."""
    C = np.asarray(channel, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("observation channel must be a square 2D array")
    if k is not None and C.shape != (int(k), int(k)):
        raise ValueError(f"observation channel must have shape ({k}, {k})")
    if not np.all(np.isfinite(C)):
        raise ValueError("observation channel contains NaN or infinite values")
    if np.any(C < -1e-12):
        raise ValueError("observation channel contains negative probabilities")
    C = np.clip(C, 0.0, None)
    row_sums = C.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("observation channel has an empty row")
    return C / row_sums


def reported_label_likelihood(
    theta: np.ndarray,
    channel: np.ndarray,
    y_reported: Sequence[int] | np.ndarray,
    *,
    average_gradient: bool = True,
    clip_min: float = 1e-12,
) -> ObservationLikelihood:
    """Return NLL and softmax-logit gradient for reported labels.

    Parameters
    ----------
    theta:
        Row-wise latent true-category probabilities, shape ``(n, k)``.
    channel:
        Observation channel ``P(reported=r | true=t)``, shape ``(k, k)``.
    y_reported:
        Observed reported label ids.
    average_gradient:
        If true, gradient is scaled by ``1/n`` so callers can use
        ``X.T @ grad_logits`` directly.  If false, callers can divide later.
    """
    theta_arr = np.asarray(theta, dtype=float)
    if theta_arr.ndim != 2:
        raise ValueError("theta must be a 2D array")
    n, k = theta_arr.shape
    C = validate_observation_channel(channel, k=k)
    y = np.asarray(y_reported).reshape(-1).astype(int, copy=False)
    if y.size != n:
        raise ValueError("y_reported must have one entry per theta row")
    if np.any((y < 0) | (y >= k)):
        raise ValueError(f"y_reported values must be in [0, {k - 1}]")

    reported_probs = theta_arr @ C
    observed = np.clip(reported_probs[np.arange(n), y], float(clip_min), 1.0)
    nll = -float(np.mean(np.log(observed)))

    scale = float(n) if average_gradient else 1.0
    grad_reported = np.zeros_like(reported_probs)
    grad_reported[np.arange(n), y] = -1.0 / (scale * observed)
    grad_theta = grad_reported @ C.T
    row_dot = np.sum(grad_theta * theta_arr, axis=1, keepdims=True)
    grad_logits = theta_arr * (grad_theta - row_dot)
    return ObservationLikelihood(
        nll=nll, grad_logits=grad_logits, reported_probs=reported_probs, observed_probs=observed
    )
