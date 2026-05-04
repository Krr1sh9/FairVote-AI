# fairvote/inference/mrp/misreport_rr.py
"""Misreport-aware RR-MRP models.

These models separate behavioural misreporting from Randomized Response noise by
using an additional transition from true preference to stated preference before
the RR channel is applied. They are baselines for scenarios such as shy-voter
behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def rr_transition_matrix(eps: float, k: int) -> np.ndarray:
    """
    k-ary randomized response transition matrix A where:
      A[s, r] = P(reported=r | stated=s, eps)

    Standard k-ary RR:
      P(keep)   = exp(eps) / (exp(eps) + k - 1)
      P(other)  = 1 / (exp(eps) + k - 1), uniformly among the other k-1 categories
    """
    if k <= 1:
        raise ValueError("k must be >= 2")
    if eps <= 0:
        raise ValueError("eps must be > 0")

    ee = float(np.exp(eps))
    p_keep = ee / (ee + (k - 1))
    p_other = 1.0 / (ee + (k - 1))

    A = np.full((k, k), p_other, dtype=float)
    np.fill_diagonal(A, p_keep)
    return A


def validate_row_stochastic(M: np.ndarray, *, atol: float = 1e-6) -> None:
    """Validate that a transition matrix can represent probabilities."""
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("misreport matrix must be square (k x k)")
    if np.any(M < -atol):
        raise ValueError("misreport matrix has negative entries")
    row_sums = M.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError("misreport matrix rows must sum to 1")


def identity_misreport(k: int) -> np.ndarray:
    """Return the no-misreporting channel used as a neutral baseline."""
    return np.eye(k, dtype=float)


def shy_misreport_matrix(k: int, shy_category: int, honesty: float) -> np.ndarray:
    """
    Simple 'shy voter' misreport model:
      - For true category == shy_category: state truth with prob=honesty,
        otherwise uniformly pick one of the other categories.
      - For other true categories: always state truth.

    Returns M[t, s] = P(stated=s | true=t).
    """
    if not (0 <= shy_category < k):
        raise ValueError("shy_category out of range")
    if not (0.0 <= honesty <= 1.0):
        raise ValueError("honesty must be in [0, 1]")

    M = np.eye(k, dtype=float)
    if k == 1:
        return M

    off = (1.0 - honesty) / (k - 1)
    M[shy_category, :] = off
    M[shy_category, shy_category] = honesty
    validate_row_stochastic(M)
    return M


def softmax_rows(Z: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax for multinomial logits."""
    Z = np.asarray(Z, dtype=float)
    Z = Z - np.max(Z, axis=1, keepdims=True)
    E = np.exp(Z)
    return E / np.sum(E, axis=1, keepdims=True)


@dataclass
class MisreportRRMultinomialModel:
    """
    Multinomial logistic regression for latent TRUE categories (theta),
    with an observation model that includes:
      TRUE -> STATED (misreport matrix M)
      STATED -> REPORTED (RR matrix A(eps))

    Observed likelihood:
      P(reported=r | x) = sum_t theta_t(x) * C[t, r]
    where C = M @ A is the composite channel TRUE -> REPORTED.

    API matches RRMultinomialModel:
      - fit(X, reported, eps, lr, steps, batch_size, verbose_every)
      - predict_theta(X) -> (n, k)
    """
    k: int
    l2: float = 1.0
    seed: int = 0
    misreport: Optional[np.ndarray] = None  # (k, k) row-stochastic

    def __post_init__(self) -> None:
        if self.k <= 1:
            raise ValueError("k must be >= 2")
        self.rng = np.random.default_rng(self.seed)
        self.W: Optional[np.ndarray] = None

        if self.misreport is None:
            self.M = identity_misreport(self.k)
        else:
            self.M = np.asarray(self.misreport, dtype=float)
            validate_row_stochastic(self.M)

    def _composite_channel(self, eps: float) -> np.ndarray:
        A = rr_transition_matrix(eps, self.k)
        # The composite channel multiplies the misreport transition (TRUE→STATED)
        # by the RR channel (STATED→REPORTED), giving a single matrix for the
        # full observation model.
        C = self.M @ A  # TRUE → STATED → REPORTED
        # Numerical hygiene: clip negative entries from floating-point drift
        # and re-normalise rows to maintain a valid probability channel.
        C = np.clip(C, 0.0, None)
        row_sums = C.sum(axis=1, keepdims=True)
        C = C / np.maximum(row_sums, 1e-12)
        return C

    def fit(
        self,
        X: np.ndarray,
        reported: np.ndarray,
        eps: float,
        *,
        lr: float = 0.05,
        steps: int = 1200,
        batch_size: int = 2048,
        verbose_every: int = 0,
    ) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(reported, dtype=int)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        n, d = X.shape
        if y.shape[0] != n:
            raise ValueError("reported must have same length as X rows")
        if np.any((y < 0) | (y >= self.k)):
            raise ValueError("reported contains out-of-range category ids")

        C = self._composite_channel(eps)  # (k, k)

        # Initialise weights small
        self.W = 0.01 * self.rng.standard_normal((d, self.k))

        # Training loop (mini-batch SGD)
        for step in range(1, steps + 1):
            idx = self.rng.integers(0, n, size=min(batch_size, n))
            Xb = X[idx]                     # (b, d)
            yb = y[idx]                     # (b,)

            logits = Xb @ self.W            # (b, k)
            theta = softmax_rows(logits)    # (b, k)

            # Gather composite channel column for each observed y:
            # c_y[b, t] = C[t, y_b]
            c_y = C[:, yb].T                # (b, k)

            # p_i = sum_t theta_it * C[t, y_i]
            p = np.sum(theta * c_y, axis=1)  # (b,)
            p = np.clip(p, 1e-12, None)

            # dL/dtheta = -(c_y / p)
            g_theta = -(c_y / p[:, None])   # (b, k)

            # softmax backprop: g_logits = theta * (g_theta - sum(theta*g_theta))
            s = np.sum(theta * g_theta, axis=1)           # (b,)
            g_logits = theta * (g_theta - s[:, None])     # (b, k)

            # Gradient for W
            grad = (Xb.T @ g_logits) / Xb.shape[0]        # (d, k)
            grad += self.l2 * self.W

            self.W -= lr * grad

            if verbose_every and (step % verbose_every == 0 or step == 1 or step == steps):
                # approximate batch NLL
                nll = float(np.mean(-np.log(p)))
                print(f"[misreport-mrp] step {step:5d}/{steps}  batch_nll={nll:.4f}")

    def predict_theta(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("Model is not fit yet.")
        X = np.asarray(X, dtype=float)
        logits = X @ self.W
        return softmax_rows(logits)
