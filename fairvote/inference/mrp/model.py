# fairvote/inference/mrp/model.py
"""Linear RR-aware MRP utilities.

This module provides a simple one-hot design matrix and multinomial model for
MRP-style inference under k-ary Randomized Response. The fitted model accounts
for the RR observation process rather than treating reported labels as true
answers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fairvote.privacy.mechanisms.kary_rr import rr_params


# =============================================================================
# One-hot encoder for categorical features
# =============================================================================

@dataclass(frozen=True)
class DesignInfo:
    feature_names: List[str]
    feature_levels: Dict[str, List[str]]
    col_slices: Dict[str, slice]
    n_cols: int
    has_intercept: bool


def build_design_matrix(
    features: Dict[str, np.ndarray],
    feature_levels: Dict[str, List[str]],
    *,
    feature_order: Optional[Sequence[str]] = None,
    intercept: bool = True,
) -> Tuple[np.ndarray, DesignInfo]:
    """
    Build a one-hot design matrix X from integer-coded categorical features.

    Args:
      features: dict feat -> np.ndarray shape (n,) of integer codes
      feature_levels: dict feat -> list of level names (defines number of levels)
      feature_order: optional order of features in the matrix
      intercept: include intercept column

    Returns:
      X: (n, d) float32
      info: DesignInfo (for reproducible encoding + poststrat)
    """
    if not features:
        raise ValueError("features cannot be empty.")
    n = None
    for k, v in features.items():
        arr = np.asarray(v)
        if arr.ndim != 1:
            raise ValueError(f"Feature '{k}' must be 1D.")
        if n is None:
            n = arr.size
        elif arr.size != n:
            raise ValueError("All features must have the same length.")

    if n is None or n <= 0:
        raise ValueError("No rows found.")

    # Determine feature order
    if feature_order is None:
        # Stable ordering for reproducibility
        feature_order = sorted(features.keys())
    else:
        feature_order = list(feature_order)

    # Validate features exist + levels exist
    for f in feature_order:
        if f not in features:
            raise KeyError(f"Missing feature '{f}' in features dict.")
        if f not in feature_levels:
            raise KeyError(f"Missing feature '{f}' in feature_levels dict.")
        L = len(feature_levels[f])
        if L < 2:
            raise ValueError(f"Feature '{f}' must have >=2 levels to be useful (got {L}).")

        x = np.asarray(features[f], dtype=int)
        if np.any((x < 0) | (x >= L)):
            raise ValueError(f"Feature '{f}' has values outside [0, {L-1}].")

    # Compute total columns
    d = (1 if intercept else 0) + sum(len(feature_levels[f]) for f in feature_order)

    # The design matrix is pre-allocated as zeros. One-hot columns are then
    # set per row.  float32 is sufficient for the downstream linear algebra.
    X = np.zeros((n, d), dtype=np.float32)
    col = 0
    if intercept:
        # The intercept column ensures the model can learn a global baseline
        # preference before feature-specific adjustments.
        X[:, 0] = 1.0
        col = 1

    col_slices: Dict[str, slice] = {}
    for f in feature_order:
        L = len(feature_levels[f])
        sl = slice(col, col + L)
        col_slices[f] = sl

        x = np.asarray(features[f], dtype=int)
        X[np.arange(n), col + x] = 1.0
        col += L

    info = DesignInfo(
        feature_names=list(feature_order),
        feature_levels={f: list(feature_levels[f]) for f in feature_order},
        col_slices=col_slices,
        n_cols=d,
        has_intercept=intercept,
    )
    return X, info


# =============================================================================
# Privacy-aware multinomial model fitted to RR-reported labels
# =============================================================================

# =============================================================================
# Row-wise softmax  (shared by the linear model)
# =============================================================================

def _softmax(z: np.ndarray) -> np.ndarray:
    # Subtract the row max for numerical stability; this does not change the
    # resulting probability vector.
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


@dataclass
class AdamState:
    m: np.ndarray
    v: np.ndarray
    t: int


class RRMultinomialModel:
    """
    Multinomial preference model trained from k-ary RR reports.

    Model:
      theta(x) = softmax(XW)  where W has shape (d, K)

    Observation under k-ary RR:
      P(report=j | x) = q + (p - q) * theta_j(x)

    We fit W by minimising:
      NLL(W) + 0.5 * l2 * ||W||^2   (optionally excluding intercept row)
    """

    def __init__(
        self,
        k: int,
        *,
        l2: float = 1.0,
        exclude_intercept_from_l2: bool = True,
        seed: int = 123,
    ):
        if not isinstance(k, int) or k < 2:
            raise ValueError("k must be an int >= 2.")
        if l2 < 0:
            raise ValueError("l2 must be >= 0.")
        self.k = k
        self.l2 = float(l2)
        self.exclude_intercept_from_l2 = bool(exclude_intercept_from_l2)
        self.rng = np.random.default_rng(seed)

        self.W: Optional[np.ndarray] = None
        self.design_info: Optional[DesignInfo] = None
        self.epsilon: Optional[float] = None

    def fit(
        self,
        X: np.ndarray,
        y_reported: np.ndarray,
        epsilon: float,
        *,
        lr: float = 0.05,
        steps: int = 1500,
        batch_size: int = 2048,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps_adam: float = 1e-8,
        verbose_every: int = 200,
    ) -> "RRMultinomialModel":
        """
        Fit model parameters W.

        Args:
          X: design matrix (n, d)
          y_reported: RR-reported labels (n,) in [0..K-1]
          epsilon: LDP epsilon used for RR
          lr: Adam learning rate
          steps: optimisation steps
          batch_size: minibatch size (use n for full-batch)
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y_reported, dtype=int)

        if X.ndim != 2:
            raise ValueError("X must be 2D.")
        n, d = X.shape
        if y.ndim != 1 or y.size != n:
            raise ValueError("y_reported must be 1D and same length as X.")
        if np.any((y < 0) | (y >= self.k)):
            raise ValueError(f"y_reported values must be in [0, {self.k-1}].")
        if steps <= 0:
            raise ValueError("steps must be > 0.")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0.")

        params = rr_params(epsilon, self.k)
        p, q = params.p, params.q
        if not (q < p):
            raise RuntimeError("RR parameters invalid (expected q < p).")

        # Initialise W with small random values to break symmetry; large values
        # would saturate the softmax and slow early training.
        W = self.rng.normal(0.0, 0.01, size=(d, self.k)).astype(np.float32)

        # Adam state
        state = AdamState(
            m=np.zeros_like(W),
            v=np.zeros_like(W),
            t=0,
        )

        # L2 mask prevents the intercept row from being regularised.  This is
        # standard practice: penalising the global mean would bias the baseline
        # preference estimate.
        l2_mask = np.ones((d, self.k), dtype=np.float32)
        if self.exclude_intercept_from_l2:
            # assume intercept is column 0 in X if present
            # (we don't strictly know; user controls that)
            # safest: don't regularise row 0
            l2_mask[0, :] = 0.0

        for step in range(1, steps + 1):
            state.t += 1
            idx = self._batch_indices(n, batch_size)
            Xb = X[idx]
            yb = y[idx]

            # Forward
            logits = Xb @ W  # (b, K)
            theta = _softmax(logits)  # (b, K)

            # RR-aware observation probability for each sample in the batch:
            # P(report = y_i | x_i) = q + (p - q) * theta_i[y_i].
            # This is the key equation linking the model to the privacy channel.
            theta_y = theta[np.arange(theta.shape[0]), yb]  # (b,)
            prob = q + (p - q) * theta_y
            # Clamp to avoid log(0) in the NLL computation.
            prob = np.clip(prob, 1e-12, 1.0)

            # Negative log-likelihood averaged over the batch.
            nll = -np.mean(np.log(prob))

            # Gradient derivation:
            # log prob = log(q + (p-q)*theta_y)
            # d log prob / d logits = a * d theta_y / d logits
            # where a = (p-q) / (q + (p-q)*theta_y)
            # and d theta_y / d logits = theta_y * (e_y - theta)
            a = (p - q) / prob  # (b,)
            # g_logits = a[:,None] * theta_y[:,None] * (onehot(y) - theta)
            g_logits = -theta  # start with (-theta)
            g_logits[np.arange(theta.shape[0]), yb] += 1.0  # (e_y - theta)
            g_logits *= (a * theta_y)[:, None]  # scale per row

            # We computed gradient of log prob; NLL gradient is negative mean of that:
            # d/dW NLL = -(1/b) X^T g_logits
            grad = -(Xb.T @ g_logits) / float(Xb.shape[0])  # (d, K)

            # L2 regularisation term applied only to non-intercept rows.
            if self.l2 > 0.0:
                grad += self.l2 * W * l2_mask

            # Bias-corrected Adam update following Kingma & Ba (2015).
            state.m = beta1 * state.m + (1.0 - beta1) * grad
            state.v = beta2 * state.v + (1.0 - beta2) * (grad * grad)

            m_hat = state.m / (1.0 - beta1 ** state.t)
            v_hat = state.v / (1.0 - beta2 ** state.t)

            W = W - lr * m_hat / (np.sqrt(v_hat) + eps_adam)

            if verbose_every and (step % verbose_every == 0 or step == 1 or step == steps):
                # report also mean prob for sanity
                mean_prob = float(np.mean(prob))
                print(f"[RR-MRP] step {step:4d}/{steps} | nll={nll:.6f} | mean P(report|x)={mean_prob:.4f}")

        self.W = W
        self.epsilon = float(epsilon)
        return self

    def predict_theta(self, X: np.ndarray) -> np.ndarray:
        """
        Predict theta(x) = P(true category | x) for each row.

        Returns (n, K) matrix.
        """
        if self.W is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X = np.asarray(X, dtype=np.float32)
        logits = X @ self.W
        return _softmax(logits)

    def predict_report_distribution(self, X: np.ndarray) -> np.ndarray:
        """
        Predict P(report=j | x) for each row given fitted epsilon.

        Returns (n, K) matrix.
        """
        if self.W is None or self.epsilon is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        # Map latent theta back to the observed report space using the known
        # RR channel parameters.  This is useful for diagnostic checks but
        # not for aggregate estimation.
        theta = self.predict_theta(X)
        params = rr_params(self.epsilon, self.k)
        p, q = params.p, params.q
        return q + (p - q) * theta

    def _batch_indices(self, n: int, batch_size: int) -> np.ndarray:
        if batch_size >= n:
            return np.arange(n, dtype=int)
        return self.rng.integers(0, n, size=batch_size, dtype=int)
