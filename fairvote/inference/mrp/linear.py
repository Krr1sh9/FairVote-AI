"""Canonical linear RR-aware MRP estimator.

This is the one authoritative implementation of the repository's *linear*
Randomized-Response-aware MRP path.  The model is honestly a regularised
multinomial regression plus post-stratification workflow.  It is MRP-style, but
it is not a full hierarchical Bayesian MRP sampler.

Latent model:
    P(true=t | x) = softmax(XW)[t]

Observation model:
    P(reported=r | x) = sum_t P(true=t | x) * A[t, r]

where A is the canonical k-ary Randomized Response channel from
:mod:`fairvote.privacy.mechanisms.kary_rr`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fairvote.inference.mrp.design import DesignInfo, DesignMatrix
from fairvote.inference.mrp.diagnostics import FitDiagnostics
from fairvote.inference.mrp.likelihood import reported_label_likelihood, softmax_rows
from fairvote.privacy.mechanisms.kary_rr import rr_transition_matrix


def validate_design_matrix(X: np.ndarray, *, name: str = "X") -> np.ndarray:
    """Validate a numeric 2-D design matrix."""
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if arr.shape[0] <= 0:
        raise ValueError(f"{name} has zero rows")
    if arr.shape[1] <= 0:
        raise ValueError(f"{name} has zero columns")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def validate_reported_labels(
    y_reported: Sequence[int] | np.ndarray, *, k: int, expected_n: int | None = None
) -> np.ndarray:
    """Validate RR-reported category labels."""
    if not isinstance(k, int) or k < 2:
        raise ValueError("k must be an int >= 2")
    raw = np.asarray(y_reported)
    if raw.ndim != 1:
        raw = raw.reshape(-1)
    if raw.size == 0:
        raise ValueError("y_reported is empty")
    if expected_n is not None and raw.size != int(expected_n):
        raise ValueError("y_reported must have one entry per row of X")
    if np.issubdtype(raw.dtype, np.floating):
        if not np.all(np.isfinite(raw)):
            raise ValueError("y_reported contains NaN or infinite values")
        if not np.all(np.equal(raw, np.floor(raw))):
            raise ValueError("y_reported must contain integer category ids")
    y = raw.astype(int, copy=False).reshape(-1)
    if np.any((y < 0) | (y >= k)):
        raise ValueError(f"y_reported values must be in [0, {k - 1}]")
    return y


def validate_weights(weights: Sequence[float] | np.ndarray, *, expected_n: int) -> np.ndarray:
    """Validate and normalise post-stratification weights."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.size != int(expected_n):
        raise ValueError("weights must have one entry per population row")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights contains NaN or infinite values")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    total = float(np.sum(w))
    if total <= 0.0:
        raise ValueError("weights must have positive sum")
    return w / total


class LinearRRMRPModel:
    """Regularised multinomial regression fitted through an RR observation channel.

    Parameters
    ----------
    k:
        Number of response categories.
    epsilon:
        Optional RR epsilon.  If omitted, pass ``epsilon`` to :meth:`fit`.
    l2:
        L2 regularisation strength.
    exclude_intercept_from_l2:
        Whether the first coefficient row is excluded from regularisation.
    seed:
        RNG seed for initialisation and mini-batching.
    """

    model_name = "linear_rr_mrp_regularized_multinomial"

    def __init__(
        self,
        k: int,
        *,
        epsilon: float | None = None,
        l2: float = 1.0,
        exclude_intercept_from_l2: bool = False,
        seed: int = 0,
    ) -> None:
        if not isinstance(k, int) or k < 2:
            raise ValueError("k must be an int >= 2")
        if l2 < 0:
            raise ValueError("l2 must be >= 0")
        self.k = int(k)
        self.epsilon = None if epsilon is None else float(epsilon)
        self.l2 = float(l2)
        self.exclude_intercept_from_l2 = bool(exclude_intercept_from_l2)
        self.seed = int(seed)
        self.W: np.ndarray | None = None
        self.A: np.ndarray | None = None
        self.fit_diagnostics: FitDiagnostics | None = None
        self.design_info: DesignInfo | DesignMatrix | None = None
        if self.epsilon is not None:
            self.A = rr_transition_matrix(self.epsilon, self.k)

    def _init_weights(self, d: int) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        return rng.normal(0.0, 0.01, size=(d, self.k)).astype(float)

    def _l2_mask(self, d: int) -> np.ndarray:
        mask = np.ones((d, self.k), dtype=float)
        if self.exclude_intercept_from_l2 and d > 0:
            mask[0, :] = 0.0
        return mask

    def fit(
        self,
        X: np.ndarray,
        y_reported: Sequence[int] | np.ndarray,
        epsilon: float | None = None,
        *,
        lr: float = 0.05,
        steps: int = 2000,
        batch_size: int = 512,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps_adam: float = 1e-8,
        verbose_every: int = 0,
        keep_history: bool = False,
        history_every: int = 1,
    ) -> FitDiagnostics:
        """Fit the model using only RR-reported labels.

        No true labels are accepted by this method.  The likelihood is the
        marginal probability of the observed *reported* label after applying the
        RR transition matrix.
        """
        X_arr = validate_design_matrix(X)
        n, d = X_arr.shape
        y = validate_reported_labels(y_reported, k=self.k, expected_n=n)
        if epsilon is not None:
            self.epsilon = float(epsilon)
        if self.epsilon is None:
            raise ValueError("epsilon must be supplied either at construction or fit time")
        if lr <= 0.0:
            raise ValueError("lr must be > 0")
        if steps <= 0:
            raise ValueError("steps must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if not (0.0 < beta1 < 1.0) or not (0.0 < beta2 < 1.0):
            raise ValueError("beta1 and beta2 must be in (0, 1)")
        if eps_adam <= 0.0:
            raise ValueError("eps_adam must be > 0")
        if history_every <= 0:
            raise ValueError("history_every must be > 0")

        A = rr_transition_matrix(self.epsilon, self.k)
        self.A = A
        W = self._init_weights(d)
        m = np.zeros_like(W)
        v = np.zeros_like(W)
        l2_mask = self._l2_mask(d)
        rng = np.random.default_rng(self.seed)
        history: list[float] = []
        started = time.perf_counter()

        for step in range(1, int(steps) + 1):
            idx = rng.integers(0, n, size=min(int(batch_size), n))
            Xb = X_arr[idx]
            yb = y[idx]

            theta = softmax_rows(Xb @ W)
            likelihood = reported_label_likelihood(theta, A, yb)
            batch_nll = likelihood.nll
            grad_W = Xb.T @ likelihood.grad_logits
            if self.l2 > 0.0:
                grad_W = grad_W + self.l2 * W * l2_mask

            m = beta1 * m + (1.0 - beta1) * grad_W
            v = beta2 * v + (1.0 - beta2) * (grad_W * grad_W)
            m_hat = m / (1.0 - beta1**step)
            v_hat = v / (1.0 - beta2**step)
            W = W - float(lr) * m_hat / (np.sqrt(v_hat) + float(eps_adam))

            if keep_history and (step == 1 or step == steps or step % int(history_every) == 0):
                history.append(batch_nll)
            if verbose_every and (step == 1 or step == steps or step % int(verbose_every) == 0):
                print(f"[linear-rr-mrp] step={step} batch_nll={batch_nll:.6f}")

        self.W = W
        final_loss = self.loss(X_arr, y)
        diagnostics = FitDiagnostics(
            steps=int(steps),
            final_loss=float(final_loss),
            runtime_sec=float(time.perf_counter() - started),
            history=np.asarray(history, dtype=float) if keep_history else None,
        )
        self.fit_diagnostics = diagnostics
        return diagnostics

    def predict_theta(self, X: np.ndarray) -> np.ndarray:
        """Predict latent true-category probabilities."""
        if self.W is None:
            raise RuntimeError("Model is not fitted")
        X_arr = validate_design_matrix(X)
        if X_arr.shape[1] != self.W.shape[0]:
            raise ValueError("X column count does not match fitted model")
        return softmax_rows(X_arr @ self.W)

    def predict_true_proba(self, X: np.ndarray) -> np.ndarray:
        """Alias for :meth:`predict_theta` used by dashboard code."""
        return self.predict_theta(X)

    def predict_reported_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict reported-label probabilities after the RR channel."""
        if self.A is None:
            if self.epsilon is None:
                raise RuntimeError("Model does not have an RR epsilon")
            self.A = rr_transition_matrix(self.epsilon, self.k)
        return self.predict_theta(X) @ self.A

    def predict_report_distribution(self, X: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for reported-label predictions."""
        return self.predict_reported_proba(X)

    def loss(self, X: np.ndarray, y_reported: Sequence[int] | np.ndarray) -> float:
        """Full-data RR-aware negative log likelihood plus L2 penalty."""
        if self.W is None:
            raise RuntimeError("Model is not fitted")
        X_arr = validate_design_matrix(X)
        y = validate_reported_labels(y_reported, k=self.k, expected_n=X_arr.shape[0])
        reported_probs = self.predict_reported_proba(X_arr)
        observed = np.clip(reported_probs[np.arange(reported_probs.shape[0]), y], 1e-12, 1.0)
        nll = -float(np.mean(np.log(observed)))
        reg = 0.0
        if self.l2 > 0.0:
            mask = self._l2_mask(self.W.shape[0])
            reg = 0.5 * self.l2 * float(np.sum((self.W * mask) * self.W))
        return nll + reg

    def poststratify(self, X_pop: np.ndarray, weights: Sequence[float] | np.ndarray) -> np.ndarray:
        """Return population-weighted latent true-category probabilities."""
        X_arr = validate_design_matrix(X_pop, name="X_pop")
        w = validate_weights(weights, expected_n=X_arr.shape[0])
        probs = self.predict_theta(X_arr)
        estimate = (w[:, None] * probs).sum(axis=0)
        estimate = np.clip(estimate, 0.0, 1.0)
        total = float(np.sum(estimate))
        if total <= 0.0:
            raise RuntimeError("poststratified probabilities sum to zero")
        return estimate / total

    def export_metadata(self, *, include_weights: bool = False) -> dict[str, Any]:
        """Return fitted-model metadata suitable for JSON export.

        ``include_weights`` is false by default to keep metadata lightweight, but
        can be enabled for exact reproducibility of fitted coefficients.
        """
        metadata: dict[str, Any] = {
            "model_name": self.model_name,
            "honest_description": "regularised multinomial regression fitted through a k-ary RR observation channel; not full Bayesian hierarchical MRP",
            "k": self.k,
            "epsilon": self.epsilon,
            "l2": self.l2,
            "exclude_intercept_from_l2": self.exclude_intercept_from_l2,
            "seed": self.seed,
            "is_fitted": self.W is not None,
            "coefficient_shape": None if self.W is None else list(self.W.shape),
            "coefficient_l2_norm": None if self.W is None else float(np.linalg.norm(self.W)),
            "fit_diagnostics": None if self.fit_diagnostics is None else self.fit_diagnostics.to_jsonable(),
        }
        if self.design_info is not None and hasattr(self.design_info, "to_jsonable"):
            metadata["design_info"] = self.design_info.to_jsonable()
        if include_weights and self.W is not None:
            metadata["weights"] = self.W.tolist()
        return metadata

    def save_metadata(self, path: str | Path, *, include_weights: bool = False) -> None:
        """Write :meth:`export_metadata` output as JSON."""
        Path(path).write_text(
            json.dumps(self.export_metadata(include_weights=include_weights), indent=2), encoding="utf-8"
        )


# Backwards-compatible public names.  Both point to the same canonical class.
RRMultinomialModel = LinearRRMRPModel
MRPRRMultinomialModel = LinearRRMRPModel
