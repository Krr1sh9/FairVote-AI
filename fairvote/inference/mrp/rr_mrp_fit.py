# fairvote/inference/mrp/rr_mrp_fit.py
"""
RR-aware MRP fitting (multinomial logistic regression with a randomized-response observation model).

Core idea
---------
We want to estimate true vote intention (latent) conditional on demographics x:

  p_true(t | x) = softmax((X W)[t])

But we do NOT observe t directly. We observe a privatised report y produced via k-ary Randomized Response:

  A[t, s] = P(y = s | t)    (row-stochastic channel matrix)

So the likelihood for one respondent i is:

  P(y_i | x_i) = sum_t p_true(t | x_i) * A[t, y_i]

We fit W by maximum likelihood (with L2 regularization) using mini-batch Adam.

This file is designed to be:
- dependency-light (numpy only)
- stable (numerically safe softmax, eps safeguards)
- reusable by both CLI + Streamlit

Typical usage
-------------
1) Build X from categorical features using DesignMatrix
2) Fit MRPRRMultinomialModel on (X, reported_choice)
3) Predict p_true for each population cell and post-stratify

Note
----
This is "RR-aware" MRP. It is not a full hierarchical Bayesian MRP;
instead it uses regularized multinomial regression, which is common in practical MRP pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------
# RR channel matrix
# ---------------------------

def rr_transition_matrix(epsilon: float, k: int) -> np.ndarray:
    """
    k-ary randomized response channel matrix A (k x k).
    Row t gives probabilities of reporting each s given true t.

    p_keep = exp(eps) / (exp(eps) + (k-1))
    p_flip = (1-p_keep)/(k-1)
    """
    eps = float(epsilon)
    k = int(k)
    if k < 2:
        raise ValueError("k must be >= 2")
    if eps <= 0:
        raise ValueError("epsilon must be > 0")

    # p_keep is the on-diagonal probability (report truthfully).
    # p_flip is the off-diagonal probability (report any other category).
    e = float(np.exp(eps))
    p_keep = e / (e + (k - 1))
    p_flip = (1.0 - p_keep) / (k - 1)

    # Fill the matrix with uniform flip probability, then overwrite the
    # diagonal with the higher truthful-report probability.
    A = np.full((k, k), p_flip, dtype=float)
    np.fill_diagonal(A, p_keep)
    return A


# ---------------------------
# Design matrix for categorical demographics
# ---------------------------

@dataclass
class FeatureSpec:
    """Levels retained for one categorical feature in the design matrix."""

    name: str
    categories: List[str]  # includes baseline category at index 0


class DesignMatrix:
    """
    Simple categorical one-hot design matrix builder with a dropped baseline per feature.

    - Adds an intercept column (all ones).
    - For each feature, uses (num_categories - 1) columns, dropping the first category as baseline.
    - Unseen categories at transform-time map to baseline.

    This is enough for MRP-style regression on uploaded CSVs (region, age_band, etc.).
    """

    def __init__(self, feature_names: Sequence[str]):
        self.feature_names = list(feature_names)
        self.specs: List[FeatureSpec] = []
        self._col_offsets: Dict[str, Tuple[int, int]] = {}  # feature -> (start, width)

    def fit(self, rows: Sequence[Dict[str, str]]) -> "DesignMatrix":
        specs: List[FeatureSpec] = []
        for fn in self.feature_names:
            vals = []
            for r in rows:
                v = str(r.get(fn, "")).strip()
                if v != "":
                    vals.append(v)
            cats = sorted(set(vals))
            if not cats:
                cats = ["(missing)"]
            specs.append(FeatureSpec(name=fn, categories=cats))
        self.specs = specs

        # compute column offsets: 1 intercept + sum(cat_count-1)
        offset = 1
        self._col_offsets = {}
        for sp in self.specs:
            width = max(0, len(sp.categories) - 1)
            self._col_offsets[sp.name] = (offset, width)
            offset += width
        return self

    @property
    def n_features(self) -> int:
        if not self.specs:
            return 1
        total = 1
        for sp in self.specs:
            total += max(0, len(sp.categories) - 1)
        return total

    def transform(self, rows: Sequence[Dict[str, str]]) -> np.ndarray:
        n = len(rows)
        d = self.n_features
        X = np.zeros((n, d), dtype=float)
        X[:, 0] = 1.0  # intercept — global baseline for all respondents

        # build lookup maps
        cat_maps: Dict[str, Dict[str, int]] = {}
        for sp in self.specs:
            cat_maps[sp.name] = {c: i for i, c in enumerate(sp.categories)}

        for i, r in enumerate(rows):
            for sp in self.specs:
                v = str(r.get(sp.name, "")).strip()
                idx = cat_maps[sp.name].get(v, 0)  # unseen categories map to baseline (index 0)
                # Index 0 is the reference/dropped category; we only set
                # indicator columns for the remaining levels.
                if idx > 0:
                    start, width = self._col_offsets[sp.name]
                    j = start + (idx - 1)
                    if 0 <= j < d:
                        X[i, j] = 1.0
        return X

    def feature_columns(self) -> List[str]:
        cols = ["intercept"]
        for sp in self.specs:
            base = sp.categories[0]
            for c in sp.categories[1:]:
                cols.append(f"{sp.name}={c} (baseline={base})")
        return cols


# ---------------------------
# RR-aware multinomial regression model
# ---------------------------

# ---------------------------------------------------------------------------
# Numerically stable softmax used throughout the linear RR-MRP fitter
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=1, keepdims=True)


def _check_y(y: np.ndarray, k: int) -> np.ndarray:
    y = np.asarray(y, dtype=int).reshape(-1)
    if y.size == 0:
        raise ValueError("y is empty")
    if np.min(y) < 0 or np.max(y) >= k:
        raise ValueError(f"y must be in [0, {k-1}]")
    return y


@dataclass
class FitInfo:
    """Training summary returned by the NumPy RR-aware MRP fitter."""

    steps: int
    final_loss: float
    history: Optional[np.ndarray] = None


class MRPRRMultinomialModel:
    """
    Fit multinomial logistic regression for p_true(t|x),
    but optimize the RR-aware likelihood for observed (reported) y.

    Parameters
    ----------
    k: number of categories
    epsilon: RR epsilon (defines channel A)
    l2: L2 regularization strength on W
    seed: RNG seed for batching
    """

    def __init__(
        self,
        *,
        k: int,
        epsilon: float,
        l2: float = 1.0,
        seed: int = 0,
    ):
        self.k = int(k)
        self.epsilon = float(epsilon)
        self.l2 = float(l2)
        self.seed = int(seed)

        self.A = rr_transition_matrix(self.epsilon, self.k)
        self.W: Optional[np.ndarray] = None  # (d, k)

    def _init_W(self, d: int) -> None:
        rng = np.random.default_rng(self.seed)
        # Small random init helps symmetry breaking
        self.W = rng.normal(0.0, 0.01, size=(d, self.k)).astype(float)

    def predict_true_proba(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("Model is not fitted")
        X = np.asarray(X, dtype=float)
        logits = X @ self.W
        return _softmax(logits)

    def predict_reported_proba(self, X: np.ndarray) -> np.ndarray:
        P = self.predict_true_proba(X)
        return P @ self.A

    def fit(
        self,
        X: np.ndarray,
        y_reported: np.ndarray,
        *,
        lr: float = 0.05,
        steps: int = 2000,
        batch_size: int = 512,
        verbose_every: int = 0,
        keep_history: bool = False,
    ) -> FitInfo:
        """
        Fit with mini-batch Adam on RR-aware negative log-likelihood.

        Loss (per batch):
          L = -mean_i log( (softmax(XW) A)[i, y_i] ) + 0.5*l2*||W||^2

        Returns FitInfo with final loss (full-data loss at end).
        """
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        if n <= 0:
            raise ValueError("X has zero rows")
        if self.W is None:
            self._init_W(d)
        assert self.W is not None

        y = _check_y(y_reported, self.k)

        rng = np.random.default_rng(self.seed)
        A = self.A
        W = self.W

        # Adam state
        m = np.zeros_like(W)
        v = np.zeros_like(W)
        beta1 = 0.9
        beta2 = 0.999
        adam_eps = 1e-8

        hist = []
        for t in range(1, int(steps) + 1):
            idx = rng.integers(0, n, size=min(int(batch_size), n))
            Xb = X[idx]
            yb = y[idx]

            # Forward pass through the composite observation model:
            # 1. theta = softmax(Xb @ W)  — latent true-category probabilities
            # 2. Q = theta @ A            — observed reported-category probabilities
            logits = Xb @ W
            P = _softmax(logits)           # (b, k)
            Q = P @ A                      # (b, k)

            # RR-aware likelihood: pick the column corresponding to each
            # observed reported label.  Clamping avoids log(0) numerics.
            qy = Q[np.arange(Q.shape[0]), yb]
            qy = np.clip(qy, 1e-12, 1.0)
            nll = -np.mean(np.log(qy))

            # Grad wrt Q: only observed column has non-zero
            b = Q.shape[0]
            grad_Q = np.zeros_like(Q)
            grad_Q[np.arange(b), yb] = -1.0 / (b * qy)

            # Q = P @ A => grad_P = grad_Q @ A^T
            grad_P = grad_Q @ A.T

            # Backprop through softmax: each row i gets
            #   grad_logits_i = P_i * (grad_P_i - <grad_P_i, P_i>)
            # where the inner product contracts over the K categories.
            s = np.sum(grad_P * P, axis=1, keepdims=True)
            grad_logits = P * (grad_P - s)  # (b, k)

            # grad_W = X^T grad_logits + l2*W
            grad_W = Xb.T @ grad_logits
            if self.l2 > 0.0:
                grad_W = grad_W + self.l2 * W

            # Bias-corrected Adam update (same formulation as Kingma & Ba).
            m = beta1 * m + (1.0 - beta1) * grad_W
            v = beta2 * v + (1.0 - beta2) * (grad_W * grad_W)
            m_hat = m / (1.0 - (beta1 ** t))
            v_hat = v / (1.0 - (beta2 ** t))
            W = W - float(lr) * m_hat / (np.sqrt(v_hat) + adam_eps)

            if keep_history or (verbose_every and (t % int(verbose_every) == 0)):
                # compute full-data loss occasionally if requested
                if keep_history:
                    hist.append(float(nll))

            if verbose_every and (t % int(verbose_every) == 0):
                print(f"[MRP-RR] step={t} batch_nll={nll:.6f}")

        # Persist the optimised weights and compute the full-data final loss
        # for logging and convergence diagnostics.
        self.W = W

        final_loss = float(self.loss(X, y))
        history = np.asarray(hist, dtype=float) if keep_history else None
        return FitInfo(steps=int(steps), final_loss=final_loss, history=history)

    def loss(self, X: np.ndarray, y_reported: np.ndarray) -> float:
        """
        Full-data RR-aware loss (negative log-likelihood + L2).
        """
        X = np.asarray(X, dtype=float)
        y = _check_y(y_reported, self.k)
        Q = self.predict_reported_proba(X)
        qy = Q[np.arange(Q.shape[0]), y]
        qy = np.clip(qy, 1e-12, 1.0)
        nll = -float(np.mean(np.log(qy)))
        reg = 0.0
        if self.l2 > 0.0 and self.W is not None:
            reg = 0.5 * self.l2 * float(np.sum(self.W * self.W))
        return nll + reg

    def poststratify(
        self,
        X_pop: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """
        Post-stratify predictions:
          p_overall = sum_j w_j * p_true(.|x_j), where weights sum to 1.

        X_pop: (m, d)
        weights: (m,) non-negative, will be normalized
        """
        X_pop = np.asarray(X_pop, dtype=float)
        w = np.asarray(weights, dtype=float).reshape(-1)
        if X_pop.shape[0] != w.size:
            raise ValueError("X_pop and weights length mismatch")
        w = np.clip(w, 0.0, np.inf)
        s = float(np.sum(w))
        if s <= 0.0:
            raise ValueError("weights must have positive sum")
        w = w / s

        # The weighted sum of per-cell predictions yields the population-level
        # aggregate estimate. Clipping and renormalisation guard against
        # floating-point drift.
        P = self.predict_true_proba(X_pop)  # (m, k)
        p = (w[:, None] * P).sum(axis=0)
        p = np.clip(p, 0.0, 1.0)
        ps = float(np.sum(p))
        if ps > 0.0:
            p /= ps
        return p


# ---------------------------
# Convenience: end-to-end fit from dict rows
# ---------------------------

def fit_rr_mrp_from_rows(
    *,
    poll_rows: Sequence[Dict[str, str]],
    response_col: str,
    feature_cols: Sequence[str],
    epsilon: float,
    k: int,
    l2: float = 1.0,
    seed: int = 0,
    lr: float = 0.05,
    steps: int = 2000,
    batch_size: int = 512,
) -> Tuple[MRPRRMultinomialModel, DesignMatrix, np.ndarray, np.ndarray]:
    """
    Build design matrix X from poll_rows[feature_cols] and fit RR-aware MRP model on y_reported.

    Returns:
      (model, design, X, y)
    """
    rows = list(poll_rows)
    design = DesignMatrix(feature_cols).fit(rows)
    X = design.transform(rows)

    # y_reported must be 0..k-1 ints
    y = np.array([int(float(str(r.get(response_col, "0")).strip())) for r in rows], dtype=int)

    model = MRPRRMultinomialModel(k=int(k), epsilon=float(epsilon), l2=float(l2), seed=int(seed))
    model.fit(X, y, lr=float(lr), steps=int(steps), batch_size=int(batch_size))
    return model, design, X, y
