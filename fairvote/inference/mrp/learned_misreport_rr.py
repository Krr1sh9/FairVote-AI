"""RR-aware MRP with a learned simple shy-voter misreport parameter.

This experimental baseline inserts a true-to-stated misreport channel before
the Randomized Response channel. The model is fitted to privatized reported
answers through the composite likelihood; synthetic true labels are not used as
training targets.
"""

# fairvote/inference/mrp/learned_misreport_rr.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fairvote.inference.mrp.likelihood import reported_label_likelihood, softmax_rows
from fairvote.inference.mrp.misreport_rr import shy_misreport_matrix
from fairvote.privacy.mechanisms.kary_rr import rr_transition_matrix


@dataclass
class LearnedShyMisreportRRMultinomialModel:
    """
    Learn a simple "shy voter" misreport parameter (honesty) from privatized data.

    Generative model:
      TRUE (latent) ~ Multinomial(theta(x; W))
      STATED        ~ MisreportMatrix M(h) applied to TRUE
      REPORTED      ~ k-ary Randomized Response A(eps) applied to STATED

    Learns:
      - W (multinomial logistic regression weights)
      - h in (0,1): honesty for one shy category

    Misreport structure:
      For true == shy_category:
        P(stated == shy_category) = h
        Otherwise uniformly among other categories
      For true != shy_category:
        stated == true (identity)
    """

    k: int
    shy_category: int
    l2: float = 1.0
    seed: int = 0
    honesty_init: float = 0.80
    honesty_lr: float = 0.02
    honesty_clip: tuple[float, float] = (1e-3, 1.0 - 1e-3)

    def __post_init__(self) -> None:
        if self.k <= 1:
            raise ValueError("k must be >= 2")
        if not (0 <= self.shy_category < self.k):
            raise ValueError("shy_category out of range")
        if not (0.0 < self.honesty_init < 1.0):
            raise ValueError("honesty_init must be in (0,1)")
        self.rng = np.random.default_rng(self.seed)
        self.W: np.ndarray | None = None
        self.honesty_: float = float(self.honesty_init)

    def _composite_channel_and_gradrow(self, eps: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          C: (k,k) composite channel TRUE -> REPORTED
          dC_row: (k,) derivative of C[shy_category, r] w.r.t honesty for each r
        """
        A = rr_transition_matrix(eps, self.k)  # STATED -> REPORTED
        s = self.shy_category
        h = float(self.honesty_)

        M = shy_misreport_matrix(self.k, s, h)  # TRUE → STATED
        C = M @ A  # TRUE → REPORTED (composite channel)

        # Derivative of C[s, r] with respect to honesty h. Only the shy-category
        # row of M depends on h, so only that row of C has a non-zero derivative.
        S = np.sum(A, axis=0) - A[s, :]
        dC_row = A[s, :] - (S / (self.k - 1.0))

        # hygiene
        C = np.clip(C, 0.0, None)
        C = C / np.maximum(C.sum(axis=1, keepdims=True), 1e-12)
        return C, dC_row

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
        """Fit on privatized reported labels through the composite channel.

        The likelihood marginalises over latent true categories and the
        true-to-stated misreport step. No true labels are supplied to this
        training method.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(reported, dtype=int)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        n, d = X.shape
        if y.shape[0] != n:
            raise ValueError("reported must have same length as X rows")
        if np.any((y < 0) | (y >= self.k)):
            raise ValueError("reported contains out-of-range category ids")

        self.W = 0.01 * self.rng.standard_normal((d, self.k))

        lo, hi = self.honesty_clip

        for step in range(1, steps + 1):
            b = min(batch_size, n)
            idx = self.rng.integers(0, n, size=b)
            Xb = X[idx]
            yb = y[idx]

            C, dC_row = self._composite_channel_and_gradrow(eps)

            logits = Xb @ self.W
            theta = softmax_rows(logits)

            likelihood = reported_label_likelihood(theta, C, yb)
            p = likelihood.observed_probs
            # Update W via gradient descent through the marginal likelihood.
            gradW = Xb.T @ likelihood.grad_logits
            gradW += self.l2 * self.W
            self.W -= lr * gradW

            # Scalar gradient for the honesty parameter h.  Only the shy-category
            # column of the Jacobian is non-zero, so the gradient simplifies to
            # a weighted mean over the batch.
            shy = self.shy_category
            dC_y = dC_row[yb]
            dp_dh = theta[:, shy] * dC_y
            g_h = -float(np.mean(dp_dh / p))
            # Clamp honesty to (0, 1) for numerical stability; the endpoints
            # correspond to degenerate channels that break the likelihood.
            self.honesty_ = float(np.clip(self.honesty_ - self.honesty_lr * g_h, lo, hi))

            if verbose_every and (step % verbose_every == 0 or step == 1 or step == steps):
                nll = float(np.mean(-np.log(p)))
                print(f"[learned-shy-mrp] step {step:5d}/{steps}  batch_nll={nll:.4f}  honesty={self.honesty_:.4f}")

    def predict_theta(self, X: np.ndarray) -> np.ndarray:
        """Predict latent true-category probabilities for feature rows."""
        if self.W is None:
            raise RuntimeError("Model is not fit yet.")
        X = np.asarray(X, dtype=float)
        return softmax_rows(X @ self.W)

    def learned_honesty(self) -> float:
        return float(self.honesty_)

    def misreport_matrix(self) -> np.ndarray:
        return shy_misreport_matrix(self.k, self.shy_category, float(self.honesty_))
