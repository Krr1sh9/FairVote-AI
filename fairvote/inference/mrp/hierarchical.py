"""Hierarchical/partial-pooling RR-aware MRP estimator.

This module is deliberately separate from :mod:`fairvote.inference.mrp.linear`.
The linear model is a regularised multinomial regression over a flat design
matrix.  The model here represents the post-stratification features as grouped
categorical factors and fits a multilevel additive logit model through the
k-ary Randomized Response observation channel.

Latent model
------------

For respondent i with demographic levels x_if, the true preference model is

    theta_i = softmax(alpha + sum_f u[f][x_if])

where ``alpha`` is a global category intercept and ``u[f][level]`` is a
feature-level varying effect.  Effects are mean-centred per feature for
identifiability.  Gaussian penalties on the varying effects act as a MAP
partial-pooling prior: sparse cells receive stronger shrinkage toward the
population mean because their likelihood contribution is weak while the prior
penalty remains active.

Observation model
-----------------

Only randomized-response reported labels are accepted:

    P(reported=r | x_i) = sum_t theta_i[t] A[t,r]

where A is the k-ary RR transition matrix.  No method in this class accepts
synthetic true labels, except offline tests that call it with whatever labels
they explicitly choose as ``y_reported``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fairvote.inference.mrp.diagnostics import FitDiagnostics
from fairvote.inference.mrp.likelihood import reported_label_likelihood, softmax_rows
from fairvote.inference.mrp.linear import validate_reported_labels, validate_weights
from fairvote.privacy.mechanisms.kary_rr import rr_transition_matrix


@dataclass(frozen=True)
class HierarchicalFeatureInfo:
    """Metadata for one categorical varying-effect block."""

    name: str
    levels: list[str]
    n_levels: int
    observed_counts: list[int]
    shrinkage_l2: float

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "levels": list(self.levels),
            "n_levels": int(self.n_levels),
            "observed_counts": [int(x) for x in self.observed_counts],
            "shrinkage_l2": float(self.shrinkage_l2),
        }


def _validate_feature_levels(
    feature_levels: Mapping[str, Sequence[str]], feature_order: Sequence[str] | None
) -> list[str]:
    if not feature_levels:
        raise ValueError("feature_levels cannot be empty")
    order = list(feature_order) if feature_order is not None else sorted(feature_levels.keys())
    if not order:
        raise ValueError("feature_order cannot be empty")
    duplicates = sorted({name for name in order if order.count(name) > 1})
    if duplicates:
        raise ValueError(f"feature_order contains duplicates: {duplicates}")
    for feature in order:
        if feature not in feature_levels:
            raise KeyError(f"Missing levels for feature {feature!r}")
        if len(feature_levels[feature]) < 1:
            raise ValueError(f"Feature {feature!r} must define at least one level")
    return order


def _validate_integer_features(
    features: Mapping[str, Sequence[int] | np.ndarray],
    feature_levels: Mapping[str, Sequence[str]],
    feature_order: Sequence[str],
) -> dict[str, np.ndarray]:
    if not features:
        raise ValueError("features cannot be empty")
    n: int | None = None
    out: dict[str, np.ndarray] = {}
    for feature in feature_order:
        if feature not in features:
            raise KeyError(f"Missing feature {feature!r}")
        raw = np.asarray(features[feature])
        if raw.ndim != 1:
            raise ValueError(f"Feature {feature!r} must be 1D")
        if raw.size == 0:
            raise ValueError(f"Feature {feature!r} is empty")
        if np.issubdtype(raw.dtype, np.floating):
            if not np.all(np.isfinite(raw)):
                raise ValueError(f"Feature {feature!r} contains NaN or infinite values")
            if not np.all(np.equal(raw, np.floor(raw))):
                raise ValueError(f"Feature {feature!r} must contain integer-coded categories")
        arr = raw.astype(int, copy=False).reshape(-1)
        n_levels = len(feature_levels[feature])
        if np.any((arr < 0) | (arr >= n_levels)):
            raise ValueError(f"Feature {feature!r} has values outside [0, {n_levels - 1}]")
        if n is None:
            n = int(arr.size)
        elif int(arr.size) != n:
            raise ValueError("All features must have the same length")
        out[feature] = arr
    return out


class HierarchicalRRMRPModel:
    """RR-aware multilevel MRP with feature-level partial pooling.

    The implementation is a deterministic empirical-Bayes/MAP optimiser rather
    than a full MCMC sampler.  That is intentional for a reproducible UG project:
    it gives a real multilevel/partial-pooling model while keeping final evidence
    runs fast and auditable.
    """

    model_name = "hierarchical_rr_mrp_partial_pooling_map"

    def __init__(
        self,
        k: int,
        *,
        epsilon: float | None = None,
        global_l2: float = 0.01,
        effect_l2: float = 1.0,
        feature_l2: Mapping[str, float] | None = None,
        seed: int = 0,
        center_effects: bool = True,
    ) -> None:
        if not isinstance(k, int) or k < 2:
            raise ValueError("k must be an int >= 2")
        if global_l2 < 0.0 or effect_l2 < 0.0:
            raise ValueError("regularisation strengths must be non-negative")
        self.k = int(k)
        self.epsilon = None if epsilon is None else float(epsilon)
        self.global_l2 = float(global_l2)
        self.effect_l2 = float(effect_l2)
        self.feature_l2 = {str(k): float(v) for k, v in (feature_l2 or {}).items()}
        if any(v < 0.0 for v in self.feature_l2.values()):
            raise ValueError("feature_l2 values must be non-negative")
        self.seed = int(seed)
        self.center_effects = bool(center_effects)

        self.feature_order: list[str] = []
        self.feature_levels: dict[str, list[str]] = {}
        self.alpha: np.ndarray | None = None
        self.effects: dict[str, np.ndarray] = {}
        self.A: np.ndarray | None = None
        self.fit_diagnostics: FitDiagnostics | None = None
        self.feature_info: dict[str, HierarchicalFeatureInfo] = {}
        if self.epsilon is not None:
            self.A = rr_transition_matrix(self.epsilon, self.k)

    def _init_parameters(self, feature_levels: Mapping[str, Sequence[str]], feature_order: Sequence[str]) -> None:
        rng = np.random.default_rng(self.seed)
        self.alpha = rng.normal(0.0, 0.01, size=self.k).astype(float)
        self.alpha -= float(np.mean(self.alpha))
        self.effects = {}
        for feature in feature_order:
            arr = rng.normal(0.0, 0.01, size=(len(feature_levels[feature]), self.k)).astype(float)
            arr -= arr.mean(axis=1, keepdims=True)
            arr -= arr.mean(axis=0, keepdims=True)
            self.effects[feature] = arr

    def _effect_l2(self, feature: str) -> float:
        return float(self.feature_l2.get(feature, self.effect_l2))

    def _logits(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
        if self.alpha is None or not self.effects:
            raise RuntimeError("Model is not fitted")
        n = next(iter(features.values())).size
        logits = np.tile(self.alpha, (n, 1))
        for feature in self.feature_order:
            logits += self.effects[feature][features[feature]]
        return logits

    def _center_all_effects(self) -> None:
        if not self.center_effects:
            return
        for feature in self.feature_order:
            eff = self.effects[feature]
            eff -= eff.mean(axis=0, keepdims=True)
            eff -= eff.mean(axis=1, keepdims=True)

    def fit(
        self,
        features: Mapping[str, Sequence[int] | np.ndarray],
        y_reported: Sequence[int] | np.ndarray,
        feature_levels: Mapping[str, Sequence[str]],
        epsilon: float | None = None,
        *,
        feature_order: Sequence[str] | None = None,
        lr: float = 0.05,
        steps: int = 2000,
        batch_size: int = 512,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps_adam: float = 1e-8,
        verbose_every: int = 0,
        keep_history: bool = False,
        history_every: int = 10,
    ) -> FitDiagnostics:
        """Fit the hierarchical model using only RR-reported labels."""
        order = _validate_feature_levels(feature_levels, feature_order)
        validated = _validate_integer_features(features, feature_levels, order)
        n = next(iter(validated.values())).size
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

        self.feature_order = list(order)
        self.feature_levels = {f: list(feature_levels[f]) for f in order}
        self._init_parameters(self.feature_levels, self.feature_order)
        assert self.alpha is not None
        self.A = rr_transition_matrix(self.epsilon, self.k)

        # Adam state for the global intercept and each varying-effect block.
        m_alpha = np.zeros_like(self.alpha)
        v_alpha = np.zeros_like(self.alpha)
        m_eff = {f: np.zeros_like(self.effects[f]) for f in self.feature_order}
        v_eff = {f: np.zeros_like(self.effects[f]) for f in self.feature_order}
        rng = np.random.default_rng(self.seed)
        history: list[float] = []
        started = time.perf_counter()

        for step in range(1, int(steps) + 1):
            idx = rng.integers(0, n, size=min(int(batch_size), n))
            batch_features = {f: arr[idx] for f, arr in validated.items()}
            yb = y[idx]

            theta = softmax_rows(self._logits(batch_features))
            likelihood = reported_label_likelihood(theta, self.A, yb)
            batch_nll = likelihood.nll
            grad_logits = likelihood.grad_logits

            grad_alpha = grad_logits.sum(axis=0)
            if self.global_l2 > 0.0:
                grad_alpha = grad_alpha + self.global_l2 * self.alpha

            # Multilevel prior penalties.  Effects are mean-centred before
            # applying the penalty so shrinkage is toward the feature-level
            # average, not toward an arbitrary baseline category.
            grad_effects: dict[str, np.ndarray] = {}
            for feature in self.feature_order:
                eff = self.effects[feature]
                grad = np.zeros_like(eff)
                np.add.at(grad, batch_features[feature], grad_logits)
                l2 = self._effect_l2(feature)
                if l2 > 0.0:
                    centered = eff - eff.mean(axis=0, keepdims=True)
                    grad = grad + l2 * centered
                grad_effects[feature] = grad

            # Adam update: alpha.
            m_alpha = beta1 * m_alpha + (1.0 - beta1) * grad_alpha
            v_alpha = beta2 * v_alpha + (1.0 - beta2) * (grad_alpha * grad_alpha)
            self.alpha = self.alpha - float(lr) * (m_alpha / (1.0 - beta1**step)) / (
                np.sqrt(v_alpha / (1.0 - beta2**step)) + float(eps_adam)
            )
            self.alpha -= float(np.mean(self.alpha))

            # Adam update: varying effects.
            for feature in self.feature_order:
                grad = grad_effects[feature]
                m_eff[feature] = beta1 * m_eff[feature] + (1.0 - beta1) * grad
                v_eff[feature] = beta2 * v_eff[feature] + (1.0 - beta2) * (grad * grad)
                self.effects[feature] = self.effects[feature] - float(lr) * (m_eff[feature] / (1.0 - beta1**step)) / (
                    np.sqrt(v_eff[feature] / (1.0 - beta2**step)) + float(eps_adam)
                )
            self._center_all_effects()

            if keep_history and (step == 1 or step == steps or step % int(history_every) == 0):
                history.append(batch_nll)
            if verbose_every and (step == 1 or step == steps or step % int(verbose_every) == 0):
                print(f"[hierarchical-rr-mrp] step={step} batch_nll={batch_nll:.6f}")

        final_loss = self.loss(validated, y)
        self.fit_diagnostics = FitDiagnostics(
            steps=int(steps),
            final_loss=float(final_loss),
            runtime_sec=float(time.perf_counter() - started),
            history=np.asarray(history, dtype=float) if keep_history else None,
        )
        self.feature_info = {}
        for feature in self.feature_order:
            counts = np.bincount(validated[feature], minlength=len(self.feature_levels[feature])).astype(int)
            self.feature_info[feature] = HierarchicalFeatureInfo(
                name=feature,
                levels=list(self.feature_levels[feature]),
                n_levels=len(self.feature_levels[feature]),
                observed_counts=[int(x) for x in counts.tolist()],
                shrinkage_l2=self._effect_l2(feature),
            )
        return self.fit_diagnostics

    def predict_theta_from_features(self, features: Mapping[str, Sequence[int] | np.ndarray]) -> np.ndarray:
        """Predict latent true-category probabilities for integer-coded features."""
        if not self.feature_order or not self.feature_levels:
            raise RuntimeError("Model is not fitted")
        validated = _validate_integer_features(features, self.feature_levels, self.feature_order)
        return softmax_rows(self._logits(validated))

    def predict_true_proba(self, features: Mapping[str, Sequence[int] | np.ndarray]) -> np.ndarray:
        return self.predict_theta_from_features(features)

    def predict_reported_proba(self, features: Mapping[str, Sequence[int] | np.ndarray]) -> np.ndarray:
        if self.A is None:
            if self.epsilon is None:
                raise RuntimeError("Model does not have an RR epsilon")
            self.A = rr_transition_matrix(self.epsilon, self.k)
        return self.predict_theta_from_features(features) @ self.A

    def loss(self, features: Mapping[str, Sequence[int] | np.ndarray], y_reported: Sequence[int] | np.ndarray) -> float:
        """Full-data negative log likelihood plus MAP prior penalty."""
        if self.alpha is None or self.A is None:
            raise RuntimeError("Model is not fitted")
        validated = _validate_integer_features(features, self.feature_levels, self.feature_order)
        n = next(iter(validated.values())).size
        y = validate_reported_labels(y_reported, k=self.k, expected_n=n)
        reported_probs = self.predict_reported_proba(validated)
        observed = np.clip(reported_probs[np.arange(n), y], 1e-12, 1.0)
        nll = -float(np.mean(np.log(observed)))
        penalty = 0.5 * self.global_l2 * float(np.sum(self.alpha * self.alpha))
        for feature in self.feature_order:
            eff = self.effects[feature]
            centered = eff - eff.mean(axis=0, keepdims=True)
            penalty += 0.5 * self._effect_l2(feature) * float(np.sum(centered * centered))
        return nll + penalty

    def poststratify(
        self,
        features: Mapping[str, Sequence[int] | np.ndarray],
        weights: Sequence[float] | np.ndarray,
    ) -> np.ndarray:
        """Return population-weighted latent true-category probabilities."""
        validated = _validate_integer_features(features, self.feature_levels, self.feature_order)
        n = next(iter(validated.values())).size
        w = validate_weights(weights, expected_n=n)
        probs = self.predict_theta_from_features(validated)
        estimate = (w[:, None] * probs).sum(axis=0)
        estimate = np.clip(estimate, 0.0, 1.0)
        total = float(np.sum(estimate))
        if total <= 0.0:
            raise RuntimeError("poststratified probabilities sum to zero")
        return estimate / total

    def export_metadata(self, *, include_parameters: bool = False) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "model_name": self.model_name,
            "honest_description": "RR-aware multilevel additive logit MRP fitted by MAP with Gaussian partial-pooling priors",
            "k": self.k,
            "epsilon": self.epsilon,
            "global_l2": self.global_l2,
            "effect_l2": self.effect_l2,
            "feature_l2": dict(self.feature_l2),
            "seed": self.seed,
            "center_effects": self.center_effects,
            "feature_order": list(self.feature_order),
            "feature_info": {f: info.to_jsonable() for f, info in self.feature_info.items()},
            "fit_diagnostics": None if self.fit_diagnostics is None else self.fit_diagnostics.to_jsonable(),
            "is_fitted": self.alpha is not None,
        }
        if self.alpha is not None:
            metadata["alpha_l2_norm"] = float(np.linalg.norm(self.alpha))
        if self.effects:
            metadata["effect_l2_norms"] = {f: float(np.linalg.norm(v)) for f, v in self.effects.items()}
        if include_parameters and self.alpha is not None:
            metadata["alpha"] = self.alpha.tolist()
            metadata["effects"] = {f: arr.tolist() for f, arr in self.effects.items()}
        return metadata

    def save_metadata(self, path: str | Path, *, include_parameters: bool = False) -> None:
        Path(path).write_text(
            json.dumps(self.export_metadata(include_parameters=include_parameters), indent=2), encoding="utf-8"
        )
