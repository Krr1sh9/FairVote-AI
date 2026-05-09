"""Misreport-aware RR-MRP estimator runners and conversion helpers."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from fairvote.inference.mrp.learned_misreport_rr import LearnedShyMisreportRRMultinomialModel
from fairvote.inference.mrp.misreport_rr import MisreportRRMultinomialModel, identity_misreport
from fairvote.simulation.bias_models import apply_misreporting

from ..config import ExperimentConfig, MethodResult, TrialConfig
from ..context import ExperimentContext
from .common import TrialData
from .linear_mrp import _poststratify_model_predictions


def misreport_to_matrix(mis: Any, k: int, *, mc_samples_per_true: int = 20_000) -> np.ndarray:
    """Convert common misreport model representations into a row-stochastic matrix."""
    if mis is None:
        return identity_misreport(k)
    if isinstance(mis, np.ndarray):
        return np.asarray(mis, dtype=float)
    for attr in ("confusion", "matrix", "M", "P", "transition", "T"):
        if hasattr(mis, attr):
            matrix = getattr(mis, attr)
            return np.asarray(matrix() if callable(matrix) else matrix, dtype=float)
    for attr in ("rows", "row_probs", "probs", "prob_rows", "row_distributions"):
        if hasattr(mis, attr):
            rows = getattr(mis, attr)
            return np.asarray(rows() if callable(rows) else rows, dtype=float)
    for meth in ("to_matrix", "as_matrix", "get_matrix"):
        if hasattr(mis, meth) and callable(getattr(mis, meth)):
            return np.asarray(getattr(mis, meth)(), dtype=float)
    for meth in ("prob", "p", "transition_prob", "p_true_to_stated", "prob_true_to_stated"):
        if hasattr(mis, meth) and callable(getattr(mis, meth)):
            f = getattr(mis, meth)
            matrix = np.zeros((k, k), dtype=float)
            for t in range(k):
                for s in range(k):
                    matrix[t, s] = float(f(t, s))
            return matrix
    try:
        rng = np.random.default_rng(424242)
        matrix = np.zeros((k, k), dtype=float)
        n = max(int(mc_samples_per_true), 2_000)
        for t in range(k):
            truth = np.full(n, t, dtype=int)
            stated = apply_misreporting(truth, mis, rng=rng)
            counts = np.bincount(stated.astype(int), minlength=k).astype(float)
            matrix[t, :] = counts / max(float(counts.sum()), 1.0)
        matrix = np.clip(matrix, 0.0, None)
        return matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1e-12)
    except Exception as exc:
        raise TypeError(f"Don't know how to convert misreport model of type {type(mis)} to a (k,k) matrix.") from exc


def mrp_misreport_rr_poststrat(
    config: ExperimentConfig,
    context: ExperimentContext,
    trial: TrialConfig,
    data: TrialData,
) -> MethodResult:
    """Oracle-style misreport-aware RR-MRP used as a diagnostic upper baseline."""
    start = time.perf_counter()
    mis_matrix = misreport_to_matrix(data.perturbation.misreport_model, config.k)
    model = MisreportRRMultinomialModel(
        k=config.k,
        l2=config.mrp_l2,
        seed=config.seed + 1999 + trial.trial,
        misreport=mis_matrix,
    )
    model.fit(
        data.X_train,
        data.perturbation.reported_categories,
        trial.epsilon,
        lr=config.mrp_lr,
        steps=config.mrp_steps,
        batch_size=config.mrp_batch_size,
        verbose_every=config.verbose_every,
    )
    overall, by_feature = _poststratify_model_predictions(
        context=context, cell_theta=model.predict_theta(context.X_cells)
    )
    return MethodResult("mrp_misreport_rr_poststrat", overall, by_feature, time.perf_counter() - start)


def oracle_known_misreport_rr_mrp(
    config: ExperimentConfig,
    context: ExperimentContext,
    trial: TrialConfig,
    data: TrialData,
) -> MethodResult:
    """Alias making the oracle-known-misreport baseline explicit in research runs."""
    result = mrp_misreport_rr_poststrat(config, context, trial, data)
    return MethodResult(
        "oracle_known_misreport_rr_mrp",
        result.estimate_overall,
        result.by_feature,
        result.runtime_sec,
        {**dict(result.extra), "oracle_known_misreport": 1},
    )


def mrp_learned_misreport_rr_poststrat(
    config: ExperimentConfig,
    context: ExperimentContext,
    trial: TrialConfig,
    data: TrialData,
) -> MethodResult:
    """Learn a simple shy-voter misreport parameter from privatized labels."""
    start = time.perf_counter()
    model = LearnedShyMisreportRRMultinomialModel(
        k=config.k,
        shy_category=config.shy_category,
        l2=config.mrp_l2,
        seed=config.seed + 2999 + trial.trial,
        honesty_init=0.80,
        honesty_lr=0.02,
    )
    model.fit(
        data.X_train,
        data.perturbation.reported_categories,
        trial.epsilon,
        lr=config.mrp_lr,
        steps=config.mrp_steps,
        batch_size=config.mrp_batch_size,
        verbose_every=config.verbose_every,
    )
    overall, by_feature = _poststratify_model_predictions(
        context=context, cell_theta=model.predict_theta(context.X_cells)
    )
    return MethodResult(
        "mrp_learned_misreport_rr_poststrat",
        overall,
        by_feature,
        time.perf_counter() - start,
        {"learned_honesty": model.learned_honesty()},
    )
