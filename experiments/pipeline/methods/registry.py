"""Estimator registry for the MRP-vs-baselines experiment."""

from __future__ import annotations

from collections.abc import Callable

from ..config import ExperimentConfig, MethodResult, TrialConfig
from ..context import ExperimentContext
from .baselines import baseline_rr_debias, oracle_true_sample_distribution, raw_reported_distribution
from .common import TrialData, build_trial_data
from .hierarchical_mrp import hierarchical_rr_mrp_poststrat
from .linear_mrp import linear_rr_no_poststrat, mrp_rr_poststrat, oracle_true_linear_mrp_poststrat
from .misreport import (
    misreport_to_matrix,
    mrp_learned_misreport_rr_poststrat,
    mrp_misreport_rr_poststrat,
    oracle_known_misreport_rr_mrp,
)
from .neural_mrp import neural_naive_reported_mrp, neural_rr_mrp, require_rr_neural_mrp_model

MethodRunner = Callable[[ExperimentConfig, ExperimentContext, TrialConfig, TrialData], MethodResult]

METHOD_REGISTRY: dict[str, MethodRunner] = {
    "oracle_true_sample_distribution": oracle_true_sample_distribution,
    "raw_reported_distribution": raw_reported_distribution,
    "baseline_rr_debias": baseline_rr_debias,
    "linear_rr_no_poststrat": linear_rr_no_poststrat,
    "mrp_rr_poststrat": mrp_rr_poststrat,
    "hierarchical_rr_mrp_poststrat": hierarchical_rr_mrp_poststrat,
    "oracle_true_linear_mrp_poststrat": oracle_true_linear_mrp_poststrat,
    "mrp_misreport_rr_poststrat": mrp_misreport_rr_poststrat,
    "oracle_known_misreport_rr_mrp": oracle_known_misreport_rr_mrp,
    "mrp_learned_misreport_rr_poststrat": mrp_learned_misreport_rr_poststrat,
    "neural_rr_mrp": neural_rr_mrp,
    "neural_naive_reported_mrp": neural_naive_reported_mrp,
}


def selected_registry(methods: list[str]) -> dict[str, MethodRunner]:
    """Return a registry subset in requested output order."""
    missing = [m for m in methods if m not in METHOD_REGISTRY]
    if missing:
        raise ValueError(f"Unknown experiment methods: {missing}")
    return {name: METHOD_REGISTRY[name] for name in methods}


__all__ = [
    "METHOD_REGISTRY",
    "MethodRunner",
    "TrialData",
    "build_trial_data",
    "selected_registry",
    "misreport_to_matrix",
    "require_rr_neural_mrp_model",
]
