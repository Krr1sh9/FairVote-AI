"""Estimator runner package.

The public API remains backwards compatible with the previous
``experiments.pipeline.methods`` module, but estimator implementations are now
split by concern: simple baselines, linear MRP, hierarchical MRP,
misreport-aware MRP, neural MRP, and the registry.
"""

from .baselines import baseline_rr_debias, oracle_true_sample_distribution, raw_reported_distribution
from .common import TrialData, build_trial_data, estimate_subgroup_distribution, estimate_subgroup_mean_proba
from .hierarchical_mrp import hierarchical_rr_mrp_poststrat
from .linear_mrp import linear_rr_no_poststrat, mrp_rr_poststrat, oracle_true_linear_mrp_poststrat
from .misreport import (
    misreport_to_matrix,
    mrp_learned_misreport_rr_poststrat,
    mrp_misreport_rr_poststrat,
    oracle_known_misreport_rr_mrp,
)
from .neural_mrp import neural_naive_reported_mrp, neural_rr_mrp, require_rr_neural_mrp_model
from .registry import METHOD_REGISTRY, MethodRunner, selected_registry

__all__ = [
    "METHOD_REGISTRY",
    "MethodRunner",
    "TrialData",
    "build_trial_data",
    "selected_registry",
    "estimate_subgroup_distribution",
    "estimate_subgroup_mean_proba",
    "oracle_true_sample_distribution",
    "raw_reported_distribution",
    "baseline_rr_debias",
    "linear_rr_no_poststrat",
    "mrp_rr_poststrat",
    "hierarchical_rr_mrp_poststrat",
    "oracle_true_linear_mrp_poststrat",
    "misreport_to_matrix",
    "mrp_misreport_rr_poststrat",
    "oracle_known_misreport_rr_mrp",
    "mrp_learned_misreport_rr_poststrat",
    "neural_rr_mrp",
    "neural_naive_reported_mrp",
    "require_rr_neural_mrp_model",
]
