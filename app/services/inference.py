"""Dashboard inference orchestration and optional dependency loading."""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

_HAS_FAIRVOTE_PRIVACY = True
try:
    from fairvote.privacy import estimate_distribution as _fv_estimate_distribution
    from fairvote.privacy import rr_transition_matrix as _fv_rr_transition_matrix

    fv_estimate_distribution: Any = _fv_estimate_distribution
    fv_rr_transition_matrix: Any = _fv_rr_transition_matrix
except Exception:
    _HAS_FAIRVOTE_PRIVACY = False
    fv_estimate_distribution = None
    fv_rr_transition_matrix = None

_HAS_RR_MRP = True
try:
    from fairvote.inference.mrp import DesignMatrix as _DesignMatrix
    from fairvote.inference.mrp import MRPRRMultinomialModel as _MRPRRMultinomialModel

    DesignMatrix: Any = _DesignMatrix
    MRPRRMultinomialModel: Any = _MRPRRMultinomialModel
except Exception:
    _HAS_RR_MRP = False
    DesignMatrix = None
    MRPRRMultinomialModel = None

_HAS_MISREPORT_RR_MRP = True
try:
    from fairvote.inference.mrp.misreport_rr import MisreportRRMultinomialModel as _MisreportRRMultinomialModel
    from fairvote.inference.mrp.misreport_rr import shy_misreport_matrix as _shy_misreport_matrix

    MisreportRRMultinomialModel: Any = _MisreportRRMultinomialModel
    shy_misreport_matrix: Any = _shy_misreport_matrix
except Exception:
    _HAS_MISREPORT_RR_MRP = False
    MisreportRRMultinomialModel = None
    shy_misreport_matrix = None

LEARNED_MRP_METHODS = {
    "Linear RR-aware MRP",
    "Neural RR-aware MRP",
    "Misreport-aware RR-MRP",
}


def torch_available() -> bool:
    """Return whether PyTorch can be found without importing it eagerly."""

    return importlib.util.find_spec("torch") is not None


def load_neural_mrp_model():
    """Lazily import the PyTorch neural MRP model only when selected."""

    from fairvote.inference.mrp.rr_neural_mrp import RRNeuralMRPModel

    return RRNeuralMRPModel


def rr_matrix(epsilon: float, k: int) -> np.ndarray:
    """Dashboard wrapper around the canonical Python RR transition matrix."""

    if not _HAS_FAIRVOTE_PRIVACY or fv_rr_transition_matrix is None:
        raise RuntimeError("FairVote privacy module is required for RR matrix calculations.")
    return np.asarray(fv_rr_transition_matrix(epsilon, k), dtype=float)


def estimate_distribution(reported: np.ndarray, epsilon: float, k: int) -> np.ndarray:
    """Dashboard wrapper around the canonical Python RR debiasing estimator."""

    if not _HAS_FAIRVOTE_PRIVACY or fv_estimate_distribution is None:
        raise RuntimeError("FairVote privacy module is required for RR debiasing.")
    out = fv_estimate_distribution(reported, epsilon=epsilon, k=k)
    return np.asarray(out, dtype=float)


def bootstrap_ci(
    reported: np.ndarray,
    epsilon: float,
    k: int,
    n_boot: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap confidence intervals for the baseline RR-debiased estimate."""

    rng = np.random.default_rng(seed)
    reported = np.asarray(reported, dtype=int).reshape(-1)
    n = int(reported.size)
    if n == 0:
        lo = np.full(k, float("nan"))
        hi = np.full(k, float("nan"))
        return lo, hi

    boot = np.empty((n_boot, k), dtype=float)
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        boot[b] = estimate_distribution(reported[idx], epsilon=epsilon, k=k)

    lo = np.quantile(boot, alpha / 2.0, axis=0)
    hi = np.quantile(boot, 1.0 - alpha / 2.0, axis=0)
    return lo, hi


def is_learned_mrp_method(method: str) -> bool:
    return str(method) in LEARNED_MRP_METHODS


def method_prefix(method: str) -> str:
    if method == "Neural RR-aware MRP":
        return "neural_mrp"
    if method == "Misreport-aware RR-MRP":
        return "misreport_mrp"
    if method == "Linear RR-aware MRP":
        return "linear_mrp"
    return "baseline"


def method_short_label(method: str) -> str:
    if method == "Neural RR-aware MRP":
        return "Neural RR-aware MRP"
    if method == "Misreport-aware RR-MRP":
        return "Misreport-aware RR-MRP"
    if method == "Linear RR-aware MRP":
        return "Linear RR-aware MRP"
    return "RR debiasing"


def available_method_options() -> list[str]:
    options = ["RR debiasing", "Linear RR-aware MRP", "Neural RR-aware MRP"]
    if _HAS_MISREPORT_RR_MRP:
        options.append("Misreport-aware RR-MRP")
    return options


def resolve_estimation_method(requested: str) -> tuple[str, str | None]:
    """Return the usable method plus a warning message when falling back.

    This pure function lets tests verify method selection without Streamlit.
    """

    method = str(requested)
    if method == "Linear RR-aware MRP" and not _HAS_RR_MRP:
        return "RR debiasing", "Linear MRP module not found. Falling back to RR debiasing."
    if method == "Misreport-aware RR-MRP" and not _HAS_MISREPORT_RR_MRP:
        return "RR debiasing", "Misreport-aware MRP module not found. Falling back to RR debiasing."
    if method == "Neural RR-aware MRP" and not torch_available():
        return (
            "RR debiasing",
            'PyTorch is not installed, so Neural RR-aware MRP cannot run. Install with `pip install -e ".[neural]"` or `pip install -e ".[dev]"`, or use a non-neural estimator. Falling back to RR debiasing.',
        )
    return method, None
