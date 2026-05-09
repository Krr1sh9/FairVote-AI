"""MRP-style estimators and post-stratification utilities for FairVote-AI.

Canonical linear path
---------------------
The repository's authoritative linear RR-aware MRP implementation is
:class:`LinearRRMRPModel` in :mod:`fairvote.inference.mrp.linear`.  It is a
regularised multinomial regression fitted through the k-ary Randomized Response
observation channel, followed by post-stratification.  It is intentionally named
and documented as MRP-style rather than a full hierarchical Bayesian MRP model.
"""

from fairvote.inference.mrp.design import DesignInfo, DesignMatrix, FeatureSpec, build_design_matrix
from fairvote.inference.mrp.diagnostics import FitDiagnostics, FitInfo
from fairvote.inference.mrp.hierarchical import HierarchicalFeatureInfo, HierarchicalRRMRPModel
from fairvote.inference.mrp.linear import (
    LinearRRMRPModel,
    MRPRRMultinomialModel,
    RRMultinomialModel,
    softmax_rows,
)
from fairvote.inference.mrp.poststratify import (
    PostStratResult,
    build_features_from_cells,
    normalise_poststrat_weights,
    poststratify,
    poststratify_overall_only,
)
from fairvote.privacy.mechanisms.kary_rr import rr_transition_matrix

__all__ = [
    "DesignInfo",
    "DesignMatrix",
    "FeatureSpec",
    "build_design_matrix",
    "FitDiagnostics",
    "FitInfo",
    "HierarchicalFeatureInfo",
    "HierarchicalRRMRPModel",
    "LinearRRMRPModel",
    "MRPRRMultinomialModel",
    "RRMultinomialModel",
    "softmax_rows",
    "PostStratResult",
    "build_features_from_cells",
    "normalise_poststrat_weights",
    "poststratify",
    "poststratify_overall_only",
    "rr_transition_matrix",
]

from fairvote.inference.mrp.misreport_rr import (  # noqa: E402
    MisreportRRMultinomialModel,
    identity_misreport,
    shy_misreport_matrix,
)

__all__ += [
    "MisreportRRMultinomialModel",
    "identity_misreport",
    "shy_misreport_matrix",
]

from fairvote.inference.mrp.learned_misreport_rr import LearnedShyMisreportRRMultinomialModel  # noqa: E402

__all__ += ["LearnedShyMisreportRRMultinomialModel"]

from fairvote.inference.mrp.rr_mrp_fit import fit_rr_mrp_from_rows  # noqa: E402

__all__ += ["fit_rr_mrp_from_rows"]

_NEURAL_EXPORTS = {
    "RRNeuralMRPFitInfo",
    "RRNeuralMRPModel",
    "NeuralRRMRPFitInfo",
    "NeuralRRMRPModel",
    "RRNeuralMRPEnsemble",
    "fit_rr_neural_mrp_ensemble",
}


def __getattr__(name: str):
    """Lazily import the optional PyTorch neural MRP implementation."""
    if name in _NEURAL_EXPORTS:
        from fairvote.inference.mrp.rr_neural_mrp import (
            NeuralRRMRPFitInfo,
            NeuralRRMRPModel,
            RRNeuralMRPFitInfo,
            RRNeuralMRPModel,
            RRNeuralMRPEnsemble,
            fit_rr_neural_mrp_ensemble,
        )

        mapping = {
            "RRNeuralMRPFitInfo": RRNeuralMRPFitInfo,
            "RRNeuralMRPModel": RRNeuralMRPModel,
            "NeuralRRMRPFitInfo": NeuralRRMRPFitInfo,
            "NeuralRRMRPModel": NeuralRRMRPModel,
            "RRNeuralMRPEnsemble": RRNeuralMRPEnsemble,
            "fit_rr_neural_mrp_ensemble": fit_rr_neural_mrp_ensemble,
        }
        globals().update(mapping)
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
