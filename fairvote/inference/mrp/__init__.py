# fairvote/inference/mrp/__init__.py
"""
MRP (Multilevel Regression and Poststratification) components for FairVote-AI.

This module provides:
- One-hot design matrix building for categorical demographics
- Privacy-aware multinomial model trained directly on k-ary RR reports
- Poststratification utilities to convert model predictions + population cell counts
  into overall and subgroup estimates
"""

from fairvote.inference.mrp.model import (
    DesignInfo,
    RRMultinomialModel,
    build_design_matrix,
)
from fairvote.inference.mrp.poststratify import (
    PostStratResult,
    build_features_from_cells,
    poststratify,
    poststratify_overall_only,
)

__all__ = [
    "DesignInfo",
    "RRMultinomialModel",
    "build_design_matrix",
    "PostStratResult",
    "build_features_from_cells",
    "poststratify",
    "poststratify_overall_only",
]

from fairvote.inference.mrp.misreport_rr import (
    MisreportRRMultinomialModel,
    identity_misreport,
    shy_misreport_matrix,
    rr_transition_matrix,
)

__all__ += [
    "MisreportRRMultinomialModel",
    "identity_misreport",
    "shy_misreport_matrix",
    "rr_transition_matrix",
]

from fairvote.inference.mrp.learned_misreport_rr import LearnedShyMisreportRRMultinomialModel

__all__ += [
    "LearnedShyMisreportRRMultinomialModel",
]

from .rr_mrp_fit import (
    MRPRRMultinomialModel,
    DesignMatrix,
    fit_rr_mrp_from_rows,
)

__all__ += [
    "MRPRRMultinomialModel",
    "DesignMatrix",
    "fit_rr_mrp_from_rows",
]


# Neural model names that are resolved lazily via __getattr__.  This set is
# kept separate from __all__ because importing PyTorch is expensive and
# optional; callers who only need the linear models should not pay that cost.
_NEURAL_EXPORTS = {
    "RRNeuralMRPFitInfo",
    "RRNeuralMRPModel",
    "NeuralRRMRPFitInfo",
    "NeuralRRMRPModel",
}


def __getattr__(name: str):
    """Lazily import the PyTorch neural MRP model only when requested.

    Importing the base MRP package should not eagerly import PyTorch. The neural
    model remains available as ``fairvote.inference.mrp.RRNeuralMRPModel`` and
    through the explicit module ``fairvote.inference.mrp.rr_neural_mrp``. It is
    intentionally not included in ``__all__`` so ``from fairvote.inference.mrp
    import *`` does not force an optional PyTorch import.
    """
    if name in _NEURAL_EXPORTS:
        from fairvote.inference.mrp.rr_neural_mrp import (
            RRNeuralMRPFitInfo,
            RRNeuralMRPModel,
            NeuralRRMRPFitInfo,
            NeuralRRMRPModel,
        )

        mapping = {
            "RRNeuralMRPFitInfo": RRNeuralMRPFitInfo,
            "RRNeuralMRPModel": RRNeuralMRPModel,
            "NeuralRRMRPFitInfo": NeuralRRMRPFitInfo,
            "NeuralRRMRPModel": NeuralRRMRPModel,
        }
        globals().update(mapping)
        return mapping[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
