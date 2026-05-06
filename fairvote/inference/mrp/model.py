"""Compatibility layer for the canonical linear RR-aware MRP implementation.

Historically this module contained a second linear RR-aware model.  To avoid
ambiguous assessment paths, the implementation is now centralised in
``fairvote.inference.mrp.linear`` and design helpers live in
``fairvote.inference.mrp.design``.
"""

from fairvote.inference.mrp.design import DesignInfo, build_design_matrix
from fairvote.inference.mrp.linear import LinearRRMRPModel, RRMultinomialModel, softmax_rows

__all__ = [
    "DesignInfo",
    "build_design_matrix",
    "LinearRRMRPModel",
    "RRMultinomialModel",
    "softmax_rows",
]
