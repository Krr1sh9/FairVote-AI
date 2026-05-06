"""
Privacy mechanisms for FairVote-AI.

Includes:
  - Client-side LDP (k-ary Randomised Response)
  - Central DP (Laplace mechanism on aggregate counts)
"""

from fairvote.privacy.mechanisms.kary_rr import (
    KaryRRParams,
    counts_from_reports,
    debias_distribution,
    invert_rr_counts,
    privatize_many,
    privatize_one,
    rr_params,
    rr_transition_matrix,
)
from fairvote.privacy.mechanisms.laplace_mechanism import (
    estimate_distribution_central_dp,
    laplace_mechanism,
)

__all__ = [
    # LDP (k-ary RR)
    "KaryRRParams",
    "rr_params",
    "rr_transition_matrix",
    "privatize_one",
    "privatize_many",
    "counts_from_reports",
    "invert_rr_counts",
    "debias_distribution",
    # Central DP (Laplace)
    "laplace_mechanism",
    "estimate_distribution_central_dp",
]
