"""
Privacy mechanisms for FairVote-AI.

Includes:
  - Client-side LDP (k-ary Randomised Response)
  - Central DP (Laplace mechanism on aggregate counts)
"""

from fairvote.privacy.mechanisms.kary_rr import (
    KaryRRParams,
    counts_from_reports,
    privatize_many,
    privatize_one,
    rr_params,
)
from fairvote.privacy.mechanisms.laplace_mechanism import (
    estimate_distribution_central_dp,
    laplace_mechanism,
)

__all__ = [
    # LDP (k-ary RR)
    "KaryRRParams",
    "rr_params",
    "privatize_one",
    "privatize_many",
    "counts_from_reports",
    # Central DP (Laplace)
    "laplace_mechanism",
    "estimate_distribution_central_dp",
]
