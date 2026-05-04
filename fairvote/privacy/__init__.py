"""
Privacy primitives and estimators for FairVote-AI.

Includes:
  - LDP: k-ary Randomised Response (privatize_one/many, estimate_distribution)
  - Central DP: Laplace mechanism on aggregate counts (estimate_distribution_central_dp)
  - Uncertainty: bootstrap confidence intervals
"""

from fairvote.privacy.estimators import (
    bootstrap_ci,
    estimate_distribution,
    estimate_distribution_from_counts,
)
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
    # LDP mechanism
    "KaryRRParams",
    "rr_params",
    "privatize_one",
    "privatize_many",
    "counts_from_reports",
    # LDP estimation
    "estimate_distribution",
    "estimate_distribution_from_counts",
    "bootstrap_ci",
    # Central DP
    "laplace_mechanism",
    "estimate_distribution_central_dp",
]
