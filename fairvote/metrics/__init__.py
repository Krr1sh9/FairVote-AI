# fairvote/metrics/__init__.py
"""
Metrics utilities (errors, fairness, subgroup evaluation) for FairVote-AI.
"""

from fairvote.metrics.group_metrics import (
    GroupError,
    group_l1_errors,
    worst_group_l1,
    weighted_group_l1,
    quantile_group_l1,
    p90_group_l1,
    correct_winner,
    rmse_per_candidate,
    overall_rmse,
    error_ratio,
)

__all__ = [
    "GroupError",
    "group_l1_errors",
    "worst_group_l1",
    "weighted_group_l1",
    "quantile_group_l1",
    "p90_group_l1",
    "correct_winner",
    "rmse_per_candidate",
    "overall_rmse",
    "error_ratio",
]

