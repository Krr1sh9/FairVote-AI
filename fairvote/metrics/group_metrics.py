"""Group-level error and fairness-audit metrics.

The metrics in this file quantify how estimation error varies across groups.
They are audit tools: a low value is evidence of better empirical behaviour in
a scenario, not a mathematical guarantee of fairness.
"""

# fairvote/metrics/group_metrics.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def l1(a: np.ndarray, b: np.ndarray) -> float:
    """Return the L1 distance between two category distributions."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sum(np.abs(a - b)))


def _normalise(v: np.ndarray) -> np.ndarray:
    """Return a valid probability vector, falling back to uniform if needed."""
    v = np.asarray(v, dtype=float)
    s = float(np.sum(v))
    if s <= 0:
        return np.full_like(v, 1.0 / max(1, v.size), dtype=float)
    return v / s


@dataclass(frozen=True)
class GroupError:
    """Per-group error record used by fairness-audit summaries."""

    name: str
    mass: float
    err_l1: float


def group_l1_errors(
    est_by_group: dict[str, np.ndarray],
    truth_by_group: dict[str, np.ndarray],
    *,
    group_masses: dict[str, float] | None = None,
    normalise_masses: bool = True,
) -> list[GroupError]:
    """
    Compute L1 errors per group.

    Groups not present in both dictionaries are skipped because there is no
    comparable estimate/truth pair for an error calculation.

    Args:
      est_by_group: group_name -> estimated distribution (K,)
      truth_by_group: group_name -> true distribution (K,)
      group_masses: group_name -> mass/proportion/count (optional; default=1 for all)
      normalise_masses: if True, masses are normalised to sum to 1 over included groups

    Returns:
      list of GroupError with mass and L1 error
    """
    # Only compare groups present in both dictionaries; a group without either
    # an estimate or a reference distribution cannot have a meaningful L1 error.
    groups = sorted(set(est_by_group.keys()) & set(truth_by_group.keys()))
    if not groups:
        return []

    masses = np.array(
        [float(group_masses.get(g, 1.0)) if group_masses is not None else 1.0 for g in groups],
        dtype=float,
    )
    masses = np.clip(masses, 0.0, None)
    if normalise_masses:
        # Normalising masses makes weighted summaries comparable across filtered
        # group sets, including major-group-only dashboard views.
        masses = _normalise(masses)

    out: list[GroupError] = []
    for g, m in zip(groups, masses, strict=False):
        e = l1(est_by_group[g], truth_by_group[g])
        out.append(GroupError(name=g, mass=float(m), err_l1=float(e)))
    return out


def worst_group_l1(
    est_by_group: dict[str, np.ndarray],
    truth_by_group: dict[str, np.ndarray],
    *,
    group_masses: dict[str, float] | None = None,
    min_mass: float = 0.0,
    normalise_masses: bool = True,
) -> float:
    """
    Worst-case group L1 error, optionally only among groups with mass >= min_mass.

    This is the "major-group worst error" metric used to avoid tiny groups dominating.
    """
    errs = group_l1_errors(est_by_group, truth_by_group, group_masses=group_masses, normalise_masses=normalise_masses)
    if not errs:
        return float("nan")
    # Filter to groups whose normalised mass exceeds the minimum.  The small
    # tolerance accounts for floating-point imprecision in the normalised masses.
    filtered = [ge for ge in errs if ge.mass >= min_mass - 1e-12]
    if not filtered:
        # If no group passes the mass filter, returning NaN is safer than
        # inventing a misleading zero-error worst case.
        return float("nan")
    return float(max(ge.err_l1 for ge in filtered))


def weighted_group_l1(
    est_by_group: dict[str, np.ndarray],
    truth_by_group: dict[str, np.ndarray],
    *,
    group_masses: dict[str, float] | None = None,
    normalise_masses: bool = True,
) -> float:
    """
    Population-weighted average group L1 error.

    This is a more stable "fairness" metric than pure worst-case.
    """
    errs = group_l1_errors(est_by_group, truth_by_group, group_masses=group_masses, normalise_masses=normalise_masses)
    if not errs:
        return float("nan")
    return float(sum(ge.mass * ge.err_l1 for ge in errs))


def quantile_group_l1(
    est_by_group: dict[str, np.ndarray],
    truth_by_group: dict[str, np.ndarray],
    *,
    q: float = 0.9,
    group_masses: dict[str, float] | None = None,
    min_mass: float = 0.0,
    normalise_masses: bool = True,
    weighted_by_mass: bool = True,
) -> float:
    """
    Robust "almost-worst" group error metric (e.g., 90th percentile).

    Two modes:
      - weighted_by_mass=True: compute quantile under a mass-weighted distribution
      - weighted_by_mass=False: unweighted quantile across groups

    Optionally filters to groups with mass >= min_mass (major groups).

    Returns:
      quantile of group errors
    """
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must be in [0, 1].")

    errs = group_l1_errors(est_by_group, truth_by_group, group_masses=group_masses, normalise_masses=normalise_masses)
    if not errs:
        return float("nan")

    errs = [ge for ge in errs if ge.mass >= min_mass - 1e-12]
    if not errs:
        return float("nan")

    values = np.array([ge.err_l1 for ge in errs], dtype=float)
    if not weighted_by_mass:
        return float(np.quantile(values, q))

    weights = np.array([ge.mass for ge in errs], dtype=float)
    weights = _normalise(weights)

    # Weighted quantile via sorted cumulative weights: walk the sorted values
    # until cumulative weight reaches q.
    order = np.argsort(values)
    v_sorted = values[order]
    w_sorted = weights[order]
    cw = np.cumsum(w_sorted)
    idx = int(np.searchsorted(cw, q, side="left"))
    idx = min(max(idx, 0), len(v_sorted) - 1)
    return float(v_sorted[idx])


def p90_group_l1(
    est_by_group: dict[str, np.ndarray],
    truth_by_group: dict[str, np.ndarray],
    *,
    group_masses: dict[str, float] | None = None,
    min_mass: float = 0.0,
    normalise_masses: bool = True,
    weighted_by_mass: bool = True,
) -> float:
    """
    Convenience: 90th percentile group L1 error.
    """
    return quantile_group_l1(
        est_by_group,
        truth_by_group,
        q=0.9,
        group_masses=group_masses,
        min_mass=min_mass,
        normalise_masses=normalise_masses,
        weighted_by_mass=weighted_by_mass,
    )


# =============================================================================
# Winner prediction
# =============================================================================


def correct_winner(
    estimate: np.ndarray,
    truth: np.ndarray,
) -> bool:
    """
    Check whether the estimated distribution predicts the same winner
    (argmax) as the ground truth distribution.

    Returns True if argmax(estimate) == argmax(truth).

    In the case of ties, numpy.argmax returns the first occurrence,
    which matches how a "called winner" would work in polling.
    """
    est = np.asarray(estimate, dtype=float)
    tru = np.asarray(truth, dtype=float)
    if est.shape != tru.shape or est.ndim != 1:
        raise ValueError("estimate and truth must be 1D arrays of the same length.")
    return bool(np.argmax(est) == np.argmax(tru))


# =============================================================================
# RMSE per candidate
# =============================================================================


def rmse_per_candidate(
    trial_estimates: list[np.ndarray],
    truth: np.ndarray,
) -> np.ndarray:
    """
    Root Mean Squared Error per candidate across Monte Carlo trials.

    Args:
      trial_estimates: list of T distribution estimates, each shape (K,)
      truth: ground truth distribution, shape (K,)

    Returns:
      np.ndarray shape (K,) where entry j = sqrt( mean_t[ (est_t[j] - truth[j])^2 ] )
    """
    truth = np.asarray(truth, dtype=float)
    if truth.ndim != 1:
        raise ValueError("truth must be a 1D array.")
    k = truth.size
    if not trial_estimates:
        return np.full(k, float("nan"))

    mat = np.array([np.asarray(e, dtype=float) for e in trial_estimates])
    if mat.ndim != 2 or mat.shape[1] != k:
        raise ValueError("All trial estimates must be 1D arrays of the same length as truth.")

    # Squared error between each trial estimate and the ground truth, per
    # candidate category.
    squared_errors = (mat - truth[np.newaxis, :]) ** 2
    return np.sqrt(np.mean(squared_errors, axis=0))


def overall_rmse(
    trial_estimates: list[np.ndarray],
    truth: np.ndarray,
) -> float:
    """
    Scalar RMSE: root of the mean squared error averaged over both
    candidates and trials.

    Returns: float
    """
    per_candidate = rmse_per_candidate(trial_estimates, truth)
    return float(np.mean(per_candidate))


# =============================================================================
# Error ratio (fairness)
# =============================================================================


def error_ratio(
    est_by_group: dict[str, np.ndarray],
    truth_by_group: dict[str, np.ndarray],
    *,
    group_masses: dict[str, float] | None = None,
    min_mass: float = 0.0,
    normalise_masses: bool = True,
) -> float:
    """
    Disparity ratio: max-group L1 error divided by min-group L1 error.

    A ratio of 1.0 means all qualifying groups have equal error under this metric.
    Values >> 1 indicate that some groups are disproportionately distorted.

    Only groups with mass >= min_mass (after optional normalisation) are
    considered.  If fewer than 2 qualifying groups exist, returns NaN.

    This addresses the fairness metric described in §5.1 of the report:
    "error ratios reflecting if minority groups are disproportionately
    distorted".
    """
    errs = group_l1_errors(
        est_by_group,
        truth_by_group,
        group_masses=group_masses,
        normalise_masses=normalise_masses,
    )
    if not errs:
        return float("nan")

    filtered = [ge for ge in errs if ge.mass >= min_mass - 1e-12]
    if len(filtered) < 2:
        return float("nan")

    errors = [ge.err_l1 for ge in filtered]
    min_err = min(errors)
    max_err = max(errors)

    if min_err <= 1e-12:
        # Avoid division by zero; if the best group has approximately zero
        # error, a finite ratio is not meaningful.
        return float("inf") if max_err > 1e-12 else 1.0

    return float(max_err / min_err)
