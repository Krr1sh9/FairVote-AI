"""Recommendation utilities for choosing among experiment configurations.

The optimiser ranks already-computed experiment summaries under explicit
privacy, utility, and group-error constraints. It does not retrain models or
claim that a chosen method is universally best outside the evaluated scenarios.
"""

# fairvote/optimisation/recommend.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import math
import csv
from pathlib import Path


# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class Candidate:
    """One typed row from an experiment summary file."""

    scenario: str
    method: str
    epsilon: float
    n_rows: int

    mean_overall_l1: float
    std_overall_l1: float
    mean_overall_mae: float
    std_overall_mae: float

    # Legacy subgroup metrics (raw worst across groups present)
    mean_worst_region_l1: float
    std_worst_region_l1: float
    mean_worst_age_l1: float
    std_worst_age_l1: float

    # New subgroup/fairness metrics (mass-aware / robust)
    mean_worst_region_l1_major: float
    std_worst_region_l1_major: float
    mean_worst_age_l1_major: float
    std_worst_age_l1_major: float

    mean_weighted_region_l1: float
    std_weighted_region_l1: float
    mean_weighted_age_l1: float
    std_weighted_age_l1: float

    mean_p90_region_l1_major: float
    std_p90_region_l1_major: float
    mean_p90_age_l1_major: float
    std_p90_age_l1_major: float

    mean_n_effective: float


@dataclass(frozen=True)
class Constraints:
    """
    Any field can be None (meaning no constraint).

    Privacy:
      epsilon_max: enforce epsilon <= epsilon_max
      epsilon_min: enforce epsilon >= epsilon_min

    Utility:
      overall_l1_max: enforce mean_overall_l1 <= overall_l1_max
      overall_mae_max: enforce mean_overall_mae <= overall_mae_max

    Fairness (legacy):
      worst_region_l1_max: enforce mean_worst_region_l1 <= worst_region_l1_max
      worst_age_l1_max: enforce mean_worst_age_l1 <= worst_age_l1_max

    Fairness (recommended, mass-aware):
      worst_region_l1_major_max: enforce mean_worst_region_l1_major <= ...
      worst_age_l1_major_max: enforce mean_worst_age_l1_major <= ...
      weighted_region_l1_max: enforce mean_weighted_region_l1 <= ...
      weighted_age_l1_max: enforce mean_weighted_age_l1 <= ...
      p90_region_l1_major_max: enforce mean_p90_region_l1_major <= ...
      p90_age_l1_major_max: enforce mean_p90_age_l1_major <= ...

    Sample size:
      min_n_effective: enforce mean_n_effective >= min_n_effective
    """
    epsilon_max: Optional[float] = None
    epsilon_min: Optional[float] = None

    overall_l1_max: Optional[float] = None
    overall_mae_max: Optional[float] = None

    worst_region_l1_max: Optional[float] = None
    worst_age_l1_max: Optional[float] = None

    worst_region_l1_major_max: Optional[float] = None
    worst_age_l1_major_max: Optional[float] = None

    weighted_region_l1_max: Optional[float] = None
    weighted_age_l1_max: Optional[float] = None

    p90_region_l1_major_max: Optional[float] = None
    p90_age_l1_major_max: Optional[float] = None

    min_n_effective: Optional[float] = None


@dataclass(frozen=True)
class Objective:
    """
    Objective for ranking feasible candidates.

    primary: which metric to minimise
    tie_breakers: additional metrics to minimise (in order).
    prefer_lower_epsilon: if True, prefer smaller epsilon after metric comparisons.
    """
    primary: str = "mean_overall_l1"
    tie_breakers: Tuple[str, ...] = (
        "mean_worst_region_l1_major",
        "mean_p90_region_l1_major",
        "mean_worst_age_l1_major",
        "mean_p90_age_l1_major",
        "mean_weighted_region_l1",
        "mean_weighted_age_l1",
        "mean_overall_mae",
    )
    prefer_lower_epsilon: bool = True


@dataclass(frozen=True)
class Recommendation:
    """Chosen candidate, or an explanation when constraints leave none."""

    scenario: str
    chosen: Optional[Candidate]
    feasible_count: int
    total_count: int
    reason_if_none: Optional[str] = None


# =============================================================================
# CSV parsing
# =============================================================================

def read_summary_csv(path: Path) -> List[Candidate]:
    """
    Read the summary.csv produced by experiments/mrp_vs_baselines.py

    This function is tolerant to older summary.csv files:
    missing columns become NaN and will be ignored by constraints/objectives unless used.
    """
    path = Path(path)
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader]

    cands: List[Candidate] = []
    for r in rows:
        n_rows = _as_int(r.get("n_rows", "0"))
        if n_rows <= 0:
            continue

        c = Candidate(
            scenario=str(r.get("scenario", "")),
            method=str(r.get("method", "")),
            epsilon=_as_float(r.get("epsilon", "nan")),
            n_rows=n_rows,

            mean_overall_l1=_as_float(r.get("mean_overall_l1", "nan")),
            std_overall_l1=_as_float(r.get("std_overall_l1", "nan")),
            mean_overall_mae=_as_float(r.get("mean_overall_mae", "nan")),
            std_overall_mae=_as_float(r.get("std_overall_mae", "nan")),

            mean_worst_region_l1=_as_float(r.get("mean_worst_region_l1", "nan")),
            std_worst_region_l1=_as_float(r.get("std_worst_region_l1", "nan")),
            mean_worst_age_l1=_as_float(r.get("mean_worst_age_l1", "nan")),
            std_worst_age_l1=_as_float(r.get("std_worst_age_l1", "nan")),

            mean_worst_region_l1_major=_as_float(r.get("mean_worst_region_l1_major", "nan")),
            std_worst_region_l1_major=_as_float(r.get("std_worst_region_l1_major", "nan")),
            mean_worst_age_l1_major=_as_float(r.get("mean_worst_age_l1_major", "nan")),
            std_worst_age_l1_major=_as_float(r.get("std_worst_age_l1_major", "nan")),

            mean_weighted_region_l1=_as_float(r.get("mean_weighted_region_l1", "nan")),
            std_weighted_region_l1=_as_float(r.get("std_weighted_region_l1", "nan")),
            mean_weighted_age_l1=_as_float(r.get("mean_weighted_age_l1", "nan")),
            std_weighted_age_l1=_as_float(r.get("std_weighted_age_l1", "nan")),

            mean_p90_region_l1_major=_as_float(r.get("mean_p90_region_l1_major", "nan")),
            std_p90_region_l1_major=_as_float(r.get("std_p90_region_l1_major", "nan")),
            mean_p90_age_l1_major=_as_float(r.get("mean_p90_age_l1_major", "nan")),
            std_p90_age_l1_major=_as_float(r.get("std_p90_age_l1_major", "nan")),

            mean_n_effective=_as_float(r.get("mean_n_effective", "nan")),
        )

        if c.scenario and c.method and _finite(c.epsilon):
            cands.append(c)

    return cands


def _as_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _as_int(x: str) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def _finite(x: float) -> bool:
    return x is not None and not math.isnan(x) and math.isfinite(x)


# =============================================================================
# Filtering + objective ranking
# =============================================================================

def filter_candidates(
    cands: Sequence[Candidate],
    *,
    constraints: Constraints,
    allowed_methods: Optional[Sequence[str]] = None,
) -> List[Candidate]:
    """Return candidates satisfying all explicit user constraints."""

    allowed = set(allowed_methods) if allowed_methods else None
    out: List[Candidate] = []
    for c in cands:
        if allowed is not None and c.method not in allowed:
            continue
        if not _satisfies(c, constraints):
            continue
        out.append(c)
    return out


def recommend_per_scenario(
    cands: Sequence[Candidate],
    *,
    constraints: Constraints,
    objective: Objective = Objective(),
    allowed_methods: Optional[Sequence[str]] = None,
    scenarios: Optional[Sequence[str]] = None,
) -> List[Recommendation]:
    """Select the lowest-objective feasible candidate within each scenario."""

    if scenarios is None:
        scenarios = sorted({c.scenario for c in cands})
    else:
        scenarios = list(scenarios)

    recs: List[Recommendation] = []
    for s in scenarios:
        pool = [c for c in cands if c.scenario == s]
        total = len(pool)
        feasible = filter_candidates(pool, constraints=constraints, allowed_methods=allowed_methods)

        if not feasible:
            reason = _explain_infeasible(pool, constraints)
            recs.append(
                Recommendation(
                    scenario=s,
                    chosen=None,
                    feasible_count=0,
                    total_count=total,
                    reason_if_none=reason,
                )
            )
            continue

        chosen = _argmin(feasible, objective)
        recs.append(
            Recommendation(
                scenario=s,
                chosen=chosen,
                feasible_count=len(feasible),
                total_count=total,
            )
        )
    return recs


def pareto_frontier(
    cands: Sequence[Candidate],
    *,
    scenario: str,
    method: Optional[str] = None,
    x: str = "epsilon",
    y: str = "mean_overall_l1",
    z: str = "mean_worst_region_l1_major",
) -> List[Candidate]:
    """
    Simple 2D+1 Pareto frontier:
      - minimise y (utility loss)
      - minimise z (fairness loss)
      - epsilon on x-axis for inspection (not used for domination unless you choose y/z as epsilon)

    Dominance:
      A dominates B if y_A <= y_B and z_A <= z_B and at least one strict.
    """
    pool = [c for c in cands if c.scenario == scenario and (method is None or c.method == method)]
    pool = [c for c in pool if _finite(_get(c, y)) and _finite(_get(c, z)) and _finite(c.epsilon)]
    front: List[Candidate] = []

    for c in pool:
        dominated = False
        for d in pool:
            if d is c:
                continue
            # Weak dominance with tolerance: d dominates c if d is no worse
            # on both objectives and strictly better on at least one.
            if (_get(d, y) <= _get(c, y) + 1e-12) and (_get(d, z) <= _get(c, z) + 1e-12):
                if (_get(d, y) < _get(c, y) - 1e-12) or (_get(d, z) < _get(c, z) - 1e-12):
                    dominated = True
                    break
        if not dominated:
            front.append(c)

    front.sort(key=lambda c: (_get(c, x), _get(c, y), _get(c, z), c.method))
    return front


def write_pareto_csv(path: Path, candidates: Sequence[Candidate]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "scenario", "method", "epsilon", "n_rows", "mean_n_effective",
        "mean_overall_l1", "std_overall_l1",
        "mean_overall_mae", "std_overall_mae",

        "mean_worst_region_l1", "std_worst_region_l1",
        "mean_worst_age_l1", "std_worst_age_l1",

        "mean_worst_region_l1_major", "std_worst_region_l1_major",
        "mean_worst_age_l1_major", "std_worst_age_l1_major",

        "mean_weighted_region_l1", "std_weighted_region_l1",
        "mean_weighted_age_l1", "std_weighted_age_l1",

        "mean_p90_region_l1_major", "std_p90_region_l1_major",
        "mean_p90_age_l1_major", "std_p90_age_l1_major",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for c in candidates:
            w.writerow({
                "scenario": c.scenario,
                "method": c.method,
                "epsilon": c.epsilon,
                "n_rows": c.n_rows,
                "mean_n_effective": c.mean_n_effective,

                "mean_overall_l1": c.mean_overall_l1,
                "std_overall_l1": c.std_overall_l1,
                "mean_overall_mae": c.mean_overall_mae,
                "std_overall_mae": c.std_overall_mae,

                "mean_worst_region_l1": c.mean_worst_region_l1,
                "std_worst_region_l1": c.std_worst_region_l1,
                "mean_worst_age_l1": c.mean_worst_age_l1,
                "std_worst_age_l1": c.std_worst_age_l1,

                "mean_worst_region_l1_major": c.mean_worst_region_l1_major,
                "std_worst_region_l1_major": c.std_worst_region_l1_major,
                "mean_worst_age_l1_major": c.mean_worst_age_l1_major,
                "std_worst_age_l1_major": c.std_worst_age_l1_major,

                "mean_weighted_region_l1": c.mean_weighted_region_l1,
                "std_weighted_region_l1": c.std_weighted_region_l1,
                "mean_weighted_age_l1": c.mean_weighted_age_l1,
                "std_weighted_age_l1": c.std_weighted_age_l1,

                "mean_p90_region_l1_major": c.mean_p90_region_l1_major,
                "std_p90_region_l1_major": c.std_p90_region_l1_major,
                "mean_p90_age_l1_major": c.mean_p90_age_l1_major,
                "std_p90_age_l1_major": c.std_p90_age_l1_major,
            })


# =============================================================================
# Internal helpers
# =============================================================================

def _get(c: Candidate, field: str) -> float:
    if field == "epsilon":
        return float(c.epsilon)
    if not hasattr(c, field):
        raise KeyError(f"Unknown field '{field}'.")
    return float(getattr(c, field))


def _satisfies(c: Candidate, cons: Constraints) -> bool:
    # Each constraint is checked independently.  The 1e-12 tolerance prevents
    # floating-point boundary rejections from excluding valid candidates.
    # epsilon
    if cons.epsilon_max is not None and c.epsilon > cons.epsilon_max + 1e-12:
        return False
    if cons.epsilon_min is not None and c.epsilon < cons.epsilon_min - 1e-12:
        return False

    # utility
    if cons.overall_l1_max is not None and _finite(c.mean_overall_l1) and c.mean_overall_l1 > cons.overall_l1_max + 1e-12:
        return False
    if cons.overall_mae_max is not None and _finite(c.mean_overall_mae) and c.mean_overall_mae > cons.overall_mae_max + 1e-12:
        return False

    # fairness (legacy)
    if cons.worst_region_l1_max is not None and _finite(c.mean_worst_region_l1) and c.mean_worst_region_l1 > cons.worst_region_l1_max + 1e-12:
        return False
    if cons.worst_age_l1_max is not None and _finite(c.mean_worst_age_l1) and c.mean_worst_age_l1 > cons.worst_age_l1_max + 1e-12:
        return False

    # fairness (major/robust)
    if cons.worst_region_l1_major_max is not None and _finite(c.mean_worst_region_l1_major) and c.mean_worst_region_l1_major > cons.worst_region_l1_major_max + 1e-12:
        return False
    if cons.worst_age_l1_major_max is not None and _finite(c.mean_worst_age_l1_major) and c.mean_worst_age_l1_major > cons.worst_age_l1_major_max + 1e-12:
        return False

    if cons.weighted_region_l1_max is not None and _finite(c.mean_weighted_region_l1) and c.mean_weighted_region_l1 > cons.weighted_region_l1_max + 1e-12:
        return False
    if cons.weighted_age_l1_max is not None and _finite(c.mean_weighted_age_l1) and c.mean_weighted_age_l1 > cons.weighted_age_l1_max + 1e-12:
        return False

    if cons.p90_region_l1_major_max is not None and _finite(c.mean_p90_region_l1_major) and c.mean_p90_region_l1_major > cons.p90_region_l1_major_max + 1e-12:
        return False
    if cons.p90_age_l1_major_max is not None and _finite(c.mean_p90_age_l1_major) and c.mean_p90_age_l1_major > cons.p90_age_l1_major_max + 1e-12:
        return False

    # sample size
    if cons.min_n_effective is not None and _finite(c.mean_n_effective) and c.mean_n_effective < cons.min_n_effective - 1e-12:
        return False

    return True


def _argmin(cands: Sequence[Candidate], obj: Objective) -> Candidate:
    # Lexicographic sort key: primary metric first, then tie-breakers, then
    # optionally epsilon to prefer stronger privacy when metrics are equal.
    def key(c: Candidate) -> Tuple:
        primary = _get(c, obj.primary)
        ties = tuple(_get(c, t) for t in obj.tie_breakers)
        if obj.prefer_lower_epsilon:
            return (primary, *ties, c.epsilon)
        return (primary, *ties)

    return min(cands, key=key)


def _explain_infeasible(pool: Sequence[Candidate], cons: Constraints) -> str:
    if not pool:
        return "No candidates available for this scenario (n_rows==0 for all)."

    def any_ok(pred) -> bool:
        return any(pred(c) for c in pool)

    checks = []

    if cons.epsilon_max is not None:
        checks.append(("epsilon_max", any_ok(lambda c: c.epsilon <= cons.epsilon_max + 1e-12)))
    if cons.epsilon_min is not None:
        checks.append(("epsilon_min", any_ok(lambda c: c.epsilon >= cons.epsilon_min - 1e-12)))

    if cons.overall_l1_max is not None:
        checks.append(("overall_l1_max", any_ok(lambda c: c.mean_overall_l1 <= cons.overall_l1_max + 1e-12)))
    if cons.overall_mae_max is not None:
        checks.append(("overall_mae_max", any_ok(lambda c: c.mean_overall_mae <= cons.overall_mae_max + 1e-12)))

    if cons.worst_region_l1_max is not None:
        checks.append(("worst_region_l1_max", any_ok(lambda c: c.mean_worst_region_l1 <= cons.worst_region_l1_max + 1e-12)))
    if cons.worst_age_l1_max is not None:
        checks.append(("worst_age_l1_max", any_ok(lambda c: c.mean_worst_age_l1 <= cons.worst_age_l1_max + 1e-12)))

    if cons.worst_region_l1_major_max is not None:
        checks.append(("worst_region_l1_major_max", any_ok(lambda c: c.mean_worst_region_l1_major <= cons.worst_region_l1_major_max + 1e-12)))
    if cons.worst_age_l1_major_max is not None:
        checks.append(("worst_age_l1_major_max", any_ok(lambda c: c.mean_worst_age_l1_major <= cons.worst_age_l1_major_max + 1e-12)))

    if cons.weighted_region_l1_max is not None:
        checks.append(("weighted_region_l1_max", any_ok(lambda c: c.mean_weighted_region_l1 <= cons.weighted_region_l1_max + 1e-12)))
    if cons.weighted_age_l1_max is not None:
        checks.append(("weighted_age_l1_max", any_ok(lambda c: c.mean_weighted_age_l1 <= cons.weighted_age_l1_max + 1e-12)))

    if cons.p90_region_l1_major_max is not None:
        checks.append(("p90_region_l1_major_max", any_ok(lambda c: c.mean_p90_region_l1_major <= cons.p90_region_l1_major_max + 1e-12)))
    if cons.p90_age_l1_major_max is not None:
        checks.append(("p90_age_l1_major_max", any_ok(lambda c: c.mean_p90_age_l1_major <= cons.p90_age_l1_major_max + 1e-12)))

    if cons.min_n_effective is not None:
        checks.append(("min_n_effective", any_ok(lambda c: c.mean_n_effective >= cons.min_n_effective - 1e-12)))

    failing = [name for (name, ok) in checks if not ok]
    if failing:
        return "No feasible candidate meets: " + ", ".join(failing)

    return "No feasible candidate meets the combined constraints (individually each constraint is satisfiable)."
