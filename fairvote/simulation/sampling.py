"""Sampling schemes used to construct synthetic polling datasets.

The functions here take a fully known simulated population and return sampled
respondents. They model data-collection bias for experiments; they do not
change the privacy mechanism or the respondent storage format.
"""

# fairvote/simulation/sampling.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from fairvote.simulation.population import Population


@dataclass(frozen=True)
class Sample:
    """
    A sampled subset of a synthetic population.

    The sample carries true categories because experiments need a scoring
    target. Production-style respondent exports should contain only randomized
    reported categories.

    A sampled subset of a population.

    - idx: indices into the Population arrays (shape: (n_sample,))
    - true_categories: realised ground-truth labels for sampled individuals
    - features: dict feature_name -> feature values for sampled individuals
    """
    idx: np.ndarray
    true_categories: np.ndarray
    features: Dict[str, np.ndarray]


def simple_random_sample(
    pop: Population,
    n: int,
    *,
    rng: Optional[np.random.Generator] = None,
    replace: bool = False,
) -> Sample:
    """
    Simple Random Sampling (SRS) over individuals.

    Args:
      pop: Population
      n: sample size
      replace: sample with replacement (False is typical for polling)
    """
    N = pop.true_categories.size
    if n <= 0:
        raise ValueError("n must be > 0")
    if (not replace) and n > N:
        raise ValueError(f"n={n} cannot exceed population size N={N} when replace=False")

    if rng is None:
        rng = np.random.default_rng()

    idx = rng.choice(N, size=n, replace=replace)
    return _make_sample(pop, idx)


def stratified_sample(
    pop: Population,
    n: int,
    strata: Sequence[str],
    *,
    rng: Optional[np.random.Generator] = None,
    allocation: str = "proportional",
    min_per_stratum: int = 0,
    replace_within: bool = False,
) -> Sample:
    """
    Stratified sampling over one or more categorical features (strata).

    You provide strata features (e.g. ["region"] or ["region","age_group"]).
    The function builds stratum cells as unique combinations and samples within each.

    allocation:
      - "proportional": sample proportionally to stratum size
      - "equal": equal sample per stratum cell (with adjustment if impossible)
      - "sqrt": Neyman-ish heuristic: proportional to sqrt(size) (oversamples small groups)

    min_per_stratum:
      - enforce a minimum number per cell (if feasible)
      - useful for fairness / subgroup estimation experiments

    Notes:
      - This function is deterministic given rng and stable ordering.
      - If min_per_stratum makes the request impossible, raises ValueError.
    """
    N = pop.true_categories.size
    if n <= 0:
        raise ValueError("n must be > 0")
    if n > N and not replace_within:
        raise ValueError(f"n={n} cannot exceed population size N={N} when replace_within=False")

    if rng is None:
        rng = np.random.default_rng()

    if not strata:
        raise ValueError("strata must be a non-empty list of feature names.")
    for f in strata:
        if f not in pop.features:
            raise KeyError(f"Unknown stratum feature '{f}'. Available: {list(pop.features.keys())}")

    # Build stratum cell ids for each person (integer-coded).  This encodes
    # the cross-classification of all strata features as a single integer
    # cell identifier using NumPy's structured-array unique-row technique.
    M = np.stack([pop.features[f].astype(int) for f in strata], axis=1)  # (N, d)
    dtype = np.dtype([(f"f{i}", M.dtype) for i in range(M.shape[1])])
    structured = M.view(dtype).reshape(-1)
    uniq, inv = np.unique(structured, return_inverse=True)  # uniq cells, inv is cell id per person

    cell_count = np.bincount(inv).astype(int)
    C = cell_count.size

    if allocation not in {"proportional", "equal", "sqrt"}:
        raise ValueError("allocation must be one of: 'proportional', 'equal', 'sqrt'")

    # Compute target sample sizes per cell
    target = _allocate_counts(cell_count, n, allocation=allocation)

    if min_per_stratum > 0:
        # Check feasibility: total minimum <= n and each cell has enough people if no replacement
        if min_per_stratum * C > n:
            raise ValueError(
                f"min_per_stratum={min_per_stratum} across C={C} strata cells exceeds n={n}"
            )

        # Enforce min and reallocate remainder
        target = _enforce_minimum(cell_count, target, n, min_per_stratum, replace_within=replace_within)

    # Now sample within each cell
    idx_out: List[int] = []
    for cell_id in range(C):
        m = int(target[cell_id])
        if m <= 0:
            continue

        members = np.where(inv == cell_id)[0]
        if members.size == 0:
            continue

        if (not replace_within) and m > members.size:
            raise ValueError(
                f"Requested m={m} from a stratum with size {members.size} (replace_within=False). "
                f"Try smaller n, different allocation, or replace_within=True."
            )

        chosen = rng.choice(members, size=m, replace=replace_within)
        idx_out.extend(chosen.tolist())

    idx = np.array(idx_out, dtype=int)
    # Shuffle to remove any ordering patterns that could leak stratum structure
    # to downstream analysis.
    rng.shuffle(idx)
    return _make_sample(pop, idx)


def biased_frame_sample(
    pop: Population,
    n: int,
    *,
    rng: Optional[np.random.Generator] = None,
    feature: str = "region",
    level_multipliers: Optional[Dict[str, float]] = None,
    replace: bool = False,
) -> Sample:
    """
    Sample from a biased sampling frame: some groups are over/under-represented.

    This models real-world issues like:
      - over-sampling urban areas (online polls)
      - under-sampling certain regions/demographics

    Args:
      feature: which categorical feature the frame bias applies to (default: region)
      level_multipliers: dict level_name -> multiplier (e.g., {"London": 1.4, "Wales": 0.7})
                         Multipliers are applied to base inclusion weights.
                         Levels not provided default to 1.0.
    """
    N = pop.true_categories.size
    if n <= 0:
        raise ValueError("n must be > 0")
    if (not replace) and n > N:
        raise ValueError(f"n={n} cannot exceed population size N={N} when replace=False")
    if rng is None:
        rng = np.random.default_rng()

    if feature not in pop.features:
        raise KeyError(f"Unknown feature '{feature}'. Available: {list(pop.features.keys())}")

    x = pop.features[feature].astype(int)
    levels = pop.feature_levels[feature]

    multipliers = np.ones(len(levels), dtype=float)
    if level_multipliers:
        name_to_idx = {name: i for i, name in enumerate(levels)}
        for name, mult in level_multipliers.items():
            if name not in name_to_idx:
                raise KeyError(f"Unknown level '{name}' for feature '{feature}'. Levels: {levels}")
            if mult <= 0:
                raise ValueError("All multipliers must be > 0.")
            multipliers[name_to_idx[name]] = float(mult)

    weights = multipliers[x].astype(float)
    weights = weights / weights.sum()

    idx = rng.choice(N, size=n, replace=replace, p=weights)
    return _make_sample(pop, idx)


def nonresponse(
    sample: Sample,
    pop: Population,
    *,
    rng: Optional[np.random.Generator] = None,
    feature_response_rates: Optional[Dict[str, Dict[str, float]]] = None,
    base_rate: float = 0.85,
) -> Sample:
    """
    Apply a non-response mechanism to an existing sample.

    Respondents are retained with probability depending on feature levels (if provided),
    otherwise at a base_rate.

    feature_response_rates format:
      {
        "age_group": {"18-24": 0.65, "65+": 0.92},
        "urbanicity": {"Urban": 0.78, "Rural": 0.90}
      }

    Levels not specified default to base_rate (clipped into (0,1]).
    """
    if rng is None:
        rng = np.random.default_rng()

    if not (0.0 < base_rate <= 1.0):
        raise ValueError("base_rate must be in (0,1].")

    n0 = sample.idx.size
    if n0 == 0:
        return sample

    keep_prob = np.full(n0, base_rate, dtype=float)

    if feature_response_rates:
        for feat, level_map in feature_response_rates.items():
            if feat not in sample.features:
                raise KeyError(
                    f"Feature '{feat}' not present in sample.features. Available: {list(sample.features.keys())}"
                )
            levels = pop.feature_levels[feat]
            name_to_idx = {name: i for i, name in enumerate(levels)}

            x = sample.features[feat].astype(int)
            for level_name, rate in level_map.items():
                if level_name not in name_to_idx:
                    raise KeyError(f"Unknown level '{level_name}' for feature '{feat}'. Levels: {levels}")
                if not (0.0 < float(rate) <= 1.0):
                    raise ValueError("Response rates must be in (0,1].")
                level_idx = name_to_idx[level_name]
                keep_prob[x == level_idx] = float(rate)

    u = rng.random(n0)
    keep = u < keep_prob

    idx = sample.idx[keep]
    return _make_sample(pop, idx)


# ---------------------------
# Internal helpers
# ---------------------------

def _make_sample(pop: Population, idx: np.ndarray) -> Sample:
    idx = np.asarray(idx, dtype=int)
    true_categories = pop.true_categories[idx]
    features = {name: arr[idx] for name, arr in pop.features.items()}
    return Sample(idx=idx, true_categories=true_categories, features=features)


def _allocate_counts(cell_count: np.ndarray, n: int, *, allocation: str) -> np.ndarray:
    C = cell_count.size
    if C == 0:
        raise ValueError("No strata cells found (unexpected).")

    if allocation == "proportional":
        weights = cell_count.astype(float)
    elif allocation == "equal":
        weights = np.ones(C, dtype=float)
    elif allocation == "sqrt":
        weights = np.sqrt(cell_count.astype(float))
    else:
        raise ValueError("Invalid allocation.")

    if weights.sum() <= 0:
        raise ValueError("Invalid stratum weights (sum <= 0).")

    weights = weights / weights.sum()
    raw = weights * n

    # Largest-remainder method: floor all allocations, then distribute the
    # leftover units one-by-one to the cells with the largest fractional
    # remainders.  This guarantees the total is exactly n.
    base = np.floor(raw).astype(int)
    remainder = raw - base
    needed = n - int(base.sum())

    if needed > 0:
        order = np.argsort(-remainder)  # descending remainder
        base[order[:needed]] += 1

    return base.astype(int)


def _enforce_minimum(
    cell_count: np.ndarray,
    target: np.ndarray,
    n: int,
    min_per_stratum: int,
    *,
    replace_within: bool,
) -> np.ndarray:
    C = cell_count.size
    out = target.copy()

    # First, set minimums
    out = np.maximum(out, min_per_stratum)

    # If sampling without replacement, ensure feasibility for each cell
    if not replace_within:
        if np.any(out > cell_count):
            # If any cell doesn't have enough members, impossible
            bad = np.where(out > cell_count)[0]
            raise ValueError(
                f"Minimum/target exceeds available members in some strata cells. "
                f"Cells: {bad.tolist()} (try reduce min_per_stratum or enable replace_within=True)."
            )

    # Adjust to sum to n
    total = int(out.sum())
    if total == n:
        return out

    if total < n:
        # Add remaining by proportional to remaining capacity (or size if replacement)
        remaining = n - total
        if replace_within:
            cap = cell_count.astype(float)
        else:
            cap = (cell_count - out).astype(float)
            cap[cap < 0] = 0.0

        if cap.sum() <= 0:
            # No capacity to add (shouldn't happen if total<n with replacement or any slack)
            return out

        weights = cap / cap.sum()
        add_raw = weights * remaining
        add = np.floor(add_raw).astype(int)
        rem = add_raw - add
        need2 = remaining - int(add.sum())
        if need2 > 0:
            order = np.argsort(-rem)
            add[order[:need2]] += 1
        out += add
        return out

    # total > n: need to subtract
    excess = total - n

    # Prefer subtracting from cells that are above min
    slack = out - min_per_stratum
    if np.all(slack <= 0):
        # All are at minimum; impossible
        raise ValueError("Cannot satisfy min_per_stratum while meeting total n (all cells at minimum).")

    # Subtract one by one from the cells that have the most headroom above
    # the minimum, preserving the minimum guarantee.
    for _ in range(excess):
        candidates = np.where(slack > 0)[0]
        if candidates.size == 0:
            raise ValueError("Cannot reduce targets to meet n without violating min_per_stratum.")
        # choose the cell with the largest slack (ties broken by largest current target)
        best = candidates[np.argmax(slack[candidates])]
        out[best] -= 1
        slack[best] -= 1

    return out
