"""Synthetic population generation for controlled polling experiments.

The generated population is deliberately artificial: it provides known latent
preferences and demographic structure so estimators can be evaluated against a
clear ground truth. These synthetic true choices are for offline evaluation
only; they are not part of the respondent collection protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Population:
    """
    Container for the generated population and evaluation-only ground truth.

    The true categories allow experiments to score estimators. They are not
    analogous to values submitted by the respondent app, where only randomized
    reports are accepted by the server.

    Synthetic population for polling experiments.

    - features: dict of feature_name -> np.ndarray of shape (N,)
      (values are integer-coded categories for each feature)
    - feature_levels: dict of feature_name -> list of human-readable level names
    - true_probs: np.ndarray of shape (N, K), where true_probs[i, j] = P(true category=j for person i)
    - true_categories: np.ndarray of shape (N,), sampled from true_probs per person (ground truth labels)
    - category_names: list of K names (e.g., "Party A", ...)
    """
    features: Dict[str, np.ndarray]
    feature_levels: Dict[str, List[str]]
    true_probs: np.ndarray
    true_categories: np.ndarray
    category_names: List[str]


# ---------------------------
# Public API
# ---------------------------

def make_realistic_uk_like_population(
    n: int,
    k: int,
    *,
    seed: int = 123,
    category_names: Optional[Sequence[str]] = None,
) -> Population:
    """
    Generate a more realistic synthetic population with correlated demographics.

    Features included (categorical):
      - region (12 UK-like regions)
      - age_group (6 buckets)
      - education (3 buckets)
      - gender (3 buckets)
      - urbanicity (3 buckets)

    Ground-truth preference model:
      - person-level multinomial (K-way) preferences generated via a softmax over logits
      - logits depend on region/age/education/gender/urbanicity + mild interactions + noise
      - produces realistic subgroup differences (useful later for bias + MRP evaluation)

    Notes:
      - The distributions are "plausible", not claiming to match official census shares.
      - This is designed to stress-test polling estimators, not to emulate a specific election.
    """
    if n <= 0:
        raise ValueError("n must be > 0")
    if k < 2:
        raise ValueError("k must be >= 2")

    rng = np.random.default_rng(seed)

    # ---- Feature levels (human readable) ----
    region_levels = [
        "London",
        "South East",
        "South West",
        "East of England",
        "West Midlands",
        "East Midlands",
        "North West",
        "North East",
        "Yorkshire & Humber",
        "Scotland",
        "Wales",
        "Northern Ireland",
    ]

    age_levels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

    edu_levels = ["No degree", "Some college/A-level", "Degree+"]
    # This is a demographic value, not a poll option. The canonical poll options
    # remain Labour, Conservative, Reform, LibDem, and Green.
    gender_levels = ["Female", "Male", "Other/Prefer not to say"]
    urban_levels = ["Urban", "Suburban", "Rural"]

    feature_levels = {
        "region": region_levels,
        "age_group": age_levels,
        "education": edu_levels,
        "gender": gender_levels,
        "urbanicity": urban_levels,
    }

    # ---- Marginal distributions (plausible defaults) ----
    # Region weights roughly shaped like population shares (not exact).
    region_w = np.array([0.13, 0.14, 0.09, 0.09, 0.09, 0.07, 0.11, 0.05, 0.08, 0.08, 0.05, 0.02], dtype=float)
    region_w = region_w / region_w.sum()

    # Age: more weight in working age + older cohorts
    age_w = np.array([0.10, 0.16, 0.16, 0.17, 0.17, 0.24], dtype=float)
    age_w = age_w / age_w.sum()

    # Gender: close to 50/50 with small "other"
    gender_w = np.array([0.495, 0.495, 0.01], dtype=float)

    # Urbanicity: most people urban/suburban
    urban_w = np.array([0.55, 0.30, 0.15], dtype=float)

    # ---- Sample base features (with correlations added later) ----
    region = rng.choice(len(region_levels), size=n, p=region_w)
    age_group = rng.choice(len(age_levels), size=n, p=age_w)
    gender = rng.choice(len(gender_levels), size=n, p=gender_w)
    urbanicity = rng.choice(len(urban_levels), size=n, p=urban_w)

    # ---- Education correlated with age + region + urbanicity ----
    # We’ll generate a "degree propensity" score and then sample education levels.
    # Higher in London / urban, lower in older cohorts, etc.
    # This gives realistic correlation structure for later MRP.
    region_degree_boost = _make_region_boost(len(region_levels), rng, high_regions={0, 1, 3}, low_regions={7, 10, 11})
    age_degree_boost = np.array([0.35, 0.30, 0.20, 0.05, -0.10, -0.25], dtype=float)  # younger -> more likely degree
    urban_degree_boost = np.array([0.20, 0.05, -0.15], dtype=float)

    # The degree propensity score combines additive effects from region,
    # age, and urbanicity, plus Gaussian noise, to model realistic
    # inter-feature correlations.  MRP benefits from this structure.
    degree_score = (
        region_degree_boost[region]
        + age_degree_boost[age_group]
        + urban_degree_boost[urbanicity]
        + rng.normal(0.0, 0.35, size=n)
    )

    # Map score -> probabilities for 3 education buckets
    # Use soft "slices": low score => No degree; mid => Some; high => Degree+
    p_degree_plus = _sigmoid(degree_score - 0.35)
    p_no_degree = _sigmoid(-(degree_score + 0.15))
    p_some = np.clip(1.0 - p_degree_plus - p_no_degree, 0.05, 0.90)
    # Renormalize
    denom = p_degree_plus + p_some + p_no_degree
    p_degree_plus /= denom
    p_some /= denom
    p_no_degree /= denom

    education = np.empty(n, dtype=int)
    u = rng.random(n)
    education[u < p_no_degree] = 0
    education[(u >= p_no_degree) & (u < p_no_degree + p_some)] = 1
    education[u >= (p_no_degree + p_some)] = 2

    # ---- Category names ----
    if category_names is None:
        category_names = [f"Option {i}" for i in range(k)]
    else:
        if len(category_names) != k:
            raise ValueError(f"category_names must have length k={k}")
        category_names = list(category_names)

    # ---- Build multinomial preference model (logits -> softmax) ----
    # The effects below create subgroup structure for MRP to learn. The values
    # are synthetic and are not intended to represent real party support.
    # Logits shape: (n, k)
    logits = np.zeros((n, k), dtype=float)

    # Global base preference (gives a realistic non-uniform baseline)
    base = rng.normal(0.0, 0.35, size=k)
    base -= base.mean()
    logits += base

    # Feature effects (each produces (n, k))
    logits += _effect_table("region", region, len(region_levels), k, rng, scale=0.55)
    logits += _effect_table("age_group", age_group, len(age_levels), k, rng, scale=0.40)
    logits += _effect_table("education", education, len(edu_levels), k, rng, scale=0.45)
    logits += _effect_table("gender", gender, len(gender_levels), k, rng, scale=0.20)
    logits += _effect_table("urbanicity", urbanicity, len(urban_levels), k, rng, scale=0.25)

    # Mild interaction: education x region (degree polarisation varies by region).
    # This makes the MRP task less trivial than estimating independent margins.
    logits += _interaction_region_education(region, education, k, rng)

    # Add person-level noise (captures unobserved factors)
    logits += rng.normal(0.0, 0.25, size=(n, k))

    # Softmax converts logits to per-person latent probabilities; the noise
    # term ensures no two people in the same cell share identical preferences.
    true_probs = _softmax(logits)
    # Realise a single concrete category per person by sampling from the
    # individual probability vector. This "true" label exists only for
    # offline evaluation.
    true_categories = _sample_categorical_rows(true_probs, rng)

    features = {
        "region": region,
        "age_group": age_group,
        "education": education,
        "gender": gender,
        "urbanicity": urbanicity,
    }

    return Population(
        features=features,
        feature_levels=feature_levels,
        true_probs=true_probs,
        true_categories=true_categories,
        category_names=list(category_names),
    )


def overall_true_distribution(pop: Population) -> np.ndarray:
    """
    Overall ground-truth distribution from the realised true_categories.
    Useful for evaluation baselines.
    """
    k = len(pop.category_names)
    return np.bincount(pop.true_categories, minlength=k).astype(float) / pop.true_categories.size


def subgroup_true_distribution(pop: Population, feature: str) -> Dict[str, np.ndarray]:
    """
    Return ground-truth distributions per level of a given feature.

    Output: dict[level_name] -> np.ndarray shape (K,)
    """
    if feature not in pop.features:
        raise KeyError(f"Unknown feature '{feature}'. Available: {list(pop.features.keys())}")

    x = pop.features[feature]
    levels = pop.feature_levels[feature]
    k = len(pop.category_names)

    out: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(levels):
        mask = (x == idx)
        if not np.any(mask):
            continue
        counts = np.bincount(pop.true_categories[mask], minlength=k).astype(float)
        out[name] = counts / counts.sum()
    return out


def poststrat_table(pop: Population, by: Sequence[str]) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
    """
    Build a post-stratification table: unique cells and their counts.

    Returns:
      cells: np.ndarray shape (C, len(by)) integer-coded feature levels
      counts: np.ndarray shape (C,) counts in each cell
      level_names: list per feature containing the names of each level

    Example:
      cells, counts, level_names = poststrat_table(pop, by=["region","age_group","education"])
    """
    for f in by:
        if f not in pop.features:
            raise KeyError(f"Unknown feature '{f}'. Available: {list(pop.features.keys())}")

    mats = [pop.features[f].astype(int) for f in by]
    M = np.stack(mats, axis=1)  # (n, d)

    # Use a structured-array view to efficiently find unique row combinations
    # across the feature columns without Python-level loops.
    dtype = np.dtype([(f"f{i}", M.dtype) for i in range(M.shape[1])])
    structured = M.view(dtype).reshape(-1)

    uniq, inv = np.unique(structured, return_inverse=True)
    counts = np.bincount(inv).astype(int)

    cells = uniq.view(M.dtype).reshape(-1, M.shape[1])
    level_names = [pop.feature_levels[f] for f in by]
    return cells, counts, level_names


# ---------------------------
# Internal helpers
# ---------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / np.sum(expz, axis=1, keepdims=True)


def _sample_categorical_rows(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Sample one categorical outcome per row from probs (n, k).
    """
    n, k = probs.shape
    # Inverse CDF trick: draw a single U(0,1) per row and compare against
    # the cumulative probability vector to select a category in O(n*k) time.
    u = rng.random(n)
    cdf = np.cumsum(probs, axis=1)
    return np.sum(cdf < u[:, None], axis=1).astype(int)


def _make_region_boost(num_regions: int, rng: np.random.Generator, *, high_regions: set, low_regions: set) -> np.ndarray:
    """
    Create a region boost vector for correlated feature generation (education propensity).
    """
    boost = rng.normal(0.0, 0.10, size=num_regions)
    for r in high_regions:
        if 0 <= r < num_regions:
            boost[r] += 0.25
    for r in low_regions:
        if 0 <= r < num_regions:
            boost[r] -= 0.20
    return boost


def _effect_table(
    name: str,
    x: np.ndarray,
    num_levels: int,
    k: int,
    rng: np.random.Generator,
    *,
    scale: float,
) -> np.ndarray:
    """
    Create a level-by-category effect table and apply it to people.

    Returns an (n, k) effect matrix.
    """
    # Level effects: (num_levels, k)
    E = rng.normal(0.0, scale, size=(num_levels, k))
    # Center each level (avoid adding huge bias across all categories)
    E = E - E.mean(axis=1, keepdims=True)
    return E[x]


def _interaction_region_education(
    region: np.ndarray,
    education: np.ndarray,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Mild region x education interaction:
      - degree holders in some regions shift preference slightly
      - no-degree in some regions shift differently
    """
    n = region.size
    out = np.zeros((n, k), dtype=float)

    # Choose a few region indices to have stronger interaction (e.g., London, South East, North West)
    special_regions = {0, 1, 6}
    deg = (education == 2)
    nodeg = (education == 0)

    # Interaction vectors (k,)
    v_deg = rng.normal(0.0, 0.18, size=k)
    v_deg -= v_deg.mean()
    v_nodeg = rng.normal(0.0, 0.18, size=k)
    v_nodeg -= v_nodeg.mean()

    is_special = np.isin(region, np.array(list(special_regions), dtype=int))
    out[is_special & deg] += v_deg
    out[is_special & nodeg] += v_nodeg
    return out
