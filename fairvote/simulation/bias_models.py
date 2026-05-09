"""Synthetic nonresponse and misreporting mechanisms for experiments.

These helpers intentionally operate on simulated truth so the project can test
how estimators behave under known bias. They should not be confused with the
real respondent app, which never receives or stores true choices.
"""

# fairvote/simulation/bias_models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from fairvote.simulation.population import Population
from fairvote.simulation.sampling import Sample


# =============================================================================
# Response / turnout (non-response) models
# =============================================================================

@dataclass(frozen=True)
class FeatureNonresponseProfile:
    """
    A simple, configurable non-response (turnout) profile.

    feature_response_rates format:
      {
        "age_group": {"18-24": 0.65, "65+": 0.92},
        "urbanicity": {"Urban": 0.78, "Rural": 0.90}
      }

    Levels not specified use base_rate.
    """
    base_rate: float = 0.85
    feature_response_rates: Optional[Dict[str, Dict[str, float]]] = None


@dataclass(frozen=True)
class PreferenceNonresponseProfile:
    """
    Non-ignorable non-response by true preference/category.

    category_response_rates:
      dict[category_index] -> response probability

    This is useful to simulate "shy voters" / "hard-to-reach supporters"
    where response depends on the target variable itself.
    """
    category_response_rates: Dict[int, float]


def apply_nonresponse(
    sample: Sample,
    pop: Population,
    *,
    rng: Optional[np.random.Generator] = None,
    feature_profile: Optional[FeatureNonresponseProfile] = None,
    preference_profile: Optional[PreferenceNonresponseProfile] = None,
) -> Sample:
    """
    Apply non-response to a Sample, returning a smaller Sample.

    This is a synthetic sampling distortion model, not a claim about any real
    electorate. It is used to test estimator robustness under controlled bias.

    - feature_profile captures turnout differences by demographics.
    - preference_profile captures turnout differences by the true category (non-ignorable).

    Response probability = base_rate (from feature_profile if provided, else 0.85)
    then overwritten (per-feature level) if specified,
    then multiplied by preference-based probability if provided.

    Notes:
      - This drops individuals (like real non-response).
      - The returned Sample preserves the original population indexing.
    """
    if rng is None:
        rng = np.random.default_rng()

    n0 = sample.idx.size
    if n0 == 0:
        return sample

    # Base probability. The default keeps most sampled respondents while still
    # allowing nonresponse to distort subgroup composition in experiments.
    base_rate = 0.85
    feature_response_rates = None
    if feature_profile is not None:
        base_rate = float(feature_profile.base_rate)
        feature_response_rates = feature_profile.feature_response_rates

    if not (0.0 < base_rate <= 1.0):
        raise ValueError("base_rate must be in (0,1].")

    # Start from a uniform base probability that all sampled respondents reply.
    p = np.full(n0, base_rate, dtype=float)

    # Feature-based adjustments overwrite the base probability for matching
    # levels, making demographic non-response explicit and reproducible.
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
                rate_f = float(rate)
                if not (0.0 < rate_f <= 1.0):
                    raise ValueError("Response rates must be in (0,1].")
                lvl = name_to_idx[level_name]
                p[x == lvl] = rate_f

    # Preference-based (non-ignorable) adjustments (multiply)
    if preference_profile is not None:
        rates = preference_profile.category_response_rates
        t = sample.true_categories.astype(int)

        mult = np.ones(n0, dtype=float)
        for cat, rate in rates.items():
            rate_f = float(rate)
            if not (0.0 < rate_f <= 1.0):
                raise ValueError("Preference response rates must be in (0,1].")
            mult[t == int(cat)] = rate_f

        # Multiplicative composition: demographic and preference effects
        # interact.  A respondent in a low-response age group who also supports
        # a "shy" category faces both penalties simultaneously.
        p = p * mult

    p = np.clip(p, 0.0, 1.0)
    keep = rng.random(n0) < p

    idx = sample.idx[keep]
    return Sample(
        idx=idx,
        true_categories=pop.true_categories[idx],
        features={name: arr[idx] for name, arr in pop.features.items()},
    )


def make_default_feature_nonresponse_profile() -> FeatureNonresponseProfile:
    """
    A plausible default non-response profile:
    - younger adults respond less
    - urban respondents slightly less likely to respond than rural/suburban
    """
    return FeatureNonresponseProfile(
        base_rate=0.85,
        feature_response_rates={
            "age_group": {"18-24": 0.70, "25-34": 0.78, "35-44": 0.85, "45-54": 0.88, "55-64": 0.90, "65+": 0.92},
            "urbanicity": {"Urban": 0.82, "Suburban": 0.87, "Rural": 0.90},
        },
    )


# =============================================================================
# Misreporting models (pre-LDP response bias)
# =============================================================================

@dataclass(frozen=True)
class MisreportModel:
    """
    Misreporting model represented as a confusion matrix C of shape (K, K):

      C[i, j] = P(stated=j | true=i)

    Rows must sum to 1.

    This models systematic response bias (e.g., "shy" supporters who claim another option).
    """
    confusion: np.ndarray  # shape (K, K)


def make_identity_misreport_model(k: int) -> MisreportModel:
    """
    No misreporting (truthful survey answers).
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    return MisreportModel(confusion=np.eye(k, dtype=float))


def make_shy_supporter_model(
    k: int,
    shy_category: int,
    *,
    honesty: float = 0.85,
    shift_to: Optional[Sequence[float]] = None,
) -> MisreportModel:
    """
    Create a misreport model where supporters of `shy_category` are less truthful.

    For true = shy_category:
      - with probability honesty: state truth
      - otherwise: state another category according to shift_to (or uniform over others)

    For all other true categories:
      - truthful (identity)

    Args:
      shift_to: optional length-K distribution (will be zeroed at shy_category and renormalised)
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    if not (0 <= shy_category < k):
        raise ValueError("shy_category out of range")
    if not (0.0 <= honesty <= 1.0):
        raise ValueError("honesty must be in [0,1]")

    # Start from the identity matrix (fully honest), then overwrite the shy
    # row to redistribute (1 - honesty) of its mass to other categories.
    C = np.eye(k, dtype=float)

    if shift_to is None:
        other = np.ones(k, dtype=float)
        other[shy_category] = 0.0
        other /= other.sum()
    else:
        other = np.array(shift_to, dtype=float)
        if other.shape != (k,):
            raise ValueError("shift_to must be length k")
        if np.any(other < 0):
            raise ValueError("shift_to must be non-negative")
        other = other.copy()
        other[shy_category] = 0.0
        s = other.sum()
        if s <= 0:
            raise ValueError("shift_to must allocate some probability to non-shy categories")
        other /= s

    C[shy_category, :] = (1.0 - honesty) * other
    C[shy_category, shy_category] = honesty
    _validate_confusion(C)
    return MisreportModel(confusion=C)


def make_general_misreport_model(confusion: np.ndarray) -> MisreportModel:
    """
    Wrap a user-provided confusion matrix after validation.
    """
    C = np.asarray(confusion, dtype=float)
    _validate_confusion(C)
    return MisreportModel(confusion=C)


def apply_misreporting(
    true_categories: np.ndarray,
    model: MisreportModel,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply misreporting to true categories to produce stated categories (pre-LDP).

    Returns:
      stated_categories: np.ndarray shape (n,)
    """
    if rng is None:
        rng = np.random.default_rng()

    t = np.asarray(true_categories, dtype=int)
    if t.ndim != 1:
        raise ValueError("true_categories must be 1D")
    if t.size == 0:
        return np.array([], dtype=int)

    C = model.confusion
    k = C.shape[0]
    if np.any((t < 0) | (t >= k)):
        raise ValueError(f"true_categories values must be in [0, {k-1}]")

    # Sample one categorical per row using inverse CDF.  This is the same
    # technique used in population.py but operates on a per-person confusion
    # matrix row rather than a preference vector.
    u = rng.random(t.size)
    out = np.empty(t.size, dtype=int)
    for i in range(t.size):
        row = C[t[i]]
        cdf = np.cumsum(row)
        out[i] = int(np.sum(cdf < u[i]))
    return out


# =============================================================================
# Privacy -> honesty/participation (scenario helpers)
# =============================================================================

def honesty_from_epsilon(
    epsilon: float,
    *,
    min_honesty: float = 0.70,
    max_honesty: float = 0.95,
    midpoint: float = 1.0,
    steepness: float = 2.0,
) -> float:
    """
    Map epsilon to an "honesty rate" for scenarios where privacy increases honesty.

    Interpretation:
      - smaller epsilon => stronger privacy => higher honesty
      - larger epsilon => weaker privacy => lower honesty

    Uses: honesty = min + (max-min) * sigmoid( steepness*(midpoint - epsilon) )
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if not (0.0 <= min_honesty <= max_honesty <= 1.0):
        raise ValueError("min_honesty/max_honesty must satisfy 0 <= min <= max <= 1")

    # Sigmoid mapping: stronger privacy (lower epsilon) yields higher honesty.
    # The steepness and midpoint parameters control how sharply the curve
    # transitions around the privacy inflection point.
    x = steepness * (midpoint - float(epsilon))
    s = 1.0 / (1.0 + np.exp(-x))
    return float(min_honesty + (max_honesty - min_honesty) * s)


def participation_from_epsilon(
    epsilon: float,
    *,
    base_rate: float = 0.80,
    max_rate: float = 0.92,
    midpoint: float = 1.0,
    steepness: float = 1.6,
) -> float:
    """
    Map epsilon to a "participation rate" for scenarios where privacy increases participation.

    Same direction:
      - smaller epsilon (more privacy) => higher participation

    participation = base + (max-base) * sigmoid( steepness*(midpoint - epsilon) )
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    if not (0.0 <= base_rate <= max_rate <= 1.0):
        raise ValueError("base_rate/max_rate must satisfy 0 <= base <= max <= 1")

    x = steepness * (midpoint - float(epsilon))
    s = 1.0 / (1.0 + np.exp(-x))
    return float(base_rate + (max_rate - base_rate) * s)


def build_shy_model_from_epsilon(
    k: int,
    shy_category: int,
    epsilon: float,
    *,
    min_honesty: float = 0.70,
    max_honesty: float = 0.95,
    midpoint: float = 1.0,
    steepness: float = 2.0,
    shift_to: Optional[Sequence[float]] = None,
) -> MisreportModel:
    """
    Convenience: create a shy-supporter misreport model where honesty depends on epsilon.
    """
    h = honesty_from_epsilon(
        epsilon,
        min_honesty=min_honesty,
        max_honesty=max_honesty,
        midpoint=midpoint,
        steepness=steepness,
    )
    return make_shy_supporter_model(k, shy_category, honesty=h, shift_to=shift_to)


# =============================================================================
# Internal validation
# =============================================================================

def _validate_confusion(C: np.ndarray) -> None:
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("confusion must be a square matrix of shape (K, K)")
    if C.shape[0] < 2:
        raise ValueError("K must be >= 2")
    # Allow small negative values from floating-point arithmetic but reject
    # deliberately negative confusion matrices.
    if np.any(C < -1e-12):
        raise ValueError("confusion matrix must be non-negative")
    row_sums = C.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError("Each row of confusion must sum to > 0")
    # A tolerance of 1e-6 accommodates cumulative floating-point rounding
    # without silently accepting grossly incorrect matrices.
    if np.max(np.abs(row_sums - 1.0)) > 1e-6:
        raise ValueError("Each row of confusion must sum to 1 (within tolerance).")
