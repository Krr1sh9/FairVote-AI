"""Research scenarios for FairVote-AI synthetic polling experiments.

The scenarios in this module make the experiment pipeline answer a sharper
question than "does the app run?": when does an RR-aware neural MRP estimator
outperform simpler RR-aware linear/poststratification baselines under local
privacy noise and sampling bias?

Scenario names deliberately separate two concepts:

* the latent population preference function, which determines whether a linear
  model should be sufficient or whether nonlinear interactions exist; and
* collection bias, such as nonresponse or pre-LDP shy-voter misreporting.

The respondent app never receives these true labels.  They exist only in the
synthetic offline benchmark so methods can be scored against known truth.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from fairvote.simulation.population import Population

SIMPLE_LINEAR_SCENARIOS = {"simple_linear"}
NONLINEAR_SCENARIOS = {
    "nonlinear_interaction",
    "education_urbanicity_interaction",
    "sparse_minority_curve",
    "nonlinear_response",
    "privacy_noise_sparse",
}
BIAS_SCENARIOS = {"no_bias", "nonresponse", "shy_fixed", "shy_privacy_helps", "privacy_tradeoff", "privacy_helps"}
VALID_SCENARIOS = SIMPLE_LINEAR_SCENARIOS | NONLINEAR_SCENARIOS | BIAS_SCENARIOS
NONRESPONSE_SCENARIOS = {"nonresponse", "shy_privacy_helps", "privacy_tradeoff", "privacy_helps"}
SHY_MISREPORT_SCENARIOS = {"shy_fixed", "shy_privacy_helps", "privacy_tradeoff", "privacy_helps"}
EPSILON_DEPENDENT_MISREPORT_SCENARIOS = {"shy_privacy_helps", "privacy_tradeoff", "privacy_helps"}


@dataclass(frozen=True)
class ScenarioInfo:
    """Human-readable metadata about one experiment scenario."""

    name: str
    truth_model: str
    collection_bias: str
    expected_neural_advantage: str


def scenario_info(name: str) -> ScenarioInfo:
    """Return metadata used in manifests and documentation."""
    if name == "simple_linear":
        return ScenarioInfo(name, "additive_linear", "none", "low")
    if name == "nonlinear_interaction":
        return ScenarioInfo(name, "region_age_and_education_urbanicity_interactions", "none", "medium_high")
    if name == "education_urbanicity_interaction":
        return ScenarioInfo(name, "education_urbanicity_interaction", "none", "medium")
    if name == "sparse_minority_curve":
        return ScenarioInfo(name, "sparse_minority_preference_curve", "none", "conditional_on_sample_size")
    if name == "nonlinear_response":
        return ScenarioInfo(name, "nonlinear_demographic_response_function", "none", "medium_high")
    if name == "privacy_noise_sparse":
        return ScenarioInfo(name, "sparse_subgroup_with_privacy_noise", "none", "conditional_on_epsilon_and_sparsity")
    if name == "nonresponse":
        return ScenarioInfo(name, "default_synthetic_population", "demographic_nonresponse", "low_medium")
    if name == "shy_fixed":
        return ScenarioInfo(name, "default_synthetic_population", "fixed_pre_ldp_misreport", "low_medium")
    if name in {"shy_privacy_helps", "privacy_tradeoff", "privacy_helps"}:
        return ScenarioInfo(
            name, "default_synthetic_population", "epsilon_dependent_nonresponse_and_misreport", "low_medium"
        )
    if name == "no_bias":
        return ScenarioInfo(name, "default_synthetic_population", "none", "low_medium")
    raise ValueError(f"Unknown scenario: {name}")


def validate_scenarios(names: Iterable[str]) -> list[str]:
    """Validate and return scenario names as a list."""
    out = [str(name).strip() for name in names if str(name).strip()]
    if not out:
        raise ValueError("at least one scenario is required")
    unknown = [name for name in out if name not in VALID_SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenario(s): {unknown}. Valid scenarios: {sorted(VALID_SCENARIOS)}")
    return out


def apply_truth_scenario(pop: Population, scenario: str, *, k: int, seed: int) -> Population:
    """Return a population whose latent preferences match the scenario.

    The base population supplies realistic correlated demographics.  This
    function resamples only ``true_probs`` and ``true_categories`` for research
    scenarios that need a controlled data-generating process.
    """
    if scenario not in VALID_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if scenario in BIAS_SCENARIOS:
        return pop

    rng = np.random.default_rng(_scenario_seed(seed, scenario))
    logits = _base_logits(pop, k, rng, scale=0.0)
    if scenario == "simple_linear":
        logits += _additive_feature_effects(pop, k, rng, scale=0.42)
    elif scenario == "nonlinear_interaction":
        logits += _additive_feature_effects(pop, k, rng, scale=0.28)
        logits += _region_age_interaction(pop, k, strength=1.10)
        logits += _education_urbanicity_interaction(pop, k, strength=0.95)
    elif scenario == "education_urbanicity_interaction":
        logits += _additive_feature_effects(pop, k, rng, scale=0.30)
        logits += _education_urbanicity_interaction(pop, k, strength=1.20)
    elif scenario == "sparse_minority_curve":
        logits += _additive_feature_effects(pop, k, rng, scale=0.25)
        logits += _sparse_minority_curve(pop, k, strength=1.45)
    elif scenario == "nonlinear_response":
        logits += _additive_feature_effects(pop, k, rng, scale=0.20)
        logits += _nonlinear_demographic_curve(pop, k, strength=1.15)
    elif scenario == "privacy_noise_sparse":
        logits += _additive_feature_effects(pop, k, rng, scale=0.22)
        logits += _sparse_minority_curve(pop, k, strength=1.25)
        logits += _education_urbanicity_interaction(pop, k, strength=0.65)
    else:  # pragma: no cover - protected by earlier validation
        raise ValueError(f"Unhandled scenario: {scenario}")

    logits += rng.normal(0.0, 0.08, size=logits.shape)
    probs = _softmax(logits)
    categories = _sample_categorical_rows(probs, rng)
    return Population(
        features={name: values.copy() for name, values in pop.features.items()},
        feature_levels={name: list(levels) for name, levels in pop.feature_levels.items()},
        true_probs=probs,
        true_categories=categories,
        category_names=list(pop.category_names),
    )


def _scenario_seed(seed: int, scenario: str) -> int:
    return int(seed) + 50_000 + sum((i + 1) * ord(ch) for i, ch in enumerate(scenario))


def _base_logits(pop: Population, k: int, rng: np.random.Generator, *, scale: float) -> np.ndarray:
    n = pop.true_categories.size
    base = np.linspace(0.55, -0.55, k, dtype=float)
    base -= base.mean()
    if scale > 0:
        base += rng.normal(0.0, scale, size=k)
        base -= base.mean()
    return np.tile(base, (n, 1)).astype(float)


def _additive_feature_effects(pop: Population, k: int, rng: np.random.Generator, *, scale: float) -> np.ndarray:
    n = pop.true_categories.size
    out = np.zeros((n, k), dtype=float)
    for feature, values in pop.features.items():
        levels = len(pop.feature_levels[feature])
        effects = rng.normal(0.0, scale, size=(levels, k))
        effects -= effects.mean(axis=1, keepdims=True)
        out += effects[np.asarray(values, dtype=int)]
    return out


def _region_age_interaction(pop: Population, k: int, *, strength: float) -> np.ndarray:
    region = np.asarray(pop.features["region"], dtype=int)
    age = np.asarray(pop.features["age_group"], dtype=int)
    n = region.size
    out = np.zeros((n, k), dtype=float)
    # London/South East younger voters and older voters in North/Wales/Scotland
    # follow different curves. A purely additive linear model cannot represent
    # this without explicit interaction columns.
    young = age <= 1
    older = age >= 4
    london_se = np.isin(region, np.array([0, 1], dtype=int))
    north_wales_scotland = np.isin(region, np.array([6, 7, 8, 9, 10], dtype=int))
    if k >= 2:
        out[london_se & young, 0] += strength
        out[london_se & young, 1] -= strength * 0.55
        out[north_wales_scotland & older, 1] += strength * 0.85
        out[north_wales_scotland & older, 0] -= strength * 0.45
    if k >= 4:
        out[london_se & young, 3] += strength * 0.45
    if k >= 5:
        out[london_se & young, 4] += strength * 0.35
        out[north_wales_scotland & older, 4] -= strength * 0.25
    out -= out.mean(axis=1, keepdims=True)
    return out


def _education_urbanicity_interaction(pop: Population, k: int, *, strength: float) -> np.ndarray:
    education = np.asarray(pop.features["education"], dtype=int)
    urban = np.asarray(pop.features["urbanicity"], dtype=int)
    out = np.zeros((education.size, k), dtype=float)
    degree_urban = (education == 2) & (urban == 0)
    no_degree_rural = (education == 0) & (urban == 2)
    if k >= 2:
        out[degree_urban, 0] += strength * 0.85
        out[degree_urban, 1] -= strength * 0.40
        out[no_degree_rural, 1] += strength
        out[no_degree_rural, 0] -= strength * 0.45
    if k >= 3:
        out[no_degree_rural, 2] += strength * 0.45
    if k >= 5:
        out[degree_urban, 4] += strength * 0.35
    out -= out.mean(axis=1, keepdims=True)
    return out


def _sparse_minority_curve(pop: Population, k: int, *, strength: float) -> np.ndarray:
    region = np.asarray(pop.features["region"], dtype=int)
    gender = np.asarray(pop.features["gender"], dtype=int)
    education = np.asarray(pop.features["education"], dtype=int)
    age = np.asarray(pop.features["age_group"], dtype=int)
    out = np.zeros((region.size, k), dtype=float)
    sparse = (gender == 2) | ((region == 11) & (education == 2)) | ((region == 7) & (age <= 1))
    if k >= 3:
        out[sparse, 2] += strength
        out[sparse, 0] -= strength * 0.35
        out[sparse, 1] -= strength * 0.30
    else:
        out[sparse, 0] += strength
        out[sparse, 1] -= strength
    if k >= 5:
        out[sparse & (age <= 2), 4] += strength * 0.50
    out -= out.mean(axis=1, keepdims=True)
    return out


def _nonlinear_demographic_curve(pop: Population, k: int, *, strength: float) -> np.ndarray:
    age = np.asarray(pop.features["age_group"], dtype=float)
    urban = np.asarray(pop.features["urbanicity"], dtype=float)
    education = np.asarray(pop.features["education"], dtype=float)
    out = np.zeros((age.size, k), dtype=float)
    age_centered = (age - np.mean(age)) / max(float(np.std(age)), 1.0)
    curve = np.sin(1.25 * age_centered) + 0.65 * (age_centered**2 - np.mean(age_centered**2))
    urban_curve = np.where(urban == 0, 0.55, np.where(urban == 2, -0.35, 0.0))
    education_curve = np.where(education == 2, 0.35 * curve, -0.20 * curve)
    if k >= 2:
        out[:, 0] += strength * (0.55 * curve + 0.45 * urban_curve)
        out[:, 1] -= strength * (0.45 * curve + 0.35 * urban_curve)
    if k >= 4:
        out[:, 3] += strength * education_curve
    if k >= 5:
        out[:, 4] += strength * (0.35 * np.cos(1.7 * age_centered) + 0.25 * (urban == 0))
    out -= out.mean(axis=1, keepdims=True)
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def _sample_categorical_rows(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    u = np.asarray(rng.random(probs.shape[0]), dtype=float)
    cdf = np.cumsum(probs, axis=1)
    return np.sum(cdf < u[:, None], axis=1).astype(int)
