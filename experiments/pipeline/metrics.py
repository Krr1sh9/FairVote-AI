"""Metric and result-row helpers for the MRP-vs-baselines experiment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from fairvote.metrics.group_metrics import correct_winner, p90_group_l1, weighted_group_l1, worst_group_l1

from .config import ExperimentConfig, TrialConfig
from .scenarios import scenario_info

METRIC_COLUMNS = [
    "overall_l1",
    "overall_linf",
    "overall_mae",
    "winner_correct",
    "worst_region_l1",
    "avg_region_l1",
    "worst_age_l1",
    "avg_age_l1",
    "worst_region_l1_major",
    "worst_age_l1_major",
    "weighted_region_l1",
    "weighted_age_l1",
    "p90_region_l1_major",
    "p90_age_l1_major",
    "worst_group_l1_major",
    "runtime_sec",
]

SUMMARY_METRICS = [
    "overall_l1",
    "overall_mae",
    "winner_correct",
    "runtime_sec",
    "worst_region_l1",
    "worst_age_l1",
    "worst_region_l1_major",
    "worst_age_l1_major",
    "weighted_region_l1",
    "weighted_age_l1",
    "p90_region_l1_major",
    "p90_age_l1_major",
    "worst_group_l1_major",
]


def l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def distribution_from_labels(labels: np.ndarray, k: int) -> np.ndarray:
    """Empirical distribution of integer labels in [0, k-1]."""
    y = np.asarray(labels, dtype=int)
    if y.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if y.size == 0:
        return np.full(k, 1.0 / k, dtype=float)
    if np.any((y < 0) | (y >= k)):
        raise ValueError(f"labels must be in [0, {k - 1}]")
    counts = np.bincount(y, minlength=k).astype(float)
    total = float(counts.sum())
    if total <= 0.0:
        return np.full(k, 1.0 / k, dtype=float)
    return counts / total


def truth_overall(pop: Any, k: int) -> np.ndarray:
    counts = np.bincount(pop.true_categories, minlength=k).astype(float)
    return counts / counts.sum()


def feature_masses_from_cells(
    *,
    cells: np.ndarray,
    counts: np.ndarray,
    by: Sequence[str],
    feature: str,
    feature_levels: Mapping[str, Sequence[str]],
) -> dict[str, float]:
    """Compute population masses for each level of a feature from poststrat cells."""
    by_list = list(by)
    if feature not in by_list:
        return {}
    j = by_list.index(feature)
    levels = list(feature_levels[feature])
    total = float(np.sum(counts))
    if total <= 0:
        return dict.fromkeys(levels, 0.0)
    return {lvl_name: float(np.sum(counts[cells[:, j] == lvl_idx]) / total) for lvl_idx, lvl_name in enumerate(levels)}


def poststrat_from_cell_theta(
    cell_theta: np.ndarray,
    cells: np.ndarray,
    counts: np.ndarray,
    *,
    by: Sequence[str],
    feature_levels: Mapping[str, Sequence[str]],
    include_features: Sequence[str],
) -> tuple[np.ndarray, dict[str, dict[str, np.ndarray]]]:
    """Poststratify cell-level category probabilities to overall and subgroup estimates."""
    total = float(np.sum(counts))
    overall = (counts[:, None] * cell_theta).sum(axis=0) / total
    by_feature: dict[str, dict[str, np.ndarray]] = {}
    by_list = list(by)
    for feat in include_features:
        if feat not in by_list:
            continue
        j = by_list.index(feat)
        levels = list(feature_levels[feat])
        out: dict[str, np.ndarray] = {}
        for lvl_idx, lvl_name in enumerate(levels):
            mask = cells[:, j] == lvl_idx
            w = counts[mask]
            tot = float(np.sum(w))
            if tot <= 0:
                continue
            out[lvl_name] = (w[:, None] * cell_theta[mask]).sum(axis=0) / tot
        by_feature[feat] = out
    return overall.astype(float), by_feature


def _group_l1s(estimates: Mapping[str, np.ndarray], truth: Mapping[str, np.ndarray]) -> list[float]:
    return [l1(est, truth[level]) for level, est in estimates.items() if level in truth]


def score_method_result(
    *,
    config: ExperimentConfig,
    trial: TrialConfig,
    method: str,
    n_effective: int,
    estimate_overall: np.ndarray,
    by_feature: Mapping[str, Mapping[str, np.ndarray]],
    truth_overall_dist: np.ndarray,
    truth_region: Mapping[str, np.ndarray],
    truth_age: Mapping[str, np.ndarray],
    region_masses: Mapping[str, float],
    age_masses: Mapping[str, float],
    runtime_sec: float,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete auditable result row for one method."""
    region_est = dict(by_feature.get("region", {}))
    age_est = dict(by_feature.get("age_group", {}))
    region_l1s = _group_l1s(region_est, truth_region)
    age_l1s = _group_l1s(age_est, truth_age)

    info = scenario_info(trial.scenario)
    worst_region_major = worst_group_l1(
        region_est, truth_region, group_masses=dict(region_masses), min_mass=config.major_mass
    )
    worst_age_major = worst_group_l1(age_est, truth_age, group_masses=dict(age_masses), min_mass=config.major_mass)
    row: dict[str, Any] = {
        "config_seed": int(config.seed),
        "random_seed": int(trial.random_seed),
        "sample_seed": int(trial.sample_seed),
        "scenario": trial.scenario,
        "trial": int(trial.trial),
        "epsilon": float(trial.epsilon),
        "sample_size": int(trial.sample_size),
        "population_n": int(config.population_n),
        "sampling": config.sampling,
        "feature_set": ",".join(config.feature_order),
        "scenario_truth_model": info.truth_model,
        "scenario_collection_bias": info.collection_bias,
        "method": method,
        "n_effective": int(n_effective),
        "skipped": 0,
        "runtime_sec": float(runtime_sec),
        "winner_correct": int(correct_winner(estimate_overall, truth_overall_dist)),
        "overall_l1": l1(estimate_overall, truth_overall_dist),
        "overall_linf": linf(estimate_overall, truth_overall_dist),
        "overall_mae": mae(estimate_overall, truth_overall_dist),
        "worst_region_l1": float(np.max(region_l1s)) if region_l1s else float("nan"),
        "avg_region_l1": float(np.mean(region_l1s)) if region_l1s else float("nan"),
        "worst_age_l1": float(np.max(age_l1s)) if age_l1s else float("nan"),
        "avg_age_l1": float(np.mean(age_l1s)) if age_l1s else float("nan"),
        "worst_region_l1_major": worst_region_major,
        "worst_age_l1_major": worst_age_major,
        "worst_group_l1_major": float(np.nanmax([worst_region_major, worst_age_major])),
        "weighted_region_l1": weighted_group_l1(region_est, truth_region, group_masses=dict(region_masses)),
        "weighted_age_l1": weighted_group_l1(age_est, truth_age, group_masses=dict(age_masses)),
        "p90_region_l1_major": p90_group_l1(
            region_est, truth_region, group_masses=dict(region_masses), min_mass=config.major_mass
        ),
        "p90_age_l1_major": p90_group_l1(age_est, truth_age, group_masses=dict(age_masses), min_mass=config.major_mass),
    }
    if extra:
        row.update(dict(extra))
    return row


def skipped_row(config: ExperimentConfig, trial: TrialConfig, method: str, n_effective: int) -> dict[str, Any]:
    """Row emitted when a trial cell has too few usable respondents."""
    info = scenario_info(trial.scenario)
    return {
        "config_seed": int(config.seed),
        "random_seed": int(trial.random_seed),
        "sample_seed": int(trial.sample_seed),
        "scenario": trial.scenario,
        "trial": int(trial.trial),
        "epsilon": float(trial.epsilon),
        "sample_size": int(trial.sample_size),
        "population_n": int(config.population_n),
        "sampling": config.sampling,
        "feature_set": ",".join(config.feature_order),
        "scenario_truth_model": info.truth_model,
        "scenario_collection_bias": info.collection_bias,
        "method": method,
        "n_effective": int(n_effective),
        "skipped": 1,
        "runtime_sec": float("nan"),
    }
