"""Summary aggregation for raw experiment trial rows."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from .config import ExperimentConfig
from .metrics import SUMMARY_METRICS


def _finite_values(rows: Iterable[Dict[str, Any]], key: str) -> np.ndarray:
    vals = np.array([r.get(key, float("nan")) for r in rows], dtype=float)
    return vals[np.isfinite(vals)]


def _stats(rows: Iterable[Dict[str, Any]], key: str) -> Tuple[float, float, int]:
    vals = _finite_values(rows, key)
    if vals.size == 0:
        return float("nan"), float("nan"), 0
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
    return mean, std, int(vals.size)


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int, n_boot: int = 1000) -> Tuple[float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    if vals.size == 1:
        return float(vals[0]), float(vals[0])
    rng = np.random.default_rng(seed)
    boot = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        boot[i] = float(np.mean(rng.choice(vals, size=vals.size, replace=True)))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def aggregate_summary(rows: List[Dict[str, Any]], config: ExperimentConfig) -> List[Dict[str, Any]]:
    """Aggregate raw trial rows into summary rows with 95% bootstrap CIs."""
    summary: List[Dict[str, Any]] = []
    for sample_size in config.sample_size_grid:
        for scenario in config.scenarios:
            for eps in config.eps_list:
                for method in config.methods:
                    sub = [
                        r
                        for r in rows
                        if int(r.get("sample_size", config.n_sample)) == int(sample_size)
                        and r.get("scenario") == scenario
                        and float(r.get("epsilon")) == float(eps)
                        and r.get("method") == method
                        and int(r.get("skipped", 0) or 0) == 0
                    ]
                    skipped = [
                        r
                        for r in rows
                        if int(r.get("sample_size", config.n_sample)) == int(sample_size)
                        and r.get("scenario") == scenario
                        and float(r.get("epsilon")) == float(eps)
                        and r.get("method") == method
                        and int(r.get("skipped", 0) or 0) == 1
                    ]
                    base: Dict[str, Any] = {
                        "config_seed": int(config.seed),
                        "random_seed": int(config.seed),
                        "trials": int(config.trials),
                        "scenario": scenario,
                        "epsilon": float(eps),
                        "sample_size": int(sample_size),
                        "population_n": int(config.population_n),
                        "sampling": config.sampling,
                        "method": method,
                        "n_rows": len(sub),
                        "n_skipped": len(skipped),
                    }
                    if not sub:
                        summary.append(base)
                        continue
                    for metric in SUMMARY_METRICS:
                        vals = _finite_values(sub, metric)
                        mean, std, n_metric = _stats(sub, metric)
                        base[f"mean_{metric}"] = mean
                        base[f"std_{metric}"] = std
                        base[f"n_{metric}"] = n_metric
                        lo, hi = _bootstrap_mean_ci(
                            vals,
                            seed=int(config.seed + 31 * int(sample_size) + round(float(eps) * 10_000) + len(method) + len(scenario)),
                        )
                        base[f"ci95_low_{metric}"] = lo
                        base[f"ci95_high_{metric}"] = hi
                    base["mean_n_effective"] = float(np.mean([r["n_effective"] for r in sub]))
                    summary.append(base)
    return summary


def required_result_columns() -> set[str]:
    """Columns that must exist in every non-skipped result row."""
    return {
        "config_seed",
        "random_seed",
        "sample_seed",
        "scenario",
        "epsilon",
        "sample_size",
        "method",
        "runtime_sec",
        "overall_l1",
        "overall_linf",
        "overall_mae",
        "winner_correct",
    }


def aggregate_paired_comparisons(
    rows: List[Dict[str, Any]],
    config: ExperimentConfig,
    *,
    neural_method: str = "neural_rr_mrp",
    linear_method: str = "mrp_rr_poststrat",
) -> List[Dict[str, Any]]:
    """Summarise paired trial deltas between neural and linear RR-aware MRP.

    Negative deltas mean the neural method has lower error than the linear
    baseline. Deltas are paired by sample size, scenario, epsilon and trial,
    which removes much of the random-sampling variation from the comparison.
    """
    if neural_method not in config.methods or linear_method not in config.methods:
        return []

    out: List[Dict[str, Any]] = []
    metrics = [
        ("overall_l1", "delta_overall_l1"),
        ("worst_group_l1_major", "delta_worst_group_l1"),
        ("worst_region_l1_major", "delta_worst_region_l1"),
        ("worst_age_l1_major", "delta_worst_age_l1"),
    ]
    for sample_size in config.sample_size_grid:
        for scenario in config.scenarios:
            for eps in config.eps_list:
                linear_by_trial: Dict[int, Dict[str, Any]] = {}
                neural_by_trial: Dict[int, Dict[str, Any]] = {}
                for row in rows:
                    if int(row.get("skipped", 0) or 0):
                        continue
                    if int(row.get("sample_size", sample_size)) != int(sample_size):
                        continue
                    if row.get("scenario") != scenario or float(row.get("epsilon")) != float(eps):
                        continue
                    method = row.get("method")
                    trial = int(row.get("trial", -1))
                    if method == linear_method:
                        linear_by_trial[trial] = row
                    elif method == neural_method:
                        neural_by_trial[trial] = row
                paired_trials = sorted(set(linear_by_trial) & set(neural_by_trial))
                base: Dict[str, Any] = {
                    "config_seed": int(config.seed),
                    "scenario": scenario,
                    "epsilon": float(eps),
                    "sample_size": int(sample_size),
                    "population_n": int(config.population_n),
                    "sampling": config.sampling,
                    "feature_set": ",".join(config.feature_order),
                    "linear_method": linear_method,
                    "neural_method": neural_method,
                    "n_paired_trials": len(paired_trials),
                }
                if not paired_trials:
                    out.append(base)
                    continue
                _add_delta_stats(base, paired_trials, linear_by_trial, neural_by_trial, metrics, seed_base=config.seed + int(sample_size) + len(scenario))
                for metric, _delta_name in metrics:
                    if f"mean_reference_{metric}" in base:
                        base[f"mean_linear_{metric}"] = base[f"mean_reference_{metric}"]
                    if f"mean_comparison_{metric}" in base:
                        base[f"mean_neural_{metric}"] = base[f"mean_comparison_{metric}"]
                out.append(base)
    return out


def aggregate_ablation_comparisons(
    rows: List[Dict[str, Any]],
    config: ExperimentConfig,
    *,
    reference_method: str = "mrp_rr_poststrat",
) -> List[Dict[str, Any]]:
    """Paired comparisons for ablation methods against canonical linear RR-MRP."""
    if reference_method not in config.methods:
        return []
    ablation_methods = [
        m
        for m in config.methods
        if m != reference_method
        and (
            m in {"linear_rr_no_poststrat", "neural_naive_reported_mrp", "baseline_rr_debias", "raw_reported_distribution"}
            or m.startswith("oracle_")
        )
    ]
    out: List[Dict[str, Any]] = []
    metrics = [("overall_l1", "delta_overall_l1"), ("worst_group_l1_major", "delta_worst_group_l1")]
    for sample_size in config.sample_size_grid:
        for scenario in config.scenarios:
            for eps in config.eps_list:
                ref_by_trial = _rows_by_trial(rows, sample_size, scenario, eps, reference_method)
                for ablation in ablation_methods:
                    ablation_by_trial = _rows_by_trial(rows, sample_size, scenario, eps, ablation)
                    paired_trials = sorted(set(ref_by_trial) & set(ablation_by_trial))
                    base: Dict[str, Any] = {
                        "config_seed": int(config.seed),
                        "scenario": scenario,
                        "epsilon": float(eps),
                        "sample_size": int(sample_size),
                        "population_n": int(config.population_n),
                        "reference_method": reference_method,
                        "ablation_method": ablation,
                        "n_paired_trials": len(paired_trials),
                    }
                    if paired_trials:
                        _add_delta_stats(base, paired_trials, ref_by_trial, ablation_by_trial, metrics, seed_base=config.seed + int(sample_size) + len(ablation))
                    out.append(base)
    return out


def aggregate_runtime_profile(rows: List[Dict[str, Any]], config: ExperimentConfig) -> List[Dict[str, Any]]:
    """Aggregate per-method runtime and failure counts for the run."""
    out: List[Dict[str, Any]] = []
    for sample_size in config.sample_size_grid:
        for scenario in config.scenarios:
            for eps in config.eps_list:
                for method in config.methods:
                    sub = [
                        r
                        for r in rows
                        if int(r.get("sample_size", config.n_sample)) == int(sample_size)
                        and r.get("scenario") == scenario
                        and float(r.get("epsilon")) == float(eps)
                        and r.get("method") == method
                    ]
                    vals = _finite_values(sub, "runtime_sec")
                    out.append(
                        {
                            "config_seed": int(config.seed),
                            "scenario": scenario,
                            "epsilon": float(eps),
                            "sample_size": int(sample_size),
                            "method": method,
                            "n_rows": len(sub),
                            "n_failures": int(sum(int(r.get("failure", 0) or 0) for r in sub)),
                            "n_skipped": int(sum(int(r.get("skipped", 0) or 0) for r in sub)),
                            "mean_runtime_sec": float(np.mean(vals)) if vals.size else float("nan"),
                            "std_runtime_sec": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0 if vals.size == 1 else float("nan"),
                            "total_runtime_sec": float(np.sum(vals)) if vals.size else 0.0,
                        }
                    )
    return out


def _rows_by_trial(rows: List[Dict[str, Any]], sample_size: int, scenario: str, eps: float, method: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        if int(row.get("skipped", 0) or 0):
            continue
        if int(row.get("sample_size", sample_size)) != int(sample_size):
            continue
        if row.get("scenario") != scenario or float(row.get("epsilon")) != float(eps):
            continue
        if row.get("method") == method:
            out[int(row.get("trial", -1))] = row
    return out


def _add_delta_stats(
    base: Dict[str, Any],
    paired_trials: List[int],
    reference_by_trial: Dict[int, Dict[str, Any]],
    comparison_by_trial: Dict[int, Dict[str, Any]],
    metrics: List[tuple[str, str]],
    *,
    seed_base: int,
) -> None:
    for metric, delta_name in metrics:
        deltas = []
        ref_vals = []
        comp_vals = []
        for trial in paired_trials:
            ref = float(reference_by_trial[trial].get(metric, float("nan")))
            comp = float(comparison_by_trial[trial].get(metric, float("nan")))
            if np.isfinite(ref) and np.isfinite(comp):
                deltas.append(comp - ref)
                ref_vals.append(ref)
                comp_vals.append(comp)
        vals = np.asarray(deltas, dtype=float)
        base[f"mean_reference_{metric}"] = float(np.mean(ref_vals)) if ref_vals else float("nan")
        base[f"mean_comparison_{metric}"] = float(np.mean(comp_vals)) if comp_vals else float("nan")
        base[f"mean_{delta_name}"] = float(np.mean(vals)) if vals.size else float("nan")
        base[f"win_rate_{delta_name}"] = float(np.mean(vals < 0.0)) if vals.size else float("nan")
        lo, hi = _bootstrap_mean_ci(vals, seed=int(seed_base + len(metric)))
        base[f"ci95_low_{delta_name}"] = lo
        base[f"ci95_high_{delta_name}"] = hi
