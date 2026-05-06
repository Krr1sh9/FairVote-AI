"""Modular, auditable runner for the MRP-vs-baselines experiment."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from typing import Any

import numpy as np

from .config import ExperimentConfig, ExperimentResult, TrialConfig
from .context import build_context
from .methods import build_trial_data, selected_registry
from .metrics import score_method_result, skipped_row
from .perturbation import apply_misreport_and_rr
from .sampling import apply_scenario_nonresponse, draw_sample
from .scenarios import scenario_info
from .summary import (
    aggregate_ablation_comparisons,
    aggregate_paired_comparisons,
    aggregate_runtime_profile,
    aggregate_summary,
)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _package_versions(names: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in names:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def sample_seed(config: ExperimentConfig, trial_index: int, sample_size: int | None = None) -> int:
    """Deterministic seed for sampling/nonresponse for a sample-size/trial cell."""
    size = int(config.n_sample if sample_size is None else sample_size)
    return int(config.seed + 10_000 * trial_index + 97 * size + 1337)


def privacy_seed(config: ExperimentConfig, trial_index: int, epsilon: float, sample_size: int | None = None) -> int:
    """Deterministic seed for pre-LDP misreporting and RR for one epsilon cell."""
    size = int(config.n_sample if sample_size is None else sample_size)
    return int(config.seed + 10_000 * trial_index + 97 * size + int(round(epsilon * 10_000)) + 7)


def minimum_effective_sample(config: ExperimentConfig) -> int:
    return max(80, int(0.05 * config.n_sample))


def execute_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run the full experiment grid and return raw rows plus summaries."""
    start = time.perf_counter()
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for sample_size in config.sample_size_grid:
        cell_config = replace(config, n_sample=int(sample_size), sample_sizes=[int(sample_size)])
        registry = selected_registry(cell_config.methods)
        for scenario in cell_config.scenarios:
            context = build_context(cell_config, scenario=scenario)
            for trial_index in range(cell_config.trials):
                rng_sample = np.random.default_rng(sample_seed(cell_config, trial_index, sample_size))
                sample = draw_sample(cell_config, context.pop, rng=rng_sample)
                sample = apply_scenario_nonresponse(sample, context.pop, scenario, rng=rng_sample)
                cell_rows, cell_failures = _run_eps_cells(cell_config, context, registry, scenario, trial_index, sample)
                rows.extend(cell_rows)
                failures.extend(cell_failures)

    summary = aggregate_summary(rows, config)
    paired = aggregate_paired_comparisons(rows, config)
    ablations = aggregate_ablation_comparisons(rows, config)
    runtime_profile = aggregate_runtime_profile(rows, config)
    manifest = build_manifest(
        config,
        rows,
        summary,
        paired,
        ablations,
        runtime_profile,
        failures,
        runtime_sec=time.perf_counter() - start,
    )
    return ExperimentResult(
        rows=rows,
        summary=summary,
        paired_comparisons=paired,
        ablations=ablations,
        runtime_profile=runtime_profile,
        failures=failures,
        config=config,
        manifest=manifest,
    )


def _run_eps_cells(
    config: ExperimentConfig,
    context: Any,
    registry: dict[str, Any],
    scenario: str,
    trial_index: int,
    sample: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    n_effective = int(sample.idx.size)
    for epsilon in config.eps_list:
        trial = TrialConfig(
            scenario=scenario,
            trial=trial_index,
            epsilon=float(epsilon),
            sample_seed=sample_seed(config, trial_index, config.n_sample),
            privacy_seed=privacy_seed(config, trial_index, float(epsilon), config.n_sample),
            sample_size=int(config.n_sample),
        )
        if n_effective < minimum_effective_sample(config):
            rows.extend(skipped_row(config, trial, method, n_effective) for method in config.methods)
            continue
        rng_privacy = np.random.default_rng(trial.privacy_seed)
        perturbation = apply_misreport_and_rr(
            config=config,
            scenario=scenario,
            true_categories=sample.true_categories.astype(int),
            epsilon=float(epsilon),
            rng=rng_privacy,
        )
        trial_data = build_trial_data(config=config, context=context, sample=sample, perturbation=perturbation)
        method_rows, method_failures = _run_methods(config, context, registry, trial, trial_data)
        rows.extend(method_rows)
        failures.extend(method_failures)
    return rows, failures


def _run_methods(
    config: ExperimentConfig,
    context: Any,
    registry: dict[str, Any],
    trial: TrialConfig,
    trial_data: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for method_name, runner in registry.items():
        method_start = time.perf_counter()
        try:
            result = runner(config, context, trial, trial_data)
            rows.append(
                score_method_result(
                    config=config,
                    trial=trial,
                    method=method_name,
                    n_effective=trial_data.n_effective,
                    estimate_overall=result.estimate_overall,
                    by_feature=result.by_feature,
                    truth_overall_dist=context.truth_overall,
                    truth_region=context.truth_region,
                    truth_age=context.truth_age,
                    region_masses=context.region_masses,
                    age_masses=context.age_masses,
                    runtime_sec=result.runtime_sec,
                    extra=result.extra,
                )
            )
        except Exception as exc:  # pragma: no cover - exercised by failure-injection tests if added
            runtime = time.perf_counter() - method_start
            failure = _failure_record(config, trial, method_name, exc, runtime)
            failures.append(failure)
            row = skipped_row(config, trial, method_name, trial_data.n_effective)
            row.update(
                {
                    "skipped": 1,
                    "failure": 1,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "runtime_sec": runtime,
                }
            )
            rows.append(row)
            if not config.continue_on_error:
                raise
    return rows, failures


def _failure_record(
    config: ExperimentConfig, trial: TrialConfig, method: str, exc: Exception, runtime_sec: float
) -> dict[str, Any]:
    return {
        "config_seed": int(config.seed),
        "sample_size": int(trial.sample_size),
        "scenario": trial.scenario,
        "trial": int(trial.trial),
        "epsilon": float(trial.epsilon),
        "method": method,
        "runtime_sec": float(runtime_sec),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
    }


def build_manifest(
    config: ExperimentConfig,
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    ablations: list[dict[str, Any]],
    runtime_profile: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    *,
    runtime_sec: float,
) -> dict[str, Any]:
    """Create a JSON manifest tying output files back to the exact config grid."""
    return {
        "experiment": "mrp_vs_baselines",
        "preset": config.preset,
        "config": config.to_dict(),
        "seed": int(config.seed),
        "scenarios": list(config.scenarios),
        "scenario_metadata": {name: scenario_info(name).__dict__ for name in config.scenarios},
        "epsilons": [float(e) for e in config.eps_list],
        "sample_sizes": [int(s) for s in config.sample_size_grid],
        "sample_size": int(config.n_sample),
        "population_n": int(config.population_n),
        "trials": int(config.trials),
        "methods": list(config.methods),
        "n_result_rows": len(rows),
        "n_summary_rows": len(summary),
        "n_paired_comparison_rows": len(paired),
        "n_ablation_rows": len(ablations),
        "n_runtime_profile_rows": len(runtime_profile),
        "n_failures": len(failures),
        "runtime_sec": float(runtime_sec),
        "provenance": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "git_sha": _git_sha(),
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "packages": _package_versions(["numpy", "pandas", "matplotlib", "torch", "flask", "streamlit"]),
        },
        "outputs": {
            "config": "config.json",
            "manifest": "manifest.json",
            "environment": "environment.json",
            "sha256sums": "sha256sums.txt",
            "raw_trials": "raw_trials.csv",
            "results_trials_legacy_alias": "results_trials.csv",
            "summary_with_ci": "summary_with_ci.csv",
            "summary_legacy_alias": "summary.csv",
            "paired_comparisons": "paired_comparisons.csv",
            "ablations": "ablations.csv",
            "runtime_profile": "runtime_profile.csv",
            "failures": "failures.csv",
            "readme": "README.md",
            "plots": "plots/",
        },
    }
