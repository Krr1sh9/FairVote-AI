"""Add repeated-trial uncertainty summaries to FairVote-AI evidence packs.

This module is a reporting layer only. It reads existing per-trial experiment
outputs and writes derived confidence-interval tables. It does not fit models,
run estimators, change metrics, or modify the raw result files.

Example
-------
python -m experiments.add_uncertainty_summaries \
  --input-dir experiments/outputs/<timestamped-run-folder>
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Existing per-trial columns used by the project evidence pack. The helper only
# summarises columns that are actually present in the input results file.
DEFAULT_METRICS: tuple[str, ...] = (
    "overall_l1",
    "overall_mae",
    "winner_correct",
    "worst_region_l1_major",
    "worst_age_l1_major",
    "p90_region_l1_major",
    "p90_age_l1_major",
    "weighted_region_l1",
    "weighted_age_l1",
    "runtime_sec",
)

GROUP_COLUMNS: tuple[str, ...] = (
    "scenario",
    "scenario_label",
    "method",
    "epsilon",
    "n_sample",
)

HIGHER_IS_BETTER = {"winner_correct"}

BASELINE_METHODS: tuple[str, ...] = (
    "baseline_rr_debias",
    "mrp_rr_poststrat",
    "mrp_misreport_rr_poststrat",
    "mrp_learned_misreport_rr_poststrat",
    "raw_reported_distribution",
)

NEURAL_METHOD = "neural_rr_mrp"


@dataclass(frozen=True)
class CISettings:
    """Configuration for repeated-trial confidence intervals."""

    z_value: float = 1.96
    confidence_level: float = 0.95


def _available_metrics(df: pd.DataFrame, requested: Sequence[str] | None = None) -> list[str]:
    metrics = list(requested or DEFAULT_METRICS)
    out: list[str] = []
    for metric in metrics:
        if metric in df.columns and pd.api.types.is_numeric_dtype(df[metric]):
            out.append(metric)
    return out


def _normalise_group_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure grouping columns exist without changing raw files on disk."""

    out = df.copy()
    if "scenario_label" not in out.columns and "scenario" in out.columns:
        out["scenario_label"] = out["scenario"]
    missing = [c for c in GROUP_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required grouping columns in results_trials.csv: {missing}")
    return out


def _sample_std(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return float("nan")
    return float(clean.std(ddof=1))


def _standard_error(std: float, n: int) -> float:
    if n <= 0 or not np.isfinite(std):
        return float("nan")
    return float(std / math.sqrt(n))


def _ci_bounds(mean: float, se: float, settings: CISettings) -> tuple[float, float, float]:
    if not np.isfinite(mean) or not np.isfinite(se):
        return float("nan"), float("nan"), float("nan")
    half_width = settings.z_value * se
    return float(mean - half_width), float(mean + half_width), float(half_width)


def _summarise_series(values: pd.Series, settings: CISettings) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    n = int(len(clean))
    mean = float(clean.mean()) if n else float("nan")
    std = _sample_std(clean)
    se = _standard_error(std, n)
    ci_low, ci_high, ci_half_width = _ci_bounds(mean, se, settings)
    return {
        "n_trials": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "ci95_half_width": ci_half_width,
    }


def build_summary_with_ci(
    df: pd.DataFrame,
    metrics: Sequence[str] | None = None,
    settings: CISettings | None = None,
) -> pd.DataFrame:
    """Build a long-form CI summary grouped by scenario/method/epsilon/n_sample."""

    settings = settings or CISettings()
    work = _normalise_group_columns(df)
    metrics_available = _available_metrics(work, metrics)
    rows: list[dict[str, object]] = []

    for key, group in work.groupby(list(GROUP_COLUMNS), dropna=False, sort=True):
        base = dict(zip(GROUP_COLUMNS, key))
        for metric in metrics_available:
            stats = _summarise_series(group[metric], settings)
            rows.append({
                **base,
                "metric": metric,
                **stats,
                "higher_is_better": int(metric in HIGHER_IS_BETTER),
                "ci_method": f"normal_approx_z_{settings.z_value:g}_over_repeated_trials",
            })
    return pd.DataFrame(rows)


def build_method_rankings_with_ci(summary_ci: pd.DataFrame) -> pd.DataFrame:
    """Rank methods by mean within each scenario/epsilon/n_sample/metric group."""

    if summary_ci.empty:
        return summary_ci.copy()

    rank_group_cols = ["scenario", "scenario_label", "epsilon", "n_sample", "metric"]
    rows: list[dict[str, object]] = []
    for key, group in summary_ci.groupby(rank_group_cols, dropna=False, sort=True):
        metric = str(key[-1])
        ascending = metric not in HIGHER_IS_BETTER
        ranked = group.sort_values(["mean", "method"], ascending=[ascending, True]).reset_index(drop=True)
        for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
            record = row.to_dict()
            record["rank_by_mean"] = rank
            rows.append(record)
    return pd.DataFrame(rows)


def _paired_neural_baseline_rows(
    df: pd.DataFrame,
    metric: str,
    baseline_method: str,
    condition_cols: Sequence[str],
    settings: CISettings,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    subset = df[df["method"].isin([NEURAL_METHOD, baseline_method])].copy()
    if subset.empty:
        return rows

    pivot_index = list(condition_cols) + ["trial"]
    pivot = subset.pivot_table(index=pivot_index, columns="method", values=metric, aggfunc="mean").reset_index()
    if NEURAL_METHOD not in pivot.columns or baseline_method not in pivot.columns:
        return rows
    pivot = pivot.dropna(subset=[NEURAL_METHOD, baseline_method])
    if pivot.empty:
        return rows
    pivot["delta_neural_minus_baseline"] = pivot[NEURAL_METHOD] - pivot[baseline_method]

    for key, group in pivot.groupby(list(condition_cols), dropna=False, sort=True):
        base = dict(zip(condition_cols, key))
        neural_stats = _summarise_series(group[NEURAL_METHOD], settings)
        baseline_stats = _summarise_series(group[baseline_method], settings)
        delta_stats = _summarise_series(group["delta_neural_minus_baseline"], settings)
        mean_delta = delta_stats["mean"]
        neural_better = (
            mean_delta > 0 if metric in HIGHER_IS_BETTER else mean_delta < 0
        ) if np.isfinite(mean_delta) else False
        rows.append({
            **base,
            "baseline_method": baseline_method,
            "neural_method": NEURAL_METHOD,
            "metric": metric,
            "paired_trials": delta_stats["n_trials"],
            "neural_mean": neural_stats["mean"],
            "neural_std": neural_stats["std"],
            "neural_se": neural_stats["se"],
            "neural_ci95_low": neural_stats["ci95_low"],
            "neural_ci95_high": neural_stats["ci95_high"],
            "baseline_mean": baseline_stats["mean"],
            "baseline_std": baseline_stats["std"],
            "baseline_se": baseline_stats["se"],
            "baseline_ci95_low": baseline_stats["ci95_low"],
            "baseline_ci95_high": baseline_stats["ci95_high"],
            "delta_mean_neural_minus_baseline": delta_stats["mean"],
            "delta_std": delta_stats["std"],
            "delta_se": delta_stats["se"],
            "delta_ci95_low": delta_stats["ci95_low"],
            "delta_ci95_high": delta_stats["ci95_high"],
            "delta_ci95_half_width": delta_stats["ci95_half_width"],
            "higher_is_better": int(metric in HIGHER_IS_BETTER),
            "neural_better_by_mean": int(neural_better),
            "ci_method": f"paired_delta_normal_approx_z_{settings.z_value:g}_over_repeated_trials",
        })
    return rows


def build_neural_comparison_with_ci(
    df: pd.DataFrame,
    metrics: Sequence[str] | None = None,
    settings: CISettings | None = None,
) -> pd.DataFrame:
    """Compare neural RR-MRP with baselines using paired per-trial deltas."""

    settings = settings or CISettings()
    work = _normalise_group_columns(df)
    metrics_available = _available_metrics(work, metrics)
    if "trial" not in work.columns:
        raise ValueError("results_trials.csv must contain a 'trial' column for paired neural comparisons")

    condition_cols = ["scenario", "scenario_label", "epsilon", "n_sample"]
    rows: list[dict[str, object]] = []
    methods_present = set(work["method"].astype(str))
    baselines = [m for m in BASELINE_METHODS if m in methods_present]
    if NEURAL_METHOD not in methods_present:
        return pd.DataFrame()

    for metric in metrics_available:
        for baseline in baselines:
            rows.extend(_paired_neural_baseline_rows(work, metric, baseline, condition_cols, settings))
    return pd.DataFrame(rows)


def _read_results(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    numeric_candidates = ["epsilon", "n_sample", "trial", *DEFAULT_METRICS]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _serialise_path_for_manifest(path: Path, *, cwd: Path | None = None) -> str:
    """Return a portable path for manifest metadata without changing results.

    The CI helper is often run from the repository root. In that case, paths in
    ``uncertainty_manifest.json`` should be repository-relative rather than
    absolute local machine paths. This keeps generated evidence portable for
    examiners while leaving all metric values unchanged.
    """

    cwd = (cwd or Path.cwd()).resolve()
    try:
        return str(path.resolve().relative_to(cwd)).replace("\\", "/")
    except ValueError:
        return str(path)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_ci_outputs_for_results_file(
    results_path: Path,
    *,
    settings: CISettings | None = None,
    metrics: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Write CI outputs next to one results_trials.csv file."""

    settings = settings or CISettings()
    df = _read_results(results_path)
    output_dir = results_path.parent
    summary_ci = build_summary_with_ci(df, metrics=metrics, settings=settings)
    rankings_ci = build_method_rankings_with_ci(summary_ci)
    neural_ci = build_neural_comparison_with_ci(df, metrics=metrics, settings=settings)

    paths = {
        "summary_with_ci": output_dir / "summary_with_ci.csv",
        "method_rankings_with_ci": output_dir / "method_rankings_with_ci.csv",
        "neural_comparison_with_ci": output_dir / "neural_comparison_with_ci.csv",
    }
    _write_csv(summary_ci, paths["summary_with_ci"])
    _write_csv(rankings_ci, paths["method_rankings_with_ci"])
    _write_csv(neural_ci, paths["neural_comparison_with_ci"])
    return paths


def _find_results_files(input_dir: Path) -> list[Path]:
    if input_dir.is_file():
        if input_dir.name != "results_trials.csv":
            raise ValueError("When --input-dir is a file it must be named results_trials.csv")
        return [input_dir]
    direct = input_dir / "results_trials.csv"
    if direct.exists():
        return [direct]
    return sorted(input_dir.rglob("results_trials.csv"))


def _load_combined_results(results_files: Sequence[Path], root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in results_files:
        df = _read_results(path)
        try:
            source_run = str(path.parent.relative_to(root))
        except ValueError:
            source_run = path.parent.name
        df["source_run"] = source_run
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _count_trials_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    work = _normalise_group_columns(df)
    return (
        work.groupby(["scenario", "scenario_label", "epsilon", "n_sample"], dropna=False)["trial"]
        .nunique()
        .reset_index(name="n_trials")
        .sort_values(["scenario", "n_sample", "epsilon"])
    )


def _update_readme(root: Path, combined_df: pd.DataFrame, outputs: dict[str, Path], settings: CISettings) -> None:
    readme = root / "README.md"
    trial_counts = _count_trials_by_condition(combined_df)
    n_results = int(len(combined_df))
    metrics = _available_metrics(combined_df)
    section_lines = [
        "",
        "## Repeated-trial uncertainty summaries",
        "",
        "This folder includes an added reporting layer for uncertainty summaries. The original `results_trials.csv`, `summary.csv`, `method_rankings.csv`, and `neural_comparison.csv` files were not edited.",
        "",
        f"Confidence intervals use the repeated stochastic trials in `results_trials.csv` with sample standard deviation, standard error `std / sqrt(n)`, and a normal-approximation 95% interval `mean ± {settings.z_value:g} * SE`.",
        "These intervals describe stochastic variability in the synthetic simulation runs. They are not a claim of external validity for real elections and they are not a formal statistical significance test.",
        "",
        f"Rows analysed across repeated-trial result files: `{n_results}`.",
        "",
        "Metrics summarised:",
        "",
        *[f"- `{m}`" for m in metrics],
        "",
        "Generated uncertainty files:",
        "",
        *[f"- `{p.relative_to(root) if p.is_relative_to(root) else p}`" for p in outputs.values()],
        "",
        "Trials by condition:",
        "",
        "| scenario | epsilon | n_sample | trials |",
        "|---|---:|---:|---:|",
    ]
    for _, row in trial_counts.iterrows():
        section_lines.append(
            f"| {row['scenario_label']} | {row['epsilon']} | {int(row['n_sample'])} | {int(row['n_trials'])} |"
        )
    section_lines.extend([
        "",
        "Limitations:",
        "",
        "- Confidence intervals are based on available repeated runs only; conditions with fewer trials have wider and less stable intervals.",
        "- The intervals reflect randomness in the synthetic population/sampling/RR/model-training pipeline, not uncertainty from real polling deployment.",
        "- Neural RR-MRP should only be described as better for a condition when the comparison tables support that condition; these files do not force or assume a neural win.",
        "",
    ])
    marker = "## Repeated-trial uncertainty summaries"
    existing = readme.read_text(encoding="utf-8") if readme.exists() else "# Evidence pack\n"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n"
    readme.write_text(existing.rstrip() + "\n" + "\n".join(section_lines).lstrip(), encoding="utf-8")


def generate_uncertainty_outputs(
    input_dir: Path,
    *,
    update_readme: bool = True,
    settings: CISettings | None = None,
    metrics: Sequence[str] | None = None,
) -> dict[str, object]:
    """Generate CI summaries for an evidence directory or a single result file."""

    settings = settings or CISettings()
    input_dir = input_dir.resolve()
    results_files = _find_results_files(input_dir)
    if not results_files:
        raise FileNotFoundError(f"No results_trials.csv files found under {input_dir}")

    per_run_outputs = []
    for result_file in results_files:
        per_run_outputs.append({
            "results_trials": result_file,
            "outputs": write_ci_outputs_for_results_file(result_file, settings=settings, metrics=metrics),
        })

    combined_outputs: dict[str, Path] = {}
    root = input_dir if input_dir.is_dir() else input_dir.parent
    if len(results_files) > 1:
        combined_df = _load_combined_results(results_files, root)
        summary_ci = build_summary_with_ci(combined_df, metrics=metrics, settings=settings)
        rankings_ci = build_method_rankings_with_ci(summary_ci)
        neural_ci = build_neural_comparison_with_ci(combined_df, metrics=metrics, settings=settings)
        combined_outputs = {
            "summary_with_ci": root / "summary_with_ci.csv",
            "method_rankings_with_ci": root / "method_rankings_with_ci.csv",
            "neural_comparison_with_ci": root / "neural_comparison_with_ci.csv",
        }
        _write_csv(summary_ci, combined_outputs["summary_with_ci"])
        _write_csv(rankings_ci, combined_outputs["method_rankings_with_ci"])
        _write_csv(neural_ci, combined_outputs["neural_comparison_with_ci"])
        if update_readme:
            _update_readme(root, combined_df, combined_outputs, settings)
    elif update_readme:
        combined_df = _load_combined_results(results_files, root)
        _update_readme(root, combined_df, per_run_outputs[0]["outputs"], settings)

    manifest = {
        "input_dir": _serialise_path_for_manifest(input_dir),
        "results_files": [_serialise_path_for_manifest(p) for p in results_files],
        "ci_method": {
            "confidence_level": settings.confidence_level,
            "z_value": settings.z_value,
            "formula": "mean +/- z * sample_std / sqrt(n_trials)",
            "std_ddof": 1,
        },
        "raw_results_modified": False,
        "per_run_outputs": [
            {
                "results_trials": _serialise_path_for_manifest(item["results_trials"]),
                "outputs": {k: _serialise_path_for_manifest(v) for k, v in item["outputs"].items()},
            }
            for item in per_run_outputs
        ],
        "combined_outputs": {k: _serialise_path_for_manifest(v) for k, v in combined_outputs.items()},
    }
    manifest_path = root / "uncertainty_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest_path"] = _serialise_path_for_manifest(manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add CI summaries to repeated-trial FairVote-AI evidence.")
    parser.add_argument(
        "--input-dir",
        default="experiments/outputs",
        help="Evidence directory or a results_trials.csv file.",
    )
    parser.add_argument(
        "--no-readme-update",
        action="store_true",
        help="Write CSV/manifest outputs but leave README.md unchanged.",
    )
    parser.add_argument(
        "--metrics",
        default="",
        help="Optional comma-separated metric list. Defaults to core evidence metrics present in results_trials.csv.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()] or None
    manifest = generate_uncertainty_outputs(
        Path(args.input_dir),
        update_readme=not args.no_readme_update,
        metrics=metrics,
    )
    print("Generated uncertainty summaries")
    print(f"- manifest: {manifest['manifest_path']}")
    for item in manifest["per_run_outputs"]:
        print(f"- per-run: {item['results_trials']}")
        for path in item["outputs"].values():
            print(f"  - {path}")
    for path in manifest["combined_outputs"].values():
        print(f"- combined: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
