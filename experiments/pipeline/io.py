"""Reproducible output helpers for experiment runs."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import ExperimentResult


def timestamped_run_dir(base: Path, name: str) -> Path:
    """Create a unique timestamped output directory with a plots subfolder."""
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    for i in range(1000):
        suffix = "" if i == 0 else f"_{i:03d}"
        run_dir = base / f"{ts}_{name}{suffix}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "plots").mkdir(parents=True, exist_ok=True)
            return run_dir
        except FileExistsError:
            continue
    ts2 = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    run_dir = base / f"{ts2}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    keys = list(fieldnames) if fieldnames is not None else sorted({k for r in rows for k in r})
    if not keys:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_experiment_outputs(run_dir: Path, result: ExperimentResult) -> None:
    """Write all final-evidence artefacts for an experiment run."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", result.config.to_dict())
    write_json(run_dir / "manifest.json", result.manifest)
    write_json(run_dir / "environment.json", result.manifest.get("provenance", {}))
    write_csv(run_dir / "raw_trials.csv", result.rows)
    # Backwards-compatible aliases used by older docs/scripts.
    write_csv(run_dir / "results_trials.csv", result.rows)
    write_csv(run_dir / "summary_with_ci.csv", result.summary)
    write_csv(run_dir / "summary.csv", result.summary)
    write_csv(run_dir / "paired_comparisons.csv", result.paired_comparisons, fieldnames=_paired_fieldnames())
    write_csv(run_dir / "ablations.csv", result.ablations, fieldnames=_ablation_fieldnames())
    write_csv(run_dir / "runtime_profile.csv", result.runtime_profile, fieldnames=_runtime_fieldnames())
    write_csv(run_dir / "failures.csv", result.failures, fieldnames=_failure_fieldnames())
    write_run_readme(run_dir / "README.md", result)
    write_sha256sums(run_dir / "sha256sums.txt", run_dir)


def write_sha256sums(path: Path, root: Path) -> None:
    """Write SHA-256 hashes for reproducibility-critical output files."""
    rows: list[str] = []
    for item in sorted(root.rglob("*")):
        if not item.is_file() or item == path:
            continue
        rel = item.relative_to(root).as_posix()
        digest = hashlib.sha256(item.read_bytes()).hexdigest()
        rows.append(f"{digest}  {rel}")
    path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


def write_run_readme(path: Path, result: ExperimentResult) -> None:
    cfg = result.config
    manifest = result.manifest
    non_skipped_rows = [row for row in result.summary if int(row.get("n_rows", 0) or 0) > 0]
    is_smoke = cfg.preset == "smoke_test" or cfg.trials < 5 or not non_skipped_rows
    evidence_note = (
        "This is a smoke/sanity run only; do not use it as final statistical evidence."
        if is_smoke
        else (
            "This run is intended as CPU-sized statistical evidence; inspect confidence intervals, "
            "iteration counts, and paired deltas before making final claims."
        )
    )
    lines = [
        "# FairVote-AI experiment run",
        "",
        f"Preset: `{cfg.preset}`",
        "",
        evidence_note,
        "",
        "## Configuration summary",
        "",
        f"- Seed: `{cfg.seed}`",
        f"- Scenarios: `{', '.join(cfg.scenarios)}`",
        f"- Epsilons: `{', '.join(str(e) for e in cfg.eps_list)}`",
        f"- Sample sizes: `{', '.join(str(s) for s in cfg.sample_size_grid)}`",
        f"- Trials per cell: `{cfg.trials}`",
        f"- Population size: `{cfg.population_n}`",
        f"- Methods: `{', '.join(cfg.methods)}`",
        f"- MRP steps: `{cfg.mrp_steps}`",
        f"- Neural steps: `{cfg.neural_steps}`",
        f"- Neural patience: `{cfg.neural_patience}`",
        "",
        "## Files",
        "",
        "- `raw_trials.csv`: one row per method × sample size × scenario × epsilon × trial.",
        "- `summary_with_ci.csv`: means, standard deviations and 95% bootstrap CIs over trials.",
        "- `paired_comparisons.csv`: paired neural-minus-linear deltas, bootstrap CIs and win rates.",
        "- `ablations.csv`: paired ablation deltas against the canonical linear RR-aware MRP baseline.",
        "- `runtime_profile.csv`: runtime/failure/skipped counts by method and condition.",
        "- `config.json`: complete reproducible configuration.",
        "- `manifest.json`: run manifest, output map, row counts, failures and total runtime.",
        "- `environment.json`: Python/platform/package provenance copied from the manifest.",
        "- `sha256sums.txt`: SHA-256 hashes for output integrity.",
        "- `failures.csv`: method-level errors if `continue_on_error=True` allowed a partial run.",
        "",
        "## Interpretation rules",
        "",
        "Negative neural-minus-linear deltas mean neural RR-MRP had lower error than the linear RR-aware MRP baseline.",
        "Treat a claim as supported only when the paired CI is mostly on the same side of zero and the win rate is convincing.",
        "Do not treat oracle baselines as deployable methods; they use synthetic true labels or known misreport structure.",
        "",
        "## Run status",
        "",
        f"- Result rows: `{manifest.get('n_result_rows')}`",
        f"- Summary rows: `{manifest.get('n_summary_rows')}`",
        f"- Paired comparison rows: `{manifest.get('n_paired_comparison_rows')}`",
        f"- Ablation rows: `{manifest.get('n_ablation_rows')}`",
        f"- Runtime profile rows: `{manifest.get('n_runtime_profile_rows')}`",
        f"- Failures: `{manifest.get('n_failures')}`",
        f"- Total runtime seconds: `{manifest.get('runtime_sec'):.3f}`",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _paired_fieldnames() -> list[str]:
    return [
        "config_seed",
        "scenario",
        "epsilon",
        "sample_size",
        "population_n",
        "sampling",
        "feature_set",
        "linear_method",
        "neural_method",
        "n_paired_trials",
        "mean_linear_overall_l1",
        "mean_neural_overall_l1",
        "mean_delta_overall_l1",
        "win_rate_delta_overall_l1",
        "ci95_low_delta_overall_l1",
        "ci95_high_delta_overall_l1",
        "mean_linear_worst_group_l1_major",
        "mean_neural_worst_group_l1_major",
        "mean_delta_worst_group_l1",
        "win_rate_delta_worst_group_l1",
        "ci95_low_delta_worst_group_l1",
        "ci95_high_delta_worst_group_l1",
    ]


def _ablation_fieldnames() -> list[str]:
    return [
        "config_seed",
        "scenario",
        "epsilon",
        "sample_size",
        "population_n",
        "reference_method",
        "ablation_method",
        "n_paired_trials",
        "mean_reference_overall_l1",
        "mean_comparison_overall_l1",
        "mean_delta_overall_l1",
        "win_rate_delta_overall_l1",
        "ci95_low_delta_overall_l1",
        "ci95_high_delta_overall_l1",
        "mean_reference_worst_group_l1_major",
        "mean_comparison_worst_group_l1_major",
        "mean_delta_worst_group_l1",
        "win_rate_delta_worst_group_l1",
        "ci95_low_delta_worst_group_l1",
        "ci95_high_delta_worst_group_l1",
    ]


def _runtime_fieldnames() -> list[str]:
    return [
        "config_seed",
        "scenario",
        "epsilon",
        "sample_size",
        "method",
        "n_rows",
        "n_failures",
        "n_skipped",
        "mean_runtime_sec",
        "std_runtime_sec",
        "total_runtime_sec",
    ]


def _failure_fieldnames() -> list[str]:
    return [
        "config_seed",
        "sample_size",
        "scenario",
        "trial",
        "epsilon",
        "method",
        "runtime_sec",
        "error_type",
        "error_message",
        "traceback",
    ]
