from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from experiments.add_uncertainty_summaries import (
    build_method_rankings_with_ci,
    build_neural_comparison_with_ci,
    build_summary_with_ci,
    generate_uncertainty_outputs,
)


def _minimal_results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"scenario": "no_bias", "scenario_label": "no_bias", "method": "neural_rr_mrp", "epsilon": 1.0, "n_sample": 100, "trial": 0, "overall_l1": 0.10, "winner_correct": 1, "runtime_sec": 2.0},
            {"scenario": "no_bias", "scenario_label": "no_bias", "method": "neural_rr_mrp", "epsilon": 1.0, "n_sample": 100, "trial": 1, "overall_l1": 0.20, "winner_correct": 1, "runtime_sec": 2.2},
            {"scenario": "no_bias", "scenario_label": "no_bias", "method": "mrp_rr_poststrat", "epsilon": 1.0, "n_sample": 100, "trial": 0, "overall_l1": 0.30, "winner_correct": 0, "runtime_sec": 0.5},
            {"scenario": "no_bias", "scenario_label": "no_bias", "method": "mrp_rr_poststrat", "epsilon": 1.0, "n_sample": 100, "trial": 1, "overall_l1": 0.40, "winner_correct": 1, "runtime_sec": 0.6},
        ]
    )


def test_summary_with_ci_uses_repeated_trials():
    summary = build_summary_with_ci(_minimal_results(), metrics=["overall_l1"])
    row = summary[(summary["method"] == "neural_rr_mrp") & (summary["metric"] == "overall_l1")].iloc[0]
    assert row["n_trials"] == 2
    assert math.isclose(row["mean"], 0.15)
    assert row["std"] > 0
    assert row["ci95_low"] < row["mean"] < row["ci95_high"]


def test_rankings_respect_metric_direction():
    summary = build_summary_with_ci(_minimal_results(), metrics=["overall_l1", "winner_correct"])
    rankings = build_method_rankings_with_ci(summary)
    best_l1 = rankings[(rankings["metric"] == "overall_l1") & (rankings["rank_by_mean"] == 1)].iloc[0]
    best_winner = rankings[(rankings["metric"] == "winner_correct") & (rankings["rank_by_mean"] == 1)].iloc[0]
    assert best_l1["method"] == "neural_rr_mrp"
    assert best_winner["method"] == "neural_rr_mrp"


def test_neural_comparison_uses_paired_deltas():
    comparison = build_neural_comparison_with_ci(_minimal_results(), metrics=["overall_l1"])
    row = comparison[comparison["baseline_method"] == "mrp_rr_poststrat"].iloc[0]
    assert row["paired_trials"] == 2
    assert math.isclose(row["delta_mean_neural_minus_baseline"], -0.2)
    assert row["neural_better_by_mean"] == 1


def test_cli_helper_writes_new_files_without_touching_results(tmp_path: Path):
    run_dir = tmp_path / "evidence" / "run"
    run_dir.mkdir(parents=True)
    results = run_dir / "results_trials.csv"
    _minimal_results().to_csv(results, index=False)
    before = results.read_text(encoding="utf-8")

    manifest = generate_uncertainty_outputs(tmp_path / "evidence", metrics=["overall_l1"], update_readme=True)

    assert (run_dir / "summary_with_ci.csv").exists()
    assert (run_dir / "method_rankings_with_ci.csv").exists()
    assert (run_dir / "neural_comparison_with_ci.csv").exists()
    assert (tmp_path / "evidence" / "uncertainty_manifest.json").exists()
    assert (tmp_path / "evidence" / "README.md").exists()
    assert results.read_text(encoding="utf-8") == before
    assert manifest["raw_results_modified"] is False
