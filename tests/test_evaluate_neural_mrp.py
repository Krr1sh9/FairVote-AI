"""Tests for experiments.evaluate_neural_mrp.

The checks cover the constrained evidence-generation wrapper without changing
experiment methodology or final result files.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from experiments import evaluate_neural_mrp as eval_mod


def test_scenario_aliases_are_normalised():
    assert eval_mod._normalise_scenarios(["no_bias", "shy_voter", "privacy_helps"]) == [
        "no_bias",
        "shy_fixed",
        "shy_privacy_helps",
    ]


@pytest.mark.parametrize("bad", [["unknown"], [""]])
def test_scenario_aliases_reject_unknown_or_empty(bad):
    with pytest.raises(ValueError):
        eval_mod._normalise_scenarios(bad)


def test_neural_comparison_marks_error_delta_direction():
    summary = [
        {
            "scenario": "no_bias",
            "epsilon": 1.0,
            "n_sample": 250,
            "method": "neural_rr_mrp",
            "n_rows": 1,
            "mean_overall_l1": 0.10,
            "mean_overall_mae": 0.02,
            "mean_weighted_region_l1": 0.11,
            "mean_weighted_age_l1": 0.20,
            "mean_p90_region_l1_major": 0.12,
            "mean_p90_age_l1_major": 0.22,
            "mean_worst_region_l1_major": 0.13,
            "mean_worst_age_l1_major": 0.23,
            "mean_runtime_sec": 2.0,
            "mean_winner_correct": 1.0,
        },
        {
            "scenario": "no_bias",
            "epsilon": 1.0,
            "n_sample": 250,
            "method": "mrp_rr_poststrat",
            "n_rows": 1,
            "mean_overall_l1": 0.15,
            "mean_overall_mae": 0.03,
            "mean_weighted_region_l1": 0.10,
            "mean_weighted_age_l1": 0.30,
            "mean_p90_region_l1_major": 0.20,
            "mean_p90_age_l1_major": 0.30,
            "mean_worst_region_l1_major": 0.21,
            "mean_worst_age_l1_major": 0.32,
            "mean_runtime_sec": 1.0,
            "mean_winner_correct": 0.0,
        },
    ]

    rows = eval_mod.build_neural_comparison(summary)
    hit = [r for r in rows if r["baseline_method"] == "mrp_rr_poststrat"]
    assert len(hit) == 1
    row = hit[0]
    assert row["delta_mean_overall_l1"] < 0
    assert row["neural_better_mean_overall_l1"] == 1
    assert row["delta_mean_runtime_sec"] > 0
    assert row["neural_better_mean_runtime_sec"] == 0
    assert row["delta_mean_winner_correct"] > 0
    assert row["neural_better_mean_winner_correct"] == 1
    assert row["complexity_supported_basic"] == 1


def test_method_rankings_rank_lower_error_first():
    summary = [
        {"scenario": "no_bias", "epsilon": 1.0, "n_sample": 250, "method": "neural_rr_mrp", "n_rows": 1, "mean_overall_l1": 0.2, "mean_weighted_region_l1": 0.3, "mean_weighted_age_l1": 0.3, "mean_runtime_sec": 2.0},
        {"scenario": "no_bias", "epsilon": 1.0, "n_sample": 250, "method": "baseline_rr_debias", "n_rows": 1, "mean_overall_l1": 0.1, "mean_weighted_region_l1": 0.4, "mean_weighted_age_l1": 0.4, "mean_runtime_sec": 0.1},
    ]
    ranks = eval_mod.build_method_rankings(summary)
    overall = [r for r in ranks if r["metric"] == "mean_overall_l1"]
    assert overall[0]["rank"] == 1
    assert overall[0]["method"] == "baseline_rr_debias"
    assert overall[1]["method"] == "neural_rr_mrp"


def test_small_preset_smoke_without_neural_for_infrastructure(tmp_path: Path, project_root: Path):
    """Fast CLI smoke test for the wrapper/output plumbing.

    Neural-enabled execution is covered by the slow/full experiment environment
    because this container may not have a usable PyTorch runtime. This test still
    checks that the script writes the expected machine-readable artefacts.
    """
    out_dir = tmp_path / "outputs"
    cmd = [
        sys.executable,
        "-m",
        "experiments.evaluate_neural_mrp",
        "--preset",
        "small",
        "--disable_neural",
        "--eps",
        "1.0",
        "--sample_sizes",
        "120",
        "--scenarios",
        "no_bias",
        "--population_n",
        "800",
        "--trials",
        "1",
        "--mrp_steps",
        "2",
        "--out_dir",
        str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(project_root), text=True, capture_output=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    run_dirs = sorted(out_dir.glob("*_neural_mrp_justification_small*"))
    assert run_dirs
    run_dir = run_dirs[-1]
    for name in [
        "config.json",
        "results_trials.csv",
        "results_trials.jsonl",
        "summary.csv",
        "summary.jsonl",
        "neural_comparison.csv",
        "method_rankings.csv",
        "neural_verdict.csv",
    ]:
        assert (run_dir / name).exists(), name

    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as f:
        methods = {row["method"] for row in csv.DictReader(f)}
    assert "baseline_rr_debias" in methods
    assert "mrp_rr_poststrat" in methods
    assert "neural_rr_mrp" not in methods
