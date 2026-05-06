from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path

from experiments.pipeline.config import ExperimentConfig, default_methods
from experiments.pipeline.io import write_csv, write_json
from experiments.pipeline.runner import execute_experiment
from experiments.pipeline.summary import required_result_columns


def _small_config() -> ExperimentConfig:
    return ExperimentConfig(
        k=5,
        eps_list=[0.5, 1.0],
        scenarios=["no_bias", "shy_fixed"],
        population_n=1_200,
        n_sample=180,
        trials=1,
        seed=777,
        sampling="srs",
        strata=["region"],
        allocation="proportional",
        min_per_stratum=0,
        biased_feature="region",
        biased_multipliers={},
        feature_order=["region", "age_group", "education", "gender", "urbanicity"],
        shy_category=0,
        shy_honesty=0.80,
        mrp_steps=3,
        mrp_lr=0.05,
        mrp_l2=1.0,
        mrp_batch_size=128,
        verbose_every=0,
        enable_neural=False,
        neural_hidden_layers=(8,),
        neural_steps=3,
        neural_lr=0.01,
        neural_batch_size=128,
        neural_seed=321,
        neural_dropout=0.0,
        neural_weight_decay=1e-4,
        major_mass=0.02,
    )


def test_pipeline_outputs_requested_methods_scenarios_and_metrics():
    config = _small_config()
    result = execute_experiment(config)

    expected_methods = set(default_methods(enable_neural=False))
    assert {row["method"] for row in result.rows} == expected_methods
    assert {row["method"] for row in result.summary} == expected_methods
    assert {row["scenario"] for row in result.rows} == set(config.scenarios)
    assert {row["scenario"] for row in result.summary} == set(config.scenarios)
    assert {float(row["epsilon"]) for row in result.rows} == set(config.eps_list)

    required = required_result_columns()
    for row in result.rows:
        assert required.issubset(row.keys()), f"missing columns for {row.get('method')}: {required - set(row)}"
        assert row["sample_size"] == config.n_sample
        assert row["config_seed"] == config.seed
        assert row["random_seed"] is not None


def test_summary_matches_raw_trial_rows():
    config = _small_config()
    result = execute_experiment(config)

    for summary_row in result.summary:
        raw = [
            row
            for row in result.rows
            if row["scenario"] == summary_row["scenario"]
            and float(row["epsilon"]) == float(summary_row["epsilon"])
            and row["method"] == summary_row["method"]
            and int(row.get("skipped", 0)) == 0
        ]
        assert int(summary_row["n_rows"]) == len(raw)
        if not raw:
            continue
        mean = sum(float(row["overall_l1"]) for row in raw) / len(raw)
        assert math.isclose(float(summary_row["mean_overall_l1"]), mean, rel_tol=1e-12, abs_tol=1e-12)


def test_manifest_and_written_files_match_run(tmp_path: Path):
    config = _small_config()
    result = execute_experiment(config)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(run_dir / "config.json", result.config.to_dict())
    write_json(run_dir / "manifest.json", result.manifest)
    write_csv(run_dir / "results_trials.csv", result.rows)
    write_csv(run_dir / "summary.csv", result.summary)

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["seed"] == config.seed
    assert manifest["config"]["n_sample"] == config.n_sample
    assert manifest["scenarios"] == config.scenarios
    assert manifest["epsilons"] == config.eps_list
    assert manifest["methods"] == config.methods
    assert manifest["n_result_rows"] == len(result.rows)
    assert manifest["n_summary_rows"] == len(result.summary)

    with (run_dir / "results_trials.csv").open(newline="", encoding="utf-8") as f:
        result_rows = list(csv.DictReader(f))
    with (run_dir / "summary.csv").open(newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))

    assert result_rows
    assert summary_rows
    assert required_result_columns().issubset(result_rows[0].keys())
    assert {row["scenario"] for row in result_rows} == set(config.scenarios)
    assert {row["method"] for row in result_rows} == set(config.methods)


def test_research_scenarios_and_oracle_methods_are_available():
    config = _small_config()
    research_config = replace(
        config,
        eps_list=[1.0],
        scenarios=["simple_linear", "nonlinear_interaction"],
        population_n=900,
        n_sample=140,
        trials=1,
        mrp_steps=2,
        methods=[
            "oracle_true_sample_distribution",
            "baseline_rr_debias",
            "linear_rr_no_poststrat",
            "mrp_rr_poststrat",
            "oracle_true_linear_mrp_poststrat",
        ],
    )
    result = execute_experiment(research_config)
    assert {row["scenario"] for row in result.rows} == {"simple_linear", "nonlinear_interaction"}
    assert {row["method"] for row in result.rows} == set(research_config.methods)
    assert all("scenario_truth_model" in row for row in result.rows)
    oracle_rows = [row for row in result.rows if row["method"].startswith("oracle_true")]
    assert oracle_rows
    assert all(int(row.get("oracle_uses_true_labels", 0)) == 1 for row in oracle_rows)


def test_paired_comparison_summary_reports_neural_minus_linear_deltas():
    config = _small_config()
    paired_config = replace(
        config,
        eps_list=[1.0],
        scenarios=["nonlinear_interaction"],
        methods=["mrp_rr_poststrat", "neural_rr_mrp"],
    )
    rows = [
        {
            "scenario": "nonlinear_interaction",
            "epsilon": 1.0,
            "trial": 0,
            "method": "mrp_rr_poststrat",
            "skipped": 0,
            "overall_l1": 0.30,
            "worst_group_l1_major": 0.50,
            "worst_region_l1_major": 0.45,
            "worst_age_l1_major": 0.50,
        },
        {
            "scenario": "nonlinear_interaction",
            "epsilon": 1.0,
            "trial": 0,
            "method": "neural_rr_mrp",
            "skipped": 0,
            "overall_l1": 0.20,
            "worst_group_l1_major": 0.40,
            "worst_region_l1_major": 0.35,
            "worst_age_l1_major": 0.40,
        },
    ]
    from experiments.pipeline.summary import aggregate_paired_comparisons

    paired = aggregate_paired_comparisons(rows, paired_config)
    assert len(paired) == 1
    assert paired[0]["n_paired_trials"] == 1
    assert math.isclose(float(paired[0]["mean_delta_overall_l1"]), -0.10, abs_tol=1e-12)
    assert math.isclose(float(paired[0]["win_rate_delta_overall_l1"]), 1.0, abs_tol=1e-12)


def test_multi_sample_summary_and_output_files_are_reproducible(tmp_path: Path):
    from experiments.pipeline.io import write_experiment_outputs

    config = replace(
        _small_config(),
        eps_list=[1.0],
        scenarios=["simple_linear"],
        population_n=900,
        n_sample=80,
        sample_sizes=[80, 120],
        trials=2,
        mrp_steps=2,
        methods=["baseline_rr_debias", "mrp_rr_poststrat"],
        preset="smoke_test",
    )
    result = execute_experiment(config)
    run_dir = tmp_path / "evidence_run"
    write_experiment_outputs(run_dir, result)

    expected = {
        "raw_trials.csv",
        "results_trials.csv",
        "summary_with_ci.csv",
        "summary.csv",
        "paired_comparisons.csv",
        "ablations.csv",
        "runtime_profile.csv",
        "config.json",
        "manifest.json",
        "failures.csv",
        "README.md",
    }
    assert expected.issubset({p.name for p in run_dir.iterdir()})

    with (run_dir / "raw_trials.csv").open(newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))
    with (run_dir / "summary_with_ci.csv").open(newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    with (run_dir / "runtime_profile.csv").open(newline="", encoding="utf-8") as f:
        runtime_rows = list(csv.DictReader(f))

    assert {int(row["sample_size"]) for row in raw_rows} == {80, 120}
    assert {int(row["sample_size"]) for row in summary_rows} == {80, 120}
    assert {int(row["sample_size"]) for row in runtime_rows} == {80, 120}
    for summary_row in summary_rows:
        matching = [
            row
            for row in raw_rows
            if row["sample_size"] == summary_row["sample_size"]
            and row["scenario"] == summary_row["scenario"]
            and float(row["epsilon"]) == float(summary_row["epsilon"])
            and row["method"] == summary_row["method"]
            and int(row.get("skipped", 0) or 0) == 0
        ]
        assert int(summary_row["n_rows"]) == len(matching)
        if matching:
            expected_mean = sum(float(row["overall_l1"]) for row in matching) / len(matching)
            assert math.isclose(float(summary_row["mean_overall_l1"]), expected_mean, rel_tol=1e-12, abs_tol=1e-12)
            assert "ci95_low_overall_l1" in summary_row
            assert "ci95_high_overall_l1" in summary_row

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["preset"] == "smoke_test"
    assert manifest["sample_sizes"] == [80, 120]
    assert manifest["outputs"]["raw_trials"] == "raw_trials.csv"
    assert manifest["outputs"]["summary_with_ci"] == "summary_with_ci.csv"
    assert manifest["n_runtime_profile_rows"] == len(runtime_rows)


def test_final_evidence_preset_has_non_minimal_defaults():
    from experiments.mrp_vs_baselines import _build_parser, _config_from_args, _explicit_options

    argv = ["--preset", "final_evidence", "--disable_neural", "--trials", "50"]
    args = _build_parser().parse_args(argv)
    args._explicit_options = _explicit_options(argv)
    config = _config_from_args(args)
    assert config.preset == "final_evidence"
    assert config.eps_list == [0.2, 0.5, 1.0, 2.0, 4.0]
    assert config.sample_size_grid == [500, 1000, 2500, 5000]
    assert "sparse_minority_curve" in config.scenarios
    assert "privacy_helps" in config.scenarios
    assert "education_urbanicity_interaction" in config.scenarios
    assert config.trials == 50
    assert "hierarchical_rr_mrp_poststrat" in config.methods
    assert config.continue_on_error is False
    assert config.mrp_steps > 5
    assert config.neural_steps > 5
