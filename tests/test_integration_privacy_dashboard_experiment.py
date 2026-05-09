"""Integration tests over respondent, dashboard parsing, and experiment pipeline.

These tests intentionally cross module boundaries. They complement focused unit
and property tests by proving that privacy-preserving submissions can be stored,
read as dashboard input, aggregated, and that a tiny experiment run emits the
expected reproducible evidence tables.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace

import pytest

pytest.importorskip("flask")

from app.parsing.upload import read_jsonl_bytes
from experiments.pipeline.config import ExperimentConfig
from experiments.pipeline.io import write_experiment_outputs
from experiments.pipeline.runner import execute_experiment
from experiments.pipeline.summary import required_result_columns
from respondent.server import create_app


@pytest.mark.integration
def test_respondent_submission_storage_results_and_dashboard_jsonl_parsing(tmp_path) -> None:
    config_path = tmp_path / "poll_config.json"
    data_path = tmp_path / "responses.jsonl"
    config_path.write_text(
        json.dumps(
            {
                "question": "Which option?",
                "options": ["A", "B", "C"],
                "epsilon": 1.0,
                "demographic_fields": [
                    {"name": "region", "label": "Region", "options": ["North", "South"], "required": False}
                ],
            }
        ),
        encoding="utf-8",
    )
    app = create_app(config_path=config_path, data_path=data_path)
    app.config["TESTING"] = True

    with app.test_client() as client:
        ok = client.post(
            "/api/respond",
            json={"perturbed_answer": 2, "demographics": {"region": "North"}},
        )
        assert ok.status_code == 201

        nested_raw = client.post(
            "/api/respond",
            json={"perturbed_answer": 1, "audit": {"events": [{"raw_answer": 0}]}},
        )
        assert nested_raw.status_code == 400

        results = client.get("/api/results")
        assert results.status_code == 200
        body = results.get_json()
        assert body["total"] == 1
        assert body["counts"] == [0, 0, 1]

    raw_jsonl = data_path.read_text(encoding="utf-8")
    assert "raw_answer" not in raw_jsonl
    assert "selected_answer" not in raw_jsonl
    assert "true_answer" not in raw_jsonl

    dashboard_rows = read_jsonl_bytes(raw_jsonl.encode("utf-8"), strict=True)
    assert len(dashboard_rows) == 1
    assert dashboard_rows[0]["perturbed_answer"] == "2"
    assert dashboard_rows[0]["region"] == "North"
    assert "timestamp" in dashboard_rows[0]


@pytest.mark.integration
def test_tiny_experiment_smoke_run_writes_reproducible_evidence_files(tmp_path) -> None:
    config = ExperimentConfig(
        k=5,
        eps_list=[1.0],
        scenarios=["simple_linear"],
        population_n=600,
        n_sample=80,
        sample_sizes=[80],
        trials=1,
        seed=2026,
        sampling="srs",
        strata=["region"],
        allocation="proportional",
        min_per_stratum=0,
        biased_feature="region",
        biased_multipliers={},
        feature_order=["region", "age_group", "education", "gender", "urbanicity"],
        shy_category=0,
        shy_honesty=0.80,
        mrp_steps=2,
        mrp_lr=0.05,
        mrp_l2=1.0,
        mrp_batch_size=128,
        verbose_every=0,
        enable_neural=False,
        neural_hidden_layers=(8,),
        neural_steps=2,
        neural_lr=0.01,
        neural_batch_size=128,
        neural_seed=123,
        neural_dropout=0.0,
        neural_weight_decay=1e-4,
        major_mass=0.02,
        methods=["baseline_rr_debias", "mrp_rr_poststrat"],
        preset="smoke_test",
    )
    result = execute_experiment(config)
    run_dir = tmp_path / "run"
    write_experiment_outputs(run_dir, result)

    expected_files = {
        "raw_trials.csv",
        "summary_with_ci.csv",
        "paired_comparisons.csv",
        "runtime_profile.csv",
        "config.json",
        "manifest.json",
        "README.md",
    }
    assert expected_files.issubset({p.name for p in run_dir.iterdir()})

    with (run_dir / "raw_trials.csv").open(newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))
    with (run_dir / "summary_with_ci.csv").open(newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))

    assert {row["method"] for row in raw_rows} == set(config.methods)
    assert {row["scenario"] for row in raw_rows} == {"simple_linear"}
    assert required_result_columns().issubset(raw_rows[0].keys())
    assert manifest["config"]["preset"] == "smoke_test"
    assert manifest["config"]["methods"] == config.methods
    assert manifest["n_result_rows"] == len(raw_rows)

    for summary_row in summary_rows:
        matching = [
            r
            for r in raw_rows
            if r["sample_size"] == summary_row["sample_size"]
            and r["scenario"] == summary_row["scenario"]
            and r["epsilon"] == summary_row["epsilon"]
            and r["method"] == summary_row["method"]
        ]
        assert int(summary_row["n_rows"]) == len(matching)
