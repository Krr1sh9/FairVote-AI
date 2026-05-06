from __future__ import annotations

import json
import zipfile
from io import BytesIO

import numpy as np
import pytest

from app.parsing.upload import read_csv_bytes, read_jsonl_bytes, read_jsonl_bytes_with_report
from app.services import inference
from app.services.category import (
    CategoryMap,
    build_category_map,
    encode_categories,
    filter_valid,
    parse_hidden_layers,
    poststratify_probabilities,
)
from app.services.exports import (
    build_group_audit_csv,
    build_overall_estimates_csv,
    build_results_bundle,
    build_results_summary_markdown,
    build_scenario_bundle,
)


def test_dashboard_csv_parsing_string_values() -> None:
    rows = read_csv_bytes(b"reported_choice,region\n0,London\n1,North\n")
    assert rows == [
        {"reported_choice": "0", "region": "London"},
        {"reported_choice": "1", "region": "North"},
    ]


def test_dashboard_jsonl_parsing_flattens_demographics_and_rejects_invalid_by_default() -> None:
    raw = (
        json.dumps({"perturbed_answer": 1, "demographics": {"region": "London", "age_band": "18-24"}}) + "\nnot-json\n"
    ).encode()
    with pytest.raises(ValueError, match="invalid JSONL on line 2"):
        read_jsonl_bytes(raw)

    report = read_jsonl_bytes_with_report(raw, strict=False)
    assert report.rows == [{"perturbed_answer": "1", "region": "London", "age_band": "18-24"}]
    assert report.invalid_lines == [2]
    assert report.rejected_count == 1


def test_dashboard_jsonl_strict_mode_reports_invalid_input() -> None:
    with pytest.raises(ValueError, match="invalid JSONL"):
        read_jsonl_bytes(b"not-json\n", strict=True)


def test_dashboard_category_mapping_and_filtering() -> None:
    cmap = build_category_map(["Lab", "Con", "Lab"], k_override=3)
    assert cmap.labels == ["Con", "Lab", "(missing_2)"]
    encoded = encode_categories(["Lab", "Con", "Unknown"], cmap)
    reported, truth, mask = filter_valid(encoded, None)
    assert reported.tolist() == [1, 0]
    assert truth is None
    assert mask.tolist() == [True, True, False]


def test_dashboard_invalid_input_helpers_reject_bad_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        parse_hidden_layers("32,-1")
    with pytest.raises(ValueError, match="weights must match"):
        poststratify_probabilities(np.ones((2, 3)), [1.0])


def test_dashboard_method_selection_falls_back_when_linear_mrp_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inference, "_HAS_RR_MRP", False)
    method, warning = inference.resolve_estimation_method("Linear RR-aware MRP")
    assert method == "RR debiasing"
    assert warning and "Falling back" in warning


def test_dashboard_overall_and_group_csv_generation() -> None:
    labels = ["A", "B"]
    p_baseline = np.array([0.6, 0.4])
    overall = build_overall_estimates_csv(
        display_labels=labels,
        p_baseline=p_baseline,
        p_true=np.array([0.55, 0.45]),
    ).decode()
    assert "category_id,label,rr_debias_p,true_p" in overall
    assert "0,A,0.6,0.55" in overall

    group_csv = build_group_audit_csv(
        [{"group": "London", "n": 10, "mass": 0.5, "major": True, "baseline_l1": 0.1}]
    ).decode()
    assert "group,n,mass,major,baseline_l1" in group_csv
    assert "London,10,0.5,True,0.1" in group_csv


def test_dashboard_result_summary_generation_contains_key_claims() -> None:
    md = build_results_summary_markdown(
        generated_at="2026-01-01 10:00:00",
        n_rows_used=100,
        epsilon=1.0,
        k=2,
        method="RR debiasing",
        group_cols=["region"],
        group_rows=[{"group": "London", "n": 50, "mass": 0.5, "major": True, "baseline_l1": 0.2}],
        group_rows_mrp=None,
        learned_l1_key="linear_mrp_l1",
        learned_method_label="Linear RR-aware MRP",
        major_mass=0.02,
        p_baseline=np.array([0.6, 0.4]),
        p_true=np.array([0.55, 0.45]),
        p_post_direct=None,
        p_mrp_post=None,
        plot_names=["overall_comparison.png"],
    ).decode()
    assert "# FairVote-AI Results Summary" in md
    assert "epsilon: 1.0" in md
    assert "Truth-based overall metrics" in md
    assert "overall_comparison.png" in md


def test_dashboard_export_bundle_contains_expected_files() -> None:
    bundle = build_results_bundle(
        overall_csv_bytes=b"category_id,label,rr_debias_p\n0,A,0.6\n",
        group_csv_bytes=b"group,n,mass,major,baseline_l1\nLondon,10,0.5,True,0.1\n",
        summary_md_bytes=b"# Summary\n",
        meta_bytes=b'{"method":"RR debiasing"}',
        plot_bytes={"overall.png": b"fake-png"},
    )
    with zipfile.ZipFile(BytesIO(bundle)) as zf:
        assert sorted(zf.namelist()) == [
            "group_audit.csv",
            "metadata.json",
            "overall_estimates.csv",
            "plots/overall.png",
            "results_summary.md",
        ]
        assert zf.read("plots/overall.png") == b"fake-png"


def test_dashboard_scenario_bundle_generation() -> None:
    bundle = build_scenario_bundle(
        poll_csv=b"region,reported_choice\nLondon,A\n",
        population_csv=b"region,count\nLondon,100\n",
        overall_csv=b"label,true_p\nA,1.0\n",
        group_csv=b"group,n\nLondon,1\n",
        summary_md=b"# Scenario\n",
        metadata={"scenario": "no_bias"},
        plot_bytes={},
    )
    with zipfile.ZipFile(BytesIO(bundle)) as zf:
        assert "synthetic_poll.csv" in zf.namelist()
        assert json.loads(zf.read("metadata.json"))["scenario"] == "no_bias"


def test_truth_columns_require_explicit_synthetic_mode() -> None:
    from app.services.upload_analysis import candidate_truth_columns, validate_truth_column_policy

    assert candidate_truth_columns(["region", "true_choice", "reported_choice"]) == ["true_choice"]
    with pytest.raises(ValueError, match="synthetic evaluation mode"):
        validate_truth_column_policy(truth_col="true_choice", synthetic_evaluation_mode=False)
    assert validate_truth_column_policy(truth_col="true_choice", synthetic_evaluation_mode=True) == "true_choice"
    assert validate_truth_column_policy(truth_col=None, synthetic_evaluation_mode=False) is None
