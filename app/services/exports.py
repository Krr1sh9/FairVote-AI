"""Export-bundle helpers for the Streamlit dashboard.

All functions are Streamlit-free and suitable for unit tests.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from datetime import datetime
from typing import Any

import numpy as np

from app.services.metrics import group_metric_summary, overall_metrics


def to_csv_bytes(rows: list[dict[str, Any]], fieldnames: list[str]) -> bytes:
    s = io.StringIO()
    w = csv.DictWriter(s, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fieldnames})
    return s.getvalue().encode("utf-8")


def build_overall_estimates_csv(
    *,
    display_labels: list[str],
    p_baseline: np.ndarray,
    p_lo: np.ndarray | None = None,
    p_hi: np.ndarray | None = None,
    p_post_direct: np.ndarray | None = None,
    p_mrp_post: np.ndarray | None = None,
    p_mrp_sample: np.ndarray | None = None,
    p_true: np.ndarray | None = None,
    learned_post_col: str = "learned_poststrat_p",
    learned_sample_col: str = "learned_sample_p",
) -> bytes:
    fieldnames = ["category_id", "label", "rr_debias_p"]
    if p_lo is not None and p_hi is not None:
        fieldnames += ["ci_low", "ci_high"]
    if p_post_direct is not None:
        fieldnames += ["direct_poststrat_p"]
    if p_mrp_post is not None:
        fieldnames += [learned_post_col]
    if p_mrp_sample is not None:
        fieldnames += [learned_sample_col]
    if p_true is not None:
        fieldnames += ["true_p"]

    rows: list[dict[str, Any]] = []
    for i, lab in enumerate(display_labels):
        row: dict[str, Any] = {"category_id": i, "label": lab, "rr_debias_p": float(p_baseline[i])}
        if p_lo is not None and p_hi is not None:
            row["ci_low"] = float(p_lo[i])
            row["ci_high"] = float(p_hi[i])
        if p_post_direct is not None:
            row["direct_poststrat_p"] = float(p_post_direct[i])
        if p_mrp_post is not None:
            row[learned_post_col] = float(p_mrp_post[i])
        if p_mrp_sample is not None:
            row[learned_sample_col] = float(p_mrp_sample[i])
        if p_true is not None:
            row["true_p"] = float(p_true[i])
        rows.append(row)
    return to_csv_bytes(rows, fieldnames)


def build_group_audit_csv(
    group_rows: list[dict[str, Any]],
    *,
    group_rows_mrp: list[dict[str, Any]] | None = None,
    learned_l1_key: str = "learned_l1",
) -> bytes:
    if not group_rows:
        return b""
    fn = ["group", "n", "mass", "major", "baseline_l1"]
    mrp_map: dict[str, float] = {}
    if group_rows_mrp is not None:
        mrp_map = {str(r["group"]): float(r.get(learned_l1_key, float("nan"))) for r in group_rows_mrp}
        fn.append(learned_l1_key)

    rows: list[dict[str, Any]] = []
    for r in group_rows:
        out: dict[str, Any] = {
            "group": r["group"],
            "n": int(r.get("n", 0)),
            "mass": float(r.get("mass", 0.0)),
            "major": bool(r.get("major", False)),
            "baseline_l1": float(r.get("baseline_l1", float("nan"))),
        }
        if mrp_map:
            out[learned_l1_key] = mrp_map.get(str(r["group"]), float("nan"))
        rows.append(out)
    return to_csv_bytes(rows, fn)


def build_results_summary_markdown(
    *,
    generated_at: str,
    n_rows_used: int,
    epsilon: float,
    k: int,
    method: str,
    group_cols: list[str],
    group_rows: list[dict[str, Any]],
    group_rows_mrp: list[dict[str, Any]] | None,
    learned_l1_key: str,
    learned_method_label: str,
    major_mass: float,
    p_baseline: np.ndarray,
    p_true: np.ndarray | None,
    p_post_direct: np.ndarray | None,
    p_mrp_post: np.ndarray | None,
    plot_names: list[str],
) -> bytes:
    has_truth = p_true is not None
    metric_label = "L1 error vs truth" if has_truth else "L1 divergence vs overall (proxy)"
    md_lines = [
        "# FairVote-AI Results Summary",
        "",
        f"- Generated: {generated_at}",
        f"- n_rows_used: {n_rows_used}",
        f"- epsilon: {float(epsilon)}",
        f"- k: {k}",
        f"- method: {method}",
        f"- group_cols: {group_cols}",
        "",
        "## Overall estimates",
        "- See: overall_estimates.csv",
        "",
        "## Group / fairness metrics",
        f"- Metric shown: {metric_label}",
    ]
    if group_rows:
        base_s = group_metric_summary(
            group_rows, metric_key="baseline_l1", major_only=True, major_mass=float(major_mass)
        )
        md_lines.append(
            f"- RR debiasing worst-major: {base_s['worst']:.6f}, p90-major: {base_s['p90']:.6f}, weighted-major: {base_s['weighted']:.6f}"
        )
        if group_rows_mrp is not None:
            mrp_s = group_metric_summary(
                group_rows_mrp, metric_key=learned_l1_key, major_only=True, major_mass=float(major_mass)
            )
            md_lines.append(
                f"- {learned_method_label} worst-major: {mrp_s['worst']:.6f}, p90-major: {mrp_s['p90']:.6f}, weighted-major: {mrp_s['weighted']:.6f}"
            )
    if has_truth:
        om = overall_metrics(p_baseline, p_true)
        md_lines.append("")
        md_lines.append("## Truth-based overall metrics")
        md_lines.append(f"- RR debiasing: overall_l1={om['overall_l1']:.6f}, overall_mae={om['overall_mae']:.6f}")
        if p_post_direct is not None:
            om2 = overall_metrics(p_post_direct, p_true)
            md_lines.append(
                f"- Direct post-strat: overall_l1={om2['overall_l1']:.6f}, overall_mae={om2['overall_mae']:.6f}"
            )
        if p_mrp_post is not None:
            om3 = overall_metrics(p_mrp_post, p_true)
            md_lines.append(
                f"- {learned_method_label} post-strat: overall_l1={om3['overall_l1']:.6f}, overall_mae={om3['overall_mae']:.6f}"
            )
    md_lines.append("")
    md_lines.append("## Plots")
    if plot_names:
        for name in sorted(plot_names):
            md_lines.append(f"- {name}")
    else:
        md_lines.append("- (No plots generated; install matplotlib.)")
    md_lines.append("")
    return ("\n".join(md_lines)).encode("utf-8")


def build_metadata_json(
    *,
    generated_at: str,
    n_rows_used: int,
    epsilon: float,
    k: int,
    method: str,
    response_col: str,
    truth_col: str | None,
    group_cols: list[str],
    major_mass: float,
    has_truth: bool,
    has_population: bool,
    extra: dict[str, Any] | None = None,
) -> bytes:
    meta: dict[str, Any] = {
        "generated_at": generated_at,
        "n_rows_used": n_rows_used,
        "epsilon": float(epsilon),
        "k": int(k),
        "method": method,
        "response_col": response_col,
        "truth_col": truth_col,
        "group_cols": group_cols,
        "major_mass": float(major_mass),
        "has_truth": bool(has_truth),
        "has_population": bool(has_population),
    }
    if extra:
        meta.update(extra)
    return json.dumps(meta, indent=2).encode("utf-8")


def build_results_bundle(
    *,
    overall_csv_bytes: bytes,
    group_csv_bytes: bytes = b"",
    summary_md_bytes: bytes,
    meta_bytes: bytes,
    plot_bytes: dict[str, bytes],
) -> bytes:
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("overall_estimates.csv", overall_csv_bytes)
        if group_csv_bytes:
            zf.writestr("group_audit.csv", group_csv_bytes)
        zf.writestr("results_summary.md", summary_md_bytes)
        zf.writestr("metadata.json", meta_bytes)
        for name, data in plot_bytes.items():
            zf.writestr(f"plots/{name}", data)
    return bundle.getvalue()


def build_plot_zip(plot_bytes: dict[str, bytes]) -> bytes:
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in plot_bytes.items():
            zf.writestr(name, data)
    return zbuf.getvalue()


def now_string() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def build_scenario_bundle(
    *,
    poll_csv: bytes,
    population_csv: bytes,
    overall_csv: bytes,
    group_csv: bytes,
    summary_md: bytes,
    metadata: dict[str, Any],
    plot_bytes: dict[str, bytes],
) -> bytes:
    """Build the scenario simulator ZIP bundle."""

    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("synthetic_poll.csv", poll_csv)
        zf.writestr("synthetic_population.csv", population_csv)
        zf.writestr("overall_comparison.csv", overall_csv)
        zf.writestr("group_audit.csv", group_csv)
        zf.writestr("summary.md", summary_md)
        zf.writestr("metadata.json", json.dumps(metadata, indent=2).encode("utf-8"))
        for name, data in plot_bytes.items():
            zf.writestr(f"plots/{name}", data)
    return bundle.getvalue()
