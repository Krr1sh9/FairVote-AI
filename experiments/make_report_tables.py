"""Build compact Markdown tables from experiment summaries.

The script reads existing outputs and formats them for reporting. It does not
rerun experiments, change metrics, or alter the final evidence CSV/JSON files.
"""

# experiments/make_report_tables.py
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import csv


# ----------------------------
# IO helpers
# ----------------------------

def _read_csv(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]


def _as_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _as_int(x: str) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def _fmt(x: float, decimals: int = 4) -> str:
    if x is None or isinstance(x, str):
        return str(x)
    if math.isnan(x):
        return "nan"
    return f"{x:.{decimals}f}"


def _group(rows: List[dict], keys: List[str]) -> Dict[Tuple[str, ...], List[dict]]:
    out: Dict[Tuple[str, ...], List[dict]] = {}
    for r in rows:
        k = tuple(str(r.get(kk, "")) for kk in keys)
        out.setdefault(k, []).append(r)
    return out


def _sort_eps(vals: List[str]) -> List[str]:
    return [x for x in sorted(vals, key=lambda v: float(v))]


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


# ----------------------------
# Table builders
# ----------------------------

def _scenario_method_table(
    rows: List[dict],
    scenario: str,
    metric: str,
    *,
    include_std: bool = True,
) -> str:
    rows_s = [r for r in rows if r.get("scenario") == scenario and _as_int(r.get("n_rows", "0")) > 0]
    if not rows_s:
        return f"### {scenario}\n\n_No rows found for this scenario._\n"

    eps_list = _sort_eps(sorted({r["epsilon"] for r in rows_s}))
    methods = sorted({r["method"] for r in rows_s})

    mean_key = f"mean_{metric}"
    std_key = f"std_{metric}"

    headers = ["method"] + [f"eps={e}" for e in eps_list]
    table_rows: List[List[str]] = []

    for m in methods:
        row = [m]
        for e in eps_list:
            hit = [r for r in rows_s if r["method"] == m and r["epsilon"] == e]
            if not hit:
                row.append("—")
                continue
            r0 = hit[0]
            mean_v = _as_float(r0.get(mean_key, "nan"))
            if include_std:
                std_v = _as_float(r0.get(std_key, "nan"))
                row.append(f"{_fmt(mean_v)} ± {_fmt(std_v)}")
            else:
                row.append(_fmt(mean_v))
        table_rows.append(row)

    return f"### {scenario}\n\n" + _md_table(headers, table_rows) + "\n"


def _pick_best(rows: List[dict], mean_metric_key: str) -> dict:
    # Pick row with minimal mean metric; tie-breaker: smallest epsilon
    best = None
    for r in rows:
        v = _as_float(r.get(mean_metric_key, "nan"))
        if math.isnan(v):
            continue
        if best is None:
            best = r
            continue
        vb = _as_float(best.get(mean_metric_key, "nan"))
        if v < vb - 1e-12:
            best = r
        elif abs(v - vb) <= 1e-12:
            if float(r["epsilon"]) < float(best["epsilon"]):
                best = r
    return best or (rows[0] if rows else {})


def _best_epsilon_table(rows: List[dict], metric: str) -> str:
    """
    For each scenario+method, pick epsilon that minimises mean_{metric}.
    """
    valid = [r for r in rows if _as_int(r.get("n_rows", "0")) > 0]
    if not valid:
        return "_No valid rows found (n_rows == 0 everywhere)._"

    grouped = _group(valid, ["scenario", "method"])
    mean_key = f"mean_{metric}"
    std_key = f"std_{metric}"

    headers = ["scenario", "method", "best_epsilon", mean_key, std_key, "n_rows", "mean_n_effective"]
    table_rows: List[List[str]] = []

    for (scenario, method), rs in sorted(grouped.items()):
        best = _pick_best(rs, mean_key)
        table_rows.append([
            scenario,
            method,
            str(best.get("epsilon", "")),
            _fmt(_as_float(best.get(mean_key, "nan"))),
            _fmt(_as_float(best.get(std_key, "nan"))),
            str(_as_int(best.get("n_rows", "0"))),
            _fmt(_as_float(best.get("mean_n_effective", "nan")), decimals=1),
        ])

    return _md_table(headers, table_rows) + "\n"


def _multi_best_table(rows: List[dict], metrics: List[str]) -> str:
    """
    Create a wide table: for each scenario+method, show best epsilon for each metric.

    This is handy for the report narrative:
    - epsilon* (utility)
    - epsilon* (fairness major worst)
    - epsilon* (weighted fairness)
    - epsilon* (p90 fairness)
    """
    valid = [r for r in rows if _as_int(r.get("n_rows", "0")) > 0]
    if not valid:
        return "_No valid rows found (n_rows == 0 everywhere)._"

    grouped = _group(valid, ["scenario", "method"])

    headers = ["scenario", "method"]
    for m in metrics:
        headers += [f"best_eps({m})", f"best_mean_{m}"]

    table_rows: List[List[str]] = []
    for (scenario, method), rs in sorted(grouped.items()):
        row = [scenario, method]
        for metric in metrics:
            best = _pick_best(rs, f"mean_{metric}")
            row.append(str(best.get("epsilon", "")))
            row.append(_fmt(_as_float(best.get(f"mean_{metric}", "nan"))))
        table_rows.append(row)

    return _md_table(headers, table_rows) + "\n"


# ----------------------------
# Main
# ----------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Turn mrp_vs_baselines summary.csv into report-ready markdown tables."
    )
    ap.add_argument(
        "--summary_csv",
        type=str,
        required=True,
        help="Path to summary.csv generated by experiments/mrp_vs_baselines.py",
    )
    ap.add_argument(
        "--out_md",
        type=str,
        default="experiments/outputs/report_tables.md",
        help="Output markdown file path.",
    )
    ap.add_argument(
        "--metric",
        type=str,
        default="overall_l1",
        choices=[
            # Utility
            "overall_l1",
            "overall_mae",
            # Legacy subgroup
            "worst_region_l1",
            "worst_age_l1",
            # Mass-aware / robust subgroup fairness (recommended)
            "worst_region_l1_major",
            "worst_age_l1_major",
            "weighted_region_l1",
            "weighted_age_l1",
            "p90_region_l1_major",
            "p90_age_l1_major",
        ],
        help="Which metric to tabulate for the per-scenario grid tables.",
    )
    ap.add_argument(
        "--no_std",
        action="store_true",
        help="If set, tables show only mean (no ± std).",
    )
    ap.add_argument(
        "--include_all",
        action="store_true",
        help="If set, also include extra sections for fairness/robust metrics (recommended).",
    )

    args = ap.parse_args()

    summary_path = Path(args.summary_csv)
    rows = _read_csv(summary_path)

    scenarios = sorted({r.get("scenario", "") for r in rows if r.get("scenario")})
    if not scenarios:
        raise SystemExit("No scenarios found in summary.csv (unexpected).")

    metric = args.metric
    include_std = not args.no_std

    parts: List[str] = []
    parts.append(f"# Report Tables — primary metric: {metric}\n")
    parts.append(f"Source: `{summary_path.as_posix()}`\n")

    # Best epsilon for the selected metric
    parts.append(f"## Best epsilon per scenario/method for `{metric}`\n")
    parts.append(_best_epsilon_table(rows, metric))

    # If requested, also show a "multi-best" overview across key metrics
    if args.include_all:
        key_metrics = [
            "overall_l1",
            "worst_region_l1_major",
            "weighted_region_l1",
            "p90_region_l1_major",
            "worst_age_l1_major",
            "weighted_age_l1",
            "p90_age_l1_major",
        ]
        parts.append("## Best epsilons across key metrics (utility + fairness)\n")
        parts.append(_multi_best_table(rows, key_metrics))

    # Full tables for selected metric
    parts.append(f"## Full tables for `{metric}`\n")
    for s in scenarios:
        parts.append(_scenario_method_table(rows, s, metric, include_std=include_std))

    # Optional extra sections: fairness metrics as separate grids
    if args.include_all:
        extra = [
            "worst_region_l1_major",
            "weighted_region_l1",
            "p90_region_l1_major",
            "worst_age_l1_major",
            "weighted_age_l1",
            "p90_age_l1_major",
        ]
        for m in extra:
            parts.append(f"## Full tables for `{m}`\n")
            for s in scenarios:
                parts.append(_scenario_method_table(rows, s, m, include_std=include_std))

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_text(out_path, "\n".join(parts))

    print(f"Wrote markdown tables to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
