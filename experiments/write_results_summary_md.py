# experiments/write_results_summary_md.py
"""
Create a report-ready RESULTS_SUMMARY.md from an mrp_vs_baselines run folder.

Reads:
  - <run_dir>/summary.csv            (required)
  - <run_dir>/learned_honesty_summary.csv (optional)
  - <run_dir>/pareto_*.csv           (optional)

Writes:
  - <run_dir>/RESULTS_SUMMARY.md
  - also copies into <run_dir>/BUNDLE/ if it exists

Usage:
  python -m experiments.write_results_summary_md --run_dir experiments/outputs/2026-01-26_203739_mrp_vs_baselines

Optional:
  python -m experiments.write_results_summary_md --run_dir ... --metric overall_l1 --eps_max 1.0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _try_float(x: object) -> float:
    try:
        return float(x)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def _fmt(x: float, ndp: int = 4) -> str:
    if x != x:  # nan
        return "nan"
    return f"{x:.{ndp}f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out) + "\n"


def _best_rows(
    summary: list[dict],
    *,
    metric: str,
    eps_max: float | None,
) -> list[dict]:
    """
    Select one "best" row per scenario.
    Preference: eps <= eps_max (if provided), then minimal metric.
    """
    by_scen: dict[str, list[dict]] = {}
    for r in summary:
        scen = r.get("scenario", "")
        if not scen:
            continue
        by_scen.setdefault(scen, []).append(r)

    best = []
    for _scen, rows in sorted(by_scen.items(), key=lambda x: x[0]):
        # filter by eps constraint if provided
        if eps_max is not None:
            rows2 = [r for r in rows if _try_float(r.get("epsilon")) <= eps_max]
            if rows2:
                rows = rows2

        # choose minimal metric, tie-break by smaller epsilon
        def keyfun(r: dict) -> tuple[float, float, str]:
            return (_try_float(r.get(metric)), _try_float(r.get("epsilon")), str(r.get("method", "")))

        rows_sorted = sorted(rows, key=keyfun)
        best.append(rows_sorted[0])
    return best


def main() -> int:
    """Write a Markdown summary from existing experiment CSV outputs."""
    ap = argparse.ArgumentParser(description="Write a report-ready RESULTS_SUMMARY.md for a run folder.")
    ap.add_argument("--run_dir", required=True, type=str, help="Run directory containing summary.csv")
    ap.add_argument(
        "--metric", default="overall_l1", type=str, help="Metric to optimise when selecting best per scenario."
    )
    ap.add_argument(
        "--eps_max", default=None, type=float, help="If provided, restrict best-choice selection to epsilon <= eps_max."
    )
    ap.add_argument("--ndp", default=4, type=int, help="Decimal places in tables.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    summary_csv = run_dir / "summary.csv"
    if not summary_csv.exists():
        raise SystemExit(f"Missing required file: {summary_csv}")

    summary = _read_csv(summary_csv)

    learned_csv = run_dir / "learned_honesty_summary.csv"
    learned = _read_csv(learned_csv) if learned_csv.exists() and learned_csv.stat().st_size > 0 else []

    pareto_files = sorted(run_dir.glob("pareto_*.csv"))

    best = _best_rows(summary, metric=args.metric, eps_max=args.eps_max)

    # Map learned honesty mean by (scenario, epsilon) if available
    learned_map: dict[tuple[str, float], dict] = {}
    for r in learned:
        scen = r.get("scenario", "")
        eps = _try_float(r.get("epsilon"))
        learned_map[(scen, eps)] = r

    md = []
    md.append("# Results summary\n\n")
    md.append(f"- Run directory: `{run_dir.as_posix()}`\n")
    md.append(f"- Summary source: `{summary_csv.as_posix()}`\n")
    md.append(f"- Best-selection metric: `{args.metric}`\n")
    if args.eps_max is not None:
        md.append(f"- Epsilon constraint used for best-selection: eps <= {args.eps_max}\n")
    md.append("\n")

    # Best per scenario table
    headers = [
        "Scenario",
        "Method",
        "Epsilon",
        "overall_l1",
        "overall_mae",
        "worst_region_l1_major",
        "worst_age_l1_major",
        "p90_region_l1_major",
        "p90_age_l1_major",
        "learned_honesty_mean",
    ]
    rows = []
    for r in best:
        scen = str(r.get("scenario", ""))
        method = str(r.get("method", ""))
        eps = _try_float(r.get("epsilon"))
        key = (scen, eps)
        hmean = "—"
        if key in learned_map:
            hmean = _fmt(_try_float(learned_map[key].get("mean_learned_honesty")), args.ndp)

        rows.append(
            [
                scen,
                method,
                _fmt(eps, 3),
                _fmt(_try_float(r.get("overall_l1")), args.ndp),
                _fmt(_try_float(r.get("overall_mae")), args.ndp),
                _fmt(_try_float(r.get("worst_region_l1_major")), args.ndp),
                _fmt(_try_float(r.get("worst_age_l1_major")), args.ndp),
                _fmt(_try_float(r.get("p90_region_l1_major")), args.ndp),
                _fmt(_try_float(r.get("p90_age_l1_major")), args.ndp),
                hmean,
            ]
        )

    md.append("## Best method per scenario (by chosen metric)\n\n")
    md.append(_md_table(headers, rows))
    md.append("\n")

    # Learned honesty section
    if learned:
        md.append("## Learned honesty (shy parameter)\n\n")
        md.append(
            "This shows the estimated honesty parameter h (higher means more truthful reporting for the shy category).\n\n"
        )
        h_headers = ["Scenario", "Epsilon", "Trials", "Mean h", "Std", "CI low", "CI high", "Median"]
        h_rows = []
        # Try to detect CI column names in the CSV
        ci_low_key = next((k for k in learned[0] if k.endswith("_low")), None)
        ci_high_key = next((k for k in learned[0] if k.endswith("_high")), None)

        for r in sorted(learned, key=lambda x: (x.get("scenario", ""), _try_float(x.get("epsilon")))):
            h_rows.append(
                [
                    str(r.get("scenario", "")),
                    _fmt(_try_float(r.get("epsilon")), 3),
                    str(r.get("n_trials", "")),
                    _fmt(_try_float(r.get("mean_learned_honesty")), args.ndp),
                    _fmt(_try_float(r.get("std_learned_honesty")), args.ndp),
                    _fmt(_try_float(r.get(ci_low_key)) if ci_low_key else float("nan"), args.ndp),
                    _fmt(_try_float(r.get(ci_high_key)) if ci_high_key else float("nan"), args.ndp),
                    _fmt(_try_float(r.get("median_learned_honesty")), args.ndp),
                ]
            )
        md.append(_md_table(h_headers, h_rows))
        md.append("\n")
    else:
        md.append("## Learned honesty (shy parameter)\n\n")
        md.append(
            "No learned honesty summary found. If you generated it, ensure `learned_honesty_summary.csv` exists in the run directory.\n\n"
        )

    # Pareto section
    if pareto_files:
        md.append("## Pareto front files\n\n")
        md.append("These CSVs contain the trade-off surfaces for privacy (epsilon) vs utility/fairness metrics.\n\n")
        for p in pareto_files:
            md.append(f"- `{p.name}`\n")
        md.append("\n")

    # Next steps for report inclusion
    md.append("## What to paste into your report\n\n")
    md.append("- The **Best method per scenario** table above (copy/paste).\n")
    md.append(
        "- The **Learned honesty** table (evidence about the fitted honesty parameter estimated from privatized reports).\n"
    )
    md.append(
        "- A short paragraph interpreting the privacy–utility–fairness trade-off (reference pareto CSVs and any plots).\n\n"
    )

    out_md = run_dir / "RESULTS_SUMMARY.md"
    out_md.write_text("".join(md), encoding="utf-8")

    # Copy into bundle if present
    bundle_dir = run_dir / "BUNDLE"
    if bundle_dir.exists() and bundle_dir.is_dir():
        (bundle_dir / "RESULTS_SUMMARY.md").write_text("".join(md), encoding="utf-8")

    print(f"Wrote: {out_md}")
    if bundle_dir.exists() and bundle_dir.is_dir():
        print(f"Copied into: {bundle_dir / 'RESULTS_SUMMARY.md'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
