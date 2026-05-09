# experiments/summarise_learned_honesty.py
"""
Summarise the learned "shy honesty" parameter from experiments/mrp_vs_baselines.py runs.

Reads results_trials.csv (per-trial rows) and produces:
  - learned_honesty_summary.csv
  - learned_honesty_summary.md

Typical usage:
  python -m experiments.summarise_learned_honesty --run_dir experiments/outputs/2026-01-26_203739_mrp_vs_baselines

Or:
  python -m experiments.summarise_learned_honesty --results_csv experiments/outputs/.../results_trials.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for row in rows for k in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _format_float(x: float, ndp: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "nan"
    return f"{float(x):.{ndp}f}"


def _bootstrap_ci_mean(
    x: np.ndarray, *, n_boot: int = 2000, alpha: float = 0.05, rng: np.random.Generator
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for the mean.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    if x.size == 1:
        m = float(x[0])
        return m, m
    n = x.size
    idx = rng.integers(0, n, size=(n_boot, n))
    means = np.mean(x[idx], axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def _md_table(rows: list[dict], headers: list[tuple[str, str]]) -> str:
    """
    headers: list of (key, title) in desired order
    """
    lines = []
    lines.append("| " + " | ".join(title for _, title in headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k, _ in headers) + " |")
    return "\n".join(lines) + "\n"


def summarise(
    *,
    results_csv: Path,
    out_dir: Path,
    method: str,
    n_boot: int,
    seed: int,
    alpha: float,
    ndp: int,
) -> tuple[Path, Path]:
    rows = _read_csv(results_csv)
    if not rows:
        raise SystemExit(f"No rows found in: {results_csv}")

    if "learned_honesty" not in rows[0]:
        # might still exist in later rows, but we will check all keys
        all_keys = set()
        for r in rows[:50]:
            all_keys.update(r.keys())
        raise SystemExit(
            "Column 'learned_honesty' not found in results_trials.csv. "
            "Make sure you ran the experiment after adding the learned method.\n"
            f"Detected columns (sample): {sorted(all_keys)}"
        )

    # Filter rows for learned method + non-skipped
    filt = []
    for r in rows:
        if r.get("method") != method:
            continue
        if str(r.get("skipped", "0")).strip() == "1":
            continue
        # require scenario + epsilon
        if "scenario" not in r or "epsilon" not in r:
            continue
        filt.append(r)

    if not filt:
        raise SystemExit(
            f"No rows for method='{method}' in {results_csv}. Did the run include the learned misreport method?"
        )

    # Group by (scenario, epsilon)
    groups: dict[tuple[str, float], list[float]] = {}
    for r in filt:
        scen = r["scenario"]
        try:
            eps = float(r["epsilon"])
        except Exception:
            continue
        try:
            h = float(r["learned_honesty"])
        except Exception:
            h = float("nan")
        groups.setdefault((scen, eps), []).append(h)

    rng = np.random.default_rng(seed)

    out_rows: list[dict] = []
    for (scen, eps), hs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        x = np.asarray(hs, dtype=float)
        x = x[np.isfinite(x)]
        n = int(x.size)
        if n == 0:
            mean = std = lo = hi = med = float("nan")
        else:
            mean = float(np.mean(x))
            std = float(np.std(x, ddof=1)) if n > 1 else 0.0
            med = float(np.median(x))
            lo, hi = _bootstrap_ci_mean(x, n_boot=n_boot, alpha=alpha, rng=rng)

        out_rows.append(
            {
                "scenario": scen,
                "epsilon": eps,
                "n_trials": n,
                "mean_learned_honesty": mean,
                "std_learned_honesty": std,
                f"ci{int((1 - alpha) * 100)}_low": lo,
                f"ci{int((1 - alpha) * 100)}_high": hi,
                "median_learned_honesty": med,
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "learned_honesty_summary.csv"
    out_md = out_dir / "learned_honesty_summary.md"

    _write_csv(out_csv, out_rows)

    # Markdown formatting
    md_rows = []
    for r in out_rows:
        md_rows.append(
            {
                "Scenario": r["scenario"],
                "Epsilon": _format_float(r["epsilon"], ndp=3),
                "N": r["n_trials"],
                "Mean h": _format_float(r["mean_learned_honesty"], ndp=ndp),
                "Std": _format_float(r["std_learned_honesty"], ndp=ndp),
                "CI low": _format_float(r[f"ci{int((1 - alpha) * 100)}_low"], ndp=ndp),
                "CI high": _format_float(r[f"ci{int((1 - alpha) * 100)}_high"], ndp=ndp),
                "Median": _format_float(r["median_learned_honesty"], ndp=ndp),
            }
        )

    md = []
    md.append("# Learned honesty summary\n")
    md.append(f"- Method: `{method}`\n")
    md.append(f"- Source: `{results_csv.as_posix()}`\n")
    md.append(f"- Bootstrap: n_boot={n_boot}, CI={(1 - alpha):.0%}, seed={seed}\n\n")
    md.append(
        _md_table(
            md_rows,
            headers=[
                ("Scenario", "Scenario"),
                ("Epsilon", "Epsilon"),
                ("N", "Trials"),
                ("Mean h", "Mean h"),
                ("Std", "Std"),
                ("CI low", "CI low"),
                ("CI high", "CI high"),
                ("Median", "Median"),
            ],
        )
    )
    out_md.write_text("".join(md), encoding="utf-8")

    return out_csv, out_md


def main() -> int:
    p = argparse.ArgumentParser(description="Summarise learned honesty (shy misreport parameter) from experiment run.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--run_dir",
        type=str,
        help="Run directory containing results_trials.csv (e.g., experiments/outputs/..._mrp_vs_baselines)",
    )
    g.add_argument("--results_csv", type=str, help="Path to results_trials.csv")

    p.add_argument("--method", type=str, default="mrp_learned_misreport_rr_poststrat", help="Method name to filter on.")
    p.add_argument("--n_boot", type=int, default=2000, help="Bootstrap resamples for CI.")
    p.add_argument("--alpha", type=float, default=0.05, help="Alpha for CI (0.05 gives 95%% CI).")
    p.add_argument("--seed", type=int, default=123, help="Seed for bootstrap sampling.")
    p.add_argument("--ndp", type=int, default=4, help="Decimal places in markdown table.")

    args = p.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
        results_csv = run_dir / "results_trials.csv"
        out_dir = run_dir
    else:
        results_csv = Path(args.results_csv)
        out_dir = results_csv.parent

    if not results_csv.exists():
        raise SystemExit(f"File not found: {results_csv}")

    out_csv, out_md = summarise(
        results_csv=results_csv,
        out_dir=out_dir,
        method=args.method,
        n_boot=args.n_boot,
        seed=args.seed,
        alpha=args.alpha,
        ndp=args.ndp,
    )

    print("Wrote:")
    print(f"- {out_csv}")
    print(f"- {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
