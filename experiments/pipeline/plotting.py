"""Optional plot generation for experiment summaries."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List


def plot_summary(run_dir: Path, summary_rows: List[Dict]) -> None:
    """Write standard summary plots when matplotlib is installed."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return
    if not summary_rows:
        return
    scenarios = sorted({r["scenario"] for r in summary_rows if int(r.get("n_rows", 0) or 0) > 0})
    methods = sorted({r["method"] for r in summary_rows if int(r.get("n_rows", 0) or 0) > 0})
    for scenario in scenarios:
        rows_s = [r for r in summary_rows if r["scenario"] == scenario and int(r.get("n_rows", 0) or 0) > 0]
        eps = sorted({float(r["epsilon"]) for r in rows_s})
        _line_plot(
            plt,
            run_dir,
            rows_s,
            methods,
            eps,
            key="mean_overall_l1",
            ylabel="Mean overall L1 error vs population truth",
            title=f"MRP vs Baselines — Overall Error ({scenario})",
            filename=f"{scenario}_overall_l1.png",
        )
        _line_plot(
            plt,
            run_dir,
            rows_s,
            methods,
            eps,
            key="mean_worst_region_l1_major",
            ylabel="Mean worst-region (major) L1 error vs population truth",
            title=f"MRP vs Baselines — Worst Region (Major) ({scenario})",
            filename=f"{scenario}_worst_region_l1_major.png",
        )


def _line_plot(plt, run_dir: Path, rows_s: List[Dict], methods: List[str], eps: List[float], *, key: str, ylabel: str, title: str, filename: str) -> None:
    plt.figure()
    for method in methods:
        xs = []
        ys = []
        for epsilon in eps:
            hit = [r for r in rows_s if float(r["epsilon"]) == epsilon and r["method"] == method]
            if not hit:
                continue
            xs.append(epsilon)
            ys.append(float(hit[0].get(key, "nan")))
        if xs:
            plt.plot(xs, ys, marker="o", label=method)
    plt.xscale("log")
    plt.xlabel("epsilon (log scale)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "plots" / filename, dpi=200)
    plt.close()
