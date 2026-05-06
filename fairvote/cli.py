# fairvote/cli.py
"""
FairVote-AI CLI

This CLI wraps your experiments + reporting scripts into a single, consistent entrypoint.

Typical usage (from project root):
  python -m fairvote.cli run mrp-vs-baselines --trials 10 --eps 0.2,0.5,1.0,2.0 --major_mass 0.02
  python -m fairvote.cli run mrp-vs-baselines --n 1000 --scenario no_bias --trials 1
  python -m fairvote.cli report tables --run_dir experiments/outputs/2026-01-26_203739_mrp_vs_baselines --metric overall_l1
  python -m fairvote.cli report honesty --run_dir experiments/outputs/2026-01-26_203739_mrp_vs_baselines
  python -m fairvote.cli report recommend --run_dir experiments/outputs/2026-01-26_203739_mrp_vs_baselines --write_pareto
  python -m fairvote.cli report bundle --run_dir experiments/outputs/2026-01-26_203739_mrp_vs_baselines
  python -m fairvote.cli report summary --run_dir experiments/outputs/2026-01-26_203739_mrp_vs_baselines

Notes:
- This file intentionally calls your existing scripts via `python -m experiments.<module> ...`
- So it stays robust even if internal functions change.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _project_root() -> Path:
    """Locate the repository root from the installed package path."""
    # Heuristic: assume this file lives at <root>/fairvote/cli.py
    return Path(__file__).resolve().parents[1]


def _run_python_module(
    module: str,
    argv: List[str],
    *,
    dry_run: bool = False,
    cwd: Optional[Path] = None,
) -> int:
    cmd = [sys.executable, "-m", module, *argv]
    printable = " ".join(cmd)

    print(f"\n[fairvote] Running:\n  {printable}\n")

    if dry_run:
        return 0

    run_cwd = str(cwd) if cwd is not None else None
    completed = subprocess.run(cmd, cwd=run_cwd, check=False)
    return int(completed.returncode)


def _ensure_exists(path: Path, what: str) -> None:
    if not path.exists():
        raise SystemExit(f"{what} not found: {path}")


def _validate_positive_int(value: int, name: str, minimum: int = 1) -> int:
    """Validate that a value is a positive integer >= minimum."""
    if not isinstance(value, int) or value < minimum:
        raise SystemExit(
            f"--{name} must be an integer >= {minimum}, got: {value}"
        )
    return value


def _validate_positive_float(value: float, name: str, minimum: float = 0.0) -> float:
    """Validate that a value is a positive float >= minimum."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        raise SystemExit(f"--{name} must be a float, got: {value}")
    
    if val < minimum:
        raise SystemExit(
            f"--{name} must be >= {minimum}, got: {val}"
        )
    return val


def _validate_probability(value: float, name: str) -> float:
    """Validate that a value is a valid probability (0.0 to 1.0)."""
    try:
        val = float(value)
    except (TypeError, ValueError):
        raise SystemExit(f"--{name} must be a float, got: {value}")
    
    if val < 0.0 or val > 1.0:
        raise SystemExit(
            f"--{name} must be in [0.0, 1.0], got: {val}"
        )
    return val


def _validate_epsilon_list(eps_str: str) -> list:
    """Parse and validate comma-separated epsilon values."""
    try:
        epsilons = [float(e.strip()) for e in eps_str.split(",")]
    except ValueError:
        raise SystemExit(
            f"--eps must be comma-separated floats, got: {eps_str}"
        )
    
    if not epsilons:
        raise SystemExit("--eps must specify at least one epsilon value")
    
    for eps in epsilons:
        if eps <= 0.0:
            raise SystemExit(
                f"All epsilon values must be > 0, got: {eps}"
            )
    
    return epsilons


def _summary_csv_from_run_dir(run_dir: Path) -> Path:
    p = run_dir / "summary.csv"
    _ensure_exists(p, "summary.csv")
    return p


def _results_trials_csv_from_run_dir(run_dir: Path) -> Path:
    p = run_dir / "results_trials.csv"
    _ensure_exists(p, "results_trials.csv")
    return p


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="fairvote", description="FairVote-AI CLI")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    p.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory to run commands from (defaults to project root).",
    )

    sub = p.add_subparsers(dest="cmd", required=True)

    # ----------------------------
    # run
    # ----------------------------
    run = sub.add_parser("run", help="Run experiments.")
    run_sub = run.add_subparsers(dest="run_cmd", required=True)

    mrp = run_sub.add_parser("mrp-vs-baselines", help="Run MRP vs baselines experiment.")
    mrp.add_argument("--trials", type=int, default=10)
    mrp.add_argument("--eps", type=str, default="0.2,0.5,1.0,2.0", help="Comma-separated eps list.")
    mrp.add_argument("--major_mass", type=float, default=0.02)
    # passthrough extras (optional)
    mrp.add_argument("--k", type=int, default=None)
    mrp.add_argument(
        "--n",
        "--n_sample",
        dest="n_sample",
        type=int,
        default=None,
        help="Sample size for experiments.mrp_vs_baselines; forwarded as --n_sample.",
    )
    mrp.add_argument("--seed", type=int, default=None)
    mrp.add_argument(
        "--scenario",
        "--scenarios",
        dest="scenarios",
        type=str,
        default=None,
        help="Scenario or comma-separated scenarios; forwarded as --scenarios.",
    )
    mrp.add_argument("--sampling", type=str, default=None, help="If supported: choose sampling scheme.")
    mrp.add_argument("--shy_category", type=int, default=None)

    # ----------------------------
    # report
    # ----------------------------
    report = sub.add_parser("report", help="Generate report artefacts from a run folder.")
    rep_sub = report.add_subparsers(dest="rep_cmd", required=True)

    tables = rep_sub.add_parser("tables", help="Generate markdown table(s) from summary.csv.")
    tables.add_argument("--run_dir", type=str, required=True)
    tables.add_argument("--metric", type=str, default="overall_l1")
    tables.add_argument("--include_all", action="store_true", default=True)
    tables.add_argument("--out_md", type=str, default=None, help="Output markdown path. Default: <run_dir>/table_<metric>.md")
    tables.add_argument("--no_std", action="store_true")
    tables.add_argument("--summary_csv", type=str, default=None, help="Optional override path to summary.csv")

    honesty = rep_sub.add_parser("honesty", help="Summarise learned honesty from results_trials.csv.")
    honesty.add_argument("--run_dir", type=str, required=True)
    honesty.add_argument("--method", type=str, default="mrp_learned_misreport_rr_poststrat")
    honesty.add_argument("--n_boot", type=int, default=2000)
    honesty.add_argument("--alpha", type=float, default=0.05, help="Alpha (0.05 => 95%% CI).")
    honesty.add_argument("--seed", type=int, default=123)
    honesty.add_argument("--ndp", type=int, default=4)

    recommend = rep_sub.add_parser("recommend", help="Pick recommended methods from summary.csv; optionally write Pareto CSVs.")
    recommend.add_argument("--run_dir", type=str, required=True)
    recommend.add_argument("--summary_csv", type=str, default=None, help="Optional override path to summary.csv")
    recommend.add_argument("--write_pareto", action="store_true")
    recommend.add_argument("--epsilon_max", type=float, default=None)
    recommend.add_argument("--worst_region_l1_major_max", type=float, default=None)
    recommend.add_argument("--worst_age_l1_major_max", type=float, default=None)
    recommend.add_argument("--overall_l1_max", type=float, default=None)
    recommend.add_argument("--overall_mae_max", type=float, default=None)

    bundle = rep_sub.add_parser("bundle", help="Build a clean BUNDLE/ folder from a run directory.")
    bundle.add_argument("--run_dir", type=str, required=True)
    bundle.add_argument("--bundle_name", type=str, default="BUNDLE")

    summary = rep_sub.add_parser("summary", help="Generate RESULTS_SUMMARY.md from a run directory.")
    summary.add_argument("--run_dir", type=str, required=True)
    summary.add_argument("--metric", type=str, default="overall_l1")
    summary.add_argument("--eps_max", type=float, default=None)
    summary.add_argument("--ndp", type=int, default=4)

    return p.parse_args()


def main() -> int:
    args = _parse_args()

    cwd = Path(args.cwd).resolve() if args.cwd else _project_root()
    dry = bool(args.dry_run)

    # Ensure we run from project root so "experiments" is importable as a module package.
    # (You already run `python -m experiments...` from project root.)
    if not cwd.exists():
        raise SystemExit(f"--cwd does not exist: {cwd}")

    # ----------------------------
    # run mrp-vs-baselines
    # ----------------------------
    if args.cmd == "run" and args.run_cmd == "mrp-vs-baselines":
        # Validate required arguments
        _validate_positive_int(args.trials, "trials", minimum=1)
        _validate_positive_float(args.major_mass, "major_mass", minimum=0.0)
        epsilons = _validate_epsilon_list(args.eps)
        
        # Validate optional arguments if provided
        if args.k is not None:
            _validate_positive_int(args.k, "k", minimum=2)
        if args.n_sample is not None:
            _validate_positive_int(args.n_sample, "n", minimum=1)
        if args.shy_category is not None:
            _validate_positive_int(args.shy_category, "shy_category", minimum=0)
        
        argv: List[str] = [
            "--trials",
            str(args.trials),
            "--eps",
            str(args.eps),
            "--major_mass",
            str(args.major_mass),
        ]
        # Optional passthrough flags if your script supports them:
        if args.k is not None:
            argv += ["--k", str(args.k)]
        if args.n_sample is not None:
            argv += ["--n_sample", str(args.n_sample)]
        if args.seed is not None:
            argv += ["--seed", str(args.seed)]
        if args.scenarios is not None:
            argv += ["--scenarios", str(args.scenarios)]
        if args.sampling is not None:
            argv += ["--sampling", str(args.sampling)]
        if args.shy_category is not None:
            argv += ["--shy_category", str(args.shy_category)]

        return _run_python_module("experiments.mrp_vs_baselines", argv, dry_run=dry, cwd=cwd)

    # ----------------------------
    # report tables
    # ----------------------------
    if args.cmd == "report" and args.rep_cmd == "tables":
        run_dir = Path(args.run_dir)
        summary_csv = Path(args.summary_csv) if args.summary_csv else _summary_csv_from_run_dir(run_dir)
        out_md = Path(args.out_md) if args.out_md else (run_dir / f"table_{args.metric}.md")

        argv = [
            "--summary_csv",
            str(summary_csv),
            "--metric",
            str(args.metric),
            "--out_md",
            str(out_md),
        ]
        if args.no_std:
            argv.append("--no_std")
        if args.include_all:
            argv.append("--include_all")

        return _run_python_module("experiments.make_report_tables", argv, dry_run=dry, cwd=cwd)

    # ----------------------------
    # report honesty
    # ----------------------------
    if args.cmd == "report" and args.rep_cmd == "honesty":
        run_dir = Path(args.run_dir)
        _results_trials_csv_from_run_dir(run_dir)  # ensure exists
        
        # Validate numeric parameters
        _validate_positive_int(args.n_boot, "n_boot", minimum=1)
        _validate_probability(args.alpha, "alpha")
        _validate_positive_int(args.ndp, "ndp", minimum=1)

        argv = [
            "--run_dir",
            str(run_dir),
            "--method",
            str(args.method),
            "--n_boot",
            str(args.n_boot),
            "--alpha",
            str(args.alpha),
            "--seed",
            str(args.seed),
            "--ndp",
            str(args.ndp),
        ]
        return _run_python_module("experiments.summarise_learned_honesty", argv, dry_run=dry, cwd=cwd)

    # ----------------------------
    # report recommend
    # ----------------------------
    if args.cmd == "report" and args.rep_cmd == "recommend":
        run_dir = Path(args.run_dir)
        summary_csv = Path(args.summary_csv) if args.summary_csv else _summary_csv_from_run_dir(run_dir)
        
        # Validate optional numeric parameters if provided
        if args.epsilon_max is not None:
            _validate_positive_float(args.epsilon_max, "epsilon_max")
        if args.worst_region_l1_major_max is not None:
            _validate_positive_float(args.worst_region_l1_major_max, "worst_region_l1_major_max")
        if args.worst_age_l1_major_max is not None:
            _validate_positive_float(args.worst_age_l1_major_max, "worst_age_l1_major_max")
        if args.overall_l1_max is not None:
            _validate_positive_float(args.overall_l1_max, "overall_l1_max")
        if args.overall_mae_max is not None:
            _validate_positive_float(args.overall_mae_max, "overall_mae_max")

        argv = ["--summary_csv", str(summary_csv)]
        if args.write_pareto:
            argv.append("--write_pareto")

        # Forward optional recommendation constraints when the CLI user provides them.
        if args.epsilon_max is not None:
            argv += ["--epsilon_max", str(args.epsilon_max)]
        if args.worst_region_l1_major_max is not None:
            argv += ["--worst_region_l1_major_max", str(args.worst_region_l1_major_max)]
        if args.worst_age_l1_major_max is not None:
            argv += ["--worst_age_l1_major_max", str(args.worst_age_l1_major_max)]
        if args.overall_l1_max is not None:
            argv += ["--overall_l1_max", str(args.overall_l1_max)]
        if args.overall_mae_max is not None:
            argv += ["--overall_mae_max", str(args.overall_mae_max)]

        return _run_python_module("experiments.recommend_from_summary", argv, dry_run=dry, cwd=cwd)

    # ----------------------------
    # report bundle
    # ----------------------------
    if args.cmd == "report" and args.rep_cmd == "bundle":
        run_dir = Path(args.run_dir)
        _ensure_exists(run_dir, "run_dir")
        argv = ["--run_dir", str(run_dir), "--bundle_name", str(args.bundle_name)]
        return _run_python_module("experiments.build_results_bundle", argv, dry_run=dry, cwd=cwd)

    # ----------------------------
    # report summary (RESULTS_SUMMARY.md)
    # ----------------------------
    if args.cmd == "report" and args.rep_cmd == "summary":
        run_dir = Path(args.run_dir)
        _summary_csv_from_run_dir(run_dir)

        argv = ["--run_dir", str(run_dir), "--metric", str(args.metric), "--ndp", str(args.ndp)]
        if args.eps_max is not None:
            argv += ["--eps_max", str(args.eps_max)]
        return _run_python_module("experiments.write_results_summary_md", argv, dry_run=dry, cwd=cwd)

    raise SystemExit("Unknown command (this should never happen).")


if __name__ == "__main__":
    raise SystemExit(main())
