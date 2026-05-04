# experiments/build_results_bundle.py
"""
Build a clean "results bundle" folder for a given mrp_vs_baselines run directory.

Copies the key artefacts into:
  <run_dir>/BUNDLE/

Includes (if present):
  - summary.csv
  - results_trials.csv
  - learned_honesty_summary.csv/.md
  - table_*.md
  - pareto_*.csv
  - plots/*.png, plots/*.pdf, plots/*.md
  - config.json (or any config*.json)
  - README_BUNDLE.md (generated)

Usage:
  python -m experiments.build_results_bundle --run_dir experiments/outputs/2026-01-26_203739_mrp_vs_baselines
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _copy_glob(src_dir: Path, pattern: str, dst_dir: Path) -> List[Path]:
    copied = []
    for p in sorted(src_dir.glob(pattern)):
        if p.is_file():
            out = dst_dir / p.name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
            copied.append(out)
    return copied


def _copy_tree_filtered(src_dir: Path, dst_dir: Path, exts: List[str]) -> List[Path]:
    copied = []
    if not src_dir.exists():
        return copied
    for p in sorted(src_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            rel = p.relative_to(src_dir)
            out = dst_dir / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)
            copied.append(out)
    return copied


def main() -> int:
    """Collect selected existing outputs into a shareable results bundle."""
    ap = argparse.ArgumentParser(description="Build a clean bundle folder containing the key result artefacts.")
    ap.add_argument("--run_dir", required=True, type=str, help="Experiment run directory (experiments/outputs/..._mrp_vs_baselines)")
    ap.add_argument("--bundle_name", default="BUNDLE", type=str, help="Name of bundle folder inside run_dir.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    bundle_dir = run_dir / args.bundle_name
    # Clean existing bundle to avoid stale files
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    copied = []

    # Core CSVs
    for name in ("summary.csv", "results_trials.csv"):
        if _copy_if_exists(run_dir / name, bundle_dir / name):
            copied.append(name)

    # Learned honesty artefacts
    for name in ("learned_honesty_summary.csv", "learned_honesty_summary.md"):
        if _copy_if_exists(run_dir / name, bundle_dir / name):
            copied.append(name)

    # Report tables (markdown)
    copied_tables = _copy_glob(run_dir, "table_*.md", bundle_dir)
    if copied_tables:
        copied.extend([p.name for p in copied_tables])

    # Pareto CSVs
    copied_pareto = _copy_glob(run_dir, "pareto_*.csv", bundle_dir)
    if copied_pareto:
        copied.extend([p.name for p in copied_pareto])

    # Config(s)
    copied_cfg = []
    copied_cfg += _copy_glob(run_dir, "config*.json", bundle_dir)
    copied_cfg += _copy_glob(run_dir, "*config*.json", bundle_dir)
    if copied_cfg:
        copied.extend([p.name for p in copied_cfg])

    # Plots folder (common extensions)
    plots_src = run_dir / "plots"
    plots_dst = bundle_dir / "plots"
    copied_plots = _copy_tree_filtered(plots_src, plots_dst, exts=[".png", ".pdf", ".md"])
    if copied_plots:
        copied.append("plots/ (png/pdf/md)")

    # Build a small README for the bundle
    readme = bundle_dir / "README_BUNDLE.md"
    lines = []
    lines.append("# Results bundle\n")
    lines.append(f"- Source run directory: `{run_dir.as_posix()}`\n")
    lines.append(f"- Bundle directory: `{bundle_dir.as_posix()}`\n\n")
    lines.append("## Contents\n\n")
    if copied:
        for item in copied:
            lines.append(f"- {item}\n")
    else:
        lines.append("- (No files copied — check that the run produced outputs.)\n")
    lines.append("\n## How to reproduce\n\n")
    lines.append("1) Run the experiment:\n")
    lines.append("   - `python -m experiments.mrp_vs_baselines ...`\n\n")
    lines.append("2) Build tables:\n")
    lines.append("   - `python -m experiments.make_report_tables --summary_csv <run_dir>/summary.csv --include_all ...`\n\n")
    lines.append("3) Learned honesty:\n")
    lines.append("   - `python -m experiments.summarise_learned_honesty --run_dir <run_dir>`\n\n")
    lines.append("4) Recommendation / Pareto:\n")
    lines.append("   - `python -m experiments.recommend_from_summary --summary_csv <run_dir>/summary.csv --write_pareto`\n")

    readme.write_text("".join(lines), encoding="utf-8")

    print("Built bundle:")
    print(f"- {bundle_dir}")
    print("Copied:")
    if copied:
        for item in copied:
            print(f"  - {item}")
    else:
        print("  - (none)")
    print(f"Wrote: {readme}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
