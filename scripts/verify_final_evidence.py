"""Fail-fast verifier for a final FairVote-AI evidence run.

Use after running `make final-evidence` or `make examiner-reproduce`:
    python -m scripts.verify_final_evidence --run_dir evidence/final/<RUN>

The verifier deliberately refuses smoke evidence, one-trial evidence, non-empty
failures, absent provenance, and unsupported neural claims.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REQUIRED_FILES = [
    "raw_trials.csv",
    "summary_with_ci.csv",
    "paired_comparisons.csv",
    "ablations.csv",
    "runtime_profile.csv",
    "failures.csv",
    "config.json",
    "manifest.json",
    "environment.json",
    "sha256sums.txt",
    "README.md",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def as_int(value: object) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except Exception:
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, help="Timestamped *_mrp_vs_baselines evidence directory")
    parser.add_argument("--min_trials", type=int, default=20, help="Minimum repeated rows per summary cell")
    parser.add_argument("--require_paired", action="store_true", help="Require non-empty paired_comparisons.csv")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    errors: list[str] = []
    for name in REQUIRED_FILES:
        if not (run_dir / name).exists():
            errors.append(f"Missing required evidence file: {name}")

    readme = (run_dir / "README.md").read_text(encoding="utf-8") if (run_dir / "README.md").exists() else ""
    lowered = readme.lower()
    if "smoke" in lowered or "sanity" in lowered or "do not use" in lowered or "non-final" in lowered:
        errors.append("README marks the run as smoke/sanity/non-final.")

    failures = read_csv(run_dir / "failures.csv")
    if failures:
        errors.append(f"failures.csv contains {len(failures)} failure row(s).")

    summary = read_csv(run_dir / "summary_with_ci.csv")
    if not summary:
        errors.append("summary_with_ci.csv is empty or unreadable.")
    else:
        trial_counts = [as_int(r.get("n_rows", r.get("trials", "0"))) for r in summary]
        min_trials = min(trial_counts) if trial_counts else 0
        if min_trials < args.min_trials:
            errors.append(
                f"Minimum repeated trials per summary cell is {min_trials}, below required {args.min_trials}."
            )
        max_skipped = max((as_int(r.get("n_skipped", "0")) for r in summary), default=0)
        if max_skipped:
            errors.append(f"summary_with_ci.csv reports skipped rows; max n_skipped={max_skipped}.")

    paired = read_csv(run_dir / "paired_comparisons.csv")
    raw = read_csv(run_dir / "raw_trials.csv")
    methods = {r.get("method", "") for r in raw}
    neural_present = any("neural" in m for m in methods)
    if (args.require_paired or neural_present) and not paired:
        errors.append("paired_comparisons.csv is empty while neural methods are present or required.")

    for name in ("manifest.json", "environment.json", "config.json"):
        p = run_dir / name
        if p.exists():
            try:
                json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                errors.append(f"{name} is not valid JSON: {exc}")

    if not (run_dir / "plots").exists():
        errors.append("plots/ directory is missing; final evidence should include figures.")

    if errors:
        raise SystemExit("Final evidence verification failed:\n- " + "\n- ".join(errors))

    print(f"OK: {run_dir} passes final-evidence structural checks.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
