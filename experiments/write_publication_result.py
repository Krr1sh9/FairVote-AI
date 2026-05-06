"""Write report-ready Markdown from an existing evidence run.

This script is deliberately conservative. It never reruns experiments and it
never invents results. If the evidence directory is a smoke run, has failures,
has only one trial per cell, or lacks paired comparisons, the generated report
says so explicitly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

METRICS = [
    "overall_l1",
    "worst_group_l1_major",
    "weighted_region_l1",
    "weighted_age_l1",
    "winner_correct",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def as_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return float("nan")


def as_int(value: object) -> int:
    try:
        return int(float(value))  # type: ignore[arg-type]
    except Exception:
        return 0


def fmt(value: float, decimals: int = 4) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.{decimals}f}"


def md_table(headers: list[str], rows: Iterable[Iterable[str]]) -> str:
    rows = [list(r) for r in rows]
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out.extend("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join(out)


def status_checks(
    run_dir: Path, summary: list[dict[str, str]], paired: list[dict[str, str]], failures: list[dict[str, str]]
) -> list[tuple[str, str, str]]:
    readme = (run_dir / "README.md").read_text(encoding="utf-8") if (run_dir / "README.md").exists() else ""
    text = readme.lower()
    min_non_skipped = min((as_int(r.get("n_rows", "0")) for r in summary), default=0)
    min_configured_trials = min((as_int(r.get("trials", "0")) for r in summary), default=0)
    max_skipped = max((as_int(r.get("n_skipped", "0")) for r in summary), default=0)
    labelled_smoke = "smoke" in text or "do not use" in text
    checks = [
        ("Run is not labelled smoke/non-final", "PASS" if not labelled_smoke else "FAIL", "README wording"),
        ("No recorded failures", "PASS" if len(failures) == 0 else "FAIL", f"{len(failures)} failure rows"),
        (
            "At least 20 non-skipped trials per summary cell",
            "PASS" if min_non_skipped >= 20 else "FAIL",
            f"minimum n_rows={min_non_skipped}; configured trials={min_configured_trials}",
        ),
        ("No skipped result cells", "PASS" if max_skipped == 0 else "WARN", f"max n_skipped={max_skipped}"),
        (
            "Neural-vs-linear paired comparisons are available",
            "PASS" if len(paired) > 0 else "WARN",
            f"{len(paired)} paired rows",
        ),
    ]
    return checks


def best_rows(summary: list[dict[str, str]], metric: str = "overall_l1") -> list[list[str]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in summary:
        if as_int(row.get("n_rows", row.get("trials", "0"))) <= 0:
            continue
        groups.setdefault((row.get("scenario", ""), row.get("method", "")), []).append(row)
    table: list[list[str]] = []
    key = f"mean_{metric}"
    for (scenario, method), rows in sorted(groups.items()):
        valid = [r for r in rows if math.isfinite(as_float(r.get(key)))]
        if not valid:
            continue
        best = min(valid, key=lambda r: (as_float(r.get(key)), as_float(r.get("epsilon"))))
        table.append(
            [
                scenario,
                method,
                str(best.get("epsilon", "")),
                str(best.get("sample_size", "")),
                fmt(as_float(best.get(key))),
                fmt(as_float(best.get(f"ci95_low_{metric}"))),
                fmt(as_float(best.get(f"ci95_high_{metric}"))),
                str(best.get("n_rows", best.get("trials", ""))),
            ]
        )
    return table


def claim_rows(summary: list[dict[str, str]], paired: list[dict[str, str]]) -> list[list[str]]:
    scenarios = {r.get("scenario", "") for r in summary}
    methods = {r.get("method", "") for r in summary}
    rows: list[list[str]] = []
    rows.append(
        [
            "RR-aware MRP improves over direct RR debiasing",
            "summary_with_ci.csv / ablations.csv",
            "SUPPORTED TO ANALYSE" if {"baseline_rr_debias", "mrp_rr_poststrat"} <= methods else "MISSING METHOD",
            "Compare mean_overall_l1 and CI by scenario; do not claim improvement unless deltas are negative and stable.",
        ]
    )
    rows.append(
        [
            "Hierarchical partial pooling improves sparse subgroup error",
            "summary_with_ci.csv",
            "SUPPORTED TO ANALYSE"
            if "hierarchical_rr_mrp_poststrat" in methods and any("sparse" in s for s in scenarios)
            else "MISSING EVIDENCE",
            "Use worst_group_l1_major in sparse scenarios; report failures as mixed results.",
        ]
    )
    rows.append(
        [
            "Neural RR-MRP beats simpler baselines in nonlinear settings",
            "paired_comparisons.csv",
            "SUPPORTED TO ANALYSE" if paired else "MISSING PAIRED COMPARISONS",
            "Only claim this if paired deltas have enough trials and confidence intervals exclude zero where relevant.",
        ]
    )
    rows.append(
        [
            "Privacy can help under strategic misreporting",
            "summary_with_ci.csv",
            "SUPPORTED TO ANALYSE" if any("privacy" in s or "shy" in s for s in scenarios) else "MISSING SCENARIO",
            "Look for intermediate-epsilon minima; do not overgeneralise to pure privacy noise scenarios.",
        ]
    )
    return rows


def write_outputs(run_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = read_csv(run_dir / "summary_with_ci.csv") or read_csv(run_dir / "summary.csv")
    paired = read_csv(run_dir / "paired_comparisons.csv")
    failures = read_csv(run_dir / "failures.csv")
    manifest = read_json(run_dir / "manifest.json")
    config = read_json(run_dir / "config.json")

    checks = status_checks(run_dir, summary, paired, failures)
    check_md = md_table(["Check", "Status", "Evidence"], checks)
    best_md = md_table(
        ["scenario", "method", "best_epsilon", "sample_size", "mean_overall_l1", "ci95_low", "ci95_high", "n_rows"],
        best_rows(summary, "overall_l1")[:80],
    )
    claims_md = md_table(
        ["Claim", "Evidence file", "Status", "Required interpretation rule"], claim_rows(summary, paired)
    )

    run_name = run_dir.name
    preset = config.get("preset") or manifest.get("preset") or "unknown"
    command = manifest.get("command") or config.get("command") or "not recorded"
    final_results = f"""# Final evidence interpretation: `{run_name}`

This file is generated from committed evidence files. It does not rerun experiments and it does not infer beyond the CSV/JSON contents.

## Provenance

- Run directory: `{run_dir}`
- Preset: `{preset}`
- Command: `{command}`

## Evidence acceptance checks

{check_md}

## Main result table: best epsilon by scenario and method

{best_md}

## Claim-to-evidence status

{claims_md}

## Suggested report wording

- If any acceptance check is `FAIL`, call this run preliminary evidence, not final evidence.
- If paired comparisons are absent, do not make neural-vs-linear superiority claims.
- If a result is mixed across scenarios, state that it is mixed rather than selecting only the favourable rows.
- Every number above comes from `summary_with_ci.csv` or `summary.csv` in this run directory.
"""
    (out_dir / "FINAL_RESULTS.md").write_text(final_results, encoding="utf-8")
    (out_dir / "CLAIM_TO_EVIDENCE_INDEX.md").write_text(
        f"# Claim-to-evidence index for `{run_name}`\n\n{claims_md}\n\n", encoding="utf-8"
    )

    paper = f"""# FairVote-AI: RR-aware MRP for locally private polling

## Abstract

FairVote-AI evaluates locally private polling estimators under k-ary Randomized Response, sampling bias, sparse subgroup structure, nonlinear response patterns and strategic misreporting. The submitted implementation includes a browser-side LDP respondent prototype, raw-answer rejection on the server, strict analyst upload validation, analytical RR debiasing, linear RR-aware MRP, hierarchical partial-pooling RR-aware MRP, optional neural RR-MRP, and a reproducible evidence pipeline.

## Method summary

The central modelling assumption is explicit: training observes only randomized-response reports. RR-aware MRP estimators optimise reported-label likelihood through the known RR transition matrix and then poststratify estimated latent choice probabilities over population cells. The hierarchical estimator adds feature-level varying effects with shrinkage, targeting sparse-cell stability without storing raw answers.

## Evidence run used for this paper draft

- Run directory: `{run_dir}`
- Preset: `{preset}`
- Command: `{command}`

## Evidence quality checks

{check_md}

## Main results

{best_md}

## Claims and support status

{claims_md}

## Limitations

This paper draft is only as strong as the evidence run above. If the run is smoke-labelled, has fewer than 20 repeated trials per cell, has failures, or lacks paired comparisons, the corresponding claims must be reported as preliminary or unsupported. Randomized response protects answer values only; demographic minimisation, rare-cell reporting and export controls are separate privacy boundaries.

## Reproducibility checklist

- `summary_with_ci.csv` or `summary.csv` present.
- `paired_comparisons.csv` present when neural claims are made.
- `failures.csv` empty for final claims.
- `manifest.json`, `environment.json` and `sha256sums.txt` present.
- Acceptance checks above pass before this text is used as a final report-ready result.
"""
    (out_dir / "fairvote_ai_results.md").write_text(paper, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, help="Evidence run directory containing summary_with_ci.csv")
    parser.add_argument("--out_dir", default=None, help="Output directory; defaults to run_dir")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"run_dir does not exist: {run_dir}")
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    write_outputs(run_dir, out_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
