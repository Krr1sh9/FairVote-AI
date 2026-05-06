"""CLI wrapper around the configuration recommendation utilities.

It reads an existing summary.csv and reports feasible candidates under user
constraints. No model fitting or experiment regeneration is performed here.
"""

# experiments/recommend_from_summary.py
from __future__ import annotations

import argparse
from pathlib import Path

from fairvote.optimisation.recommend import (
    Constraints,
    Objective,
    pareto_frontier,
    read_summary_csv,
    recommend_per_scenario,
    write_pareto_csv,
)


def _parse_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _print_recommendations(recs):
    print("===================================")
    print("Recommendation(s) from summary.csv")
    print("===================================")
    for r in recs:
        print(f"\nScenario: {r.scenario}")
        print(f"  Candidates: {r.total_count} | Feasible: {r.feasible_count}")
        if r.chosen is None:
            print(f"  -> No feasible choice. Reason: {r.reason_if_none}")
            continue
        c = r.chosen
        print("  -> Chosen:")
        print(f"     method   = {c.method}")
        print(f"     epsilon  = {c.epsilon}")
        print(f"     n_rows   = {c.n_rows}")
        print(f"     mean_n_effective        = {c.mean_n_effective:.1f}")
        print(f"     mean_overall_l1         = {c.mean_overall_l1:.6f}")
        print(f"     mean_worst_region_l1    = {c.mean_worst_region_l1:.6f}")
        print(f"     mean_worst_age_l1       = {c.mean_worst_age_l1:.6f}")
        print(f"     mean_overall_mae        = {c.mean_overall_mae:.6f}")
        # New / recommended fairness metrics
        print(f"     mean_worst_region_l1_major = {c.mean_worst_region_l1_major:.6f}")
        print(f"     mean_p90_region_l1_major   = {c.mean_p90_region_l1_major:.6f}")
        print(f"     mean_weighted_region_l1    = {c.mean_weighted_region_l1:.6f}")
        print(f"     mean_worst_age_l1_major    = {c.mean_worst_age_l1_major:.6f}")
        print(f"     mean_p90_age_l1_major      = {c.mean_p90_age_l1_major:.6f}")
        print(f"     mean_weighted_age_l1       = {c.mean_weighted_age_l1:.6f}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Recommend epsilon/method from mrp_vs_baselines summary.csv under constraints."
    )
    ap.add_argument(
        "--summary_csv",
        type=str,
        required=True,
        help="Path to summary.csv (from experiments/mrp_vs_baselines.py).",
    )

    # Which scenarios/methods to consider
    ap.add_argument("--scenarios", type=str, default="", help="Comma list; default = all in file.")
    ap.add_argument("--methods", type=str, default="", help="Comma list; default = all in file.")

    # Constraints (privacy + utility)
    ap.add_argument("--epsilon_max", type=float, default=None)
    ap.add_argument("--epsilon_min", type=float, default=None)
    ap.add_argument("--overall_l1_max", type=float, default=None)
    ap.add_argument("--overall_mae_max", type=float, default=None)

    # Constraints (legacy fairness)
    ap.add_argument("--worst_region_l1_max", type=float, default=None)
    ap.add_argument("--worst_age_l1_max", type=float, default=None)

    # Constraints (recommended fairness: mass-aware / robust)
    ap.add_argument("--worst_region_l1_major_max", type=float, default=None)
    ap.add_argument("--worst_age_l1_major_max", type=float, default=None)
    ap.add_argument("--weighted_region_l1_max", type=float, default=None)
    ap.add_argument("--weighted_age_l1_max", type=float, default=None)
    ap.add_argument("--p90_region_l1_major_max", type=float, default=None)
    ap.add_argument("--p90_age_l1_major_max", type=float, default=None)

    # Sample size constraint
    ap.add_argument("--min_n_effective", type=float, default=None)

    # Objective
    ap.add_argument(
        "--primary",
        type=str,
        default="mean_overall_l1",
        choices=[
            "mean_overall_l1",
            "mean_overall_mae",
            "mean_worst_region_l1",
            "mean_worst_age_l1",
            "mean_worst_region_l1_major",
            "mean_worst_age_l1_major",
            "mean_weighted_region_l1",
            "mean_weighted_age_l1",
            "mean_p90_region_l1_major",
            "mean_p90_age_l1_major",
        ],
        help="Primary metric to minimise.",
    )
    ap.add_argument(
        "--tie_breakers",
        type=str,
        default="mean_worst_region_l1_major,mean_p90_region_l1_major,mean_weighted_region_l1,mean_worst_age_l1_major,mean_p90_age_l1_major,mean_weighted_age_l1,mean_overall_mae",
        help="Comma list of fields to minimise after primary.",
    )
    ap.add_argument(
        "--prefer_lower_epsilon",
        action="store_true",
        help="Prefer smaller epsilon as a final tie-breaker (default True).",
    )
    ap.add_argument(
        "--prefer_higher_epsilon",
        action="store_true",
        help="If set, do NOT prefer lower epsilon; epsilon won't be used as a tie-breaker.",
    )

    # Pareto outputs
    ap.add_argument(
        "--write_pareto",
        action="store_true",
        help="Write pareto frontier CSV per scenario (and per method if you pass --methods).",
    )
    ap.add_argument(
        "--pareto_y",
        type=str,
        default="mean_overall_l1",
        choices=[
            "mean_overall_l1",
            "mean_overall_mae",
            "mean_worst_region_l1",
            "mean_worst_age_l1",
            "mean_worst_region_l1_major",
            "mean_worst_age_l1_major",
            "mean_weighted_region_l1",
            "mean_weighted_age_l1",
            "mean_p90_region_l1_major",
            "mean_p90_age_l1_major",
        ],
        help="Pareto y-axis metric to minimise.",
    )
    ap.add_argument(
        "--pareto_z",
        type=str,
        default="mean_worst_region_l1_major",
        choices=[
            "mean_overall_l1",
            "mean_overall_mae",
            "mean_worst_region_l1",
            "mean_worst_age_l1",
            "mean_worst_region_l1_major",
            "mean_worst_age_l1_major",
            "mean_weighted_region_l1",
            "mean_weighted_age_l1",
            "mean_p90_region_l1_major",
            "mean_p90_age_l1_major",
        ],
        help="Pareto z-axis metric to minimise.",
    )

    args = ap.parse_args()

    summary_path = Path(args.summary_csv)
    cands = read_summary_csv(summary_path)

    scenarios = _parse_list(args.scenarios) if args.scenarios.strip() else None
    methods = _parse_list(args.methods) if args.methods.strip() else None

    cons = Constraints(
        epsilon_max=args.epsilon_max,
        epsilon_min=args.epsilon_min,
        overall_l1_max=args.overall_l1_max,
        overall_mae_max=args.overall_mae_max,
        worst_region_l1_max=args.worst_region_l1_max,
        worst_age_l1_max=args.worst_age_l1_max,
        worst_region_l1_major_max=args.worst_region_l1_major_max,
        worst_age_l1_major_max=args.worst_age_l1_major_max,
        weighted_region_l1_max=args.weighted_region_l1_max,
        weighted_age_l1_max=args.weighted_age_l1_max,
        p90_region_l1_major_max=args.p90_region_l1_major_max,
        p90_age_l1_major_max=args.p90_age_l1_major_max,
        min_n_effective=args.min_n_effective,
    )

    prefer_lower = True
    if args.prefer_higher_epsilon:
        prefer_lower = False
    if args.prefer_lower_epsilon:
        prefer_lower = True

    obj = Objective(
        primary=args.primary,
        tie_breakers=tuple(_parse_list(args.tie_breakers)),
        prefer_lower_epsilon=prefer_lower,
    )

    recs = recommend_per_scenario(
        cands,
        constraints=cons,
        objective=obj,
        allowed_methods=methods,
        scenarios=scenarios,
    )
    _print_recommendations(recs)

    # Pareto frontier write-out
    if args.write_pareto:
        out_dir = summary_path.parent
        scen_list = scenarios if scenarios is not None else sorted({c.scenario for c in cands})

        # If methods specified, write separate pareto per method too
        method_list: list[str | None] = [None] if methods is None else list(methods)

        for s in scen_list:
            for m in method_list:
                front = pareto_frontier(
                    cands,
                    scenario=s,
                    method=m,
                    y=args.pareto_y,
                    z=args.pareto_z,
                )
                if not front:
                    continue
                name = f"pareto_{s}.csv" if m is None else f"pareto_{s}_{m}.csv"
                write_pareto_csv(out_dir / name, front)
                print(f"\nWrote Pareto CSV: {out_dir / name}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
