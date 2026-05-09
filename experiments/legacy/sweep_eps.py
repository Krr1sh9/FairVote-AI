# experiments/sweep_eps.py
"""Run epsilon sweeps for privacy-utility evaluation.

This script compares estimators across Randomized Response privacy levels,
sampling settings, and bias scenarios. It writes machine-readable outputs for
analysis and reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from fairvote.privacy import estimate_distribution, estimate_distribution_central_dp, privatize_many
from fairvote.simulation.bias_models import (
    apply_misreporting,
    apply_nonresponse,
    build_shy_model_from_epsilon,
    make_default_feature_nonresponse_profile,
    make_identity_misreport_model,
    make_shy_supporter_model,
)
from fairvote.simulation.population import (
    Population,
    make_realistic_uk_like_population,
    subgroup_true_distribution,
)
from fairvote.simulation.sampling import (
    biased_frame_sample,
    simple_random_sample,
    stratified_sample,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _parse_eps_list(s: str) -> list[float]:
    """Parse epsilon values without changing the requested sweep grid."""
    vals = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError("No epsilons parsed. Provide --eps like '0.1,0.2,0.5,1.0'.")
    if any(e <= 0 for e in vals):
        raise ValueError("All epsilons must be > 0.")
    return vals


def _parse_strata_list(s: str) -> list[str]:
    out = [x.strip() for x in s.split(",") if x.strip()]
    if not out:
        raise ValueError("No strata parsed. Provide --strata like 'region,age_group'.")
    return out


def _parse_level_multipliers(s: str) -> dict[str, float]:
    """
    Parse "London=1.4,Wales=0.7,Scotland=1.1" into dict.
    """
    s = s.strip()
    if not s:
        return {}
    out: dict[str, float] = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("Invalid multiplier format. Use 'Level=1.2,Other=0.8'.")
        name, val = item.split("=", 1)
        name = name.strip()
        val_f = float(val.strip())
        if val_f <= 0:
            raise ValueError("Multipliers must be > 0.")
        out[name] = val_f
    return out


def _run_dir(base: Path) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_path = base / f"{ts}_sweep_eps"
    run_path.mkdir(parents=True, exist_ok=False)
    (run_path / "plots").mkdir(parents=True, exist_ok=True)
    return run_path


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def _linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _error_ratio(l1_list: list) -> float:
    """Ratio of max to min group L1 error (fairness disparity)."""
    if len(l1_list) < 2:
        return float("nan")
    mn = min(l1_list)
    mx = max(l1_list)
    if mn <= 1e-12:
        return float("inf") if mx > 1e-12 else 1.0
    return float(mx / mn)


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _estimate_subgroup(
    reported: np.ndarray,
    eps: float,
    k: int,
    subgroup_mask: np.ndarray,
) -> np.ndarray:
    if not np.any(subgroup_mask):
        return np.full(k, 1.0 / k, dtype=float)
    return estimate_distribution(reported[subgroup_mask], eps, k)


def _estimate_subgroup_central_dp(
    true_categories: np.ndarray,
    eps: float,
    k: int,
    subgroup_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if not np.any(subgroup_mask):
        return np.full(k, 1.0 / k, dtype=float)
    return estimate_distribution_central_dp(true_categories[subgroup_mask], eps, k, rng=rng)


# -----------------------------------------------------------------------------
# Experiment core
# -----------------------------------------------------------------------------


def run_sweep(
    pop: Population,
    *,
    k: int,
    eps_list: list[float],
    n_samples: list[int],
    trials: int,
    seed: int,
    sampling: str,
    strata: list[str],
    allocation: str,
    min_per_stratum: int,
    biased_feature: str,
    biased_multipliers: dict[str, float],
    scenario: str,
    shy_category: int,
    shy_honesty: float,
) -> tuple[list[dict], list[dict]]:
    """
    Returns:
      - rows: per (trial, epsilon) metrics
      - summary: aggregated per epsilon (mean/std)
    """
    np.random.default_rng(seed)

    # Ground-truth subgroup distributions (population truth)
    truth_region = subgroup_true_distribution(pop, "region")
    truth_age = subgroup_true_distribution(pop, "age_group")

    # Also keep the ordered level names for consistent aggregation
    region_levels = pop.feature_levels["region"]
    age_levels = pop.feature_levels["age_group"]

    import itertools

    rows: list[dict] = []

    for n_sample, t in itertools.product(n_samples, range(trials)):
        target_n_sample = n_sample
        # Separate RNG per trial and n_sample (reproducible)
        rng_t = np.random.default_rng(seed + 10_000 * n_sample + t)

        # 1) Draw sample
        if sampling == "srs":
            sample = simple_random_sample(pop, n_sample, rng=rng_t, replace=False)
        elif sampling == "stratified":
            sample = stratified_sample(
                pop,
                n_sample,
                strata=strata,
                rng=rng_t,
                allocation=allocation,
                min_per_stratum=min_per_stratum,
                replace_within=False,
            )
        elif sampling == "biased":
            sample = biased_frame_sample(
                pop,
                n_sample,
                rng=rng_t,
                feature=biased_feature,
                level_multipliers=biased_multipliers if biased_multipliers else None,
                replace=False,
            )
        else:
            raise ValueError("Invalid sampling mode")

        # 2) Apply nonresponse if scenario includes it
        if scenario in {"nonresponse", "shy_privacy_helps"}:
            feature_profile = make_default_feature_nonresponse_profile()
            sample = apply_nonresponse(sample, pop, rng=rng_t, feature_profile=feature_profile)

        # If sample collapses too much, skip trial (rare but possible under strong nonresponse)
        if sample.idx.size < max(50, int(0.05 * n_sample)):
            # Record as skipped; continue
            for eps in eps_list:
                rows.append(
                    {
                        "trial": t,
                        "epsilon": eps,
                        "n_target": int(target_n_sample),
                        "n_effective": int(sample.idx.size),
                        "skipped": 1,
                    }
                )
            continue

        # True categories of the respondents (pre-misreport)
        true_cats = sample.true_categories.astype(int)

        # Precompute subgroup masks (aligned to respondent arrays)
        region_vals = sample.features["region"].astype(int)
        age_vals = sample.features["age_group"].astype(int)

        for eps in eps_list:
            rng_eps = np.random.default_rng(seed + 10_000 * t + int(round(eps * 10_000)))

            # 3) Misreporting scenarios (pre-LDP)
            if scenario == "no_bias":
                mis = make_identity_misreport_model(k)
                stated = true_cats
            elif scenario == "shy_fixed":
                mis = make_shy_supporter_model(k, shy_category, honesty=shy_honesty)
                stated = apply_misreporting(true_cats, mis, rng=rng_eps)
            elif scenario == "shy_privacy_helps":
                # Honesty increases as epsilon decreases (more privacy => more honesty)
                mis = build_shy_model_from_epsilon(k, shy_category, eps)
                stated = apply_misreporting(true_cats, mis, rng=rng_eps)
            elif scenario == "nonresponse":
                mis = make_identity_misreport_model(k)
                stated = true_cats
            else:
                raise ValueError("Invalid scenario")

            # 4) Apply client-side LDP (k-ary RR). The simulator may know the
            # stated synthetic answer, but estimators below receive only the
            # privatized reported category.
            reported = privatize_many(stated, eps, k, rng=rng_eps)

            # 5) Estimate overall distribution from reported
            theta_hat_ldp = estimate_distribution(reported, eps, k)

            # 6) Ground-truth target (population truth): compare to population subgroup truths
            # Overall truth in population (not just the sample)
            theta_true_pop = np.bincount(pop.true_categories, minlength=k).astype(float)
            theta_true_pop = theta_true_pop / theta_true_pop.sum()

            # =================================================================
            # Method A: LDP (k-ary RR debias)
            # =================================================================
            ldp_overall_l1 = _l1(theta_hat_ldp, theta_true_pop)
            ldp_overall_linf = _linf(theta_hat_ldp, theta_true_pop)
            ldp_overall_mae = _mae(theta_hat_ldp, theta_true_pop)
            ldp_correct_winner = int(np.argmax(theta_hat_ldp) == np.argmax(theta_true_pop))

            # Subgroup estimation (LDP): estimate from RR-reported slices
            ldp_region_est = {}
            ldp_region_l1s = []
            for lvl_idx, lvl_name in enumerate(region_levels):
                if lvl_name not in truth_region:
                    continue
                mask = region_vals == lvl_idx
                est = _estimate_subgroup(reported, eps, k, mask)
                ldp_region_est[lvl_name] = est
                ldp_region_l1s.append(_l1(est, truth_region[lvl_name]))

            ldp_age_est = {}
            ldp_age_l1s = []
            for lvl_idx, lvl_name in enumerate(age_levels):
                if lvl_name not in truth_age:
                    continue
                mask = age_vals == lvl_idx
                est = _estimate_subgroup(reported, eps, k, mask)
                ldp_age_est[lvl_name] = est
                ldp_age_l1s.append(_l1(est, truth_age[lvl_name]))

            # Error ratios (max-group / min-group L1)
            ldp_region_ratio = _error_ratio(ldp_region_l1s)
            ldp_age_ratio = _error_ratio(ldp_age_l1s)

            rows.append(
                {
                    "trial": t,
                    "epsilon": float(eps),
                    "method": "ldp_rr_debias",
                    "n_target": int(target_n_sample),
                    "n_effective": int(sample.idx.size),
                    "skipped": 0,
                    "overall_l1": ldp_overall_l1,
                    "overall_linf": ldp_overall_linf,
                    "overall_mae": ldp_overall_mae,
                    "correct_winner": ldp_correct_winner,
                    "worst_region_l1": float(np.max(ldp_region_l1s)) if ldp_region_l1s else float("nan"),
                    "avg_region_l1": float(np.mean(ldp_region_l1s)) if ldp_region_l1s else float("nan"),
                    "worst_age_l1": float(np.max(ldp_age_l1s)) if ldp_age_l1s else float("nan"),
                    "avg_age_l1": float(np.mean(ldp_age_l1s)) if ldp_age_l1s else float("nan"),
                    "error_ratio_region": ldp_region_ratio,
                    "error_ratio_age": ldp_age_ratio,
                }
            )

            # =================================================================
            # Method B: Central DP (Laplace on aggregate counts)
            #   Trust model: the collector sees raw stated categories,
            #   computes exact counts, then adds Laplace noise.
            # =================================================================
            rng_cdp = np.random.default_rng(seed + 10_000 * t + int(round(eps * 10_000)) + 99)
            theta_hat_cdp = estimate_distribution_central_dp(stated, eps, k, rng=rng_cdp)

            cdp_overall_l1 = _l1(theta_hat_cdp, theta_true_pop)
            cdp_overall_linf = _linf(theta_hat_cdp, theta_true_pop)
            cdp_overall_mae = _mae(theta_hat_cdp, theta_true_pop)

            # Subgroup estimation (Central DP): Laplace noise on true subgroup counts
            cdp_region_l1s = []
            for lvl_idx, lvl_name in enumerate(region_levels):
                if lvl_name not in truth_region:
                    continue
                mask = region_vals == lvl_idx
                rng_sub = np.random.default_rng(seed + 10_000 * t + int(round(eps * 10_000)) + 200 + lvl_idx)
                est = _estimate_subgroup_central_dp(stated, eps, k, mask, rng_sub)
                cdp_region_l1s.append(_l1(est, truth_region[lvl_name]))

            cdp_age_l1s = []
            for lvl_idx, lvl_name in enumerate(age_levels):
                if lvl_name not in truth_age:
                    continue
                mask = age_vals == lvl_idx
                rng_sub = np.random.default_rng(seed + 10_000 * t + int(round(eps * 10_000)) + 300 + lvl_idx)
                est = _estimate_subgroup_central_dp(stated, eps, k, mask, rng_sub)
                cdp_age_l1s.append(_l1(est, truth_age[lvl_name]))

            cdp_correct_winner = int(np.argmax(theta_hat_cdp) == np.argmax(theta_true_pop))
            cdp_region_ratio = _error_ratio(cdp_region_l1s)
            cdp_age_ratio = _error_ratio(cdp_age_l1s)

            rows.append(
                {
                    "trial": t,
                    "epsilon": float(eps),
                    "method": "central_dp_laplace",
                    "n_target": int(target_n_sample),
                    "n_effective": int(sample.idx.size),
                    "skipped": 0,
                    "overall_l1": cdp_overall_l1,
                    "overall_linf": cdp_overall_linf,
                    "overall_mae": cdp_overall_mae,
                    "correct_winner": cdp_correct_winner,
                    "worst_region_l1": float(np.max(cdp_region_l1s)) if cdp_region_l1s else float("nan"),
                    "avg_region_l1": float(np.mean(cdp_region_l1s)) if cdp_region_l1s else float("nan"),
                    "worst_age_l1": float(np.max(cdp_age_l1s)) if cdp_age_l1s else float("nan"),
                    "avg_age_l1": float(np.mean(cdp_age_l1s)) if cdp_age_l1s else float("nan"),
                    "error_ratio_region": cdp_region_ratio,
                    "error_ratio_age": cdp_age_ratio,
                }
            )

    # Aggregate summary per (epsilon, method, n_target)
    methods = sorted({r.get("method", "ldp_rr_debias") for r in rows if r.get("skipped", 0) == 0})
    summary: list[dict] = []

    # We must preserve the order of eps_list and n_samples
    for eps in eps_list:
        for method in methods:
            for n_val in n_samples:
                sub = [
                    r
                    for r in rows
                    if r.get("epsilon") == float(eps)
                    and r.get("method", "ldp_rr_debias") == method
                    and r.get("n_target", n_val) == n_val
                    and r.get("skipped", 0) == 0
                ]
                if not sub:
                    summary.append({"epsilon": float(eps), "method": method, "n_target": n_val, "n_rows": 0})
                    continue

                stats_rows = list(sub)

                def stats(key: str, rows_for_stats=stats_rows) -> tuple[float, float]:
                    vals = np.array([r[key] for r in rows_for_stats], dtype=float)
                    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0

                m_l1, s_l1 = stats("overall_l1")
                m_wreg, s_wreg = stats("worst_region_l1")
                m_wage, s_wage = stats("worst_age_l1")
                m_mae, s_mae = stats("overall_mae")

                # Correct winner probability
                cw_vals = np.array([r.get("correct_winner", 0) for r in sub], dtype=float)
                cw_prob = float(np.mean(cw_vals))

                # RMSE (per-trial squared L1, then root-mean)
                l1_vals = np.array([r["overall_l1"] for r in sub], dtype=float)
                rmse_overall = float(np.sqrt(np.mean(l1_vals**2)))

                # Error ratio stats
                m_ratio_reg, s_ratio_reg = stats("error_ratio_region")
                m_ratio_age, s_ratio_age = stats("error_ratio_age")

                summary.append(
                    {
                        "epsilon": float(eps),
                        "method": method,
                        "n_target": n_val,
                        "n_rows": len(sub),
                        "mean_overall_l1": m_l1,
                        "std_overall_l1": s_l1,
                        "mean_overall_mae": m_mae,
                        "std_overall_mae": s_mae,
                        "rmse_overall": rmse_overall,
                        "correct_winner_prob": cw_prob,
                        "mean_worst_region_l1": m_wreg,
                        "std_worst_region_l1": s_wreg,
                        "mean_worst_age_l1": m_wage,
                        "std_worst_age_l1": s_wage,
                        "mean_error_ratio_region": m_ratio_reg,
                        "std_error_ratio_region": s_ratio_reg,
                        "mean_error_ratio_age": m_ratio_age,
                        "std_error_ratio_age": s_ratio_age,
                    }
                )

    return rows, summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_summary(run_dir: Path, summary: list[dict]) -> None:
    # Optional plotting (matplotlib). If not installed, silently skip.
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    valid = [r for r in summary if r.get("n_rows", 0) > 0]
    if not valid:
        return

    methods = sorted({r.get("method", "ldp_rr_debias") for r in valid})
    method_labels = {
        "ldp_rr_debias": "LDP (k-ary RR)",
        "central_dp_laplace": "Central DP (Laplace)",
    }
    method_styles = {
        "ldp_rr_debias": {"marker": "o", "linestyle": "-"},
        "central_dp_laplace": {"marker": "s", "linestyle": "--"},
    }

    n_targets_set = sorted({r.get("n_target") for r in valid if "n_target" in r})
    epsilons_set = sorted({r["epsilon"] for r in valid})

    is_1d_sweep = len(n_targets_set) <= 1

    if is_1d_sweep:
        # --- 1D Mode: Overall L1 vs epsilon ---
        plt.figure()
        for m in methods:
            rows_m = [r for r in valid if r.get("method", "ldp_rr_debias") == m]
            if not rows_m:
                continue
            eps = [r["epsilon"] for r in rows_m]
            y = [r["mean_overall_l1"] for r in rows_m]
            style = method_styles.get(m, {"marker": "o", "linestyle": "-"})
            plt.plot(eps, y, label=method_labels.get(m, m), **style)
        plt.xscale("log")
        plt.xlabel("epsilon (log scale)")
        plt.ylabel("Mean overall L1 error vs population truth")
        plt.title("Privacy–Utility Curve: Central DP vs LDP")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "plots" / "overall_l1_vs_epsilon.png", dpi=200)
        plt.close()

        # --- 1D Mode: Worst region L1 vs epsilon ---
        plt.figure()
        for m in methods:
            rows_m = [r for r in valid if r.get("method", "ldp_rr_debias") == m]
            if not rows_m:
                continue
            eps = [r["epsilon"] for r in rows_m]
            y = [r["mean_worst_region_l1"] for r in rows_m]
            style = method_styles.get(m, {"marker": "o", "linestyle": "-"})
            plt.plot(eps, y, label=method_labels.get(m, m), **style)
        plt.xscale("log")
        plt.xlabel("epsilon (log scale)")
        plt.ylabel("Mean worst-region L1 error vs population truth")
        plt.title("Privacy–Fairness Curve: Worst Region Error")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "plots" / "worst_region_l1_vs_epsilon.png", dpi=200)
        plt.close()

        # --- 1D Mode: Correct winner probability vs epsilon ---
        plt.figure()
        for m in methods:
            rows_m = [r for r in valid if r.get("method", "ldp_rr_debias") == m]
            if not rows_m:
                continue
            eps = [r["epsilon"] for r in rows_m]
            y = [r.get("correct_winner_prob", float("nan")) for r in rows_m]
            if all(v != v for v in y):  # all NaN
                continue
            style = method_styles.get(m, {"marker": "o", "linestyle": "-"})
            plt.plot(eps, y, label=method_labels.get(m, m), **style)
        plt.xscale("log")
        plt.xlabel("epsilon (log scale)")
        plt.ylabel("P(correct winner)")
        plt.title("Correct Winner Probability vs Privacy Budget")
        plt.ylim(-0.05, 1.05)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "plots" / "correct_winner_vs_epsilon.png", dpi=200)
        plt.close()

        # --- 1D Mode: Error ratio (region) vs epsilon ---
        plt.figure()
        for m in methods:
            rows_m = [r for r in valid if r.get("method", "ldp_rr_debias") == m]
            if not rows_m:
                continue
            eps = [r["epsilon"] for r in rows_m]
            y = [r.get("mean_error_ratio_region", float("nan")) for r in rows_m]
            if all(v != v for v in y):
                continue
            style = method_styles.get(m, {"marker": "o", "linestyle": "-"})
            plt.plot(eps, y, label=method_labels.get(m, m), **style)
        plt.xscale("log")
        plt.xlabel("epsilon (log scale)")
        plt.ylabel("Error ratio (max-region / min-region L1)")
        plt.title("Fairness Disparity Ratio vs Privacy Budget")
        plt.axhline(y=1.0, color="gray", linestyle=":", alpha=0.6, label="Perfect fairness")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "plots" / "error_ratio_vs_epsilon.png", dpi=200)
        plt.close()

    else:
        # --- 2D Mode: Sample Complexity Curves (X = n_sample, Y = L1 error, lines = epsilon) ---
        for metric, title, filename in [
            ("mean_overall_l1", "Sample Complexity (Overall L1 Error)", "overall_l1_vs_nsample.png"),
            ("mean_worst_region_l1", "Sample Complexity (Worst Region L1 Error)", "worst_region_vs_nsample.png"),
        ]:
            plt.figure(figsize=(10, 6))
            colors = plt.cm.viridis(np.linspace(0, 1, len(epsilons_set)))

            for m in methods:
                for i, eps in enumerate(epsilons_set):
                    rows_me = [r for r in valid if r.get("method", "") == m and r["epsilon"] == eps]
                    if not rows_me:
                        continue
                    x_n = [r["n_target"] for r in rows_me]
                    y_err = [r[metric] for r in rows_me]

                    label = f"{method_labels.get(m, m)} (ε={eps})"
                    style = method_styles.get(m, {"marker": "o", "linestyle": "-"})
                    plt.plot(x_n, y_err, label=label, color=colors[i], **style)

            plt.xscale("log")
            plt.xlabel("Sample Size (n_target, log scale)")
            plt.ylabel(title)
            plt.title(title)
            # Create legend outside the plot area
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.savefig(run_dir / "plots" / filename, dpi=200)
            plt.close()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main() -> int:
    """Sweep epsilon/sample settings for aggregate and subgroup metrics."""
    p = argparse.ArgumentParser(description="Sweep epsilon and measure privacy–utility + subgroup error.")
    p.add_argument("--k", type=int, default=5, help="Number of categories (K).")
    p.add_argument("--eps", type=str, default="0.1,0.2,0.5,1.0,2.0,4.0", help="Comma-separated epsilons.")
    p.add_argument("--population_n", type=int, default=100_000, help="Synthetic population size.")
    p.add_argument("--n_samples", type=str, default="5000", help="Comma-separated sample sizes (e.g. 1000,5000).")
    p.add_argument("--trials", type=int, default=10, help="Number of Monte Carlo trials.")
    p.add_argument("--seed", type=int, default=123, help="Base RNG seed.")
    p.add_argument("--out_dir", type=str, default="experiments/outputs", help="Base output directory.")

    # Sampling
    p.add_argument("--sampling", type=str, default="srs", choices=["srs", "stratified", "biased"])
    p.add_argument(
        "--strata", type=str, default="region", help="Strata for stratified sampling, e.g. 'region,age_group'"
    )
    p.add_argument("--allocation", type=str, default="proportional", choices=["proportional", "equal", "sqrt"])
    p.add_argument("--min_per_stratum", type=int, default=0, help="Min per stratum cell (stratified only).")

    # Biased frame sampling
    p.add_argument("--biased_feature", type=str, default="region", help="Feature to bias in 'biased' sampling mode.")
    p.add_argument("--multipliers", type=str, default="", help="Level multipliers, e.g. 'London=1.4,Wales=0.7'")

    # Scenario
    p.add_argument(
        "--scenario",
        type=str,
        default="no_bias",
        choices=["no_bias", "nonresponse", "shy_fixed", "shy_privacy_helps"],
        help="Bias scenario.",
    )
    p.add_argument("--shy_category", type=int, default=0, help="Which category is 'shy' (for shy scenarios).")
    p.add_argument("--shy_honesty", type=float, default=0.80, help="Honesty for shy_fixed (0..1).")

    args = p.parse_args()

    eps_list = _parse_eps_list(args.eps)
    n_samples = [int(x.strip()) for x in args.n_samples.split(",") if x.strip()]
    strata = _parse_strata_list(args.strata)
    multipliers = _parse_level_multipliers(args.multipliers)

    # Build population
    pop = make_realistic_uk_like_population(args.population_n, args.k, seed=args.seed)

    # Create run directory
    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    run_dir = _run_dir(base_out)

    # Save config snapshot
    config = {
        "k": args.k,
        "eps": eps_list,
        "population_n": args.population_n,
        "n_samples": n_samples,
        "trials": args.trials,
        "seed": args.seed,
        "sampling": args.sampling,
        "strata": strata,
        "allocation": args.allocation,
        "min_per_stratum": args.min_per_stratum,
        "biased_feature": args.biased_feature,
        "multipliers": multipliers,
        "scenario": args.scenario,
        "shy_category": args.shy_category,
        "shy_honesty": args.shy_honesty,
    }
    _write_json(run_dir / "config.json", config)

    # Run sweep
    rows, summary = run_sweep(
        pop,
        k=args.k,
        eps_list=eps_list,
        n_samples=n_samples,
        trials=args.trials,
        seed=args.seed,
        sampling=args.sampling,
        strata=strata,
        allocation=args.allocation,
        min_per_stratum=args.min_per_stratum,
        biased_feature=args.biased_feature,
        biased_multipliers=multipliers,
        scenario=args.scenario,
        shy_category=args.shy_category,
        shy_honesty=args.shy_honesty,
    )

    # Save outputs
    _write_csv(run_dir / "results_trials.csv", rows)
    _write_csv(run_dir / "summary.csv", summary)

    # Optional plots
    _plot_summary(run_dir, summary)

    # Print where outputs are
    print(f"Saved run to: {run_dir}")
    print(f"- {run_dir / 'summary.csv'}")
    print(f"- {run_dir / 'results_trials.csv'}")
    print(f"- {run_dir / 'plots'} (if matplotlib installed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
