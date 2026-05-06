# experiments/sensitivity_analysis.py
"""
Sensitivity Analysis for FairVote-AI

Tests robustness of findings to population assumptions by running the epsilon
sweep under three configurations:

  1. baseline    — default UK-like population (moderate correlation)
  2. weak_corr   — demographics weakly predict vote preference
  3. strong_nr   — severe non-response bias (young/urban much less likely to respond)

Outputs:
  - sensitivity_comparison.csv   — side-by-side summary for all configs
  - plots/sensitivity_*.png      — overlay plots comparing configs

This experiment checks whether the main findings are sensitive to
  assumptions about demographic-vote correlation and non-response.
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
    FeatureNonresponseProfile,
    apply_nonresponse,
)
from fairvote.simulation.population import (
    Population,
    make_realistic_uk_like_population,
    subgroup_true_distribution,
)
from fairvote.simulation.sampling import simple_random_sample

# =============================================================================
# Population variants
# =============================================================================


def _make_weak_correlation_population(n: int, k: int, seed: int) -> Population:
    """
    Population where demographics only weakly predict preferences.
    Uses the same structure as the default but with much smaller effect scales.
    """
    rng = np.random.default_rng(seed)

    from fairvote.simulation.population import (
        _effect_table,
        _make_region_boost,
        _sample_categorical_rows,
        _sigmoid,
        _softmax,
    )

    region_levels = [
        "London",
        "South East",
        "South West",
        "East of England",
        "West Midlands",
        "East Midlands",
        "North West",
        "North East",
        "Yorkshire & Humber",
        "Scotland",
        "Wales",
        "Northern Ireland",
    ]
    age_levels = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    edu_levels = ["No degree", "Some college/A-level", "Degree+"]
    # Demographic "Other/Prefer not to say" is retained only for gender; it is not
    # used as a party/poll option in the canonical five-option setup.
    gender_levels = ["Female", "Male", "Other/Prefer not to say"]
    urban_levels = ["Urban", "Suburban", "Rural"]

    feature_levels = {
        "region": region_levels,
        "age_group": age_levels,
        "education": edu_levels,
        "gender": gender_levels,
        "urbanicity": urban_levels,
    }

    region_w = np.array([0.13, 0.14, 0.09, 0.09, 0.09, 0.07, 0.11, 0.05, 0.08, 0.08, 0.05, 0.02])
    region_w /= region_w.sum()
    age_w = np.array([0.10, 0.16, 0.16, 0.17, 0.17, 0.24])
    age_w /= age_w.sum()
    gender_w = np.array([0.495, 0.495, 0.01])
    urban_w = np.array([0.55, 0.30, 0.15])

    region = rng.choice(len(region_levels), size=n, p=region_w)
    age_group = rng.choice(len(age_levels), size=n, p=age_w)
    gender = rng.choice(len(gender_levels), size=n, p=gender_w)
    urbanicity = rng.choice(len(urban_levels), size=n, p=urban_w)

    region_degree_boost = _make_region_boost(len(region_levels), rng, high_regions={0, 1, 3}, low_regions={7, 10, 11})
    age_degree_boost = np.array([0.35, 0.30, 0.20, 0.05, -0.10, -0.25])
    urban_degree_boost = np.array([0.20, 0.05, -0.15])
    degree_score = (
        region_degree_boost[region]
        + age_degree_boost[age_group]
        + urban_degree_boost[urbanicity]
        + rng.normal(0, 0.35, n)
    )
    p_dp = _sigmoid(degree_score - 0.35)
    p_nd = _sigmoid(-(degree_score + 0.15))
    p_sm = np.clip(1.0 - p_dp - p_nd, 0.05, 0.90)
    denom = p_dp + p_sm + p_nd
    p_dp /= denom
    p_sm /= denom
    p_nd /= denom
    education = np.empty(n, dtype=int)
    u = rng.random(n)
    education[u < p_nd] = 0
    education[(u >= p_nd) & (u < p_nd + p_sm)] = 1
    education[u >= (p_nd + p_sm)] = 2

    category_names = [f"Option {i}" for i in range(k)]
    logits = np.zeros((n, k))
    base = rng.normal(0.0, 0.35, size=k)
    base -= base.mean()
    logits += base

    # Weak-correlation setting: scale factors are 0.1x the default.
    logits += _effect_table("region", region, len(region_levels), k, rng, scale=0.08)
    logits += _effect_table("age_group", age_group, len(age_levels), k, rng, scale=0.06)
    logits += _effect_table("education", education, len(edu_levels), k, rng, scale=0.07)
    logits += _effect_table("gender", gender, len(gender_levels), k, rng, scale=0.03)
    logits += _effect_table("urbanicity", urbanicity, len(urban_levels), k, rng, scale=0.04)
    logits += rng.normal(0.0, 0.25, size=(n, k))

    true_probs = _softmax(logits)
    true_categories = _sample_categorical_rows(true_probs, rng)

    features = {
        "region": region,
        "age_group": age_group,
        "education": education,
        "gender": gender,
        "urbanicity": urbanicity,
    }
    return Population(
        features=features,
        feature_levels=feature_levels,
        true_probs=true_probs,
        true_categories=true_categories,
        category_names=category_names,
    )


def _make_severe_nonresponse_profile() -> FeatureNonresponseProfile:
    """Much stronger non-response bias than the default."""
    return FeatureNonresponseProfile(
        base_rate=0.75,
        feature_response_rates={
            "age_group": {
                "18-24": 0.35,  # very low (vs default 0.70)
                "25-34": 0.55,  # low    (vs default 0.78)
                "35-44": 0.72,
                "45-54": 0.82,
                "55-64": 0.88,
                "65+": 0.92,
            },
            "urbanicity": {
                "Urban": 0.65,  # lower (vs default 0.82)
                "Suburban": 0.80,
                "Rural": 0.88,
            },
        },
    )


# =============================================================================
# Sweep runner (simplified version of sweep_eps.run_sweep for sensitivity)
# =============================================================================


def _l1(a, b):
    return float(np.sum(np.abs(np.asarray(a) - np.asarray(b))))


def _run_config(
    config_name: str,
    pop: Population,
    k: int,
    eps_list: list[float],
    n_sample: int,
    trials: int,
    seed: int,
    nr_profile=None,
) -> list[dict]:
    """Run a simplified sweep for one configuration."""
    theta_true = np.bincount(pop.true_categories, minlength=k).astype(float)
    theta_true /= theta_true.sum()

    truth_region = subgroup_true_distribution(pop, "region")
    region_levels = pop.feature_levels["region"]

    results: list[dict] = []

    for t in range(trials):
        rng = np.random.default_rng(seed + 10_000 * t)
        sample = simple_random_sample(pop, n_sample, rng=rng, replace=False)

        if nr_profile is not None:
            sample = apply_nonresponse(sample, pop, rng=rng, feature_profile=nr_profile)

        if sample.idx.size < 50:
            continue

        true_cats = sample.true_categories.astype(int)
        region_vals = sample.features["region"].astype(int)

        for eps in eps_list:
            rng_eps = np.random.default_rng(seed + 10_000 * t + int(round(eps * 10_000)))

            # LDP
            reported = privatize_many(true_cats, eps, k, rng=rng_eps)
            theta_ldp = estimate_distribution(reported, eps, k)
            ldp_l1 = _l1(theta_ldp, theta_true)
            ldp_cw = int(np.argmax(theta_ldp) == np.argmax(theta_true))

            # Region subgroup errors
            region_l1s = []
            for idx, name in enumerate(region_levels):
                if name not in truth_region:
                    continue
                mask = region_vals == idx
                if not np.any(mask):
                    continue
                est = estimate_distribution(reported[mask], eps, k)
                region_l1s.append(_l1(est, truth_region[name]))

            worst_reg = float(np.max(region_l1s)) if region_l1s else float("nan")

            # Central DP
            rng_cdp = np.random.default_rng(seed + 10_000 * t + int(round(eps * 10_000)) + 99)
            theta_cdp = estimate_distribution_central_dp(true_cats, eps, k, rng=rng_cdp)
            cdp_l1 = _l1(theta_cdp, theta_true)
            cdp_cw = int(np.argmax(theta_cdp) == np.argmax(theta_true))

            results.append(
                {
                    "config": config_name,
                    "trial": t,
                    "epsilon": float(eps),
                    "method": "ldp_rr_debias",
                    "n_effective": int(sample.idx.size),
                    "overall_l1": ldp_l1,
                    "correct_winner": ldp_cw,
                    "worst_region_l1": worst_reg,
                }
            )
            results.append(
                {
                    "config": config_name,
                    "trial": t,
                    "epsilon": float(eps),
                    "method": "central_dp_laplace",
                    "n_effective": int(sample.idx.size),
                    "overall_l1": cdp_l1,
                    "correct_winner": cdp_cw,
                    "worst_region_l1": float("nan"),
                }
            )

    return results


def _aggregate(rows: list[dict]) -> list[dict]:
    """Aggregate per (config, method, epsilon)."""
    keys = set()
    for r in rows:
        keys.add((r["config"], r["method"], r["epsilon"]))

    summary = []
    for config, method, eps in sorted(keys):
        sub = [r for r in rows if r["config"] == config and r["method"] == method and r["epsilon"] == eps]
        l1s = np.array([r["overall_l1"] for r in sub])
        cws = np.array([r["correct_winner"] for r in sub])
        n_eff = np.array([r["n_effective"] for r in sub])

        summary.append(
            {
                "config": config,
                "method": method,
                "epsilon": eps,
                "n_trials": len(sub),
                "mean_overall_l1": float(np.mean(l1s)),
                "std_overall_l1": float(np.std(l1s, ddof=1)) if len(l1s) > 1 else 0.0,
                "rmse_overall": float(np.sqrt(np.mean(l1s**2))),
                "correct_winner_prob": float(np.mean(cws)),
                "mean_n_effective": float(np.mean(n_eff)),
            }
        )
    return summary


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _plot_comparison(run_dir: Path, summary: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    configs = sorted({r["config"] for r in summary})
    config_styles = {
        "baseline": {"color": "#2196F3", "marker": "o"},
        "weak_corr": {"color": "#FF9800", "marker": "s"},
        "strong_nr": {"color": "#E91E63", "marker": "^"},
    }

    for method in ["ldp_rr_debias", "central_dp_laplace"]:
        method_label = "LDP" if "ldp" in method else "Central DP"

        # Overall L1
        fig, ax = plt.subplots(figsize=(8, 5))
        for cfg in configs:
            rows = [r for r in summary if r["config"] == cfg and r["method"] == method]
            if not rows:
                continue
            eps = [r["epsilon"] for r in rows]
            y = [r["mean_overall_l1"] for r in rows]
            style = config_styles.get(cfg, {"color": "gray", "marker": "x"})
            ax.plot(eps, y, label=cfg, **style, linestyle="-")
        ax.set_xscale("log")
        ax.set_xlabel("ε (privacy budget)")
        ax.set_ylabel("Mean overall L1 error")
        ax.set_title(f"Sensitivity Analysis — {method_label}: Overall Error")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "plots" / f"sensitivity_l1_{method}.png", dpi=200)
        plt.close(fig)

        # Correct winner probability
        fig, ax = plt.subplots(figsize=(8, 5))
        for cfg in configs:
            rows = [r for r in summary if r["config"] == cfg and r["method"] == method]
            if not rows:
                continue
            eps = [r["epsilon"] for r in rows]
            y = [r["correct_winner_prob"] for r in rows]
            style = config_styles.get(cfg, {"color": "gray", "marker": "x"})
            ax.plot(eps, y, label=cfg, **style, linestyle="-")
        ax.set_xscale("log")
        ax.set_xlabel("ε (privacy budget)")
        ax.set_ylabel("P(correct winner)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(f"Sensitivity Analysis — {method_label}: Correct Winner")
        ax.legend()
        fig.tight_layout()
        fig.savefig(run_dir / "plots" / f"sensitivity_cw_{method}.png", dpi=200)
        plt.close(fig)


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    """Run a small sensitivity grid over privacy/noise settings."""
    p = argparse.ArgumentParser(description="Sensitivity analysis: test robustness to population assumptions.")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--eps", type=str, default="0.1,0.2,0.5,1.0,2.0,4.0")
    p.add_argument("--population_n", type=int, default=100_000)
    p.add_argument("--n_sample", type=int, default=5000)
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str, default="experiments/outputs")
    args = p.parse_args()

    eps_list = [float(x.strip()) for x in args.eps.split(",") if x.strip()]
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = Path(args.out_dir) / f"{ts}_sensitivity"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    # Config 1: Baseline (default population)
    print("Config 1/3: baseline (default population)...")
    pop_baseline = make_realistic_uk_like_population(args.population_n, args.k, seed=args.seed)
    rows_baseline = _run_config("baseline", pop_baseline, args.k, eps_list, args.n_sample, args.trials, args.seed)

    # Config 2: Weak correlation
    print("Config 2/3: weak_corr (weak demographic-vote correlation)...")
    pop_weak = _make_weak_correlation_population(args.population_n, args.k, seed=args.seed)
    rows_weak = _run_config("weak_corr", pop_weak, args.k, eps_list, args.n_sample, args.trials, args.seed)

    # Config 3: Severe non-response
    print("Config 3/3: strong_nr (severe non-response bias)...")
    nr_profile = _make_severe_nonresponse_profile()
    rows_strong = _run_config(
        "strong_nr", pop_baseline, args.k, eps_list, args.n_sample, args.trials, args.seed, nr_profile=nr_profile
    )

    all_rows = rows_baseline + rows_weak + rows_strong
    summary = _aggregate(all_rows)

    # Save
    _write_csv(run_dir / "sensitivity_trials.csv", all_rows)
    _write_csv(run_dir / "sensitivity_comparison.csv", summary)

    config_snapshot = {
        "k": args.k,
        "eps": eps_list,
        "population_n": args.population_n,
        "n_sample": args.n_sample,
        "trials": args.trials,
        "seed": args.seed,
        "configs": ["baseline", "weak_corr", "strong_nr"],
    }
    (run_dir / "config.json").write_text(json.dumps(config_snapshot, indent=2), encoding="utf-8")

    # Plots
    _plot_comparison(run_dir, summary)

    print(f"\nSensitivity analysis complete. Output: {run_dir}")
    print(f"  - {run_dir / 'sensitivity_comparison.csv'}")
    print(f"  - {run_dir / 'plots'}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
