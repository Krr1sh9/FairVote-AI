"""Small end-to-end sanity run for the core Randomized Response estimator.

This script is intended as a quick smoke check of the privacy mechanism and
aggregate debiasing, not as evidence for the final experiment conclusions.
"""

# experiments/sanity_run.py
from __future__ import annotations

import argparse
from typing import List

import numpy as np

from fairvote.privacy import bootstrap_ci, estimate_distribution, privatize_many


def _parse_probs(s: str, k: int) -> np.ndarray:
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    if len(parts) != k:
        raise ValueError(f"--probs must contain exactly {k} comma-separated values.")
    probs = np.array([float(x) for x in parts], dtype=float)
    if np.any(probs < 0):
        raise ValueError("--probs must be non-negative.")
    total = float(probs.sum())
    if total <= 0:
        raise ValueError("--probs must sum to > 0.")
    return probs / total


def _default_probs(k: int) -> np.ndarray:
    # A simple, non-uniform distribution for sanity checking.
    # If k differs, create a smooth-ish distribution and renormalize.
    base = np.linspace(1.0, 2.0, num=k, dtype=float)
    base = base / base.sum()
    return base


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sanity run: true -> LDP (k-ary RR) -> debiased estimate + bootstrap CI"
    )
    parser.add_argument("--k", type=int, default=5, help="Number of categories (K).")
    parser.add_argument("--epsilon", type=float, default=1.0, help="LDP epsilon (> 0).")
    parser.add_argument("--n", type=int, default=5000, help="Number of respondents.")
    parser.add_argument("--seed", type=int, default=123, help="RNG seed.")
    parser.add_argument(
        "--probs",
        type=str,
        default="",
        help="Comma-separated true distribution of length K (e.g., '0.1,0.2,0.3,0.2,0.2').",
    )
    parser.add_argument(
        "--n_boot",
        type=int,
        default=1000,
        help="Bootstrap resamples for CI (>= 200 recommended).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="CI alpha (0.05 gives 95%% intervals).",
    )

    args = parser.parse_args(argv)

    if args.k < 2:
        raise ValueError("--k must be >= 2")
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be > 0")
    if args.n <= 0:
        raise ValueError("--n must be > 0")

    rng = np.random.default_rng(args.seed)

    if args.probs.strip():
        theta_true = _parse_probs(args.probs, args.k)
    else:
        theta_true = _default_probs(args.k)

    # 1) Generate true categories according to theta_true.  These simulate
    #    the respondents' actual (unobserved in practice) preferences.
    true_categories = rng.choice(args.k, size=args.n, p=theta_true)

    # 2) Apply k-ary Randomized Response (client-side LDP).  After this step,
    #    the true categories are no longer accessible to the analyst.
    reported_categories = privatize_many(true_categories, args.epsilon, args.k, rng=rng)

    # 3) Debias to estimate the true distribution using only the reported data.
    theta_hat = estimate_distribution(reported_categories, args.epsilon, args.k)

    # 4) Bootstrap CI on the debiased estimate.  The bootstrap resamples the
    #    reported (not true) data, giving frequentist coverage despite RR noise.
    ci_low, ci_high = bootstrap_ci(
        reported_categories,
        args.epsilon,
        args.k,
        n_boot=args.n_boot,
        alpha=args.alpha,
        rng=rng,
    )

    # Print results
    ci_level = int(round((1.0 - args.alpha) * 100))
    header = f"Sanity Run (k={args.k}, epsilon={args.epsilon}, n={args.n}, seed={args.seed})"
    print("=" * len(header))
    print(header)
    print("=" * len(header))
    print()

    print("Category |  True p   |  Est p    |  CI Low   |  CI High")
    print("---------+----------+----------+----------+----------")
    for j in range(args.k):
        print(
            f"{j:8d} | {theta_true[j]:8.4f} | {theta_hat[j]:8.4f} | {ci_low[j]:8.4f} | {ci_high[j]:8.4f}"
        )
    print()

    l1_err = float(np.sum(np.abs(theta_hat - theta_true)))
    linf_err = float(np.max(np.abs(theta_hat - theta_true)))
    print(f"L1 error:   {l1_err:.6f}")
    print(f"Linf error: {linf_err:.6f}")
    print(f"CI level:   {ci_level}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
