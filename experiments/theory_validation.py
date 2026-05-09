"""Theory checks for k-ary Randomized Response and RR-aware estimators.

The functions in this module are intentionally lightweight enough to run in CI
while still documenting the mathematical properties used in the report:
privacy ratio, unbiased inverse before clipping, variance of reported counts,
and Monte Carlo confidence-interval coverage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from fairvote.privacy import bootstrap_ci, estimate_distribution
from fairvote.privacy.mechanisms.kary_rr import privatize_many, rr_params, rr_transition_matrix


def privacy_ratio(epsilon: float, k: int) -> float:
    """Maximum P(M(x)=r)/P(M(x')=r) over k-ary RR inputs and reports."""
    A = rr_transition_matrix(epsilon, k)
    ratios = []
    for x in range(k):
        for xp in range(k):
            if x == xp:
                continue
            for r in range(k):
                ratios.append(A[x, r] / max(A[xp, r], 1e-300))
    return float(max(ratios))


def expected_report_distribution(theta: np.ndarray, epsilon: float, k: int) -> np.ndarray:
    """Return E[reported distribution] = theta A."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape != (k,):
        raise ValueError(f"theta must have shape ({k},)")
    if np.any(theta < 0) or not np.isclose(theta.sum(), 1.0):
        raise ValueError("theta must be a probability vector")
    return theta @ rr_transition_matrix(epsilon, k)


def analytic_debiased_variance(theta: np.ndarray, epsilon: float, n: int) -> np.ndarray:
    """Approximate category-wise variance of the unclipped RR inverse estimator.

    The inverse estimator is (reported_j - q)/(p-q).  For multinomial reported
    counts, Var(reported_j) = pi_j(1-pi_j)/n where pi = theta A.
    """
    k = int(np.asarray(theta).size)
    params = rr_params(epsilon, k)
    pi = expected_report_distribution(theta, epsilon, k)
    return pi * (1.0 - pi) / max(int(n), 1) / ((params.p - params.q) ** 2)


def monte_carlo_unbiasedness(
    theta: np.ndarray,
    *,
    epsilon: float,
    n: int,
    reps: int,
    seed: int = 123,
    clip: bool = False,
    renormalize: bool = False,
) -> dict[str, Any]:
    """Monte Carlo estimate of RR inverse bias and variance."""
    theta = np.asarray(theta, dtype=float)
    k = int(theta.size)
    rng = np.random.default_rng(seed)
    estimates = np.empty((int(reps), k), dtype=float)
    for r in range(int(reps)):
        true = rng.choice(k, size=int(n), p=theta)
        reported = privatize_many(true, epsilon=epsilon, k=k, rng=rng)
        estimates[r] = estimate_distribution(reported, epsilon, k, clip=clip, renormalize=renormalize)
    mean = estimates.mean(axis=0)
    variance = estimates.var(axis=0, ddof=1) if reps > 1 else np.zeros(k)
    return {
        "theta": theta.tolist(),
        "epsilon": float(epsilon),
        "n": int(n),
        "reps": int(reps),
        "clip": bool(clip),
        "renormalize": bool(renormalize),
        "mean_estimate": mean.tolist(),
        "bias": (mean - theta).tolist(),
        "max_abs_bias": float(np.max(np.abs(mean - theta))),
        "empirical_variance": variance.tolist(),
        "analytic_variance": analytic_debiased_variance(theta, epsilon, n).tolist(),
    }


def bootstrap_coverage(
    theta: np.ndarray,
    *,
    epsilon: float,
    n: int,
    reps: int,
    n_boot: int,
    seed: int = 456,
) -> dict[str, Any]:
    """Estimate percentile bootstrap marginal coverage for the RR inverse."""
    theta = np.asarray(theta, dtype=float)
    k = int(theta.size)
    rng = np.random.default_rng(seed)
    covered = np.zeros(k, dtype=int)
    for _ in range(int(reps)):
        true = rng.choice(k, size=int(n), p=theta)
        reported = privatize_many(true, epsilon=epsilon, k=k, rng=rng)
        lo, hi = bootstrap_ci(reported, epsilon, k, n_boot=int(n_boot), rng=rng)
        covered += ((lo <= theta) & (theta <= hi)).astype(int)
    return {
        "theta": theta.tolist(),
        "epsilon": float(epsilon),
        "n": int(n),
        "reps": int(reps),
        "n_boot": int(n_boot),
        "marginal_coverage": (covered / max(int(reps), 1)).tolist(),
        "min_marginal_coverage": float(np.min(covered / max(int(reps), 1))),
    }



def epsilon_k_grid_checks(*, quick: bool = False) -> list[dict[str, Any]]:
    """Validate RR privacy ratio and unbiased inverse over epsilon/k grid."""
    eps_grid = [0.2, 0.5, 1.0, 2.0] if quick else [0.1, 0.2, 0.5, 1.0, 2.0, 4.0]
    k_grid = [2, 5] if quick else [2, 3, 5, 7]
    rows: list[dict[str, Any]] = []
    for k in k_grid:
        theta = np.arange(k, 0, -1, dtype=float)
        theta = theta / theta.sum()
        for epsilon in eps_grid:
            ratio = privacy_ratio(float(epsilon), int(k))
            pi = expected_report_distribution(theta, float(epsilon), int(k))
            A = rr_transition_matrix(float(epsilon), int(k))
            recovered = np.linalg.solve(A.T, pi)
            rows.append(
                {
                    "k": int(k),
                    "epsilon": float(epsilon),
                    "privacy_ratio": float(ratio),
                    "exp_epsilon": float(np.exp(epsilon)),
                    "privacy_ratio_abs_error": float(abs(ratio - np.exp(epsilon))),
                    "inverse_recovery_l1": float(np.sum(np.abs(recovered - theta))),
                }
            )
    return rows


def clipping_bias_check(theta: np.ndarray, *, epsilon: float, n: int, reps: int, seed: int = 789) -> dict[str, Any]:
    """Compare unclipped inverse to clipped/renormalised estimator bias."""
    unclipped = monte_carlo_unbiasedness(theta, epsilon=epsilon, n=n, reps=reps, seed=seed, clip=False, renormalize=False)
    clipped = monte_carlo_unbiasedness(theta, epsilon=epsilon, n=n, reps=reps, seed=seed, clip=True, renormalize=True)
    return {
        "epsilon": float(epsilon),
        "n": int(n),
        "reps": int(reps),
        "unclipped_max_abs_bias": float(unclipped["max_abs_bias"]),
        "clipped_renormalized_max_abs_bias": float(clipped["max_abs_bias"]),
        "bias_increase_from_clipping": float(clipped["max_abs_bias"] - unclipped["max_abs_bias"]),
    }

def run_theory_validation(*, out_dir: Path, quick: bool = False) -> dict[str, Any]:
    """Run deterministic theory checks and write JSON/Markdown artefacts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    theta = np.array([0.42, 0.25, 0.18, 0.10, 0.05], dtype=float)
    epsilon = 1.0
    n = 900 if quick else 2500
    reps = 40 if quick else 200
    n_boot = 250 if quick else 1000
    ratio = privacy_ratio(epsilon, theta.size)
    grid_checks = epsilon_k_grid_checks(quick=quick)
    results = {
        "privacy_ratio": ratio,
        "exp_epsilon": float(np.exp(epsilon)),
        "privacy_ratio_matches_exp_epsilon": bool(np.isclose(ratio, np.exp(epsilon), rtol=1e-12)),
        "expected_report_distribution": expected_report_distribution(theta, epsilon, theta.size).tolist(),
        "unbiasedness": monte_carlo_unbiasedness(theta, epsilon=epsilon, n=n, reps=reps, seed=111),
        "coverage": bootstrap_coverage(theta, epsilon=epsilon, n=max(300, n // 2), reps=max(8, reps // 8), n_boot=n_boot, seed=222),
        "epsilon_k_grid": grid_checks,
        "max_grid_privacy_ratio_abs_error": float(max(row["privacy_ratio_abs_error"] for row in grid_checks)),
        "max_grid_inverse_recovery_l1": float(max(row["inverse_recovery_l1"] for row in grid_checks)),
        "clipping_bias": clipping_bias_check(theta, epsilon=0.5, n=max(300, n // 2), reps=max(20, reps // 2), seed=333),
    }
    (out_dir / "theory_validation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (out_dir / "theory_validation.md").write_text(_markdown_summary(results), encoding="utf-8")
    return results


def _markdown_summary(results: dict[str, Any]) -> str:
    u = results["unbiasedness"]
    c = results["coverage"]
    return "\n".join(
        [
            "# Theory validation run",
            "",
            f"- Max RR privacy ratio: `{results['privacy_ratio']:.6f}`; exp(epsilon): `{results['exp_epsilon']:.6f}`.",
            f"- Privacy-ratio check passed: `{results['privacy_ratio_matches_exp_epsilon']}`.",
            f"- Unclipped inverse max absolute Monte Carlo bias: `{u['max_abs_bias']:.6f}` over `{u['reps']}` repetitions.",
            f"- Bootstrap minimum marginal coverage: `{c['min_marginal_coverage']:.3f}` over `{c['reps']}` repetitions.",
            f"- Epsilon/k grid max privacy-ratio absolute error: `{results['max_grid_privacy_ratio_abs_error']:.3e}`.",
            f"- Epsilon/k grid max inverse-recovery L1 error: `{results['max_grid_inverse_recovery_l1']:.3e}`.",
            f"- Clipped+renormalised estimator max bias in the finite-sample check: `{results['clipping_bias']['clipped_renormalized_max_abs_bias']:.6f}`.",
            "",
            "The unclipped inverse is the theoretical unbiased estimator. Production reports clip and renormalise finite-sample estimates, so clipped outputs trade exact unbiasedness for valid probability vectors. The grid checks validate this relationship over multiple epsilon and category-count settings.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run FairVote-AI RR theory validation checks.")
    parser.add_argument("--out_dir", type=Path, default=Path("evidence/final/theory"))
    parser.add_argument("--quick", action="store_true", help="Use a small deterministic run suitable for CI.")
    args = parser.parse_args(argv)
    run_theory_validation(out_dir=args.out_dir, quick=args.quick)
    print(f"Wrote theory validation artefacts to {args.out_dir}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
