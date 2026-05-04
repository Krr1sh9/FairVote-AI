# experiments/mrp_vs_baselines.py
"""Compare RR debiasing, MRP baselines, and neural RR-MRP on simulations.

The experiment generates synthetic populations, applies sampling/bias and
k-ary Randomized Response, then evaluates estimators with aggregate, subgroup,
winner-correctness, and runtime metrics. Synthetic true labels are used only for
evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np

from fairvote.inference.mrp import RRMultinomialModel, build_design_matrix
from fairvote.inference.mrp.misreport_rr import MisreportRRMultinomialModel, identity_misreport
from fairvote.inference.mrp.learned_misreport_rr import LearnedShyMisreportRRMultinomialModel
from fairvote.metrics.group_metrics import correct_winner, worst_group_l1, weighted_group_l1, p90_group_l1
from fairvote.privacy import estimate_distribution, privatize_many
from fairvote.simulation.bias_models import (
    apply_misreporting,
    apply_nonresponse,
    build_shy_model_from_epsilon,
    make_default_feature_nonresponse_profile,
    make_shy_supporter_model,
)
from fairvote.simulation.population import (
    make_realistic_uk_like_population,
    subgroup_true_distribution,
    poststrat_table,
)
from fairvote.simulation.sampling import (
    biased_frame_sample,
    simple_random_sample,
    stratified_sample,
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _ts_run_dir(base: Path, name: str) -> Path:
    """Create a unique timestamped output directory.

    On Windows it's easy to double-launch the script within the same second (e.g., clicking a UI button twice),
    which would otherwise collide. We resolve collisions by adding a numeric suffix.
    """
    base.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    for i in range(0, 1000):
        suffix = "" if i == 0 else f"_{i:03d}"
        run_dir = base / f"{ts}_{name}{suffix}"
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            (run_dir / "plots").mkdir(parents=True, exist_ok=True)
            return run_dir
        except FileExistsError:
            continue

    # Extremely unlikely fallback: include microseconds
    ts2 = datetime.now().strftime("%Y-%m-%d_%H%M%S_%f")
    run_dir = base / f"{ts2}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_eps_list(s: str) -> List[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("Provide --eps like '0.2,0.5,1.0'.")
    if any(e <= 0 for e in vals):
        raise ValueError("All epsilons must be > 0.")
    return vals


def _l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(a - b)))


def _linf(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def _distribution_from_labels(labels: np.ndarray, k: int) -> np.ndarray:
    """Empirical distribution of integer labels in [0, k-1]."""
    y = np.asarray(labels, dtype=int)
    if y.ndim != 1:
        raise ValueError("labels must be a 1D array")
    if y.size == 0:
        return np.full(k, 1.0 / k, dtype=float)
    if np.any((y < 0) | (y >= k)):
        raise ValueError(f"labels must be in [0, {k - 1}]")
    counts = np.bincount(y, minlength=k).astype(float)
    total = float(counts.sum())
    if total <= 0.0:
        return np.full(k, 1.0 / k, dtype=float)
    return counts / total


def _parse_hidden_layers(s: str) -> Tuple[int, ...]:
    """Parse CLI hidden layer sizes such as '32' or '64,32'."""
    raw = str(s).strip()
    if raw.lower() in {"", "none", "linear"}:
        return tuple()
    out: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        width = int(item)
        if width <= 0:
            raise ValueError("Neural hidden layer sizes must be positive integers.")
        out.append(width)
    if not out:
        raise ValueError("Provide --neural_hidden_layers like '32' or '64,32'.")
    return tuple(out)


def _experiment_methods(enable_neural: bool) -> List[str]:
    methods = [
        "raw_reported_distribution",
        "baseline_rr_debias",
        "mrp_rr_poststrat",
        "mrp_misreport_rr_poststrat",
        "mrp_learned_misreport_rr_poststrat",
    ]
    if enable_neural:
        methods.append("neural_rr_mrp")
    return methods


def _require_rr_neural_mrp_model():
    """Import the optional PyTorch neural MRP only when the experiment needs it."""
    try:
        from fairvote.inference.mrp.rr_neural_mrp import RRNeuralMRPModel
    except Exception as exc:  # pragma: no cover - depends on optional torch environment
        raise RuntimeError(
            "Neural RR-MRP is enabled, but the PyTorch model could not be imported. "
            'Install the final supported project dependencies with `pip install -e ".[dev,ai,streamlit,respondent]"` or rerun with --disable_neural.'
        ) from exc
    return RRNeuralMRPModel


def _base_result_row(
    *,
    scenario: str,
    trial: int,
    eps: float,
    method: str,
    n_effective: int,
    estimate_overall: np.ndarray,
    truth_overall: np.ndarray,
    runtime_sec: float,
) -> dict:
    """Common overall metrics for one estimator."""
    return {
        "scenario": scenario,
        "trial": trial,
        "epsilon": float(eps),
        "method": method,
        "n_effective": int(n_effective),
        "skipped": 0,
        "runtime_sec": float(runtime_sec),
        "winner_correct": int(correct_winner(estimate_overall, truth_overall)),
        "overall_l1": _l1(estimate_overall, truth_overall),
        "overall_linf": _linf(estimate_overall, truth_overall),
        "overall_mae": _mae(estimate_overall, truth_overall),
    }




# ---------------- Misreport model conversion helpers ----------------
def _misreport_to_matrix(mis, k: int, *, mc_samples_per_true: int = 20000) -> np.ndarray:
    """
    Convert various misreport model representations into a (k, k) float matrix.

    Interpretation:
      M[t, s] = P(stated=s | true=t)

    Supports:
      - numpy array already
      - objects exposing .matrix / .M / .P (attr may be array or callable)
      - objects exposing .to_matrix() / .as_matrix()
      - objects exposing callable prob-style methods: prob(t,s), p(t,s), transition_prob(t,s)
      - fallback: empirical estimation using apply_misreporting(...) for unknown MisreportModel objects

    The empirical fallback is deterministic given fixed RNG seed inside this function.
    """
    if mis is None:
        return identity_misreport(k)

    # Already a matrix
    if isinstance(mis, np.ndarray):
        return np.asarray(mis, dtype=float)

    # Common attribute names
    for attr in ("matrix", "M", "P", "transition", "T"):
        if hasattr(mis, attr):
            M = getattr(mis, attr)
            M = M() if callable(M) else M
            return np.asarray(M, dtype=float)

    # Common container names (rows / row_probs / probs)
    for attr in ("rows", "row_probs", "probs", "prob_rows", "row_distributions"):
        if hasattr(mis, attr):
            M = getattr(mis, attr)
            M = M() if callable(M) else M
            return np.asarray(M, dtype=float)

    # Common method names
    for meth in ("to_matrix", "as_matrix", "get_matrix"):
        if hasattr(mis, meth) and callable(getattr(mis, meth)):
            return np.asarray(getattr(mis, meth)(), dtype=float)

    # Prob-style callable: prob(true, stated)
    for meth in ("prob", "p", "transition_prob", "p_true_to_stated", "prob_true_to_stated"):
        if hasattr(mis, meth) and callable(getattr(mis, meth)):
            f = getattr(mis, meth)
            M = np.zeros((k, k), dtype=float)
            for t in range(k):
                for s in range(k):
                    M[t, s] = float(f(t, s))
            return M

    # -----------------------------
    # Fallback: empirical estimation
    # -----------------------------
    try:
        rng = np.random.default_rng(424242)
        M = np.zeros((k, k), dtype=float)
        n = int(mc_samples_per_true)
        n = max(n, 2000)

        for t in range(k):
            truth = np.full(n, t, dtype=int)
            stated = apply_misreporting(truth, mis, rng=rng)
            counts = np.bincount(stated.astype(int), minlength=k).astype(float)
            row = counts / max(float(counts.sum()), 1.0)
            M[t, :] = row

        # numeric hygiene
        M = np.clip(M, 0.0, None)
        rs = M.sum(axis=1, keepdims=True)
        M = M / np.maximum(rs, 1e-12)
        return M
    except Exception as e:
        raise TypeError(
            f"Don't know how to convert misreport model of type {type(mis)} to a (k,k) matrix. "
            "Tried common attributes/methods and empirical fallback via apply_misreporting(...). "
            f"Empirical fallback failed with: {e}"
        ) from e



def _truth_overall(pop, k: int) -> np.ndarray:
    counts = np.bincount(pop.true_categories, minlength=k).astype(float)
    return counts / counts.sum()


def _estimate_subgroup_baseline(
    reported: np.ndarray, eps: float, k: int, mask: np.ndarray
) -> np.ndarray:
    if not np.any(mask):
        return np.full(k, 1.0 / k, dtype=float)
    return estimate_distribution(reported[mask], eps, k)


def _feature_masses_from_cells(
    *,
    cells: np.ndarray,
    counts: np.ndarray,
    by: Sequence[str],
    feature: str,
    feature_levels: Dict[str, List[str]],
) -> Dict[str, float]:
    """
    Compute population masses for each level of a feature using the poststrat table.
    Returns level_name -> proportion (sums to 1 across all levels).
    """
    by_list = list(by)
    if feature not in by_list:
        return {}
    j = by_list.index(feature)
    levels = feature_levels[feature]

    total = float(np.sum(counts))
    if total <= 0:
        return {lvl: 0.0 for lvl in levels}

    out: Dict[str, float] = {}
    for lvl_idx, lvl_name in enumerate(levels):
        m = (cells[:, j] == lvl_idx)
        out[lvl_name] = float(np.sum(counts[m]) / total)
    return out


def _poststrat_from_cell_theta(
    cell_theta: np.ndarray,
    cells: np.ndarray,
    counts: np.ndarray,
    *,
    by: Sequence[str],
    feature_levels: Dict[str, List[str]],
    include_features: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, Dict[str, np.ndarray]]]:
    """
    Fast poststratification once you already have:
      - cell_theta (C, K) = predicted true-category distribution per cell
      - cells (C, d) = integer-coded feature levels for each cell
      - counts (C,) = population counts per cell
    """
    total = float(np.sum(counts))
    overall = (counts[:, None] * cell_theta).sum(axis=0) / total

    by_feature: Dict[str, Dict[str, np.ndarray]] = {}
    by_list = list(by)

    for feat in include_features:
        if feat not in by_list:
            continue
        j = by_list.index(feat)
        levels = feature_levels[feat]
        out: Dict[str, np.ndarray] = {}
        for lvl_idx, lvl_name in enumerate(levels):
            m = (cells[:, j] == lvl_idx)
            w = counts[m]
            tot = float(np.sum(w))
            if tot <= 0:
                continue
            est = (w[:, None] * cell_theta[m]).sum(axis=0) / tot
            out[lvl_name] = est
        by_feature[feat] = out

    return overall.astype(float), by_feature


def _plot_summary(run_dir: Path, summary_rows: List[dict]) -> None:
    # Optional plotting.
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if not summary_rows:
        return

    scenarios = sorted({r["scenario"] for r in summary_rows if r.get("n_rows", 0) and int(r["n_rows"]) > 0})
    methods = sorted({r["method"] for r in summary_rows if r.get("n_rows", 0) and int(r["n_rows"]) > 0})

    for scenario in scenarios:
        rows_s = [r for r in summary_rows if r["scenario"] == scenario and int(r.get("n_rows", 0) or 0) > 0]
        if not rows_s:
            continue

        eps = sorted({float(r["epsilon"]) for r in rows_s})

        # Overall L1 vs epsilon for each method
        plt.figure()
        for m in methods:
            ys = []
            xs = []
            for e in eps:
                hit = [r for r in rows_s if float(r["epsilon"]) == e and r["method"] == m]
                if not hit:
                    continue
                xs.append(e)
                ys.append(float(hit[0].get("mean_overall_l1", "nan")))
            if xs:
                plt.plot(xs, ys, marker="o", label=m)
        plt.xscale("log")
        plt.xlabel("epsilon (log scale)")
        plt.ylabel("Mean overall L1 error vs population truth")
        plt.title(f"MRP vs Baselines — Overall Error ({scenario})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "plots" / f"{scenario}_overall_l1.png", dpi=200)
        plt.close()

        # Worst region L1 major vs epsilon
        plt.figure()
        for m in methods:
            ys = []
            xs = []
            for e in eps:
                hit = [r for r in rows_s if float(r["epsilon"]) == e and r["method"] == m]
                if not hit:
                    continue
                xs.append(e)
                ys.append(float(hit[0].get("mean_worst_region_l1_major", "nan")))
            if xs:
                plt.plot(xs, ys, marker="o", label=m)
        plt.xscale("log")
        plt.xlabel("epsilon (log scale)")
        plt.ylabel("Mean worst-region (major) L1 error vs population truth")
        plt.title(f"MRP vs Baselines — Worst Region (Major) ({scenario})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "plots" / f"{scenario}_worst_region_l1_major.png", dpi=200)
        plt.close()


# -----------------------------------------------------------------------------
# Experiment core
# -----------------------------------------------------------------------------

def run_experiment(
    *,
    k: int,
    eps_list: List[float],
    scenarios: List[str],
    population_n: int,
    n_sample: int,
    trials: int,
    seed: int,
    sampling: str,
    strata: List[str],
    allocation: str,
    min_per_stratum: int,
    biased_feature: str,
    biased_multipliers: Dict[str, float],
    feature_order: List[str],
    # bias params
    shy_category: int,
    shy_honesty: float,
    # mrp hyperparams
    mrp_steps: int,
    mrp_lr: float,
    mrp_l2: float,
    mrp_batch_size: int,
    verbose_every: int,
    # neural MRP hyperparams
    enable_neural: bool,
    neural_hidden_layers: Tuple[int, ...],
    neural_steps: int,
    neural_lr: float,
    neural_batch_size: int,
    neural_seed: int,
    neural_dropout: float,
    neural_weight_decay: float,
    # fairness metric params
    major_mass: float,
) -> Tuple[List[dict], List[dict]]:
    if enable_neural:
        RRNeuralMRPModel = _require_rr_neural_mrp_model()
    else:
        RRNeuralMRPModel = None

    pop = make_realistic_uk_like_population(population_n, k, seed=seed)

    truth_overall = _truth_overall(pop, k)
    truth_region = subgroup_true_distribution(pop, "region")
    truth_age = subgroup_true_distribution(pop, "age_group")

    # Poststrat table must include ALL features used in the model
    by = list(feature_order)
    cells, cell_counts, _level_names = poststrat_table(pop, by=by)

    # Feature masses (population shares) for fairness metrics
    region_masses = _feature_masses_from_cells(
        cells=cells, counts=cell_counts.astype(float), by=by, feature="region", feature_levels=pop.feature_levels
    )
    age_masses = _feature_masses_from_cells(
        cells=cells, counts=cell_counts.astype(float), by=by, feature="age_group", feature_levels=pop.feature_levels
    )

    # Pre-encode poststrat design matrix (fast)
    cell_features = {f: cells[:, j].astype(int) for j, f in enumerate(by)}
    X_cells, _design_info_cells = build_design_matrix(
        cell_features,
        pop.feature_levels,
        feature_order=feature_order,
        intercept=True,
    )

    include_features = ["region", "age_group"]

    rows: List[dict] = []

    for scenario in scenarios:
        if scenario not in {"no_bias", "nonresponse", "shy_fixed", "shy_privacy_helps"}:
            raise ValueError(f"Unknown scenario: {scenario}")

        for t in range(trials):
            rng_t = np.random.default_rng(seed + 10_000 * t + 1337)

            # ---- Sampling stage ----
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
                raise ValueError("sampling must be one of: srs, stratified, biased")

            # ---- Nonresponse scenario ----
            if scenario in {"nonresponse", "shy_privacy_helps"}:
                feature_profile = make_default_feature_nonresponse_profile()
                sample = apply_nonresponse(sample, pop, rng=rng_t, feature_profile=feature_profile)

            n_eff = int(sample.idx.size)
            if n_eff < max(80, int(0.05 * n_sample)):
                for eps in eps_list:
                    for method in _experiment_methods(enable_neural):
                        rows.append({
                            "scenario": scenario,
                            "trial": t,
                            "epsilon": float(eps),
                            "method": method,
                            "n_effective": n_eff,
                            "skipped": 1,
                        })
                continue

            true_cats = sample.true_categories.astype(int)

            # Features for model training
            train_features = {f: sample.features[f].astype(int) for f in feature_order}
            X_train, _design_info_train = build_design_matrix(
                train_features,
                pop.feature_levels,
                feature_order=feature_order,
                intercept=True,
            )

            # For baseline subgroup estimates (from respondent sample slices)
            region_vals = sample.features["region"].astype(int)
            age_vals = sample.features["age_group"].astype(int)
            region_levels = pop.feature_levels["region"]
            age_levels = pop.feature_levels["age_group"]

            for eps in eps_list:
                rng_eps = np.random.default_rng(seed + 10_000 * t + int(round(eps * 10_000)) + 7)

                # ---- Misreport scenario (pre-LDP) ----
                # We also keep the misreport object around for the misreport-aware model.
                mis = None
                if scenario in {"no_bias", "nonresponse"}:
                    stated = true_cats
                elif scenario == "shy_fixed":
                    mis = make_shy_supporter_model(k, shy_category, honesty=shy_honesty)
                    stated = apply_misreporting(true_cats, mis, rng=rng_eps)
                elif scenario == "shy_privacy_helps":
                    mis = build_shy_model_from_epsilon(k, shy_category, eps)
                    stated = apply_misreporting(true_cats, mis, rng=rng_eps)
                else:
                    raise ValueError("Invalid scenario")

                # ---- Apply RR (client-side) ----
                reported = privatize_many(stated, eps, k, rng=rng_eps)

                # =========================
                # Method 0: Raw reported distribution (no RR correction)
                # =========================
                method_start = time.perf_counter()
                theta_hat_raw = _distribution_from_labels(reported, k)

                raw_region_est: Dict[str, np.ndarray] = {}
                raw_region_l1s = []
                for lvl_idx, lvl_name in enumerate(region_levels):
                    if lvl_name not in truth_region:
                        continue
                    msk = (region_vals == lvl_idx)
                    est = _distribution_from_labels(reported[msk], k)
                    raw_region_est[lvl_name] = est
                    raw_region_l1s.append(_l1(est, truth_region[lvl_name]))

                raw_age_est: Dict[str, np.ndarray] = {}
                raw_age_l1s = []
                for lvl_idx, lvl_name in enumerate(age_levels):
                    if lvl_name not in truth_age:
                        continue
                    msk = (age_vals == lvl_idx)
                    est = _distribution_from_labels(reported[msk], k)
                    raw_age_est[lvl_name] = est
                    raw_age_l1s.append(_l1(est, truth_age[lvl_name]))

                raw_row = _base_result_row(
                    scenario=scenario,
                    trial=t,
                    eps=eps,
                    method="raw_reported_distribution",
                    n_effective=n_eff,
                    estimate_overall=theta_hat_raw,
                    truth_overall=truth_overall,
                    runtime_sec=time.perf_counter() - method_start,
                )
                raw_row.update({
                    "worst_region_l1": float(np.max(raw_region_l1s)) if raw_region_l1s else float("nan"),
                    "avg_region_l1": float(np.mean(raw_region_l1s)) if raw_region_l1s else float("nan"),
                    "worst_age_l1": float(np.max(raw_age_l1s)) if raw_age_l1s else float("nan"),
                    "avg_age_l1": float(np.mean(raw_age_l1s)) if raw_age_l1s else float("nan"),
                    "worst_region_l1_major": worst_group_l1(
                        raw_region_est, truth_region, group_masses=region_masses, min_mass=major_mass
                    ),
                    "worst_age_l1_major": worst_group_l1(
                        raw_age_est, truth_age, group_masses=age_masses, min_mass=major_mass
                    ),
                    "weighted_region_l1": weighted_group_l1(raw_region_est, truth_region, group_masses=region_masses),
                    "weighted_age_l1": weighted_group_l1(raw_age_est, truth_age, group_masses=age_masses),
                    "p90_region_l1_major": p90_group_l1(
                        raw_region_est, truth_region, group_masses=region_masses, min_mass=major_mass
                    ),
                    "p90_age_l1_major": p90_group_l1(
                        raw_age_est, truth_age, group_masses=age_masses, min_mass=major_mass
                    ),
                })
                rows.append(raw_row)

                # =========================
                # Method A: Baseline RR debias (overall + subgroup-by-slicing)
                # =========================
                method_start = time.perf_counter()
                theta_hat_base = estimate_distribution(reported, eps, k)

                # subgroup baseline estimates
                region_est: Dict[str, np.ndarray] = {}
                region_l1s = []
                for lvl_idx, lvl_name in enumerate(region_levels):
                    if lvl_name not in truth_region:
                        continue
                    msk = (region_vals == lvl_idx)
                    est = _estimate_subgroup_baseline(reported, eps, k, msk)
                    region_est[lvl_name] = est
                    region_l1s.append(_l1(est, truth_region[lvl_name]))

                age_est: Dict[str, np.ndarray] = {}
                age_l1s = []
                for lvl_idx, lvl_name in enumerate(age_levels):
                    if lvl_name not in truth_age:
                        continue
                    msk = (age_vals == lvl_idx)
                    est = _estimate_subgroup_baseline(reported, eps, k, msk)
                    age_est[lvl_name] = est
                    age_l1s.append(_l1(est, truth_age[lvl_name]))

                # fairness metrics (baseline)
                worst_region_major = worst_group_l1(region_est, truth_region, group_masses=region_masses, min_mass=major_mass)
                worst_age_major = worst_group_l1(age_est, truth_age, group_masses=age_masses, min_mass=major_mass)

                weighted_region = weighted_group_l1(region_est, truth_region, group_masses=region_masses)
                weighted_age = weighted_group_l1(age_est, truth_age, group_masses=age_masses)

                p90_region_major = p90_group_l1(region_est, truth_region, group_masses=region_masses, min_mass=major_mass)
                p90_age_major = p90_group_l1(age_est, truth_age, group_masses=age_masses, min_mass=major_mass)

                rows.append({
                    "scenario": scenario,
                    "trial": t,
                    "epsilon": float(eps),
                    "method": "baseline_rr_debias",
                    "n_effective": n_eff,
                    "skipped": 0,
                    "runtime_sec": time.perf_counter() - method_start,
                    "winner_correct": int(correct_winner(theta_hat_base, truth_overall)),

                    "overall_l1": _l1(theta_hat_base, truth_overall),
                    "overall_linf": _linf(theta_hat_base, truth_overall),
                    "overall_mae": _mae(theta_hat_base, truth_overall),

                    # legacy subgroup metrics (raw across all groups present in sample)
                    "worst_region_l1": float(np.max(region_l1s)) if region_l1s else float("nan"),
                    "avg_region_l1": float(np.mean(region_l1s)) if region_l1s else float("nan"),
                    "worst_age_l1": float(np.max(age_l1s)) if age_l1s else float("nan"),
                    "avg_age_l1": float(np.mean(age_l1s)) if age_l1s else float("nan"),

                    # new fairness metrics (population-mass-aware / robust)
                    "worst_region_l1_major": worst_region_major,
                    "worst_age_l1_major": worst_age_major,
                    "weighted_region_l1": weighted_region,
                    "weighted_age_l1": weighted_age,
                    "p90_region_l1_major": p90_region_major,
                    "p90_age_l1_major": p90_age_major,
                })

                # =========================
                # Method B: RR-aware MRP + poststratification
                # =========================
                method_start = time.perf_counter()
                model = RRMultinomialModel(k, l2=mrp_l2, seed=seed + 999 + t)
                model.fit(
                    X_train,
                    reported,
                    eps,
                    lr=mrp_lr,
                    steps=mrp_steps,
                    batch_size=mrp_batch_size,
                    verbose_every=verbose_every,
                )

                cell_theta = model.predict_theta(X_cells)
                theta_hat_mrp, by_feat = _poststrat_from_cell_theta(
                    cell_theta,
                    cells,
                    cell_counts.astype(float),
                    by=by,
                    feature_levels=pop.feature_levels,
                    include_features=include_features,
                )

                mrp_region_est = by_feat.get("region", {})
                mrp_age_est = by_feat.get("age_group", {})

                mrp_region_l1s = [
                    _l1(est, truth_region[lvl])
                    for (lvl, est) in mrp_region_est.items()
                    if lvl in truth_region
                ]
                mrp_age_l1s = [
                    _l1(est, truth_age[lvl])
                    for (lvl, est) in mrp_age_est.items()
                    if lvl in truth_age
                ]

                mrp_worst_region_major = worst_group_l1(mrp_region_est, truth_region, group_masses=region_masses, min_mass=major_mass)
                mrp_worst_age_major = worst_group_l1(mrp_age_est, truth_age, group_masses=age_masses, min_mass=major_mass)

                mrp_weighted_region = weighted_group_l1(mrp_region_est, truth_region, group_masses=region_masses)
                mrp_weighted_age = weighted_group_l1(mrp_age_est, truth_age, group_masses=age_masses)

                mrp_p90_region_major = p90_group_l1(mrp_region_est, truth_region, group_masses=region_masses, min_mass=major_mass)
                mrp_p90_age_major = p90_group_l1(mrp_age_est, truth_age, group_masses=age_masses, min_mass=major_mass)

                rows.append({
                    "scenario": scenario,
                    "trial": t,
                    "epsilon": float(eps),
                    "method": "mrp_rr_poststrat",
                    "n_effective": n_eff,
                    "skipped": 0,
                    "runtime_sec": time.perf_counter() - method_start,
                    "winner_correct": int(correct_winner(theta_hat_mrp, truth_overall)),

                    "overall_l1": _l1(theta_hat_mrp, truth_overall),
                    "overall_linf": _linf(theta_hat_mrp, truth_overall),
                    "overall_mae": _mae(theta_hat_mrp, truth_overall),

                    # legacy subgroup metrics (raw)
                    "worst_region_l1": float(np.max(mrp_region_l1s)) if mrp_region_l1s else float("nan"),
                    "avg_region_l1": float(np.mean(mrp_region_l1s)) if mrp_region_l1s else float("nan"),
                    "worst_age_l1": float(np.max(mrp_age_l1s)) if mrp_age_l1s else float("nan"),
                    "avg_age_l1": float(np.mean(mrp_age_l1s)) if mrp_age_l1s else float("nan"),

                    # new fairness metrics
                    "worst_region_l1_major": mrp_worst_region_major,
                    "worst_age_l1_major": mrp_worst_age_major,
                    "weighted_region_l1": mrp_weighted_region,
                    "weighted_age_l1": mrp_weighted_age,
                    "p90_region_l1_major": mrp_p90_region_major,
                    "p90_age_l1_major": mrp_p90_age_major,
                })

                # =========================
                # Method C: Misreport-aware RR-MRP + poststrat (NEW)
                #   TRUE -> STATED (misreport) -> REPORTED (RR/LDP)
                # =========================
                method_start = time.perf_counter()
                # Convert the misreport model into the transition matrix expected by the estimator.
                mis_mat = _misreport_to_matrix(mis, k)

                model2 = MisreportRRMultinomialModel(k, l2=mrp_l2, seed=seed + 1999 + t, misreport=mis_mat)
                model2.fit(
                    X_train,
                    reported,
                    eps,
                    lr=mrp_lr,
                    steps=mrp_steps,
                    batch_size=mrp_batch_size,
                    verbose_every=verbose_every,
                )

                cell_theta2 = model2.predict_theta(X_cells)
                theta_hat_mrp2, by_feat2 = _poststrat_from_cell_theta(
                    cell_theta2,
                    cells,
                    cell_counts.astype(float),
                    by=by,
                    feature_levels=pop.feature_levels,
                    include_features=include_features,
                )

                mrp2_region_est = by_feat2.get("region", {})
                mrp2_age_est = by_feat2.get("age_group", {})

                mrp2_region_l1s = [
                    _l1(est, truth_region[lvl])
                    for (lvl, est) in mrp2_region_est.items()
                    if lvl in truth_region
                ]
                mrp2_age_l1s = [
                    _l1(est, truth_age[lvl])
                    for (lvl, est) in mrp2_age_est.items()
                    if lvl in truth_age
                ]

                mrp2_worst_region_major = worst_group_l1(mrp2_region_est, truth_region, group_masses=region_masses, min_mass=major_mass)
                mrp2_worst_age_major = worst_group_l1(mrp2_age_est, truth_age, group_masses=age_masses, min_mass=major_mass)

                mrp2_weighted_region = weighted_group_l1(mrp2_region_est, truth_region, group_masses=region_masses)
                mrp2_weighted_age = weighted_group_l1(mrp2_age_est, truth_age, group_masses=age_masses)

                mrp2_p90_region_major = p90_group_l1(mrp2_region_est, truth_region, group_masses=region_masses, min_mass=major_mass)
                mrp2_p90_age_major = p90_group_l1(mrp2_age_est, truth_age, group_masses=age_masses, min_mass=major_mass)

                rows.append({
                    "scenario": scenario,
                    "trial": t,
                    "epsilon": float(eps),
                    "method": "mrp_misreport_rr_poststrat",
                    "n_effective": n_eff,
                    "skipped": 0,
                    "runtime_sec": time.perf_counter() - method_start,
                    "winner_correct": int(correct_winner(theta_hat_mrp2, truth_overall)),

                    "overall_l1": _l1(theta_hat_mrp2, truth_overall),
                    "overall_linf": _linf(theta_hat_mrp2, truth_overall),
                    "overall_mae": _mae(theta_hat_mrp2, truth_overall),

                    "worst_region_l1": float(np.max(mrp2_region_l1s)) if mrp2_region_l1s else float("nan"),
                    "avg_region_l1": float(np.mean(mrp2_region_l1s)) if mrp2_region_l1s else float("nan"),
                    "worst_age_l1": float(np.max(mrp2_age_l1s)) if mrp2_age_l1s else float("nan"),
                    "avg_age_l1": float(np.mean(mrp2_age_l1s)) if mrp2_age_l1s else float("nan"),

                    "worst_region_l1_major": mrp2_worst_region_major,
                    "worst_age_l1_major": mrp2_worst_age_major,
                    "weighted_region_l1": mrp2_weighted_region,
                    "weighted_age_l1": mrp2_weighted_age,
                    "p90_region_l1_major": mrp2_p90_region_major,
                    "p90_age_l1_major": mrp2_p90_age_major,
                })

                # =========================
                # Method D: Learned shy-misreport RR-MRP + poststrat (NEW)
                #   Learns honesty from privatized data (no oracle misreport model)
                # =========================
                method_start = time.perf_counter()
                model3 = LearnedShyMisreportRRMultinomialModel(
                    k=k,
                    shy_category=shy_category,
                    l2=mrp_l2,
                    seed=seed + 2999 + t,
                    honesty_init=0.80,
                    honesty_lr=0.02,
                )
                model3.fit(
                    X_train,
                    reported,
                    eps,
                    lr=mrp_lr,
                    steps=mrp_steps,
                    batch_size=mrp_batch_size,
                    verbose_every=verbose_every,
                )

                cell_theta3 = model3.predict_theta(X_cells)
                theta_hat_mrp3, by_feat3 = _poststrat_from_cell_theta(
                    cell_theta3,
                    cells,
                    cell_counts.astype(float),
                    by=by,
                    feature_levels=pop.feature_levels,
                    include_features=include_features,
                )

                mrp3_region_est = by_feat3.get("region", {})
                mrp3_age_est = by_feat3.get("age_group", {})

                mrp3_region_l1s = [
                    _l1(est, truth_region[lvl])
                    for (lvl, est) in mrp3_region_est.items()
                    if lvl in truth_region
                ]
                mrp3_age_l1s = [
                    _l1(est, truth_age[lvl])
                    for (lvl, est) in mrp3_age_est.items()
                    if lvl in truth_age
                ]

                mrp3_worst_region_major = worst_group_l1(
                    mrp3_region_est, truth_region, group_masses=region_masses, min_mass=major_mass
                )
                mrp3_worst_age_major = worst_group_l1(
                    mrp3_age_est, truth_age, group_masses=age_masses, min_mass=major_mass
                )

                mrp3_weighted_region = weighted_group_l1(mrp3_region_est, truth_region, group_masses=region_masses)
                mrp3_weighted_age = weighted_group_l1(mrp3_age_est, truth_age, group_masses=age_masses)

                mrp3_p90_region_major = p90_group_l1(
                    mrp3_region_est, truth_region, group_masses=region_masses, min_mass=major_mass
                )
                mrp3_p90_age_major = p90_group_l1(
                    mrp3_age_est, truth_age, group_masses=age_masses, min_mass=major_mass
                )

                rows.append({
                    "scenario": scenario,
                    "trial": t,
                    "epsilon": float(eps),
                    "method": "mrp_learned_misreport_rr_poststrat",
                    "n_effective": n_eff,
                    "skipped": 0,
                    "runtime_sec": time.perf_counter() - method_start,
                    "winner_correct": int(correct_winner(theta_hat_mrp3, truth_overall)),
                    "learned_honesty": model3.learned_honesty(),

                    "overall_l1": _l1(theta_hat_mrp3, truth_overall),
                    "overall_linf": _linf(theta_hat_mrp3, truth_overall),
                    "overall_mae": _mae(theta_hat_mrp3, truth_overall),

                    "worst_region_l1": float(np.max(mrp3_region_l1s)) if mrp3_region_l1s else float("nan"),
                    "avg_region_l1": float(np.mean(mrp3_region_l1s)) if mrp3_region_l1s else float("nan"),
                    "worst_age_l1": float(np.max(mrp3_age_l1s)) if mrp3_age_l1s else float("nan"),
                    "avg_age_l1": float(np.mean(mrp3_age_l1s)) if mrp3_age_l1s else float("nan"),

                    "worst_region_l1_major": mrp3_worst_region_major,
                    "worst_age_l1_major": mrp3_worst_age_major,
                    "weighted_region_l1": mrp3_weighted_region,
                    "weighted_age_l1": mrp3_weighted_age,
                    "p90_region_l1_major": mrp3_p90_region_major,
                    "p90_age_l1_major": mrp3_p90_age_major,
                })

                # =========================
                # Method E: Neural RR-aware MRP + poststrat
                #   Learns latent P(true | x) from privatized RR reports only.
                # =========================
                if enable_neural:
                    if RRNeuralMRPModel is None:  # defensive; should be set above when enabled
                        raise RuntimeError("RRNeuralMRPModel was not imported despite enable_neural=True")

                    method_start = time.perf_counter()
                    neural_model = RRNeuralMRPModel(
                        k=k,
                        epsilon=eps,
                        hidden_layers=neural_hidden_layers,
                        dropout=neural_dropout,
                        weight_decay=neural_weight_decay,
                        seed=int(neural_seed) + 3999 + t,
                    )
                    neural_info = neural_model.fit(
                        X_train,
                        reported,
                        lr=neural_lr,
                        steps=neural_steps,
                        batch_size=neural_batch_size,
                        verbose_every=verbose_every,
                    )

                    cell_theta_neural = neural_model.predict_true_proba(X_cells)
                    theta_hat_neural, by_feat_neural = _poststrat_from_cell_theta(
                        cell_theta_neural,
                        cells,
                        cell_counts.astype(float),
                        by=by,
                        feature_levels=pop.feature_levels,
                        include_features=include_features,
                    )

                    neural_region_est = by_feat_neural.get("region", {})
                    neural_age_est = by_feat_neural.get("age_group", {})

                    neural_region_l1s = [
                        _l1(est, truth_region[lvl])
                        for (lvl, est) in neural_region_est.items()
                        if lvl in truth_region
                    ]
                    neural_age_l1s = [
                        _l1(est, truth_age[lvl])
                        for (lvl, est) in neural_age_est.items()
                        if lvl in truth_age
                    ]

                    neural_worst_region_major = worst_group_l1(
                        neural_region_est, truth_region, group_masses=region_masses, min_mass=major_mass
                    )
                    neural_worst_age_major = worst_group_l1(
                        neural_age_est, truth_age, group_masses=age_masses, min_mass=major_mass
                    )

                    neural_weighted_region = weighted_group_l1(
                        neural_region_est, truth_region, group_masses=region_masses
                    )
                    neural_weighted_age = weighted_group_l1(
                        neural_age_est, truth_age, group_masses=age_masses
                    )

                    neural_p90_region_major = p90_group_l1(
                        neural_region_est, truth_region, group_masses=region_masses, min_mass=major_mass
                    )
                    neural_p90_age_major = p90_group_l1(
                        neural_age_est, truth_age, group_masses=age_masses, min_mass=major_mass
                    )

                    rows.append({
                        "scenario": scenario,
                        "trial": t,
                        "epsilon": float(eps),
                        "method": "neural_rr_mrp",
                        "n_effective": n_eff,
                        "skipped": 0,
                        "runtime_sec": time.perf_counter() - method_start,
                        "winner_correct": int(correct_winner(theta_hat_neural, truth_overall)),
                        "neural_final_loss": neural_info.final_loss,

                        "overall_l1": _l1(theta_hat_neural, truth_overall),
                        "overall_linf": _linf(theta_hat_neural, truth_overall),
                        "overall_mae": _mae(theta_hat_neural, truth_overall),

                        "worst_region_l1": float(np.max(neural_region_l1s)) if neural_region_l1s else float("nan"),
                        "avg_region_l1": float(np.mean(neural_region_l1s)) if neural_region_l1s else float("nan"),
                        "worst_age_l1": float(np.max(neural_age_l1s)) if neural_age_l1s else float("nan"),
                        "avg_age_l1": float(np.mean(neural_age_l1s)) if neural_age_l1s else float("nan"),

                        "worst_region_l1_major": neural_worst_region_major,
                        "worst_age_l1_major": neural_worst_age_major,
                        "weighted_region_l1": neural_weighted_region,
                        "weighted_age_l1": neural_weighted_age,
                        "p90_region_l1_major": neural_p90_region_major,
                        "p90_age_l1_major": neural_p90_age_major,
                    })


    # ---- Summary aggregation ----
    summary: List[dict] = []

    def _stats(sub: List[dict], key: str) -> Tuple[float, float]:
        vals = np.array([r[key] for r in sub], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float("nan"), float("nan")
        mean = float(np.mean(vals))
        std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        return mean, std

    methods = _experiment_methods(enable_neural)

    for scenario in scenarios:
        for eps in eps_list:
            for method in methods:
                sub = [
                    r for r in rows
                    if r.get("scenario") == scenario
                    and float(r.get("epsilon")) == float(eps)
                    and r.get("method") == method
                    and r.get("skipped", 0) == 0
                ]
                if not sub:
                    summary.append({
                        "scenario": scenario,
                        "epsilon": float(eps),
                        "method": method,
                        "n_rows": 0,
                    })
                    continue

                m_overall_l1, s_overall_l1 = _stats(sub, "overall_l1")
                m_overall_mae, s_overall_mae = _stats(sub, "overall_mae")
                m_winner, s_winner = _stats(sub, "winner_correct")
                m_runtime, s_runtime = _stats(sub, "runtime_sec")
                m_wr, s_wr = _stats(sub, "worst_region_l1")
                m_wa, s_wa = _stats(sub, "worst_age_l1")

                m_wr_major, s_wr_major = _stats(sub, "worst_region_l1_major")
                m_wa_major, s_wa_major = _stats(sub, "worst_age_l1_major")

                m_w_region, s_w_region = _stats(sub, "weighted_region_l1")
                m_w_age, s_w_age = _stats(sub, "weighted_age_l1")

                m_p90_region, s_p90_region = _stats(sub, "p90_region_l1_major")
                m_p90_age, s_p90_age = _stats(sub, "p90_age_l1_major")

                summary.append({
                    "scenario": scenario,
                    "epsilon": float(eps),
                    "method": method,
                    "n_rows": len(sub),

                    "mean_overall_l1": m_overall_l1,
                    "std_overall_l1": s_overall_l1,
                    "mean_overall_mae": m_overall_mae,
                    "std_overall_mae": s_overall_mae,
                    "mean_winner_correct": m_winner,
                    "std_winner_correct": s_winner,
                    "mean_runtime_sec": m_runtime,
                    "std_runtime_sec": s_runtime,

                    # legacy
                    "mean_worst_region_l1": m_wr,
                    "std_worst_region_l1": s_wr,
                    "mean_worst_age_l1": m_wa,
                    "std_worst_age_l1": s_wa,

                    # new
                    "mean_worst_region_l1_major": m_wr_major,
                    "std_worst_region_l1_major": s_wr_major,
                    "mean_worst_age_l1_major": m_wa_major,
                    "std_worst_age_l1_major": s_wa_major,

                    "mean_weighted_region_l1": m_w_region,
                    "std_weighted_region_l1": s_w_region,
                    "mean_weighted_age_l1": m_w_age,
                    "std_weighted_age_l1": s_w_age,

                    "mean_p90_region_l1_major": m_p90_region,
                    "std_p90_region_l1_major": s_p90_region,
                    "mean_p90_age_l1_major": m_p90_age,
                    "std_p90_age_l1_major": s_p90_age,

                    "mean_n_effective": float(np.mean([r["n_effective"] for r in sub])),
                })

    return rows, summary


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _parse_multipliers(s: str) -> Dict[str, float]:
    s = s.strip()
    if not s:
        return {}
    out: Dict[str, float] = {}
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError("Use --multipliers like 'London=1.4,Wales=0.7'")
        name, val = item.split("=", 1)
        v = float(val.strip())
        if v <= 0:
            raise ValueError("Multipliers must be > 0.")
        out[name.strip()] = v
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Compare baseline RR debias vs RR-aware MRP + poststratification.")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--eps", type=str, default="0.2,0.5,1.0,2.0")
    p.add_argument("--scenarios", type=str, default="no_bias,nonresponse,shy_privacy_helps")
    p.add_argument("--population_n", type=int, default=100_000)
    p.add_argument("--n_sample", type=int, default=5_000)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out_dir", type=str, default="experiments/outputs")

    # Sampling
    p.add_argument("--sampling", choices=["srs", "stratified", "biased"], default="srs")
    p.add_argument("--strata", type=str, default="region")
    p.add_argument("--allocation", choices=["proportional", "equal", "sqrt"], default="proportional")
    p.add_argument("--min_per_stratum", type=int, default=0)
    p.add_argument("--biased_feature", type=str, default="region")
    p.add_argument("--multipliers", type=str, default="")

    # Bias params
    p.add_argument("--shy_category", type=int, default=0)
    p.add_argument("--shy_honesty", type=float, default=0.80)

    # MRP params
    p.add_argument("--features", type=str, default="region,age_group,education,gender,urbanicity")
    p.add_argument("--mrp_steps", type=int, default=1200)
    p.add_argument("--mrp_lr", type=float, default=0.05)
    p.add_argument("--mrp_l2", type=float, default=1.0)
    p.add_argument("--mrp_batch_size", type=int, default=2048)
    p.add_argument("--verbose_every", type=int, default=0, help="Set >0 to print training logs periodically.")

    # Neural RR-MRP params
    p.add_argument(
        "--disable_neural",
        action="store_true",
        help="Disable the PyTorch RR-aware neural MRP estimator.",
    )
    p.add_argument(
        "--neural_hidden_layers",
        type=str,
        default="16",
        help="Comma-separated hidden layer widths for neural RR-MRP, e.g. '16' or '32,16'.",
    )
    p.add_argument("--neural_steps", type=int, default=200)
    p.add_argument("--neural_lr", type=float, default=0.01)
    p.add_argument("--neural_batch_size", type=int, default=512)
    p.add_argument("--neural_seed", type=int, default=321)
    p.add_argument("--neural_dropout", type=float, default=0.0)
    p.add_argument("--neural_weight_decay", type=float, default=1e-4)

    # Fairness metric params
    p.add_argument(
        "--major_mass",
        type=float,
        default=0.02,
        help="Only treat groups with population share >= major_mass as 'major groups' for worst/p90 metrics.",
    )

    args = p.parse_args()

    eps_list = _parse_eps_list(args.eps)
    scenarios = _parse_list(args.scenarios)
    strata = _parse_list(args.strata)
    feature_order = _parse_list(args.features)
    neural_hidden_layers = _parse_hidden_layers(args.neural_hidden_layers)
    multipliers = _parse_multipliers(args.multipliers)

    base_out = Path(args.out_dir)
    base_out.mkdir(parents=True, exist_ok=True)
    run_dir = _ts_run_dir(base_out, "mrp_vs_baselines")

    config = {
        "k": args.k,
        "eps": eps_list,
        "scenarios": scenarios,
        "population_n": args.population_n,
        "n_sample": args.n_sample,
        "trials": args.trials,
        "seed": args.seed,

        "sampling": args.sampling,
        "strata": strata,
        "allocation": args.allocation,
        "min_per_stratum": args.min_per_stratum,
        "biased_feature": args.biased_feature,
        "multipliers": multipliers,

        "shy_category": args.shy_category,
        "shy_honesty": args.shy_honesty,

        "features": feature_order,
        "mrp_steps": args.mrp_steps,
        "mrp_lr": args.mrp_lr,
        "mrp_l2": args.mrp_l2,
        "mrp_batch_size": args.mrp_batch_size,
        "verbose_every": args.verbose_every,

        "enable_neural": not args.disable_neural,
        "neural_hidden_layers": list(neural_hidden_layers),
        "neural_steps": args.neural_steps,
        "neural_lr": args.neural_lr,
        "neural_batch_size": args.neural_batch_size,
        "neural_seed": args.neural_seed,
        "neural_dropout": args.neural_dropout,
        "neural_weight_decay": args.neural_weight_decay,

        "major_mass": args.major_mass,
        "methods": _experiment_methods(not args.disable_neural),
    }
    _write_json(run_dir / "config.json", config)

    rows, summary = run_experiment(
        k=args.k,
        eps_list=eps_list,
        scenarios=scenarios,
        population_n=args.population_n,
        n_sample=args.n_sample,
        trials=args.trials,
        seed=args.seed,

        sampling=args.sampling,
        strata=strata,
        allocation=args.allocation,
        min_per_stratum=args.min_per_stratum,
        biased_feature=args.biased_feature,
        biased_multipliers=multipliers,

        feature_order=feature_order,
        shy_category=args.shy_category,
        shy_honesty=args.shy_honesty,

        mrp_steps=args.mrp_steps,
        mrp_lr=args.mrp_lr,
        mrp_l2=args.mrp_l2,
        mrp_batch_size=args.mrp_batch_size,
        verbose_every=args.verbose_every,

        enable_neural=not args.disable_neural,
        neural_hidden_layers=neural_hidden_layers,
        neural_steps=args.neural_steps,
        neural_lr=args.neural_lr,
        neural_batch_size=args.neural_batch_size,
        neural_seed=args.neural_seed,
        neural_dropout=args.neural_dropout,
        neural_weight_decay=args.neural_weight_decay,

        major_mass=args.major_mass,
    )

    _write_csv(run_dir / "results_trials.csv", rows)
    _write_csv(run_dir / "summary.csv", summary)
    _plot_summary(run_dir, summary)

    print(f"Saved run to: {run_dir}")
    print(f"- {run_dir / 'summary.csv'}")
    print(f"- {run_dir / 'results_trials.csv'}")
    print(f"- {run_dir / 'plots'} (if matplotlib installed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())