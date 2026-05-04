# experiments/generate_poll_csv.py
"""
Generate a realistic synthetic polling CSV suitable for the Streamlit "Upload & Estimate" demo.

This script creates TWO files:
  1) poll CSV: respondent rows with demographics + reported (privatised) choice
  2) population CSV: population counts by demographic group (for post-stratification)

Why this is realistic:
- Population has region x age-band structure (with unequal group sizes)
- True vote intention varies by region and age (via a softmax/logit model)
- Optional nonresponse bias (response rates vary by group)
- Optional "shy voter" misreporting before privacy, with honesty depending on epsilon
- Randomized Response (k-ary RR) applied to produce reported_choice for privacy

Example:
  python -m experiments.generate_poll_csv --out_dir app/data --scenario shy_privacy_helps --epsilon 0.5 --n 5000

Then in Streamlit:
  - Upload poll CSV
  - Upload population CSV (optional but recommended)
  - Set response column = reported_choice, group columns = region + age_band, epsilon = the same epsilon used here

Notes:
- If your project provides fairvote.privacy.privatize_many (recommended), we'll use it.
  Otherwise we use a correct fallback RR implementation.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


# ----------------------------
# Optional integration with your library
# ----------------------------

def _try_import_privatize_many():
    try:
        from fairvote.privacy import privatize_many  # type: ignore
        return privatize_many
    except Exception:
        return None


# ----------------------------
# Synthetic population + preferences
# ----------------------------

REGIONS = ["London", "South East", "Midlands", "North West", "Scotland", "Wales"]
AGE_BANDS = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

# Region weights (roughly plausible; sum not required to be 1)
REGION_WEIGHTS = np.array([0.18, 0.20, 0.22, 0.20, 0.12, 0.08], dtype=float)
AGE_WEIGHTS = np.array([0.11, 0.16, 0.17, 0.18, 0.17, 0.21], dtype=float)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def _make_population_counts(
    *,
    total_pop: int,
    rng: np.random.Generator,
) -> Dict[Tuple[str, str], int]:
    """
    Create integer population counts for each (region, age_band) group.

    We build a region x age table by:
      - allocating total_pop across regions using REGION_WEIGHTS
      - within each region, allocating across age bands using AGE_WEIGHTS + region-specific jitter
    """
    total_pop = int(total_pop)
    if total_pop <= 0:
        raise ValueError("total_pop must be positive")

    # Allocate across regions
    rw = REGION_WEIGHTS / REGION_WEIGHTS.sum()
    region_counts = rng.multinomial(total_pop, rw)

    counts: Dict[Tuple[str, str], int] = {}
    for r_idx, region in enumerate(REGIONS):
        n_r = int(region_counts[r_idx])

        # Add mild region-specific age skew (Dirichlet around AGE_WEIGHTS)
        alpha = 50.0 * (AGE_WEIGHTS / AGE_WEIGHTS.sum())
        # small region-specific perturbation
        jitter = rng.normal(0.0, 0.02, size=len(AGE_BANDS))
        base = np.clip(AGE_WEIGHTS + jitter, 0.01, None)
        base = base / base.sum()

        # Mix: mostly base, slightly Dirichlet noise
        dirichlet = rng.dirichlet(alpha)
        aw = 0.75 * base + 0.25 * dirichlet
        aw = aw / aw.sum()

        age_counts = rng.multinomial(n_r, aw)
        for a_idx, age in enumerate(AGE_BANDS):
            counts[(region, age)] = int(age_counts[a_idx])

    # Fix any rounding drift (should be exact from multinomial, but keep safe)
    drift = total_pop - sum(counts.values())
    if drift != 0:
        # Adjust the largest group
        g_max = max(counts, key=lambda k: counts[k])
        counts[g_max] += drift
    return counts


def _make_true_preferences(
    *,
    k: int,
    rng: np.random.Generator,
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Create per-group true vote distributions p(true_choice | region, age_band).

    We use:
      logits = global_logits + region_effect[r] + age_effect[a]
      p = softmax(logits)

    This produces structured, realistic variation across demographics.
    """
    k = int(k)
    if k < 2:
        raise ValueError("k must be >= 2")

    # global base: not uniform; plausible multi-party split
    global_logits = rng.normal(0.0, 0.6, size=k)

    # region and age effects: small, structured
    region_effects = {r: rng.normal(0.0, 0.35, size=k) for r in REGIONS}
    age_effects = {a: rng.normal(0.0, 0.30, size=k) for a in AGE_BANDS}

    prefs: Dict[Tuple[str, str], np.ndarray] = {}
    for r in REGIONS:
        for a in AGE_BANDS:
            logits = global_logits + region_effects[r] + age_effects[a]
            p = _softmax(logits)
            prefs[(r, a)] = p
    return prefs


# ----------------------------
# Bias models (nonresponse + shy voter)
# ----------------------------

def _response_rate_for_group(region: str, age_band: str) -> float:
    """
    Nonresponse model: response probabilities vary by group.
    (You can justify this in your writeup: harder-to-reach groups, survey fatigue, etc.)
    """
    # baseline
    p = 0.75

    # Younger and very old respond less
    if age_band in ("18-24", "65+"):
        p -= 0.18
    if age_band == "25-34":
        p -= 0.08

    # Region-specific friction
    if region in ("London", "South East"):
        p -= 0.06
    if region in ("Scotland",):
        p -= 0.03

    return float(np.clip(p, 0.25, 0.95))


def _honesty_from_epsilon(epsilon: float) -> float:
    """
    Model "privacy helps honesty":
    - smaller epsilon = more privacy = higher honesty
    - larger epsilon = less privacy = lower honesty

    Returns honesty h in (0,1).
    """
    eps = float(epsilon)
    # Tuned to give a clear gradient across eps in your typical sweep (0.2..2.0)
    h_min = 0.55
    h_max = 0.92
    beta = 0.55
    h = h_min + (h_max - h_min) * math.exp(-eps / beta)
    return float(np.clip(h, 0.0, 1.0))


def _is_shy_group(region: str, age_band: str) -> bool:
    """
    Which demographic groups exhibit "shy voter" behaviour.
    Keep this simple but defensible.
    """
    return (age_band in ("55-64", "65+")) and (region in ("London", "South East", "Midlands"))


def _apply_shy_misreport(
    *,
    true_choice: int,
    p_true: np.ndarray,
    shy_category: int,
    honesty: float,
    rng: np.random.Generator,
) -> int:
    """
    If the respondent's true choice is the shy category, they may misreport (pre-privacy)
    with probability (1 - honesty). If they misreport, they choose among other categories
    proportional to p_true (excluding the shy category).
    """
    if int(true_choice) != int(shy_category):
        return int(true_choice)

    h = float(honesty)
    if rng.random() < h:
        return int(true_choice)

    k = int(p_true.size)
    probs = np.asarray(p_true, dtype=float).copy()
    probs[int(shy_category)] = 0.0
    s = float(probs.sum())
    if s <= 0:
        # fallback: uniform among non-shy
        probs = np.ones(k, dtype=float)
        probs[int(shy_category)] = 0.0
        probs /= probs.sum()
    else:
        probs /= s
    return int(rng.choice(k, p=probs))


# ----------------------------
# Randomized Response (k-ary RR)
# ----------------------------

def _rr_privatize_fallback(
    stated: np.ndarray,
    *,
    epsilon: float,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    k-ary RR fallback:
      with prob p_keep, report stated
      else report uniformly among other k-1 categories
    """
    eps = float(epsilon)
    k = int(k)
    p_keep = math.exp(eps) / (math.exp(eps) + (k - 1))
    stated = np.asarray(stated, dtype=int)

    n = int(stated.size)
    out = stated.copy()

    flip = rng.random(n) > p_keep
    if np.any(flip):
        # For each flipped, pick a random category != stated
        for i in np.where(flip)[0]:
            t = int(stated[i])
            # sample from {0..k-1} \ {t}
            r = int(rng.integers(0, k - 1))
            out[i] = r if r < t else r + 1
    return out


def privatize_many(
    stated: np.ndarray,
    *,
    epsilon: float,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Prefer fairvote.privacy.privatize_many if present; otherwise use fallback.
    Your codebase uses epsilon= (not eps=), per your tests.
    """
    fn = _try_import_privatize_many()
    if fn is not None:
        # Your project likely uses numpy's global RNG; set it for determinism if caller desires.
        return np.asarray(fn(stated, epsilon=epsilon, k=k), dtype=int)
    return _rr_privatize_fallback(stated, epsilon=epsilon, k=k, rng=rng)


# ----------------------------
# Main generation
# ----------------------------

@dataclass
class Generated:
    poll_rows: List[dict]
    pop_rows: List[dict]


def generate(
    *,
    n: int,
    k: int,
    epsilon: float,
    scenario: str,
    shy_category: int,
    include_truth: bool,
    include_stated: bool,
    seed: int,
    total_pop: int,
) -> Generated:
    """Generate a synthetic poll CSV with evaluation-only truth columns."""

    rng = np.random.default_rng(int(seed))

    pop_counts = _make_population_counts(total_pop=int(total_pop), rng=rng)
    prefs = _make_true_preferences(k=int(k), rng=rng)

    # Build group sampling frame
    groups = list(pop_counts.keys())
    counts = np.array([pop_counts[g] for g in groups], dtype=float)
    weights = counts / counts.sum()

    poll_rows: List[dict] = []

    # For nonresponse: accept/reject with group response rate
    def accept(region: str, age: str) -> bool:
        if scenario == "nonresponse":
            return rng.random() < _response_rate_for_group(region, age)
        return True

    # For shy voter scenario
    honesty = _honesty_from_epsilon(epsilon) if scenario == "shy_privacy_helps" else 1.0

    # Sample until we have n respondents after nonresponse
    while len(poll_rows) < int(n):
        g_idx = int(rng.choice(len(groups), p=weights))
        region, age = groups[g_idx]

        if not accept(region, age):
            continue

        p_true = prefs[(region, age)]
        true_choice = int(rng.choice(k, p=p_true))

        # stated_choice begins as the true choice. In shy-voter scenarios,
        # the misreport model may alter it before the RR privacy step.
        stated_choice = true_choice
        if scenario == "shy_privacy_helps" and _is_shy_group(region, age):
            stated_choice = _apply_shy_misreport(
                true_choice=true_choice,
                p_true=p_true,
                shy_category=int(shy_category),
                honesty=float(honesty),
                rng=rng,
            )

        poll_rows.append(
            {
                "respondent_id": len(poll_rows) + 1,
                "region": region,
                "age_band": age,
                "true_choice": true_choice,
                "stated_choice": stated_choice,
            }
        )

    # Apply k-ary Randomized Response to stated_choice to get reported_choice.
    # This is the single privacy-preserving transformation; everything after
    # this point sees only the privatised value.
    stated_arr = np.array([r["stated_choice"] for r in poll_rows], dtype=int)
    reported_arr = privatize_many(stated_arr, epsilon=float(epsilon), k=int(k), rng=rng)

    for i, rep in enumerate(reported_arr.tolist()):
        poll_rows[i]["reported_choice"] = int(rep)
        poll_rows[i]["epsilon"] = float(epsilon)
        poll_rows[i]["k"] = int(k)
        poll_rows[i]["scenario"] = str(scenario)
        if scenario == "shy_privacy_helps":
            poll_rows[i]["shy_category"] = int(shy_category)
            poll_rows[i]["honesty"] = float(honesty)

    # Drop truth/stated if not requested.  In a real deployment, the server
    # would never receive these columns; they exist here only for offline
    # evaluation of the debiasing estimator.
    if not include_truth:
        for r in poll_rows:
            r.pop("true_choice", None)
    if not include_stated:
        for r in poll_rows:
            r.pop("stated_choice", None)

    pop_rows: List[dict] = []
    for (region, age), c in sorted(pop_counts.items(), key=lambda x: (x[0][0], x[0][1])):
        pop_rows.append({"region": region, "age_band": age, "count": int(c)})

    return Generated(poll_rows=poll_rows, pop_rows=pop_rows)


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    p = argparse.ArgumentParser(description="Generate a synthetic poll CSV + population CSV for Streamlit demo.")
    p.add_argument("--out_dir", type=str, default="app/data", help="Output directory to write CSV files.")
    p.add_argument("--n", type=int, default=5000, help="Number of respondent rows to generate (after nonresponse).")
    p.add_argument("--k", type=int, default=5, help="Number of categories (parties).")
    p.add_argument("--epsilon", type=float, default=1.0, help="Randomized Response epsilon used to privatize.")
    p.add_argument(
        "--scenario",
        type=str,
        default="no_bias",
        choices=["no_bias", "nonresponse", "shy_privacy_helps"],
        help="Bias scenario to generate.",
    )
    p.add_argument("--shy_category", type=int, default=0, help="Which category is 'shy' (only used in shy_privacy_helps).")
    p.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility.")
    p.add_argument("--total_pop", type=int, default=200_000, help="Synthetic population size used for population.csv.")
    p.add_argument("--include_truth", action="store_true", help="Include true_choice column (synthetic evaluation only).")
    p.add_argument("--include_stated", action="store_true", help="Include stated_choice column (pre-privacy).")
    p.add_argument("--poll_name", type=str, default=None, help="Override poll CSV filename.")
    p.add_argument("--pop_name", type=str, default="population.csv", help="Population CSV filename.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)

    # default poll filename encodes scenario and epsilon
    poll_name = args.poll_name
    if poll_name is None:
        ts = np.datetime64("now").astype(str).replace(":", "").replace("-", "")
        poll_name = f"poll_{args.scenario}_eps{args.epsilon:g}_n{args.n}_{ts}.csv"

    gen = generate(
        n=int(args.n),
        k=int(args.k),
        epsilon=float(args.epsilon),
        scenario=str(args.scenario),
        shy_category=int(args.shy_category),
        include_truth=bool(args.include_truth),
        include_stated=bool(args.include_stated),
        seed=int(args.seed),
        total_pop=int(args.total_pop),
    )

    poll_path = out_dir / poll_name
    pop_path = out_dir / str(args.pop_name)

    write_csv(poll_path, gen.poll_rows)
    write_csv(pop_path, gen.pop_rows)

    print("Wrote poll CSV:", poll_path)
    print("Wrote population CSV:", pop_path)
    print("\nSuggested Streamlit settings:")
    print(" - response column: reported_choice")
    print(" - group columns: region, age_band")
    print(" - epsilon:", args.epsilon)
    if args.include_truth:
        print(" - truth column (optional): true_choice")

    if args.scenario == "shy_privacy_helps":
        print(" - shy_category:", args.shy_category)
        print(" - honesty used (derived from epsilon):", _honesty_from_epsilon(float(args.epsilon)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
