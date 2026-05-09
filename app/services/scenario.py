"""Pure synthetic-scenario helpers used by the Streamlit demo."""

from __future__ import annotations

from typing import Any

import numpy as np

from fairvote.privacy.mechanisms.kary_rr import privatize_one


def party_labels_for_k(k: int) -> list[str]:
    if k == 5:
        return ["Labour", "Conservative", "Reform", "LibDem", "Green"]
    return [f"Party_{i}" for i in range(k)]


def region_labels(n: int) -> list[str]:
    base = ["London", "North", "Midlands", "South", "East", "Wales", "Scotland", "NI"]
    return base[:n] if n <= len(base) else base + [f"Region_{i}" for i in range(len(base), n)]


def age_labels(n: int) -> list[str]:
    base = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    return base[:n] if n <= len(base) else base + [f"Age_{i}" for i in range(len(base), n)]


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits)
    e = np.exp(x)
    return e / np.sum(e)


def prefs_for_cell(party_labels: list[str], region: str, age: str, rng: np.random.Generator) -> np.ndarray:
    """Structured-but-simple preference model for the scenario demo."""

    k = len(party_labels)
    if k == 5 and party_labels == ["Labour", "Conservative", "Reform", "LibDem", "Green"]:
        base = np.array([0.34, 0.30, 0.12, 0.08, 0.16], dtype=float)
        logits = np.log(base + 1e-12)

        if age in ("18-24", "25-34"):
            logits += np.array([0.10, -0.10, 0.05, 0.10, -0.05])
        elif age in ("55-64", "65+"):
            logits += np.array([-0.10, 0.15, 0.05, -0.10, 0.00])

        if region == "London":
            logits += np.array([0.10, -0.05, 0.10, 0.00, -0.05])
        if region == "North":
            logits += np.array([0.05, -0.05, 0.00, 0.00, 0.00])
        if region == "South":
            logits += np.array([-0.05, 0.10, 0.05, 0.00, -0.10])
        if region == "Scotland":
            logits += np.array([0.00, 0.00, 0.00, 0.00, 0.15])

        logits += rng.normal(0.0, 0.05, size=5)
        return softmax(logits)

    alpha = np.ones(k, dtype=float) * 3.0
    return rng.dirichlet(alpha)


def generate_population(
    regions: list[str], ages: list[str], total_pop: int, rng: np.random.Generator
) -> list[dict[str, str]]:
    """Generate a small population count table for region x age cells."""

    del rng  # kept for API stability and future stochastic population generation
    reg_w = np.linspace(1.3, 0.7, num=len(regions))
    reg_w = reg_w / reg_w.sum()
    age_w = np.linspace(1.2, 0.8, num=len(ages))
    age_w = age_w / age_w.sum()

    rows = []
    denom = float((reg_w.reshape(-1, 1) * age_w.reshape(1, -1)).sum())
    for r_i, r in enumerate(regions):
        for a_i, a in enumerate(ages):
            w = float(reg_w[r_i] * age_w[a_i])
            c = max(1, int(round(total_pop * w / denom)))
            rows.append({"region": r, "age_band": a, "count": str(c)})
    return rows


def sample_from_population(pop_rows: list[dict[str, str]], n: int, rng: np.random.Generator) -> list[tuple[str, str]]:
    keys = []
    weights = []
    for r in pop_rows:
        keys.append((str(r.get("region", "")).strip(), str(r.get("age_band", "")).strip()))
        try:
            c = float(r.get("count", "0"))
        except Exception:
            c = 0.0
        weights.append(max(0.0, c))
    w = np.asarray(weights, dtype=float)
    w = w / w.sum() if w.sum() > 0 else np.full(len(keys), 1.0 / len(keys))
    idx = rng.choice(len(keys), size=int(n), replace=True, p=w)
    return [keys[i] for i in idx]


def apply_nonresponse(region: str, age: str, base: float, rng: np.random.Generator) -> bool:
    """Return True if respondent remains in the sample."""

    nr = float(base)
    if age in ("18-24", "25-34"):
        nr += 0.12
    if age in ("65+", "55-64"):
        nr -= 0.05
    if region in ("London",):
        nr += 0.05
    if region in ("North", "Midlands"):
        nr += 0.03
    nr = float(np.clip(nr, 0.02, 0.60))
    return bool(rng.random() >= nr)


def apply_misreport(true_idx: int, k: int, honesty: float, rng: np.random.Generator) -> int:
    honesty = float(np.clip(honesty, 0.0, 1.0))
    if rng.random() < honesty:
        return int(true_idx)
    other = [j for j in range(k) if j != true_idx]
    return int(rng.choice(other))


def apply_shy_effect(
    true_idx: int, shy_idx: int, k: int, shy_base: float, epsilon: float, rng: np.random.Generator
) -> int:
    """Simple 'privacy helps' misreport model for one shy category."""

    if true_idx != shy_idx:
        return int(true_idx)
    p = float(shy_base) * (float(epsilon) / (float(epsilon) + 1.0))
    p = float(np.clip(p, 0.0, 0.95))
    if rng.random() >= p:
        return int(true_idx)
    other = [j for j in range(k) if j != true_idx]
    return int(rng.choice(other))


def generate_scenario_poll(
    *,
    scenario: str,
    n_resp: int,
    epsilon: float,
    k: int,
    n_regions: int,
    n_ages: int,
    total_pop: int,
    seed: int,
    nonresponse_base: float,
    shy_base: float,
    honesty: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    """Generate poll and population rows for the scenario simulator."""

    rng = np.random.default_rng(int(seed))
    parties = party_labels_for_k(int(k))
    regions = region_labels(int(n_regions))
    ages = age_labels(int(n_ages))
    pop_rows = generate_population(regions, ages, int(total_pop), rng)
    demos = sample_from_population(pop_rows, int(n_resp), rng)
    shy_idx = 1 if "Conservative" in parties else 0

    poll_rows: list[dict[str, str]] = []
    for reg, age in demos:
        if scenario == "nonresponse" and not apply_nonresponse(reg, age, float(nonresponse_base), rng):
            continue
        p = prefs_for_cell(parties, reg, age, rng)
        true_idx = int(rng.choice(len(parties), p=p))

        declared_idx = true_idx
        if scenario == "misreport":
            declared_idx = apply_misreport(true_idx, len(parties), float(honesty), rng)
        elif scenario == "shy_privacy_helps":
            declared_idx = apply_shy_effect(true_idx, shy_idx, len(parties), float(shy_base), float(epsilon), rng)

        reported_idx = privatize_one(declared_idx, float(epsilon), len(parties), rng)
        poll_rows.append(
            {
                "region": reg,
                "age_band": age,
                "true_choice": parties[true_idx],
                "declared_choice": parties[declared_idx],
                "reported_choice": parties[reported_idx],
                "epsilon": str(float(epsilon)),
            }
        )

    meta = {"parties": parties, "regions": regions, "ages": ages, "shy_idx": shy_idx}
    return poll_rows, pop_rows, meta
