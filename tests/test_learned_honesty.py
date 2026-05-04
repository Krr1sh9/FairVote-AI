# tests/test_learned_honesty.py
import numpy as np
import pytest


def _sample_from_row_stochastic(rng: np.random.Generator, row_probs: np.ndarray) -> int:
    row_probs = np.asarray(row_probs, dtype=float)
    row_probs = row_probs / row_probs.sum()
    return int(rng.choice(len(row_probs), p=row_probs))


def _simulate_true_to_reported(
    *,
    rng: np.random.Generator,
    k: int,
    eps: float,
    theta: np.ndarray,
    shy_category: int,
    honesty: float,
    n: int,
) -> np.ndarray:
    """
    Simulate:
      TRUE -> STATED (shy misreport) -> REPORTED (k-ary RR)
    using the same matrix forms used by the model.
    """
    from fairvote.inference.mrp.misreport_rr import rr_transition_matrix, shy_misreport_matrix

    theta = np.asarray(theta, dtype=float)
    theta = theta / theta.sum()

    M = shy_misreport_matrix(k, shy_category, honesty)   # TRUE -> STATED
    A = rr_transition_matrix(eps, k)                    # STATED -> REPORTED
    C = M @ A                                           # TRUE -> REPORTED

    true = rng.choice(k, size=n, p=theta)
    reported = np.empty(n, dtype=int)
    for i, t in enumerate(true):
        reported[i] = _sample_from_row_stochastic(rng, C[t])
    return reported


def test_learned_honesty_drops_in_shy_scenario():
    """
    If there is shy misreport (honesty < 1), the learned honesty should
    usually end up noticeably below 1.
    """
    try:
        from fairvote.inference.mrp.learned_misreport_rr import LearnedShyMisreportRRMultinomialModel
    except Exception:
        pytest.skip("LearnedShyMisreportRRMultinomialModel not available")

    rng = np.random.default_rng(7)
    k = 5
    eps = 0.8
    n = 4000
    shy = 0

    # Simple global model: intercept-only features
    X = np.ones((n, 1), dtype=float)

    theta = np.array([0.18, 0.12, 0.30, 0.22, 0.18], dtype=float)
    reported = _simulate_true_to_reported(
        rng=rng, k=k, eps=eps, theta=theta, shy_category=shy, honesty=0.60, n=n
    )

    model = LearnedShyMisreportRRMultinomialModel(
        k=k,
        shy_category=shy,
        l2=0.5,
        seed=42,
        honesty_init=0.90,
        honesty_lr=0.03,
    )

    model.fit(X, reported, eps, lr=0.08, steps=500, batch_size=512, verbose_every=0)
    h = model.learned_honesty()

    assert 0.0 < h < 1.0
    # Should move down meaningfully from ~1 in a shy setting
    assert h < 0.90


def test_learned_honesty_stays_high_when_no_misreport():
    """
    In a no-bias setting (honesty = 1), learned honesty should remain high.
    """
    try:
        from fairvote.inference.mrp.learned_misreport_rr import LearnedShyMisreportRRMultinomialModel
    except Exception:
        pytest.skip("LearnedShyMisreportRRMultinomialModel not available")

    rng = np.random.default_rng(9)
    k = 5
    eps = 0.8
    n = 4000
    shy = 0

    X = np.ones((n, 1), dtype=float)
    theta = np.array([0.18, 0.12, 0.30, 0.22, 0.18], dtype=float)

    reported = _simulate_true_to_reported(
        rng=rng, k=k, eps=eps, theta=theta, shy_category=shy, honesty=1.00, n=n
    )

    model = LearnedShyMisreportRRMultinomialModel(
        k=k,
        shy_category=shy,
        l2=0.5,
        seed=99,
        honesty_init=0.95,
        honesty_lr=0.03,
    )

    model.fit(X, reported, eps, lr=0.08, steps=500, batch_size=512, verbose_every=0)
    h = model.learned_honesty()

    assert 0.0 < h < 1.0
    # In no-bias, it should stay quite high (allowing some estimation noise)
    assert h > 0.85
