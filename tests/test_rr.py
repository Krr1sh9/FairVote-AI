# tests/test_rr.py
import numpy as np
import pytest


def test_rr_transition_matrix_properties():
    """
    RR transition matrix must be row-stochastic and match the closed-form probs.
    """
    try:
        from fairvote.inference.mrp.misreport_rr import rr_transition_matrix
    except Exception:
        pytest.skip("rr_transition_matrix not available at fairvote.inference.mrp.misreport_rr")

    k = 7
    eps = 1.3
    A = rr_transition_matrix(eps, k)

    assert A.shape == (k, k)
    assert np.all(np.isfinite(A))
    assert np.all(A >= -1e-12)
    assert np.all(A <= 1.0 + 1e-12)

    row_sums = A.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-10)

    p_keep = np.exp(eps) / (np.exp(eps) + (k - 1))
    p_flip = (1.0 - p_keep) / (k - 1)

    diag = np.diag(A)
    assert np.allclose(diag, p_keep, atol=1e-10)

    off = A.copy()
    np.fill_diagonal(off, np.nan)
    off_vals = off[~np.isnan(off)]
    assert np.allclose(off_vals, p_flip, atol=1e-10)
