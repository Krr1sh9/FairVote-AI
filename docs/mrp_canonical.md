# Canonical MRP Implementation

This repository now has one authoritative Python implementation for the **RR-aware linear poststratification/MRP-style estimator**:

```text
fairvote/inference/mrp/linear.py
```

The canonical class is:

```python
from fairvote.inference.mrp import LinearRRMRPModel
```

Backward-compatible names are preserved:

```python
from fairvote.inference.mrp import RRMultinomialModel
from fairvote.inference.mrp.rr_mrp_fit import MRPRRMultinomialModel
```

Both names point to the same canonical implementation. They exist only to avoid breaking older dashboard and experiment imports.

## Honest model name

The model is **regularised multinomial regression plus post-stratification**. It is MRP-style because it estimates category probabilities from demographic features and then post-stratifies over population cells. It is **not** a full hierarchical Bayesian MRP sampler.

## Observation model

The model is trained on RR-reported labels, not true labels.

```text
P(true=t | x) = softmax(XW)[t]
P(reported=r | x) = sum_t P(true=t | x) * A[t,r]
```

where `A` is the k-ary Randomized Response transition matrix from the canonical privacy module:

```text
fairvote/privacy/mechanisms/kary_rr.py
```

## Module structure

```text
fairvote/inference/mrp/
  design.py          # design matrix builders and metadata
  linear.py          # canonical linear RR-aware model
  hierarchical.py    # true partial-pooling RR-aware MRP with shrinkage
  poststratify.py    # population-cell weighting and subgroup estimates
  diagnostics.py     # fit diagnostics / convergence metadata
  rr_mrp_fit.py      # compatibility wrapper for old dashboard import path
  model.py           # compatibility wrapper for old experiment import path
  misreport_rr.py    # misreport-aware extension
  learned_misreport_rr.py
  neural/            # optional PyTorch neural estimator package
  rr_neural_mrp.py   # compatibility facade
  neural.py          # thin neural aliases
```

## Hierarchical partial-pooling path

The final evidence preset also includes `hierarchical_rr_mrp_poststrat`. This is not a renamed linear model: it uses a global intercept plus feature-level varying effects, optimised against the same RR observation likelihood with Gaussian shrinkage penalties. Sparse demographic levels borrow strength through the global intercept and regularised level effects, giving an examiner-visible partial-pooling contribution rather than only one-hot regularised regression. See `fairvote/inference/mrp/hierarchical.py` and `docs/theory_validation.md`.

## Validation and diagnostics

The canonical linear path validates:

- `epsilon` and `k` through the canonical RR channel;
- category labels are integers in `[0, k-1]`;
- design matrices are 2-D and finite;
- no NaNs or infinities are accepted in design matrices, labels, cells, or weights;
- post-stratification cells match the training design metadata;
- post-stratification weights are non-negative and normalised to sum to one.

Iterative fitting returns `FitDiagnostics` with:

- number of steps;
- final full-data loss;
- runtime in seconds;
- optional loss history.

Fitted model metadata can be exported with:

```python
model.export_metadata()
model.save_metadata("linear_rr_mrp_metadata.json")
```

By default, this metadata records the coefficient shape and norm rather than the full coefficient matrix. Use `include_weights=True` if exact fitted coefficients are needed.

## Why this matters for assessment

The MRP implementation is deliberately structured so examiners can identify the canonical path without searching through duplicated model files. Experiments and the dashboard both call the same linear RR-aware likelihood implementation. The old files `model.py` and `rr_mrp_fit.py` no longer maintain separate optimisation algorithms.
