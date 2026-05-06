# Theory validation for FairVote-AI

This note is the assessment theoretical counterpart to `experiments/theory_validation.py` and `tests/test_theory_validation.py`.  It states exactly which claims the implementation relies on, the assumptions under which those claims are valid, and where the practical estimators deliberately depart from the ideal mathematics.

## 1. k-ary randomized response mechanism

For a true answer `Y ∈ {0, …, k-1}` and privacy parameter `ε > 0`, FairVote-AI uses symmetric k-ary randomized response:

```text
P(Z = z | Y = y) = p  if z = y
P(Z = z | Y = y) = q  if z ≠ y
p = exp(ε) / (exp(ε) + k - 1)
q = 1 / (exp(ε) + k - 1)
```

For any two possible true answers `y, y'` and any reported answer `z`,

```text
P(Z=z|Y=y) / P(Z=z|Y=y') ≤ p/q = exp(ε).
```

Therefore the client-side mechanism is ε-local differentially private. The browser and server tests enforce the implementation boundary: the server accepts only perturbed answers and rejects raw-answer fields before storage.

## 2. Reported distribution and unbiased inversion

Let `θ` be the true category distribution and `r` the reported distribution. With the symmetric RR transition matrix `A`, where diagonal entries are `p` and off-diagonal entries are `q`,

```text
r = A θ.
```

For category `j`, this reduces to

```text
r_j = q + (p - q) θ_j.
```

The debiased estimator is therefore

```text
θ̂_j = (r̂_j - q) / (p - q).
```

Before finite-sample clipping/renormalisation, this estimator is unbiased because `E[r̂_j] = r_j`.

## 3. Variance and finite-sample limits

For `n` independent respondents, `n r̂_j` is binomial with success probability `r_j`, so the ideal variance is

```text
Var(θ̂_j) = r_j(1-r_j) / (n (p-q)^2).
```

The variance inflates as `ε` decreases because `p-q` shrinks. This is the core privacy-accuracy tradeoff tested in the robustness preset. In practice FairVote-AI clips negative values and renormalises category probabilities. This makes the finite-sample estimator stable and valid as a probability vector, but it introduces small boundary bias at small `n` or small `ε`; the Monte Carlo theory-validation script measures this effect explicitly.

## 4. RR-aware likelihood

For MRP, the model predicts latent true-answer probabilities `π_i = P(Y_i | x_i)` for respondent features `x_i`. The observed likelihood is not `P(Y_i=z_i)`; it is the randomized-response channel likelihood:

```text
P(Z_i=z_i | x_i) = Σ_y P(Z_i=z_i | Y_i=y) P(Y_i=y | x_i).
```

This is the main methodological safeguard against training on perturbed responses as if they were truthful labels. `tests/test_mrp_canonical.py` checks this boundary.

## 5. Hierarchical partial-pooling MRP

The new deployable hierarchical estimator is `hierarchical_rr_mrp_poststrat`, implemented in `fairvote/inference/mrp/hierarchical.py`. For category `c` and respondent feature vector `x`, it uses

```text
η_c(x) = α_c + Σ_f β_{f, level_f(x), c}
π_c(x) = softmax_c(η(x)).
```

The objective is the negative RR-channel log-likelihood plus Gaussian shrinkage penalties:

```text
L = -Σ_i log P(Z_i | x_i) + λ_α ||α||² + λ_β Σ_f Σ_l ||β_{f,l}||².
```

Each feature-level effect is mean-centred after every update. This provides genuine partial pooling: sparse demographic cells borrow strength through the global intercept and the shrinkage prior instead of being estimated independently. Poststratification then averages model-predicted true probabilities over the target population cells.

## 6. Validation artefacts

Run:

```bash
python -m experiments.theory_validation --out_dir evidence/final/theory --quick
```

The script writes:

- `theory_validation.json`: machine-readable privacy-ratio, unbiasedness, variance, and interval-coverage checks.
- `theory_validation.md`: human-readable summary for the report appendix.

The unit tests verify the analytic privacy ratio, reported-distribution identity, Monte Carlo unbiasedness tolerance, and bootstrap interval coverage behaviour.

## 7. Claims not made

The local privacy guarantee applies to the randomized answer, not to demographic attributes. Demographics can still identify rare combinations, so the respondent server now exposes `/api/privacy-report` to highlight rare cells before individual-level export. This distinction is essential: FairVote-AI is a privacy-preserving polling prototype, not a blanket anonymisation system.
