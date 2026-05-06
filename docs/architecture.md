# System Architecture

FairVote-AI is organised as a respondent-facing collection system plus an analysis pipeline for privacy-aware inference and evaluation.

The project should be described as **AI-assisted**, not as an entirely AI-based system. The privacy mechanism is k-ary Randomized Response. The AI component is the RR-aware Neural MRP model in the inference layer.

## Layer diagram

```text
┌─────────────────────────────────────────────────────────────────────┐
│                     ANALYST DASHBOARD / EXPERIMENTS                 │
│  Streamlit app, CLI experiments, CSV/JSONL outputs, plots           │
│  app/streamlit_app.py, experiments/                                 │
├─────────────────────────────────────────────────────────────────────┤
│                     PRIVACY-AWARE INFERENCE LAYER                   │
│  RR debiasing                                                       │
│  Linear RR-aware MRP                                                │
│  Misreport-aware RR-MRP                                             │
│  RR-aware Neural MRP                                                │
│  fairvote/privacy/estimators.py, fairvote/inference/mrp/            │
├─────────────────────────────────────────────────────────────────────┤
│                         PRIVACY MECHANISM                           │
│  Local Differential Privacy: k-ary Randomized Response                                │
│  Central DP baseline: Laplace mechanism on aggregate counts         │
│  fairvote/privacy/mechanisms/                                       │
├─────────────────────────────────────────────────────────────────────┤
│                SIMULATION, SAMPLING, AND BIAS MODELS                │
│  Synthetic population, sampling frames, nonresponse, misreporting   │
│  fairvote/simulation/                                               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       RESPONDENT APPLICATION                         │
│  Browser-side k-ary RR, Flask API, JSONL response storage            │
│  respondent/                                                         │
└─────────────────────────────────────────────────────────────────────┘
```

## Application-mode data flow

```text
Respondent browser
  1. User selects a true answer locally.
  2. Browser applies k-ary Randomized Response.
  3. Browser sends only the privatized reported answer and demographics.

Flask server
  4. Rejects payloads containing `true_answer` or other true/raw answer fields.
  5. Stores perturbed_answer and demographics in JSONL.

Analyst layer
  6. Estimates aggregate vote shares from privatized reported answers.
  7. Optionally runs MRP/post-stratification if demographic features and population cells are supplied.
```

## Simulation-mode data flow

```text
Synthetic population
  -> sample respondents
  -> apply nonresponse or misreporting scenario
  -> apply Randomized Response
  -> fit estimators using reported labels only
  -> evaluate against simulator truth after fitting
```

The simulator contains true labels because it creates the population. Those labels are for evaluation. They are not required in real polling mode and are not used as training targets for RR-aware Neural MRP.

## Module map

### `fairvote/privacy/`

| File | Purpose |
|---|---|
| `mechanisms/kary_rr.py` | k-ary Randomized Response local-DP mechanism |
| `mechanisms/laplace_mechanism.py` | Central-DP aggregate-count baseline |
| `estimators.py` | RR debiasing estimator and bootstrap confidence intervals |

Randomized Response is not AI. It is the Local Differential Privacy mechanism and the observation channel used by the inference models.

### `fairvote/inference/mrp/`

| File | Purpose |
|---|---|
| `design.py` | Categorical design-matrix builders and design metadata |
| `linear.py` | **Canonical** RR-aware linear poststratification/MRP-style estimator: regularised multinomial regression through the RR observation channel |
| `poststratify.py` | Population-cell weighting, subgroup estimates, and post-stratification validation |
| `diagnostics.py` | Fit diagnostics such as final loss, runtime, steps, and optional loss history |
| `model.py` | Thin compatibility wrapper for old experiment imports; no separate linear model logic |
| `rr_mrp_fit.py` | Thin compatibility wrapper for old dashboard imports; no separate linear model logic |
| `fairvote/inference/mrp/neural/` | Optional PyTorch RR-aware Neural MRP package; `rr_neural_mrp.py` is a compatibility facade |
| `neural.py` | Thin aliases for RR-aware Neural MRP classes |
| `misreport_rr.py` | Misreport-aware RR-MRP extension for simulated shy-voter settings |
| `learned_misreport_rr.py` | Learned honesty/misreport extension for MRP experiments |

The canonical linear model is documented in `docs/mrp_canonical.md`. It is MRP-style regularised regression plus post-stratification, not a full hierarchical Bayesian MRP sampler.

### `fairvote/simulation/`

| File | Purpose |
|---|---|
| `population.py` | Synthetic population generation and population truth summaries |
| `sampling.py` | Simple random, stratified, and biased-frame sampling |
| `bias_models.py` | Nonresponse, shy-voter, and privacy-helps-honesty scenarios |

### `fairvote/metrics/`

| File | Purpose |
|---|---|
| `group_metrics.py` | Overall, winner, worst-group, weighted, p90, RMSE, and error-ratio metrics |

### `experiments/`

| File | Purpose |
|---|---|
| `experiments/legacy/sweep_eps.py` | Local Differential Privacy vs central-DP epsilon/sample-size sweeps |
| `mrp_vs_baselines.py` | Thin CLI wrapper for the main estimator comparison |
| `pipeline/` | Modular experiment engine: typed config, sampling, perturbation, method registry, metrics, summaries, manifests and plotting |
| `experiments/legacy/evaluate_neural_mrp.py` | Legacy compatibility helper for older neural-only checks; not the canonical final-evidence path |
| `experiments/legacy/sensitivity_analysis.py` | Sensitivity checks under different population/bias assumptions |

## RR-aware Neural MRP

The neural model estimates a latent vote distribution from features:

```text
P_theta(true=t | x) = softmax(neural_network(x))
```

It is trained through the known Randomized Response channel:

```text
P_theta(reported=r | x)
  = sum_t P_theta(true=t | x) P_RR(reported=r | true=t)
```

The loss is the negative log-likelihood of the observed privatized reports. This is the key privacy-compatible design choice: training uses reported labels, not true labels.

The model is post-stratified by applying `predict_true_proba()` to population cells and weighting by cell counts. This gives an aggregate estimate of latent vote shares.

## Why baselines remain necessary

The neural model is compared to RR debiasing and RR-aware linear poststratification/MRP because the extra complexity is not automatically justified. RR-aware Neural MRP may help when demographic effects are nonlinear, but it can also overfit or underperform when sample size is small, epsilon is low, or the demographic signal is weak.

The architecture treats RR-aware Neural MRP as one estimator in a comparison pipeline, not as the default answer. Final claims should come from a current `experiments.mrp_vs_baselines --preset final_evidence` run and should be interpreted through `summary_with_ci.csv`, `paired_comparisons.csv`, `ablations.csv` and `runtime_profile.csv`.

## Main limitations reflected in the architecture

- Local Differential Privacy protects answer values but does not provide anonymity by itself.
- Demographic fields are not randomized by the current respondent client.
- Fairness metrics audit subgroup error; they do not guarantee fair estimates.
- Synthetic experiments test behaviour under chosen assumptions; they do not prove real election accuracy.
