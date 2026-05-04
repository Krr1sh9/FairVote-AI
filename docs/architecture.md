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
│  Local DP: k-ary Randomized Response                                │
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

Randomized Response is not AI. It is the local privacy mechanism and the observation channel used by the inference models.

### `fairvote/inference/mrp/`

| File | Purpose |
|---|---|
| `model.py` | Linear multinomial MRP model used by experiments |
| `rr_mrp_fit.py` | Linear RR-aware MRP fitting used by the dashboard |
| `rr_neural_mrp.py` | PyTorch RR-aware Neural MRP model |
| `poststratify.py` | Post-stratification utilities |
| `misreport_rr.py` | Misreport-aware RR-MRP model for simulated shy-voter settings |
| `learned_misreport_rr.py` | Learned honesty/misreport extension for MRP experiments |

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
| `sweep_eps.py` | LDP vs central-DP epsilon/sample-size sweeps |
| `mrp_vs_baselines.py` | Main estimator comparison including neural RR-MRP |
| `evaluate_neural_mrp.py` | Dedicated experiment for deciding whether neural RR-MRP is justified |
| `sensitivity_analysis.py` | Sensitivity checks under different population/bias assumptions |

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

The neural model is compared to RR debiasing and linear RR-aware MRP because the extra complexity is not automatically justified. Neural MRP may help when demographic effects are nonlinear, but it can also overfit or underperform when sample size is small, epsilon is low, or the demographic signal is weak.

The included final evidence pack shows exactly why this comparison matters. In `experiments/outputs/final_neural_evidence/`, linear RR-aware MRP has lower mean overall L1 than neural RR-MRP (`0.176` versus `0.204`), while neural RR-MRP has lower averaged worst-group L1 but worse weighted group L1, p90 group L1, winner correctness, and runtime. The architecture therefore treats neural MRP as one estimator in a comparison pipeline, not as the default answer.

## Main limitations reflected in the architecture

- Local DP protects answer values but does not provide anonymity by itself.
- Demographic fields are not randomized by the current respondent client.
- Fairness metrics audit subgroup error; they do not guarantee fair estimates.
- Synthetic experiments test behaviour under chosen assumptions; they do not prove real election accuracy.
