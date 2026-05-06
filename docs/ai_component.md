# AI Component: RR-aware Neural MRP

This document defines the AI component in FairVote-AI and keeps the claim deliberately narrow.

## What counts as AI here

The AI component is the **RR-aware Neural MRP** model implemented in:

```text
`fairvote/inference/mrp/neural/` (facade: `fairvote/inference/mrp/rr_neural_mrp.py`)
```

Randomized Response is not AI. It is the Local Differential Privacy mechanism used by the respondent browser. The neural model is one estimator in the comparison benchmark; it is not treated as the default or automatically superior method.

## Observation model

The neural network maps respondent features to a latent distribution over true choices:

```text
P_theta(true = t | x)
```

The server never needs true labels in real-data training mode. Training uses the reported privatized labels and the known Randomized Response channel:

```text
P_theta(reported = r | x)
  = sum_t P_theta(true = t | x) P_RR(reported = r | true = t)
```

The training loss is reported-label negative log-likelihood plus regularisation. Synthetic true labels, where available, are used only after fitting for evaluation metrics such as Brier score and aggregate/subgroup error.

## Diagnostics exposed

The model exposes report-ready diagnostics:

- training loss history;
- validation reported-label NLL;
- early stopping and patience;
- number of steps completed;
- runtime and device metadata;
- optional checkpoint and metadata export;
- synthetic-only Brier score where true labels are available;
- entropy/uncertainty summaries;
- optional multi-seed ensemble support.

See [`neural_rr_mrp_diagnostics.md`](neural_rr_mrp_diagnostics.md) for the detailed API and interpretation guidance.

## Why compare against simpler baselines

A neural estimator adds hyperparameters, runtime, dependency cost and overfitting risk. The project therefore compares it with:

- raw reported distribution;
- direct RR debiasing;
- RR-aware linear poststratification/MRP;
- misreport-aware model variants where the scenario includes behavioural misreporting;
- oracle baselines in synthetic experiments.

The research question is not “does AI improve polling?” It is:

> When does RR-aware Neural MRP improve over simpler RR-aware linear/poststratification baselines under Local Differential Privacy and sampling bias?

## Expected useful cases

RR-aware Neural MRP has a principled reason to help when:

- the synthetic truth has nonlinear demographic interactions;
- subgroup preference curves are not well represented by additive linear effects;
- there is enough sample size and privacy signal to estimate those effects;
- validation reported-label NLL improves without unstable overfitting.

## Expected weak cases

The simpler RR-aware linear poststratification/MRP baseline is preferable when:

- the true data-generating process is approximately linear/additive;
- sample size is small;
- epsilon is low and privacy noise dominates the signal;
- validation NLL does not improve;
- runtime or reproducibility matters more than flexible modelling.

## Evidence route

Use the final-evidence pipeline, not historical development evidence folders:

```bash
python -m experiments.mrp_vs_baselines --preset smoke_test      # sanity check only
python -m experiments.mrp_vs_baselines --preset final_evidence  # final-style evidence
```

Final-style outputs are written to `experiments/outputs/<timestamp>/` and include `raw_trials.csv`, `summary_with_ci.csv`, `paired_comparisons.csv`, `ablations.csv`, `runtime_profile.csv`, `config.json`, `manifest.json` and a run README.

Interpret neural claims using [`evidence_interpretation.md`](evidence_interpretation.md). A defensible conclusion may be positive, negative or mixed, depending on paired deltas, confidence intervals, win rates, subgroup error and runtime.

## Non-claims

The neural model does not:

- recover individual true votes;
- remove privacy noise;
- guarantee better accuracy than simpler baselines;
- guarantee fairness;
- prove real-election forecasting accuracy;
- make the respondent app production election software.
