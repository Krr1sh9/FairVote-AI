# AI Component: RR-aware Neural MRP

This document defines exactly what the AI component in FairVote-AI is and what it is not.

## What is AI in this project?

The AI component is the **RR-aware Neural Multilevel Regression and Post-stratification model** implemented in:

```text
fairvote/inference/mrp/rr_neural_mrp.py
```

It is a PyTorch neural network used for aggregate inference from privatized polling data.

Randomized Response is not the AI component. Randomized Response is the local differential privacy mechanism used during data collection.

## Model inputs and outputs

### Input

The model receives a feature matrix derived from respondent demographics or poll metadata, for example:

```text
age_group, region, education, gender, urbanicity
```

These features are encoded into a numeric design matrix before fitting.

### Latent output

The neural network outputs a latent distribution over true vote categories:

```text
P_theta(true vote = t | demographic features x)
```

This distribution is latent because real respondent true votes are not observed by the server and are not required for training.

### Observation model

The observed label is the privatized answer produced by k-ary Randomized Response:

```text
P_RR(reported = r | true = t)
```

The model therefore predicts reported labels by marginalising over latent true labels:

```text
P_theta(reported = r | x)
  = sum_t P_theta(true = t | x) P_RR(reported = r | true = t)
```

### Training objective

The model is trained by minimising the negative log-likelihood of the reported privatized labels:

```text
loss(theta) = -sum_i log P_theta(reported_i | x_i) + regularisation
```

It does not train against true votes in real polling mode.

## Why this is privacy-compatible

The respondent server stores only privatized reported answers and demographics. The neural model trains on those same reported answers. It does not require the server to receive, store, or reconstruct individual true votes.

In synthetic experiments, true labels exist because the simulator generates them. They are used only after model fitting to evaluate aggregate and subgroup error. They are not used as training labels for the neural model.

## Why compare against simpler baselines?

A neural model adds complexity: more hyperparameters, longer runtime, and greater overfitting risk. It is therefore compared against:

- raw reported distribution,
- RR debiasing,
- linear RR-aware MRP,
- misreport-aware RR-MRP where the simulated misreport model is available.

The neural model is justified only if the experiment results show an accuracy or subgroup-error improvement large enough to justify the extra complexity. It may fail under low epsilon, small sample sizes, weak demographic signal, or poor hyperparameter choices.

## Result interpretation from the final evidence pack

The repository includes a final evidence folder at:

```text
experiments/outputs/final_neural_evidence/
```

This evidence pack is **computationally constrained final-style evidence**, not the exhaustive full preset. The attempted exhaustive full preset was too slow for the available execution environment, so the included run uses all four epsilon values and all four bias scenarios, but fewer trials and training steps. Its configuration is recorded in `config.json` and summarised in `experiments/outputs/final_neural_evidence/README.md`.

The included run used:

```text
trials = 1
MRP steps = 5
neural steps = 5
population_n = 5000
epsilons = 0.2, 0.5, 1.0, 2.0
sample sizes = 500, 1000
scenarios = no_bias, nonresponse, shy_fixed, shy_privacy_helps
```

The main generated findings are:

| Method | Mean overall L1 ↓ | Winner correctness ↑ | Mean runtime sec ↓ |
|---|---:|---:|---:|
| Raw reported distribution | **0.163** | 0.250 | 0.000 |
| RR debiasing | 0.430 | 0.250 | 0.001 |
| Linear RR-aware MRP | 0.176 | 0.344 | 0.005 |
| Neural RR-aware MRP | 0.204 | 0.125 | 0.030 |
| Misreport-aware RR-MRP | 0.211 | **0.906** | 0.768 |
| Learned misreport RR-MRP | 0.229 | 0.094 | 0.006 |

These outputs do **not** show that neural RR-MRP is generally superior to simpler baselines. Raw reported distribution had the lowest average overall L1 in this constrained synthetic run, but it is not RR-corrected. Among RR-aware corrected/model-based estimators, linear RR-aware MRP achieved lower average overall L1 than neural RR-MRP in the included evidence pack. Neural RR-MRP had a lower averaged worst-group L1 than linear MRP, but it was worse on weighted group L1, p90 group L1, winner correctness, and runtime.

The appropriate conclusion is therefore cautious: neural RR-MRP is a genuine privacy-compatible AI estimator, but the generated evidence does not justify replacing linear RR-aware MRP with the neural model as the default estimator. It should be presented as an evaluated extension whose usefulness depends on the setting.

## What the model does not claim

The neural model does not:

- recover individual true votes,
- remove privacy noise,
- guarantee better accuracy than simpler baselines,
- guarantee fairness,
- prove real election forecasting accuracy from synthetic experiments alone.

Its purpose is narrower: to test whether nonlinear RR-aware MRP improves aggregate and subgroup estimation under specified simulation conditions.

## Relevant commands

Small smoke experiment:

```bash
python -m experiments.evaluate_neural_mrp --preset small
```

Full neural-justification experiment:

```bash
python -m experiments.evaluate_neural_mrp --preset full
```

Standard MRP comparison:

```bash
python -m experiments.mrp_vs_baselines
```

Disable neural MRP in the standard comparison:

```bash
python -m experiments.mrp_vs_baselines --disable_neural
```
