# Evidence Interpretation

This document explains how to read the experiment outputs without overclaiming. It is the canonical interpretation guide for `raw_trials.csv`, `summary_with_ci.csv`, `paired_comparisons.csv`, `ablations.csv`, and `runtime_profile.csv`.

## Core principle

The experiments do not try to prove that AI is always better. They test a conditional question:

> When does RR-aware Neural MRP improve over simpler RR-aware linear/poststratification baselines under Local Differential Privacy and sampling bias?

A mixed or negative result is still useful. If the RR-aware linear poststratification/MRP baseline is more accurate or much faster in simple settings, that supports the conclusion that neural complexity is not always justified.

## Output files

| File | How to interpret it |
|---|---|
| `raw_trials.csv` | One row per method/condition/trial. Use for detailed checks, paired comparisons and debugging. |
| `summary_with_ci.csv` | Mean metrics and uncertainty summaries over repeated trials. Use for report tables. |
| `paired_comparisons.csv` | Neural-minus-linear paired deltas and bootstrap confidence intervals. Use for the main neural-vs-linear claim. |
| `ablations.csv` | Comparisons against ablation methods such as no-poststratification or naive neural training. Use to explain why modelling choices matter. |
| `runtime_profile.csv` | Runtime by method/condition. Use to discuss practicality. |
| `config.json` | Exact configuration needed to reproduce the run. |
| `manifest.json` | Run metadata, row counts, method/scenario lists and output file inventory. |
| `failures.csv` | Any method failures. Empty or absent failures should be stated. |

## Main metrics

### Overall L1 error

Overall L1 error measures the absolute distance between the estimated aggregate distribution and the synthetic true aggregate distribution. Lower is better.

Use it to answer: **How close is the estimated overall vote-share distribution?**

Limitations:

- it can hide subgroup errors;
- it depends on synthetic truth;
- small true margins can make winner correctness unstable even when L1 is small.

### Worst-group L1 error

Worst-group L1 is the largest subgroup error across audited demographic groups. Lower is better.

Use it to answer: **Which method has the worst subgroup failure mode?**

Limitations:

- it is sensitive to small or sparse groups;
- it should be interpreted alongside group sizes and weighted group error;
- it is an audit metric, not a fairness guarantee.

### Weighted group L1 and P90 group L1

Weighted group L1 averages subgroup errors using group weights. P90 group L1 reports a high-error percentile. Lower is better for both.

Use them to avoid relying only on the single worst group.

### Winner correctness

Winner correctness records whether the estimated top option matches the synthetic true top option. Higher is better.

Limitations:

- it is coarse;
- it can be unstable when the true margin is small;
- it does not measure calibration or distributional accuracy.

### Reported-label negative log-likelihood

Reported-label NLL is used by the neural model to measure how well the model explains privatized reported labels through the RR observation channel. Lower is better.

Use it as a convergence and validation diagnostic. Do not treat it as direct proof of real vote accuracy.

### Brier score

Brier score is a synthetic-only calibration metric comparing predicted probabilities against known true labels. Lower is better.

It is not available for real respondent data because true labels are not collected.

### Entropy / uncertainty

Entropy summarises how uncertain a model's predicted distribution is. High entropy can be reasonable under strong privacy noise or sparse data. Low entropy is not automatically good if the model is confidently wrong.

### Runtime

Runtime is reported per method and condition. Lower is better for practical deployment, but runtime must be interpreted alongside accuracy and uncertainty.

A method that is slightly more accurate but much slower may not be preferable for the final project conclusion unless the accuracy gain is clear and relevant.

## Interpreting neural-minus-linear comparisons

`paired_comparisons.csv` is the main file for the neural research question.

Typical columns include:

- scenario;
- epsilon;
- sample size;
- metric;
- mean neural-minus-linear delta;
- bootstrap 95% confidence interval;
- win rate by trial.

For error metrics such as L1:

```text
negative delta = neural lower error than linear
positive delta = neural higher error than linear
```

For winner correctness or other higher-is-better metrics, check the column definition before interpreting sign.

A strong neural-help claim needs more than one favourable number. Look for:

- repeated-trial win rate above chance;
- paired confidence interval that is mostly on the favourable side;
- validation NLL that does not indicate unstable training;
- runtime overhead that is acceptable;
- consistency across relevant nonlinear scenarios, not only one lucky trial.

## What RR-aware Neural MRP proves

The neural component proves that the project implements a privacy-compatible learned estimator that:

- predicts latent true-vote probabilities from features;
- trains on privatized reported labels through the known RR channel;
- does not require true labels in real-data training mode;
- exposes diagnostics such as validation NLL, early stopping, runtime, Brier score and entropy.

## What RR-aware Neural MRP does not prove

The neural component does not prove that:

- neural models are always better than RR-aware linear poststratification/MRP;
- synthetic results transfer directly to real elections;
- the respondent app is production election infrastructure;
- subgroup fairness is guaranteed;
- low epsilon estimates are reliable for sparse subgroups.

If the evidence shows that RR-aware Neural MRP underperforms RR-aware linear poststratification/MRP in simple scenarios, state that clearly. That is not a project failure; it is a valid empirical result.

## Interpreting simple vs nonlinear scenarios

| Scenario type | Expected conclusion pattern |
|---|---|
| Simple linear/no-bias | Linear RR-aware MRP should usually be competitive. Neural wins here need careful scrutiny because the extra complexity may not be justified. |
| Nonlinear interaction | RR-aware Neural MRP has a principled reason to help. A neural advantage here is meaningful if paired deltas and validation diagnostics support it. |
| Nonresponse | Poststratification should matter. Compare no-poststratification ablations against poststratified methods. |
| Sparse/privacy-noise scenarios | Expect instability. Worst-group metrics, entropy and confidence intervals matter more than single aggregate L1 values. |
| Misreport/shy-voter scenarios | Known-misreport or oracle baselines help interpret how much unmodelled reporting bias matters. Do not claim these scenarios prove real voter psychology. |

## Limitations of synthetic data

Synthetic simulations are useful because they provide known truth and controlled bias mechanisms. They are limited because:

- the data-generating assumptions are chosen by the project;
- real voters may not follow the simulated preference or response functions;
- demographics may be more complex or more identifying in real data;
- real sampling frames and turnout mechanisms are not modelled fully;
- confidence intervals over synthetic trials describe simulation variability, not real-election uncertainty.

The final report should therefore phrase conclusions as:

> Under the tested synthetic assumptions, method X performed better/worse than method Y on metric Z.

Avoid phrasing such as:

> This proves the method will predict real elections accurately.

## Report-safe conclusion templates

Use wording like:

- "The neural model is technically valid because it trains through the RR observation channel on privatized reported labels."
- "The evidence does not show universal neural superiority; it identifies the conditions where neural flexibility is or is not useful."
- "Linear RR-aware MRP remains the stronger default when the data-generating process is approximately linear or sample size/privacy signal is limited."
- "Worst-group improvements should be interpreted cautiously where subgroup counts are small."
- "Synthetic oracle baselines are upper-bound diagnostics, not deployable real-data methods."
