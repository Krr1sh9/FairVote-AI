# Research Contribution: When Does Neural RR-MRP Help?

This project is not intended to claim that adding a neural network automatically improves private polling.  The sharper research question is:

> **When does RR-aware Neural MRP improve over simpler RR-aware linear/poststratification baselines under Local Differential Privacy and sampling bias?**

The experiment pipeline now supports scenarios and ablations designed to answer that question with evidence rather than with a generic "AI" claim.

## Hypothesis

RR-aware Neural MRP should help mainly when the latent preference function contains nonlinear demographic structure that a regularised additive RR-aware linear poststratification/MRP-style model cannot represent well, for example:

- a **region × age** interaction;
- an **education × urbanicity** interaction;
- a **sparse minority subgroup** with a distinct preference curve;
- a nonlinear demographic response curve;
- privacy noise that interacts with subgroup sparsity.

The simpler RR-aware linear model should be preferable when the data-generating process is additive, when the sample is too small for a neural model to estimate interactions reliably, or when high privacy noise dominates the signal.

## Scenario set

The main scenarios are:

| Scenario | Purpose | Expected result if the methods behave sensibly |
|---|---|---|
| `simple_linear` | Additive demographic truth model. | Linear RR-MRP should be sufficient; neural should not consistently win. |
| `nonlinear_interaction` | Strong region × age and education × urbanicity interactions. | Neural RR-MRP may improve, especially with enough sample size and moderate epsilon. |
| `education_urbanicity_interaction` | Focused two-feature nonlinear interaction. | Neural may improve over additive RR-aware linear poststratification/MRP if signal survives RR noise. |
| `sparse_minority_curve` | Sparse subgroup has a distinct preference curve. | Neural may help only if enough subgroup observations remain; it may overfit when sparse. |
| `nonlinear_response` | Smooth nonlinear demographic response function. | Neural may help where additive linear effects are misspecified. |
| `privacy_noise_sparse` | Sparse subgroup plus privacy noise. | Tests whether privacy noise removes the signal neural models need. |
| `nonresponse` | Demographic nonresponse bias. | Poststratification should matter; neural benefit depends on nonlinear residual structure. |
| `shy_fixed` | Fixed pre-LDP misreporting for a shy category. | Known-misreport oracle should bound achievable correction. |
| `shy_privacy_helps` / `privacy_tradeoff` | Epsilon-dependent misreporting plus nonresponse. | Tests privacy/utility/social-desirability trade-offs. |

## Baselines and ablations

The `research` method preset includes methods that separate engineering claims from research claims:

| Method | Role |
|---|---|
| `oracle_true_sample_distribution` | Synthetic upper diagnostic: sample aggregate using true labels before RR. |
| `raw_reported_distribution` | Naive lower baseline: treats privatized reports as truth. |
| `baseline_rr_debias` | Analytical RR debiasing without poststratification. |
| `linear_rr_no_poststrat` | RR-aware linear model without population reweighting. |
| `mrp_rr_poststrat` | Main RR-aware linear/poststratification baseline. |
| `oracle_true_linear_mrp_poststrat` | Synthetic upper diagnostic: RR-aware linear poststratification/MRP fitted to true labels. |
| `mrp_misreport_rr_poststrat` | Existing oracle-style known-misreport RR-MRP. |
| `oracle_known_misreport_rr_mrp` | Explicit alias for the known-misreport oracle baseline. |
| `mrp_learned_misreport_rr_poststrat` | Learns a simple shy-voter honesty parameter from privatized reports. |
| `neural_rr_mrp` | Main RR-aware neural model, trained through the RR observation channel. |
| `neural_naive_reported_mrp` | Neural ablation that treats privatized reports as true labels; useful for showing whether RR-aware loss matters. |

Demographic-subset ablations use the existing `--features` option, for example:

```bash
python -m experiments.mrp_vs_baselines \
  --methods research \
  --features region,age_group \
  --scenarios simple_linear,nonlinear_interaction
```

Compare this with the full feature set:

```bash
python -m experiments.mrp_vs_baselines \
  --methods research \
  --features region,age_group,education,gender,urbanicity \
  --scenarios simple_linear,nonlinear_interaction
```

## Paired comparison output

Every run with both `mrp_rr_poststrat` and `neural_rr_mrp` now writes:

```text
paired_comparisons.csv
```

This file compares neural and linear methods on the same scenario, epsilon, and trial.  Negative deltas mean neural has lower error.  Important columns include:

- `mean_delta_overall_l1` = neural overall L1 minus linear overall L1;
- `mean_delta_worst_group_l1` = neural worst-major-group L1 minus linear worst-major-group L1;
- `win_rate_delta_overall_l1` = fraction of paired trials where neural improves overall L1;
- `ci95_low_delta_overall_l1` and `ci95_high_delta_overall_l1` = bootstrap 95% confidence interval for the paired mean delta.

This matters because a neural model should not be judged on a single unpaired run.  The paired table supports claims such as:

- neural helps in nonlinear interaction scenarios at moderate epsilon;
- neural does not help in simple additive scenarios;
- neural loses when privacy noise or subgroup sparsity overwhelms signal;
- RR-aware linear poststratification/MRP is preferable when it is simpler and statistically competitive.

## Suggested evidence run

A final-report-quality run should use multiple trials and the `research` preset:

```bash
python -m experiments.mrp_vs_baselines \
  --methods research \
  --scenarios simple_linear,nonlinear_interaction,education_urbanicity_interaction,sparse_minority_curve,nonresponse,shy_privacy_helps \
  --eps 0.2,0.5,1.0,2.0 \
  --trials 30 \
  --population_n 100000 \
  --n_sample 3000 \
  --mrp_steps 800 \
  --neural_steps 800 \
  --neural_patience 40
```

If this is computationally too expensive, run a smaller pilot first and clearly label it as constrained evidence.

## What would support the research contribution?

Evidence supports the contribution if:

1. `simple_linear` shows little or no neural advantage over `mrp_rr_poststrat`;
2. nonlinear scenarios show lower paired neural error under at least some epsilon/sample-size conditions;
3. `neural_naive_reported_mrp` performs worse than `neural_rr_mrp`, showing that the RR-aware likelihood matters;
4. oracle baselines bound what could be achieved with unobserved true labels or known misreporting;
5. confidence intervals and win rates show the result is not a one-off trial accident.

Evidence refutes or limits the contribution if neural does not improve on nonlinear scenarios, if gains vanish under repeated trials, or if simple RR-aware linear poststratification/MRP is consistently similar at lower runtime.  That is still a valid final-year research outcome if it is reported honestly.

---

## Final evidence preset

The research question should be evaluated with the final evidence preset rather than one-off minimal runs:

```bash
python -m experiments.mrp_vs_baselines --preset final_evidence
```

This preset runs the neural-vs-linear question across privacy levels, sample sizes, simple and nonlinear truth models, nonresponse, and shy-voter/misreport conditions. It writes `summary_with_ci.csv`, `paired_comparisons.csv`, `ablations.csv`, and `runtime_profile.csv` so the final report can make an honest conclusion such as:

- neural RR-MRP helps under nonlinear demographic interactions if paired deltas are negative with convincing intervals and win rates;
- RR-aware linear poststratification/MRP is preferable in simple/additive scenarios if neural deltas are near zero or positive;
- the neural method is not justified where the runtime cost rises without a stable accuracy gain.

Use `--trials 50` or `--trials 100` for stronger evidence if time permits.
