# Experiment Protocol

This protocol is the canonical description of how final evidence runs should be generated and interpreted. It separates smoke checks from evidence suitable for the final report.

## Research question

> When does RR-aware Neural MRP improve over simpler RR-aware linear/poststratification baselines under Local Differential Privacy and sampling bias?

## Hypotheses

| ID | Hypothesis | Evidence that would support it | Evidence that would refute or weaken it |
|---|---|---|---|
| H1 | In simple approximately linear demographic settings, RR-aware linear poststratification/MRP should be sufficient and may outperform RR-aware Neural MRP. | Linear has lower or similar L1/worst-group L1, lower runtime, and overlapping paired CIs. | Neural consistently improves with meaningful paired CIs despite the simple data-generating process. |
| H2 | In nonlinear interaction settings, RR-aware Neural MRP can improve subgroup or aggregate error when sample size and epsilon provide enough signal. | Negative neural-minus-linear paired L1 deltas, positive win rate, validation NLL improvement, acceptable runtime. | Neural has unstable validation loss, worse L1, worse worst-group error or substantially higher runtime without accuracy gain. |
| H3 | Under strong privacy noise and sparse subgroups, neural complexity may overfit or fail to recover signal. | Higher entropy, worse validation NLL, worse paired deltas for neural at low epsilon/small sample sizes. | Neural consistently improves sparse subgroup metrics with stable validation behaviour. |
| H4 | Misreport-aware or oracle models should show the upper-bound value of modelling the correct observation process. | Oracle baselines improve over practical estimators, especially in misreport scenarios. | Oracle methods do not improve, suggesting the scenario or metric may not capture the intended bias. |

## Scenarios

| Scenario | Purpose | Expected estimator behaviour |
|---|---|---|
| `simple_linear` / `no_bias` | Linear demographic signal with no intentional nonresponse/misreporting. | Linear RR-aware MRP should be competitive; neural should not be assumed to help. |
| `nonresponse` | Sampling bias caused by unequal response probabilities. | Poststratification should help compared with raw/sample-only estimates. |
| `nonlinear_interaction` | Region-age or similar nonlinear demographic interaction. | RR-aware Neural MRP has a principled reason to help if enough signal remains after RR noise. |
| `education_urbanicity_interaction` | Interaction between education and urbanicity. | Tests whether nonlinear feature combinations matter. |
| `sparse_minority_curve` | Sparse subgroup with a different preference curve. | Tests worst-group behaviour and sparse subgroup stability. |
| `nonlinear_response` | Nonlinear response mechanism. | Tests interaction between sampling bias and model flexibility. |
| `privacy_noise_sparse` | Privacy noise combined with subgroup sparsity. | Tests whether neural complexity becomes harmful under limited signal. |
| `shy_fixed` / misreport | Fixed simulated misreporting before RR perturbation. | Misreport-aware/oracle baselines should expose the cost of unmodelled bias. |
| `shy_privacy_helps` / `privacy_tradeoff` | Simulated privacy-related honesty trade-off. | Used for cautious discussion; not proof of real voter psychology. |

## Estimators and baselines

| Method | Role | Uses demographics | RR-aware | Uses true labels in real-data training? |
|---|---|---:|---:|---:|
| `raw_reported_distribution` | Descriptive baseline over privatized reports | No | No | No |
| `baseline_rr_debias` | Aggregate RR inversion baseline | No | Yes | No |
| `linear_rr_no_poststrat` | Ablation: RR-aware linear model without poststratification | Yes | Yes | No |
| `mrp_rr_poststrat` | Canonical RR-aware linear poststratification/MRP-style baseline | Yes | Yes | No |
| `neural_rr_mrp` | Main learned nonlinear estimator | Yes | Yes | No |
| `neural_naive_reported_mrp` | Ablation: neural model that does not use RR-aware loss | Yes | No / naive | No true labels, but treats reports incorrectly |
| `mrp_misreport_rr_poststrat` | Known-form misreport-aware baseline | Yes | RR + misreport model | No |
| `mrp_learned_misreport_rr_poststrat` | Learned misreport extension | Yes | RR + learned honesty | No |
| `oracle_true_sample_distribution` | Upper-bound sample estimate using synthetic truth | No | N/A | Synthetic only |
| `oracle_true_linear_mrp_poststrat` | Upper-bound MRP using synthetic truth | Yes | N/A | Synthetic only |
| `oracle_known_misreport_rr_mrp` | Upper-bound known-misreport model | Yes | RR + oracle misreport | Synthetic only |

Oracle baselines are not available in real respondent mode. They are used only to interpret synthetic experiments.

## Metrics

| Metric | Interpretation | Lower/higher better | Notes |
|---|---|---|---|
| Overall L1 | Absolute error between estimated and true aggregate distribution. | Lower | Main aggregate utility metric. |
| Worst-group L1 | Largest subgroup error across audited groups. | Lower | Sensitive to sparse groups; should be interpreted with group sizes. |
| Weighted group L1 | Group error averaged by group/population weight. | Lower | Less dominated by rare groups than worst-group L1. |
| P90 group L1 | 90th percentile subgroup error. | Lower | Robust high-error summary. |
| Winner correctness | Whether the top estimated option matches the synthetic truth. | Higher | Coarse and unstable when true margins are close. |
| Reported-label NLL | Neural likelihood of privatized reports through the RR channel. | Lower | Training/validation diagnostic, not direct election accuracy. |
| Brier score | Synthetic-only probability calibration against true labels. | Lower | Not available for real respondent labels. |
| Entropy | Uncertainty of predicted true-vote distribution. | Contextual | High entropy may be appropriate under low epsilon. |
| Runtime | Method runtime in seconds. | Lower | Used to discuss practicality, not accuracy. |

## Presets and trial counts

| Preset | Purpose | Typical settings | Evidence status |
|---|---|---|---|
| `smoke_test` | Fast sanity check during development. | Few trials, small sample sizes, reduced training. | Not final evidence. |
| `medium_evidence` | Draft-report or development evidence. | Moderate trial count and sample sizes. | Useful for debugging and preliminary analysis. |
| `final_evidence` | Submission-quality repeated-trial evidence. | Epsilons `0.2,0.5,1.0,2.0`; sample sizes `500,1000,2500`; scenarios `simple_linear,nonresponse,nonlinear_interaction,shy_fixed`; at least 30 trials. | Preferred report evidence. |

Final evidence should not use minimal settings such as `--mrp_steps 5` or `--neural_steps 5` unless the run is clearly labelled as a smoke test.

## Random seeds and reproducibility

Each run writes `config.json` and `manifest.json`. Raw result rows include run context such as:

- preset name;
- config seed;
- random seed;
- sample seed;
- scenario;
- epsilon;
- sample size;
- trial number;
- method name;
- runtime.

The seed policy is deterministic: a run can be regenerated from its config. Changing trial count, scenario list, epsilon grid or method list should be treated as a new run and documented in the run-level `README.md`.

## Output files

A final-evidence run should write:

| File | Purpose |
|---|---|
| `raw_trials.csv` | One row per method per trial condition with metrics and runtime. |
| `summary_with_ci.csv` | Aggregated means, standard errors and 95% confidence intervals over trials. |
| `paired_comparisons.csv` | Neural-minus-linear paired deltas, bootstrap intervals and win rates. |
| `ablations.csv` | Ablation comparisons against canonical baselines where applicable. |
| `runtime_profile.csv` | Runtime mean/dispersion by method/scenario/epsilon/sample size. |
| `failures.csv` | Non-fatal method failures, if any. Empty or absent failures should be stated. |
| `config.json` | Exact run configuration. |
| `manifest.json` | Run metadata, file list and row counts. |
| `README.md` | Human-readable explanation of the run and whether it is smoke/medium/final evidence. |

## Success criteria

The research question is supported only if the evidence is conditional and paired:

- neural-minus-linear deltas are negative for the relevant error metric;
- win rates are above chance across repeated trials;
- bootstrap 95% confidence intervals do not strongly contradict the claimed effect;
- neural validation NLL and diagnostics do not indicate unstable training;
- runtime overhead is acknowledged.

A defensible conclusion may be negative or mixed. A finding that RR-aware linear poststratification/MRP is preferable in most settings is still a valid research outcome.

## Exclusion and failure rules

- A failed method invocation should be recorded in `failures.csv` rather than silently removed.
- If `--fail_fast` is used, the run is for debugging and should not be treated as final evidence.
- Conditions with too few successful trials should be excluded from final claims or clearly labelled incomplete.
- Oracle baselines must be labelled synthetic-only.
- Smoke-test outputs must not be presented as final evidence.

## Runtime constraints

Final evidence is computationally heavier because it multiplies scenarios, epsilons, sample sizes, trials and methods. If full evidence is too slow, use a two-tier plan:

1. run the full grid without neural methods to verify schema, baselines and runtime;
2. run the neural comparison on the most relevant scenarios and state the reduced grid explicitly.

Do not hide runtime limits. State them in the run-level `README.md` and in the final report.
