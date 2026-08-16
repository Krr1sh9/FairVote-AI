# FairVote-AI

Synthetic polling with Local Differential Privacy, Randomised Response and poststratification.

FairVote-AI is a synthetic simulation study of how the Local Differential Privacy budget, sample size and demographic sampling bias affect population-level polling estimates. A fixed six-cell synthetic population defines a known population truth, allowing three estimators to be evaluated directly against it. A small decision-tree recommender predicts the L1 error of each estimator for a given poll and selects the estimator with the lowest predicted error using only inputs available at prediction time.

Everything here is simulated. There are no real respondents and no validation claim for real elections or populations.

## Research question

How do the Local Differential Privacy budget, sample size and demographic sampling bias affect the accuracy of k-ary Randomised Response estimates, and when can simple cell-wise poststratification improve population-level estimates?

## Method

A synthetic population of two regions by three age groups gives six demographic cells over three response options, with fixed known weights and preferences. Its analytical truth is `(0.3925, 0.4200, 0.1875)`.

Each poll samples respondents under one of three bias conditions (`none`, `moderate`, `strong`), privatises each answer with k-ary Randomised Response at privacy budget `epsilon`, and estimates the population response shares.

k-ary Randomised Response reports the true category with probability `p` and each other category with probability `q`, where:

```text
p = exp(epsilon) / (exp(epsilon) + k - 1)
q = 1 / (exp(epsilon) + k - 1)
```

Since:

```text
p / q = exp(epsilon)
```

the likelihood ratio for any reported category under any two possible true categories is bounded by `exp(epsilon)`. The mechanism therefore satisfies Local Differential Privacy with privacy budget `epsilon`.

Smaller `epsilon` means stronger privacy and more randomisation.

Three estimators are compared:

- Raw privatised reports: the uncorrected category shares from the privatised reports.
- Overall RR correction: analytical Randomised Response inversion applied to the whole sample, followed by clipping and renormalisation to form a valid distribution.
- Poststratified RR estimate: raw per-cell Randomised Response inversions weighted by the known population cell shares, followed by a single clipping-and-renormalisation step after the cells are combined.

If a demographic cell contains no sampled respondents, that cell uses the whole-sample raw Randomised Response inversion as a fallback before the final weighted estimate is formed.

This is simple cell-wise poststratification using known population shares. It is not Multilevel Regression and Poststratification, or MRP.

Error is measured using L1 error and maximum absolute category error against the known synthetic population truth.

## Experiment

The full experiment grid crosses:

- four privacy budgets: `0.25`, `0.5`, `1.0`, `2.0`
- four sample sizes: `250`, `500`, `1000`, `2000`
- three bias conditions: `none`, `moderate`, `strong`
- 30 repetitions per configuration

This gives:

- 1,440 synthetic polls
- 4,320 estimator-level result rows
- 144 summary rows

Each synthetic poll uses a fresh deterministic NumPy generator initialised from `[seed, epsilon_index, size_index, bias_index]`, where `seed` is `1000 + repetition`.

## Recommender

Three shallow `DecisionTreeRegressor` models are trained, one for each estimator's L1 error.

The model configuration is:

```text
max_depth = 4
min_samples_leaf = 20
random_state = 42
```

The models use exactly six features:

```text
epsilon
n_respondents
demographic_imbalance
cell_proportion_min
cell_proportion_max
cell_proportion_std
```

These represent the privacy setting, sample size and demographic composition available to the recommender at prediction time.

The three prediction targets are:

```text
raw_frequencies_l1
rr_debiased_l1
poststratified_l1
```

The audit fields `seed`, `bias` and `fallback_cells` are excluded from the model features. The population truth is not a model input, and the measured L1 errors are used as training targets rather than input features.

The estimator with the lowest predicted L1 error is selected. When the gap between the two lowest predicted L1 errors is within the empirical approximate-tie threshold, the result is reported as an approximate tie rather than as a clear preference.

Full-run evaluation uses five-fold `GroupKFold` grouped by seed, together with leave-one-epsilon-out evaluation.

Grouping by seed keeps polls associated with the same repetition seed out of both the training and test sets within a fold.

The leave-one-epsilon-out evaluation removes one privacy budget from training at a time and evaluates the models on that excluded budget.

Regret is defined as:

```text
actual L1 error of the selected estimator
-
lowest actual L1 error available for that poll
```

An oracle that selects the estimator with the lowest actual per-poll error therefore has zero regret.

## Results

Full-run results, scoped to this synthetic simulation:

| Strategy | Mean selected L1 | Mean regret |
| --- | ---: | ---: |
| Recommender, grouped by seed | 0.1793 | 0.0294 |
| Recommender, held-out epsilon | 0.1920 | 0.0420 |
| Always raw privatised reports | 0.2073 | 0.0573 |
| Always overall RR correction | 0.2724 | 0.1225 |
| Always poststratified | 0.2574 | 0.1074 |
| Oracle best per poll | 0.1500 | 0.0000 |

No single estimator is best for every synthetic poll.

Within this simulation, the grouped recommender has a lower mean selected L1 error than each fixed-estimator strategy. The oracle remains better because it selects using the actual per-poll errors, which are unavailable to the recommender at prediction time.

The leave-one-epsilon-out results are weaker than the grouped results, indicating that generalising to a held-out privacy budget is more difficult within this experiment grid.

Exact-best estimator selection accuracy is `0.588` for grouped cross-validation and `0.526` for leave-one-epsilon-out evaluation. Regret is reported alongside exact-best accuracy because several polls have estimators with very similar errors.

These results apply only to the implemented synthetic population, experiment grid and estimator definitions. They do not establish that any estimator or recommendation strategy is universally superior.

## Installation

Python 3.11 or later is required.

Create a virtual environment:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project and all dependencies required for experiments, the application, the recommender and development checks:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[experiments,app,ai,dev]"
```

Verify the installed dependency environment:

```bash
python -m pip check
```

## Usage

Run the quality checks and tests:

```bash
python -m ruff check .
python -m ruff format --check .
python -m compileall fairvote experiments app.py
python -m pytest -q
```

Run the full statistical experiment:

```bash
python -m experiments.run_experiments
```

Run the full recommender experiment:

```bash
python -m experiments.train_selector
```

Both full pipelines write their generated outputs under `results/`.

Run the quick pipelines in a separate verification directory:

```bash
python -m experiments.run_experiments --quick --output-dir verification_results/quick
python -m experiments.train_selector --quick --output-dir verification_results/quick
```

Run the interactive Streamlit demonstration:

```bash
streamlit run app.py
```

The recommender builds its own dataset from the shared experiment grid, so the statistical experiment does not need to run first.

The Streamlit recommendation panel uses:

```text
results/ai/selector_dataset.csv
```

If that file is absent, the statistical simulation still runs and the application displays the command required to generate the recommender dataset.

## Repository structure

```text
fairvote/     simulation, Randomised Response, estimators, poststratification, study and AI selector
experiments/  statistical experiment, recommender pipeline and plots
tests/        53 pytest tests
results/      generated CSV files, metrics and figures
app.py        Streamlit demonstration
```

## Tests

The final suite contains 53 pytest tests.

The suite focuses on the main scientific, statistical and integration behaviour rather than maximising a coverage percentage.

It covers:

- Randomised Response parameters and transition behaviour
- statistical agreement of Randomised Response reports with the theoretical channel
- increased debiased-estimator variability at lower privacy budgets
- analytical Randomised Response inversion
- clipping and renormalisation
- poststratification
- empty-cell fallback behaviour
- synthetic population construction
- demographic sampling bias
- fixed-seed reproducibility
- integration of the three estimators
- L1 and maximum absolute error
- shared experiment-grid consistency
- recommender feature isolation
- deterministic decision-tree training
- recommendation and approximate-tie handling
- grouped validation
- leave-one-epsilon-out validation
- regret and baseline calculations
- generated result consistency
- Streamlit recommender integration

GitHub Actions runs dependency checks, Ruff formatting and linting checks, compilation, pytest, the quick statistical pipeline and the quick recommender pipeline using Python 3.11.

## Generated outputs

The committed full study contains 12 generated artefacts.

Numerical outputs:

```text
results/experiment_results.csv
results/summary.csv
results/ai/selector_dataset.csv
results/ai/selector_predictions.csv
results/ai/selector_metrics.json
```

Figures:

```text
results/plots/l1_vs_epsilon.png
results/plots/l1_vs_sample_size.png
results/plots/l1_by_bias.png
results/plots/ai_selector_baselines.png
results/plots/ai_selector_tree_raw_frequencies.png
results/plots/ai_selector_tree_rr_debiased.png
results/plots/ai_selector_tree_poststratified.png
```

The full selector prediction file contains 2,880 rows:

- 1,440 grouped cross-validation predictions
- 1,440 leave-one-epsilon-out predictions

## Reproducibility

The experiment grid and random-number generation are deterministic.

Using the same source code, experiment configuration, dependency environment and seeds should reproduce the numerical outputs to normal floating-point precision.

A fresh full regeneration was compared with the committed CSV and JSON outputs and produced no numerical differences within a tolerance of `1e-12`.

The seven generated figures were also regenerated and manually inspected.

This is not a claim of universal byte-for-byte reproducibility across different operating systems, dependency versions or rendering environments.

## Limitations

The population is small, synthetic and fully specified, and its population cell weights are exact by construction.

Only four privacy budgets, four sample sizes and three bias conditions are studied.

The poststratification method is simple cell-wise poststratification and is not MRP.

The population demographic weights are assumed to be known.

The recommender uses three shallow decision trees trained only on the implemented synthetic experiment grid.

The recommender has not been validated as an estimator-selection system for real polling.

Leave-one-epsilon-out evaluation tests transfer only between the privacy budgets included in this experiment design.

The results are not evidence about real elections, arbitrary populations or arbitrary privacy settings.

No result should be interpreted as showing that one estimator is universally superior.

## Privacy and scope

Randomised Response applies privacy-preserving randomisation to each synthetic categorical response before aggregation.

The polling interface used by the study exposes demographic cell indices and privatised reported categories rather than the original individual preference categories.

No real respondent data is collected, processed or stored by this study.

The Local Differential Privacy guarantee describes the implemented Randomised Response mechanism. It should not be interpreted as a complete privacy or security analysis of a deployed polling system.

A real deployment would require additional consideration of data collection, metadata, repeated participation, implementation security, consent and applicable legal or regulatory requirements.

## Licence

MIT. See `LICENSE`.
