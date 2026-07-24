# FairVote-AI

**Privacy-preserving polling with Randomised Response and poststratification**

A controlled simulation of how local privacy noise, sample size and demographic sampling bias affect
population-level estimates from a poll. A fixed six-cell synthetic population supplies a known ground
truth, so every estimator can be scored exactly. Everything here is simulated. There are no real
respondents, no collected data and no predictive or machine-learning component anywhere in the code.

## Research question

> How do the Local Differential Privacy budget, polling sample size and demographic sampling bias
> affect the accuracy of estimates produced using k-ary Randomised Response, and when can simple
> cell-wise poststratification improve population-level estimates?

## Motivation and scope

Sensitive surveys face two problems at once. Individual answers carry disclosure risk, and the people
who respond are often not representative of the population being measured. Randomised Response
addresses the first by perturbing each answer at source, at the cost of added variance.
Poststratification addresses the second by reweighting demographic cells towards known population
shares. Both are well established on their own. The main question here is how the two corrections
behave when privacy noise and sampling bias are present at the same time.

The work focuses on a controlled synthetic experiment rather than a complete polling system. A
synthetic population is defined, samples are drawn either representatively or with demographic bias,
k-ary Randomised Response is applied, and three estimators are compared against the known distribution
over a factorial grid with fixed seeds.

## Method

### Synthetic population

Two regions and three age groups give six demographic cells over three response categories. Weights
and within-cell preferences are fixed constants in `fairvote/simulation/population.py`. Nothing is
fitted or learned.

| Cell | Weight | Option A | Option B | Option C |
|---|---:|---:|---:|---:|
| North / 18-34 | 0.10 | 0.55 | 0.30 | 0.15 |
| North / 35-54 | 0.15 | 0.45 | 0.35 | 0.20 |
| North / 55+ | 0.20 | 0.30 | 0.50 | 0.20 |
| South / 18-34 | 0.15 | 0.60 | 0.25 | 0.15 |
| South / 35-54 | 0.20 | 0.40 | 0.40 | 0.20 |
| South / 55+ | 0.20 | 0.20 | 0.60 | 0.20 |

Weighting the cell preferences by the population weights gives the analytical truth
`(0.3925, 0.4200, 0.1875)`. Because the population is specified rather than sampled, this truth is
exact.

### Sampling bias

Bias is introduced by multiplying each cell's population weight by a fixed age-group inclusion factor
and renormalising the result into sampling probabilities. The moderate and strong conditions
progressively under-represent the 55+ age group while increasing the share of respondents aged 18-34.

| Bias | 18-34 | 35-54 | 55+ | Resulting 18-34 share of sample |
|---|---:|---:|---:|---:|
| `none` | 1.0 | 1.0 | 1.0 | 0.250 |
| `moderate` | 1.8 | 1.0 | 0.6 | 0.433 |
| `strong` | 3.5 | 1.0 | 0.3 | 0.651 |

Preferences differ sharply by age, so the strong condition shifts the sample well away from the
population it is meant to describe. Closing that gap is what poststratification is being asked to do.

### k-ary Randomised Response

For `k` categories and a privacy budget `epsilon > 0`, a respondent reports their true category with
probability `p` and any one specific alternative with probability `q`:

```text
p = exp(epsilon) / (exp(epsilon) + k - 1)
q = 1            / (exp(epsilon) + k - 1)
```

Since `p / q = exp(epsilon)`, no single report distinguishes any two true categories by more than that
factor, which is the local differential privacy guarantee. Smaller epsilon means stronger privacy and
noisier reports. `rr_params` uses the equivalent `exp(-epsilon)` form, which stays finite for large
budgets where the direct form would overflow. The mechanism works for any `k >= 2`, although three
categories are used throughout.

Inverting the channel gives an estimate of the latent distribution. For reported frequency `f_j`:

```text
theta_hat_j = (f_j - q) / (p - q)
```

This estimator is unbiased but unconstrained, so in finite samples it can fall outside the probability
simplex. Where a valid distribution is needed, `project_distribution` clips to `[0, 1]` and
renormalises to sum to one. Only the corrected estimates are projected. Raw privatised frequencies are
counts divided by their total, so they already form a distribution.

### The three estimators

| Method key | Label | Correction applied |
|---|---|---|
| `raw_frequencies` | Raw privatised reports | none |
| `rr_debiased` | Overall RR correction | Randomised Response channel |
| `poststratified` | Poststratified RR estimate | RR channel and demographic composition |

`raw_frequencies` is a naive baseline that ignores the mechanism. `rr_debiased` inverts the channel
across the whole sample and projects the result, which removes the perturbation but leaves any
demographic skew in place.

`poststratified` treats each cell separately. Every non-empty cell receives a raw inversion with
`clip=False, renormalize=False`, and the cell estimates are combined as
`sum(population_weight[c] * raw_cell_estimate[c])`. The projection is applied after the cell estimates
are combined. Applying it within each cell would affect the comparison because clipping is
non-linear.

A cell with no sampled respondents falls back to the raw whole-sample inversion. The count of such
cells is returned as `fallback_cells` and shown in the Streamlit diagnostics, but it is not a column
in the experiment tables.

This is simple cell-wise weighting towards known population shares, not multilevel regression and
poststratification. No model is fitted and no information is pooled across cells.

### Metrics

```text
L1 error                        = sum(abs(estimate - truth))
maximum absolute category error = max(abs(estimate - truth))
```

L1 error measures total distributional error. The maximum absolute error identifies the worst single
category. These are the only two metrics used.

## Experiment design

| Variable | Full run | Quick run |
|---|---|---|
| Epsilon | 0.25, 0.5, 1.0, 2.0 | 0.5, 1.0 |
| Respondents | 250, 500, 1000, 2000 | 500, 1000 |
| Sampling bias | none, moderate, strong | none, moderate, strong |
| Repetitions | 30 (seeds 1000-1029) | 3 (seeds 1000-1002) |
| Configurations | 48 | 12 |
| Result rows | 4,320 | 108 |
| Summary rows | 144 | 36 |

Each repetition runs one synthetic poll and scores all three estimators against it, so the row counts
are configurations times repetitions times three methods.

`experiment_results.csv` holds one row per repetition and method, with columns `epsilon`,
`n_respondents`, `bias`, `seed`, `method`, `l1_error` and `max_abs_error`. `summary.csv` aggregates
over repetitions with columns `method`, `bias`, `epsilon`, `n_respondents`, `n_repetitions`,
`l1_mean`, `l1_std`, `max_abs_error_mean` and `max_abs_error_std`.

Three figures are produced. `l1_vs_epsilon.png` plots mean L1 error against epsilon with the sample
size fixed at 1000. `l1_vs_sample_size.png` plots mean L1 error against sample size with epsilon fixed
at 1.0. Both use one panel per bias level and one line per estimator. `l1_by_bias.png` is a boxplot of
the individual repetition-level L1 errors at epsilon 1.0 and n = 1000, grouped by bias level, so the
spread across seeds stays visible. Its title reports the number of repetitions actually present, which
is 30 after a full run and 3 after a quick one.

## Streamlit demonstration

`app.py` is a single-page interface for exploring one poll at a time. The sidebar offers epsilon,
number of respondents, bias level and a random seed. The first three are restricted to the values used
in the experiment, and `evaluate_poll` rejects anything outside them.

The page shows the chosen settings, a bar chart and table comparing the true distribution against all
three estimates, a table of both error metrics per estimator, and a collapsed diagnostics panel with
per-cell sample counts, population weights, sample proportions, the L1 demographic imbalance between
sample and population, and the empty-cell fallback count.

## Repository structure

```text
.
├── .github/workflows/tests.yml
├── .gitattributes
├── .gitignore
├── LICENSE
├── README.md
├── app.py
├── pyproject.toml
├── experiments/
│   ├── __init__.py
│   ├── plots.py
│   └── run_experiments.py
├── fairvote/
│   ├── __init__.py
│   ├── metrics.py
│   ├── poststratification.py
│   ├── study.py
│   ├── privacy/
│   │   ├── estimators.py
│   │   └── mechanisms/kary_rr.py
│   └── simulation/
│       ├── population.py
│       └── sampling.py
└── tests/
```

`fairvote/study.py` holds the shared workflow that runs one poll and scores every estimator, and is
used by both the experiment runner and the app. There is no `results/` directory here. It is created
the first time an experiment is run.

## Installation

The package requires Python 3.11 or later. The automated workflow tests it on Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[experiments,app,dev]"
```

On Windows, activate the environment with `.\.venv\Scripts\Activate.ps1` in PowerShell, or
`.venv\Scripts\activate.bat` in Command Prompt.

NumPy is required by the core package. The optional dependency groups install the packages needed for
experiments, the Streamlit demonstration and development checks.

## Tests and checks

```bash
python -m pytest -q
python -m pytest -q --cov=fairvote --cov=experiments
python -m ruff check .
python -m ruff format --check .
```

The test suite covers algebraic properties of the Randomised Response channel, fixed-seed statistical
checks with predefined tolerances, population and sampling invariants, poststratification including a
hand-calculated case and the empty-cell fallback, and end-to-end runs of the experiment in quick mode.
The GitHub Actions workflow repeats these on Python 3.11 and also runs `pip check`, byte-compilation
and a quick experiment writing to a separate `ci_results` directory.

## Running the experiments

```bash
python -m experiments.run_experiments --quick
python -m experiments.run_experiments
```

Quick mode provides a faster end-to-end check of the pipeline. Three repetitions are too few to draw
conclusions from, so the full grid is the one to use for analysis. `--output-dir PATH` writes
elsewhere. Both CSVs are removed and rewritten on each run, and any `.png` already in the plots
directory is deleted before the three figures are written.

## Generated outputs

None of the following are stored in the repository. They appear only after an experiment is run, under
the output directory, which defaults to `results/`:

```text
results/experiment_results.csv
results/summary.csv
results/plots/l1_vs_epsilon.png
results/plots/l1_vs_sample_size.png
results/plots/l1_by_bias.png
```

All values in these files are generated by the experiment runner. Because no result files are included
at this stage, this README does not report experimental findings.

## Reproducibility

The experiment runner uses explicit NumPy random generators and deterministic seeds derived from each
configuration. Each stream comes from the repetition seed together with the indices of the epsilon,
sample size and bias level, so every cell of the grid gets a distinct stream. No global random state
is used. Using the same environment and seeds produces the same experiment outputs, and one of the
tests runs quick mode twice and checks that the two result tables match.

## Limitations, privacy and ethics

The population is small, coarse and entirely known, which is convenient for measurement and unlike any
real survey. Poststratification here is given the exact population weights. In practice those weights
are themselves estimates, and error in them would propagate into the result. Empty cells use a simple
whole-sample fallback. Any conclusion drawn from these experiments applies to this simulation and not
to real elections or populations.

No personal data of any kind is involved. Respondents, their demographics and their opinions are all
generated from the specification in `fairvote/simulation/population.py`. Inside `run_synthetic_poll`,
true categories are drawn into a local array, passed through Randomised Response, and then not
returned. `PollSample` carries only cell indices and privatised categories, so no estimator can reach
the unperturbed answers. The only privacy property claimed is that of the Randomised Response
mechanism as implemented.

## Licence

Licensed under the MIT License. See `LICENSE`.
