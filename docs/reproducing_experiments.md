# Reproducing Experiments

This guide explains how to reproduce the main FairVote-AI experiments.

The experiments are designed to compare privacy-aware estimators, not to assume that the neural model is best. In particular, the RR-aware Neural MRP model is evaluated against simpler baselines because the added complexity must be justified by evidence.

## Prerequisites

Use Python 3.14.4 for final local verification.

Windows PowerShell from the project root:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,ai,streamlit,respondent]"
```

macOS/Linux from the project root:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,ai,streamlit,respondent]"
```

PyTorch is intentionally optional in the package design, but the final supported verification install includes the `ai` extra because RR-aware Neural MRP is the AI component. If your platform needs a specific PyTorch wheel, install the appropriate `torch` build first, then rerun the full install command above.


## CI and local verification

GitHub Actions installs the full development environment:

```bash
pip install -e ".[dev,ai,streamlit,respondent]"
```


The CI workflow also installs Node.js so that the JavaScript Randomized Response tests in `tests/test_respondent_rr_js.py` run rather than being silently skipped. CI runs the normal pytest suite, targeted neural tests, respondent privacy tests, JavaScript RR tests, experiment checks, and dashboard syntax checks.

Very slow experiment checks are intentionally left as local-only so that CI does not run large experiment grids on every push. To run the slow MRP integration test locally:

```bash
FV_RUN_SLOW=1 pytest tests/test_mrp_vs_baselines.py -q
```


On Windows PowerShell:

```powershell
$env:FV_RUN_SLOW="1"
pytest tests/test_mrp_vs_baselines.py -q
Remove-Item Env:FV_RUN_SLOW
```


If Node.js is not installed locally, install it before running the JavaScript RR tests. On Windows:

```powershell
winget install OpenJS.NodeJS.LTS
```



## 1. LDP / central-DP epsilon sweep

This experiment sweeps epsilon and compares Local Differential Privacy through Randomized Response against a central-DP aggregate-count baseline.

```bash
python -m experiments.sweep_eps \
  --k 5 \
  --eps "0.1,0.2,0.5,1.0,2.0,4.0" \
  --population_n 100000 \
  --n_samples "5000" \
  --trials 50 \
  --seed 123 \
  --scenario no_bias
```


With nonresponse bias:

```bash
python -m experiments.sweep_eps \
  --scenario nonresponse \
  --trials 50
```


With shy-supporter misreporting where privacy can improve honesty:

```bash
python -m experiments.sweep_eps \
  --scenario shy_privacy_helps \
  --shy_category 0 \
  --trials 50
```


2D sweep over sample size and epsilon:

```bash
python -m experiments.sweep_eps \
  --eps "0.5,1.0,2.0" \
  --n_samples "500,1000,2000,5000,10000" \
  --trials 30
```


Typical outputs:

- `summary.csv` — aggregated metrics by epsilon and method.
- `results_trials.csv` — per-trial metrics.
- `plots/` — utility/fairness visualisations where generated.

## 2. MRP vs baselines

This experiment compares the main inference methods:

- raw reported distribution,
- RR debiasing,
- linear RR-aware MRP,
- misreport-aware RR-MRP where available,
- learned-misreport RR-MRP,
- neural RR-aware MRP.

```bash
python -m experiments.mrp_vs_baselines
```


This command requires the optional `ai` extra/PyTorch unless neural is disabled. Disable neural MRP for a faster baseline-only check that does not require PyTorch:

```bash
python -m experiments.mrp_vs_baselines --disable_neural
```


Useful small smoke command:

```bash
python -m experiments.mrp_vs_baselines \
  --trials 1 \
  --eps 1.0 \
  --scenarios no_bias \
  --population_n 800 \
  --n_sample 120 \
  --mrp_steps 5 \
  --mrp_batch_size 128 \
  --neural_steps 5 \
  --neural_batch_size 128 \
  --neural_hidden_layers 8
```


## 3. Neural RR-MRP justification experiment

This is the most important experiment for justifying the AI component. It varies epsilon, sample size, and bias scenario to test whether neural RR-aware MRP improves overall or subgroup estimation enough to justify its added complexity.

The experiment answers:

- Does neural MRP improve overall vote share estimation?
- Does neural MRP improve subgroup estimation?
- Does it help more under nonlinear or biased scenarios?
- How much runtime does it add?
- When does it fail?

### Small preset

Use this for a fast smoke check:

```bash
pip install -e ".[ai]"
python -m experiments.evaluate_neural_mrp --preset small
```


The small preset is not enough for final claims. It exists to confirm that the pipeline runs and writes the expected files. In the development verification run, the small preset produced `config.json`, `results_trials.csv`, `summary.csv`, `neural_comparison.csv`, `method_rankings.csv`, and `neural_verdict.csv`.

### Full preset

Use this for report evidence:

```bash
python -m experiments.evaluate_neural_mrp --preset full
```


The full preset varies:

- epsilon: `0.2, 0.5, 1.0, 2.0`,
- sample size: `500, 1000, 5000, 10000`,
- scenario: `no_bias`, `nonresponse`, `shy_voter`, `privacy_helps`.

Override example:

```bash
python -m experiments.evaluate_neural_mrp \
  --preset full \
  --trials 10 \
  --neural_hidden_layers "64,32" \
  --neural_steps 600
```


### Outputs

The run folder is written under `experiments/outputs/` and contains:

- `config.json` — exact run configuration.
- `results_trials.csv` / `results_trials.jsonl` — per-trial method metrics.
- `summary.csv` / `summary.jsonl` — aggregated metrics by scenario, epsilon, sample size, and method.
- `neural_comparison.csv` / `neural_comparison.jsonl` — neural-vs-baseline deltas.
- `method_rankings.csv` — per-condition method rankings.
- `neural_verdict.csv` / `neural_verdict.json` — aggregate win rates and mean deltas.

In `neural_comparison.csv`, error deltas are `neural - baseline`, so negative values favour the neural model. Runtime deltas are also `neural - baseline`, so positive values mean the neural model is slower.

### Included final evidence pack

The repository includes a generated evidence folder at:

```text
experiments/outputs/final_neural_evidence/
```


This folder contains:

- `README.md` — how the evidence pack was generated and its limitations.
- `config.json` — exact run configuration.
- `results_trials.csv` / `results_trials.jsonl` — raw per-trial results.
- `summary.csv` / `summary.jsonl` — aggregated method metrics.
- `neural_comparison.csv` / `neural_comparison.jsonl` — neural-vs-baseline deltas.
- `method_rankings.csv` — method ranks by metric and condition.
- `neural_verdict.csv` / `neural_verdict.json` — aggregate neural win/loss summaries.
- `plots/` — generated dissertation-ready plots and plot README.

The included evidence pack was generated with the following computationally constrained command:

```bash
python -m experiments.evaluate_neural_mrp \
  --preset full \
  --eps 0.2,0.5,1.0,2.0 \
  --sample_sizes 500,1000 \
  --scenarios no_bias,nonresponse,shy_voter,privacy_helps \
  --population_n 5000 \
  --trials 1 \
  --mrp_steps 5 \
  --neural_steps 5 \
  --neural_hidden_layers 16 \
  --mrp_batch_size 512 \
  --neural_batch_size 512 \
  --out_dir experiments/outputs/final_neural_evidence_run
```


This run covers all intended epsilon values and bias scenarios, but it is not the exhaustive full preset. It uses one trial per condition and reduced training steps, so it should be treated as computationally constrained final-style evidence.

The generated summary shows:

| Method | Mean overall L1 ↓ | Winner correctness ↑ | Mean runtime sec ↓ |
|---|---:|---:|---:|
| Raw reported distribution | **0.163** | 0.250 | 0.000 |
| RR debiasing | 0.430 | 0.250 | 0.001 |
| Linear RR-aware MRP | 0.176 | 0.344 | 0.005 |
| Neural RR-aware MRP | 0.204 | 0.125 | 0.030 |
| Misreport-aware RR-MRP | 0.211 | **0.906** | 0.768 |
| Learned misreport RR-MRP | 0.229 | 0.094 | 0.006 |

The result should be reported cautiously: neural RR-MRP is a valid privacy-compatible AI model, but this evidence pack does not show that it is generally better than linear RR-aware MRP.

### Preferred exhaustive full preset

If compute time is available, the preferred exhaustive command is:

```bash
python -m experiments.evaluate_neural_mrp \
  --preset full \
  --out_dir experiments/outputs/final_neural_mrp_full
```


Use the generated `summary.csv`, `neural_comparison.csv`, `method_rankings.csv`, and `neural_verdict.csv` from that full run for final quantitative claims if it completes.

## 4. Sensitivity analysis

```bash
python -m experiments.sensitivity_analysis \
  --k 5 \
  --eps "0.1,0.5,1.0,2.0,4.0" \
  --trials 20 \
  --seed 42
```


This checks whether findings change under alternative population and bias assumptions.

## 5. Report tables and recommendations

Generate report tables from a run summary CSV:

```bash
python -m experiments.make_report_tables \
  --summary_csv experiments/outputs/<timestamp>/summary.csv \
  --out_md experiments/outputs/<timestamp>/report_tables.md
```


Find a privacy/sample-size configuration satisfying constraints:

```bash
python -m experiments.recommend_from_summary \
  --summary_csv experiments/outputs/<timestamp>/summary.csv \
  --epsilon_max 2.0 \
  --overall_l1_max 0.15
```

## Reproducibility notes

- Experiments use deterministic seeds where supplied.
- Each experiment saves a `config.json` snapshot for traceability.
- Results may vary slightly across package versions and hardware.
- Synthetic true labels are used only for evaluation after fitting.
- Real respondent data should not contain true labels.
