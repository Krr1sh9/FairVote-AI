# FairVote-AI

FairVote-AI is an **AI-assisted privacy-preserving polling framework**. It combines browser-side Local Differential Privacy with RR-aware statistical and neural inference methods for aggregate and subgroup polling estimates.

The privacy mechanism is **k-ary Randomized Response (RR)**. Randomized Response is **not AI**; it is the mechanism that perturbs each respondent's answer before it leaves the browser. The AI component is the **RR-aware Neural MRP model**, a PyTorch model that learns a nonlinear mapping from demographic features to a latent vote distribution while training only on privatized RR reports.

This project is a final-year research prototype for studying privacy/utility/fairness trade-offs under simulation and uploaded poll data. It is not a production election-forecasting system.

---

For a short examiner-focused run guide, see [`EXAMINER_RUN_GUIDE.md`](EXAMINER_RUN_GUIDE.md).

## Core idea

```text
Respondent browser
  -> on-device k-ary Randomized Response
  -> Flask server stores only privatized reported answers
  -> analyst dashboard / experiments estimate aggregate vote shares
  -> RR debiasing, linear RR-aware MRP, misreport-aware MRP, and neural RR-aware MRP are compared
```

The neural model uses the observation model:

```text
P_theta(true vote | demographics) = softmax(neural_network(demographics))
P(reported answer | true vote)    = known k-ary Randomized Response channel
P(reported answer | demographics) = sum_t P_theta(true=t | demographics) P_RR(reported | true=t)
```

Training maximises the marginal likelihood of the **privatized reported answers**. It does not require true votes from real respondents.

---

## AI component

The AI component is implemented in:

```text
fairvote/inference/mrp/rr_neural_mrp.py
```

The model learns:

```text
demographic/features matrix X
  -> neural network
  -> latent P(true vote | X)
  -> Randomized Response observation channel
  -> likelihood of observed privatized report
```

True labels are used only in synthetic simulations to evaluate error after fitting. In real polling mode, the respondent server and dashboard do not require true votes.

---

## Model comparison

| Method | Uses demographics | Neural | Accounts for RR | Requires true votes | Main role |
|---|---:|---:|---:|---:|---|
| Raw reported distribution | No | No | No | No | Descriptive baseline over privatized reports; not RR-corrected |
| RR debiasing | No | No | Yes | No | Simple aggregate baseline using the known RR channel |
| Linear RR-aware MRP | Yes | No | Yes | No | Statistical MRP baseline with post-stratification |
| Neural RR-aware MRP | Yes | Yes | Yes | No | AI component; tests whether nonlinear demographic modelling helps under RR noise |
| Misreport-aware RR-MRP | Yes | No | Yes + misreport model | No | Bias-aware baseline for simulated shy-voter/misreporting settings |
| Learned misreport-aware RR-MRP | Yes | No | Yes + learned honesty parameter | No | Extension for simulated privacy/“shy voter” settings |

The neural model is evaluated against simpler baselines. The project does **not** assume that neural MRP is automatically better.

---

## Final experiment evidence

The final evidence pack included in this repository is located at:

```text
experiments/outputs/final_neural_evidence/
```

It contains:

```text
config.json
results_trials.csv
results_trials.jsonl
summary.csv
summary.jsonl
neural_comparison.csv
neural_comparison.jsonl
method_rankings.csv
neural_verdict.csv
neural_verdict.json
plots/
```

This evidence pack is **computationally constrained final-style evidence**, not the exhaustive full preset. The full preset was attempted but was too slow for the available execution environment. The included evidence run covers all epsilon values and all bias scenarios, but it uses one trial per condition and reduced training steps.

Configuration used:

```text
epsilons:       0.2, 0.5, 1.0, 2.0
sample sizes:   500, 1000
scenarios:      no_bias, nonresponse, shy_fixed, shy_privacy_helps
trials:         1
population_n:   5000
MRP steps:      5
neural steps:   5
neural layers:  16
```

### Findings from the included evidence pack

The generated results do **not** show that neural RR-MRP is generally superior to simpler baselines.

Across the constrained evidence run:

| Method | Mean overall L1 ↓ | Winner correctness ↑ | Mean runtime sec ↓ |
|---|---:|---:|---:|
| Raw reported distribution | **0.163** | 0.250 | 0.000 |
| RR debiasing | 0.430 | 0.250 | 0.001 |
| Linear RR-aware MRP | 0.176 | 0.344 | 0.005 |
| Neural RR-aware MRP | 0.204 | 0.125 | 0.030 |
| Misreport-aware RR-MRP | 0.211 | **0.906** | 0.768 |
| Learned misreport RR-MRP | 0.229 | 0.094 | 0.006 |

Key interpretation:

- The raw reported distribution had the lowest average overall L1 in this constrained synthetic run, but it is a descriptive baseline over privatized reports rather than an RR-corrected estimator of the latent true distribution.
- Among the RR-aware corrected/model-based estimators, linear RR-aware MRP had the best average overall L1 in the constrained evidence run.
- Neural RR-MRP improved over RR debiasing on average, but did not beat linear RR-aware MRP overall.
- Neural RR-MRP had lower average worst-group L1 than linear MRP in the included analysis, but was worse on weighted group L1, p90 group L1, winner correctness, and runtime.
- Misreport-aware RR-MRP had the highest winner correctness in this constrained run, but was much slower and should not be over-interpreted because there was only one trial per condition.
- The evidence supports a cautious conclusion: neural RR-MRP is a valid privacy-compatible AI estimator, but the included results do not justify claiming that it is generally better than linear RR-aware MRP.

For stronger dissertation evidence, run the fuller experiment locally and replace the constrained evidence tables with the resulting `summary.csv`, `neural_comparison.csv`, `method_rankings.csv`, and `neural_verdict.csv`.

---

## Key features

### Privacy-preserving respondent collection

- Browser-side k-ary Randomized Response implemented in the respondent client.
- The Flask server receives and stores `perturbed_answer`, not the respondent's true answer.
- Requests containing `true_answer` or other true/raw answer fields such as `true_choice`, `selected_answer`, or `raw_vote` are rejected by the server.

### Inference methods

- RR debiasing using the known Randomized Response channel.
- Linear RR-aware MRP with post-stratification.
- Misreport-aware RR-MRP for simulated shy-voter settings where a misreport model is available.
- Learned misreport-aware RR-MRP for simulated privacy/honesty settings.
- RR-aware Neural MRP for nonlinear demographic effects, evaluated against simpler baselines rather than assumed superior.

### Simulation and evaluation

- Synthetic UK-like population generator.
- Sampling, nonresponse, and shy-voter/misreporting scenarios.
- Metrics for overall error, correct winner, worst-group error, weighted group error, p90 group error, and runtime.
- Dedicated neural-justification experiment to test when the neural model helps and when it fails.

### Dashboard

- Streamlit analyst dashboard for uploaded CSV/JSONL responses.
- User-selectable inference methods including RR debiasing, linear RR-aware MRP, neural RR-aware MRP, and misreport-aware RR-MRP where available.
- True labels are optional and intended only for synthetic evaluation, not for real respondent data.

---

## Project structure

| Path | Purpose |
|---|---|
| `fairvote/privacy/` | Randomized Response, RR debiasing, and central-DP baseline mechanisms |
| `fairvote/inference/mrp/` | Linear, misreport-aware, learned-misreport, and neural RR-aware MRP models |
| `fairvote/simulation/` | Synthetic population, sampling, nonresponse, and misreporting models |
| `fairvote/metrics/` | Overall and subgroup error metrics |
| `respondent/` | Flask respondent server and browser-side RR client |
| `app/` | Streamlit analyst dashboard |
| `experiments/` | Simulation and model-comparison scripts |
| `experiments/outputs/final_neural_evidence/` | Included computationally constrained final evidence pack |
| `docs/` | Architecture, ethics/privacy, API, and reproduction notes |
| `tests/` | Unit and smoke tests |

---

## Windows quickstart

If you are using Windows PowerShell, start with the beginner-safe guide:

```text
QUICKSTART_WINDOWS.md
```

It includes virtual environment setup, PowerShell execution-policy fixes, Node.js setup for JavaScript RR tests, respondent-app checks, dashboard usage, and troubleshooting. Note that `source .venv/bin/activate` is for Linux/macOS, not Windows PowerShell.

---

## Installation

FairVote-AI targets **Python 3.14.4** for final submission. The package metadata requires Python `>=3.14,<3.15`. The base install is intentionally lightweight and does not install PyTorch. Install the optional `ai` extra when you want to run RR-aware Neural MRP. If you use another Python version, treat it as untested for this submission.

From the project root, create and activate a Python 3.14.4 virtual environment.

macOS/Linux:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip
```

Windows PowerShell final supported setup:

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev,ai,streamlit,respondent]"
```

You can also run `python --version` after activation to confirm Python 3.14.4.

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

Base package:

```bash
pip install -e .
```

Development/test install without AI:

```bash
pip install -e ".[dev]"
```

AI-enabled install for neural MRP:

```bash
pip install -e ".[ai]"
```

Dashboard install:

```bash
pip install -e ".[streamlit]"
```

Respondent server install:

```bash
pip install -e ".[respondent]"
```

Preferred full development install:

```bash
pip install -e ".[dev,ai,streamlit,respondent]"
```

If your platform needs a specific PyTorch wheel, install the appropriate `torch` build first using the official PyTorch install selector, then install the project with the `ai` extra.

---

## Running tests

Run the full test suite from the project root:

```bash
pip install -e ".[dev,ai,streamlit,respondent]"
pytest -q
```

Run targeted neural tests:

```bash
pytest tests/test_rr_neural_mrp.py -q
```

Run experiment-related tests:

```bash
pytest tests/test_evaluate_neural_mrp.py -q
pytest tests/test_mrp_vs_baselines.py -q
```

Run the slow MRP integration test when preparing the final submission:

```bash
FV_RUN_SLOW=1 pytest tests/test_mrp_vs_baselines.py -q
```

---

## Running the respondent server

```bash
python respondent/server.py --port 5001
```

Open `http://127.0.0.1:5001`. Responses are written to:

```text
respondent/data/responses.jsonl
```

This file should contain privatized reported answers and demographics. It should not contain true answers.

---

## Running the dashboard

```bash
streamlit run app/streamlit_app.py
```

In the dashboard, upload a CSV or JSONL response file, select the reported-answer column, select demographic feature columns, choose an inference method, and run estimation. For real polling data, do not provide true labels. If synthetic data includes a true-label column, the dashboard treats it as evaluation-only.

---


### CI note

GitHub Actions installs the full development extras and Node.js, then runs the Python test suite plus targeted neural, respondent, JavaScript RR, experiment, and dashboard checks. Very slow experiment checks are local-only; run them with `FV_RUN_SLOW=1`.


## Running experiments

### Small neural MRP smoke experiment

```bash
python -m experiments.evaluate_neural_mrp --preset small
```

This verifies the experiment pipeline and output files. It should not be treated as final evidence.

### Included final evidence configuration

The evidence pack in `experiments/outputs/final_neural_evidence/` was generated with:

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

This is computationally constrained evidence, not the exhaustive full preset.

### Preferred full preset for stronger dissertation evidence

```bash
python -m experiments.evaluate_neural_mrp \
  --preset full \
  --out_dir experiments/outputs/final_neural_mrp_full
```

The full preset is computationally heavier. Use its generated `summary.csv`, `neural_comparison.csv`, `method_rankings.csv`, and `neural_verdict.csv` if you want stronger final performance claims.

### Standard MRP/baseline comparison

```bash
python -m experiments.mrp_vs_baselines
```

Use `--disable_neural` for a baseline-only run without PyTorch:

```bash
python -m experiments.mrp_vs_baselines --disable_neural
```

### LDP vs central-DP sweep

```bash
python -m experiments.sweep_eps --trials 10 --eps 0.5,1.0,2.0 --n_samples 1000,5000,10000
```

---

## Privacy and ethics notes

- Randomized Response is the privacy mechanism, not the AI component.
- Local Differential Privacy protects the submitted answer value, but it does not guarantee full respondent anonymity.
- Demographic combinations, small groups, timing, IP/network metadata, and deployment logs still require care.
- True labels in synthetic CSVs are evaluation-only. Real respondent exports should not contain true votes.
- Fairness is audited with subgroup error metrics; it is not guaranteed by the model or privacy mechanism.

---

## Limitations

- The included final evidence pack is computationally constrained. It uses one trial per condition and reduced training steps.
- Neural MRP can overfit, underfit, or be slower than linear MRP. It is included because it may capture nonlinear demographic structure, not because it is guaranteed to perform better.
- Synthetic validation is not proof of real election accuracy. It tests performance under the assumptions of the simulator.
- LDP adds noise. Low epsilon values can make all estimators inaccurate, especially for small subgroups.
- Fairness is audited with subgroup metrics; it is not guaranteed.

---

## License

This project is distributed under the MIT license. See `LICENSE`.


## Respondent privacy testing

For a beginner-safe dashboard verification checklist, see [`docs/dashboard_manual_test.md`](docs/dashboard_manual_test.md). For a beginner-safe respondent web app verification checklist, see [`docs/respondent_manual_test.md`](docs/respondent_manual_test.md). For additional manual checks of browser-side Randomized Response, the localStorage duplicate-submission guard, audit mode (`?audit=1`), and server rejection of `true_answer` / true raw-answer fields, see [`docs/respondent_privacy_testing.md`](docs/respondent_privacy_testing.md).
