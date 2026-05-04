# FairVote-AI Documentation

FairVote-AI is an AI-assisted privacy-preserving polling framework for studying privacy, utility, and subgroup fairness under Local Differential Privacy.

The privacy mechanism is k-ary Randomized Response. Randomized Response is not AI; it is the local privacy layer that perturbs each respondent's answer before collection. The AI component is the RR-aware Neural MRP model, which learns a nonlinear relationship between demographic features and latent vote intention while training only on privatized reported answers.

## Quick start

### Prerequisites

- Python 3.14.4
- pip
- PyTorch only for neural MRP experiments or neural dashboard inference

### Installation

The supported final-submission setup is Python 3.14.4 with the full development, AI, dashboard, and respondent extras.

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

Optional narrower installs are available for specific tasks, but the command above is the supported all-in-one setup for final verification.

### Running the analyst dashboard

```bash
streamlit run app/streamlit_app.py
```

### Running the respondent server

```bash
python respondent/server.py --port 5001
```

Open `http://localhost:5001`. Edit `respondent/poll_config.json` to configure the question, options, demographic fields, and privacy parameter epsilon.

### Running experiments

```bash
# LDP / central-DP epsilon sweep
python -m experiments.sweep_eps --trials 20 --eps "0.1,0.5,1.0,2.0,4.0"

# MRP and baseline comparison; neural MRP requires the ai extra/PyTorch
python -m experiments.mrp_vs_baselines

# Dedicated neural-justification experiment: fast smoke preset
python -m experiments.evaluate_neural_mrp --preset small

# Dedicated neural-justification experiment: preferred full preset
python -m experiments.evaluate_neural_mrp --preset full
```

Results are saved to `experiments/outputs/<timestamp>/`.

### Running tests

```bash
python -m pytest tests/ -q
```

## Documentation map

| File | Purpose |
|---|---|
| `architecture.md` | System architecture and module map |
| `ai_component.md` | Precise explanation of the RR-aware Neural MRP model |
| `ethics_and_privacy.md` | Privacy, ethics, limitations, and sustainability |
| `reproducing_experiments.md` | Commands and outputs for experiments |
| `api_reference.md` | Respondent server API |
| `dashboard_manual_test.md` | Beginner-safe manual verification checklist for dashboard uploads, inference methods, evidence checks, and demo screenshots |
| `respondent_manual_test.md` | Beginner-safe manual verification checklist for the respondent web app |

## Key components

- **Local Differential Privacy**: k-ary Randomized Response with unbiased RR debiasing estimator.
- **AI inference**: RR-aware Neural MRP trained on privatized reported labels through the RR observation model.
- **Statistical baselines**: RR debiasing and linear RR-aware MRP.
- **Bias scenarios**: nonresponse, shy-voter misreporting, and privacy-helps-honesty simulations.
- **Fairness metrics**: worst-group, weighted, p90, error ratio, and correct-winner metrics.
- **Dashboard**: Streamlit interface for uploaded real or synthetic polling data.

## Final neural-MRP evidence interpretation

The included evidence pack is located at:

```text
experiments/outputs/final_neural_evidence/
```

It contains `config.json`, raw trial results, summary tables, neural-vs-baseline comparisons, method rankings, verdict files, and plots. This is **computationally constrained final-style evidence**, not the exhaustive full preset. It uses all four epsilon values and all four bias scenarios, but only one trial per condition and reduced model training steps.

The generated evidence shows that neural RR-MRP is a real privacy-compatible AI estimator, but it is not generally superior to the simpler linear RR-aware MRP baseline in this run; note that the raw reported distribution is an uncorrected descriptive baseline over privatized reports:

| Method | Mean overall L1 ↓ | Winner correctness ↑ | Mean runtime sec ↓ |
|---|---:|---:|---:|
| Raw reported distribution | **0.163** | 0.250 | 0.000 |
| RR debiasing | 0.430 | 0.250 | 0.001 |
| Linear RR-aware MRP | 0.176 | 0.344 | 0.005 |
| Neural RR-aware MRP | 0.204 | 0.125 | 0.030 |
| Misreport-aware RR-MRP | 0.211 | **0.906** | 0.768 |
| Learned misreport RR-MRP | 0.229 | 0.094 | 0.006 |

The correct conclusion is conditional: neural RR-MRP is a valid AI-assisted extension, but added neural complexity is not automatically justified. In the included evidence pack, among RR-aware corrected/model-based estimators, linear RR-aware MRP is stronger overall, while neural RR-MRP is useful mainly as an evaluated privacy-compatible alternative.

## Important limitations

The neural model is not assumed to be better. It is evaluated against simpler baselines because added model complexity must be justified empirically. Synthetic results are not proof of real election accuracy, and fairness metrics audit disparities rather than guaranteeing fairness. The included evidence pack is computationally constrained, so it should be described as final-style evidence for this submission rather than as an exhaustive full-preset result.


## Respondent app privacy testing

See [`dashboard_manual_test.md`](dashboard_manual_test.md) for a beginner-safe dashboard verification checklist. See [`respondent_manual_test.md`](respondent_manual_test.md) for a beginner-safe respondent app verification checklist. See [`respondent_privacy_testing.md`](respondent_privacy_testing.md) for additional manual checks covering browser-side Randomized Response, the localStorage duplicate-submission guard, debug/audit mode, and server rejection of `true_answer` / true raw-answer fields.
