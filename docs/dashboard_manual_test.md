# FairVote-AI Dashboard Manual Test Checklist

This checklist verifies the Streamlit analyst dashboard path for a fresh examiner/demo run. It is intentionally manual because the dashboard is interactive.

The dashboard is for **analysis of privatized reports**. It is not production election software, and it does not prove real-world election accuracy.

## 1. Start the dashboard

From the project root, with the virtual environment activated:

```bash
streamlit run app/streamlit_app.py
```

Expected result:

- Streamlit opens a local browser tab, usually at `http://localhost:8501`.
- The page title is `FairVote-AI`.
- The main tabs include `Upload & Estimate`, `Scenario Simulator`, `Simulation & Runs`, `Recommendations`, and `About`.

If Streamlit is missing, install the dashboard dependencies:

```bash
pip install -e ".[streamlit,ai]"
```

For the full development environment:

```bash
pip install -e ".[dev,ai,streamlit,respondent]"
```

## 2. Test synthetic CSV upload

In the `Upload & Estimate` tab:

1. Click the poll upload box.
2. Upload one of the synthetic CSV files from:

```text
app/data/
```

Example files include names like:

```text
poll_no_bias_...
poll_nonresponse_...
poll_shy_privacy_helps_...
```

Expected result:

- The dashboard reports that the poll CSV loaded successfully.
- The sidebar shows column-selection controls.

Important: these CSV files are **synthetic demonstration data**. Columns such as `true_choice` are included only for evaluation. Real respondent exports should not contain true votes.

## 3. Test population CSV upload

Upload the population CSV:

```text
app/data/population.csv
```

Expected result:

- The dashboard reports that the population CSV loaded successfully.
- Post-stratification controls become available in the sidebar.

If using learned MRP post-stratification, select post-stratification keys that match the feature columns used by the learned model.

## 4. Select the reported-answer column

For synthetic CSV files, select:

```text
reported_choice
```

For respondent-server JSONL exports, select:

```text
perturbed_answer
```

Expected result:

- The dashboard detects the number of unique response categories.
- The selected column should contain privatized reported answers, not true answers.

## 5. Confirm `true_choice` is optional and evaluation-only

If a synthetic CSV contains `true_choice`, the sidebar may automatically select it as the optional true-choice column.

This is correct only for synthetic evaluation.

For real respondent data:

- set the true-choice column to `(none)`;
- do not upload true answers;
- use respondent JSONL exports containing `perturbed_answer` and demographics.

Expected wording in the dashboard:

```text
Optional: true choice column (only for evaluation / synthetic data)
```

## 6. Test RR debiasing

In the sidebar:

1. Select `RR debiasing`.
2. Confirm `epsilon` is set correctly, usually from the uploaded data or manually.
3. Run/inspect the estimate.

Expected result:

- The dashboard displays an estimated vote-share table.
- If synthetic truth is selected, it may display true-error metrics.
- If no truth is selected, it should not require true labels.

## 7. Test linear RR-aware MRP

In the sidebar:

1. Select `Linear RR-aware MRP`.
2. Select demographic feature columns such as `region`, `age_band`, or other available columns.
3. Use modest demo settings:

```text
training steps = 50
batch size = 512
learning rate = default
seed = 0
```

Expected result:

- The model fits without requiring true votes.
- The dashboard displays a sample-averaged MRP estimate.
- If population data and matching post-stratification keys are supplied, it displays a post-stratified estimate.

## 8. Test neural RR-aware MRP

In the sidebar:

1. Select `Neural RR-aware MRP`.
2. Select demographic feature columns.
3. Use recommended demo settings:

```text
training steps = 50
batch size = 512
model size = Small: 16
learning rate = default
seed = 0
dropout = 0.0
weight decay = default
```

Expected result:

- The dashboard warns that neural MRP is a learned model and should be compared against baselines.
- The model fits without requiring true votes.
- The dashboard displays the neural MRP estimate.

Do not claim that neural MRP is automatically better. In the included constrained evidence pack, neural MRP was valid and privacy-compatible but did not consistently outperform linear RR-aware MRP.

## 9. Test misreport-aware MRP, if available

If the option appears:

1. Select `Misreport-aware RR-MRP`.
2. Select feature columns.
3. Choose the shy/misreported category and honesty value.
4. Use modest demo settings:

```text
training steps = 50
batch size = 512
```

Expected result:

- The model fits and displays an estimate.
- It is clearly a model of possible behavioural misreporting before Randomized Response, not a privacy mechanism by itself.

## 10. Test respondent JSONL upload

First run the respondent app and submit one or more test responses:

```bash
python respondent/server.py --port 5001
```

Then upload:

```text
respondent/data/responses.jsonl
```

Expected result:

- The dashboard loads the JSONL file.
- The available reported-answer column is `perturbed_answer`.
- Demographic fields appear as flattened columns.
- There should be no `true_answer`, `true_choice`, `selected_answer`, or `raw_vote` column.

If the file is empty, submit a test response in the respondent app first.

## 11. Test fairness/subgroup metrics

Select group columns such as:

```text
region
age_band
```

Expected result:

- If synthetic `true_choice` is selected, subgroup metrics are true error metrics.
- If no truth is selected, the dashboard shows divergence from the overall estimate as a robustness proxy.

Do not describe this as a fairness guarantee. It is a subgroup-error audit.

## 12. Test exports and evidence consistency

If the dashboard offers downloads/exports, verify that exported result columns use method-specific names such as:

```text
baseline_rr_p
linear_mrp_sample_p
neural_mrp_sample_p
misreport_mrp_sample_p
```

Expected result:

- The output makes clear which method produced each estimate.
- No true labels are required for real respondent data.

## Recommended demo settings

For a fast examiner demo:

```text
Synthetic poll CSV: app/data/poll_no_bias_*.csv
Population CSV: app/data/population.csv
Reported column: reported_choice
True column: true_choice only for synthetic evaluation
Feature columns: region, age_band if available
Group columns: region, age_band if available
RR debiasing: default settings
Linear MRP: training steps = 50, batch size = 512
Neural MRP: Small: 16, training steps = 50, batch size = 512
Misreport-aware MRP: training steps = 50, batch size = 512
```

## Screenshots to capture for report or video

Recommended screenshots:

1. Dashboard home with `Upload & Estimate` tab visible.
2. Successful synthetic CSV upload.
3. Population CSV upload and post-stratification controls.
4. Sidebar method selector showing RR debiasing, linear MRP, neural MRP, and misreport-aware MRP if available.
5. Neural MRP settings showing `Small: 16`, `training steps = 50`, and `batch size = 512`.
6. Neural MRP warning explaining it should be compared against baselines.
7. Estimate table for RR debiasing.
8. Estimate table for linear RR-aware MRP.
9. Estimate table for neural RR-aware MRP.
10. Fairness/subgroup metric section.
11. Final evidence folder or plots from `experiments/outputs/final_neural_evidence/`.

## Known limitations

- The dashboard is an analyst tool, not a production election system.
- True labels are allowed only for synthetic evaluation.
- Real respondent JSONL exports should contain privatized answers only.
- Neural MRP can overfit, underfit, or perform worse than linear MRP.
- Fairness metrics audit subgroup error; they do not guarantee fairness.
- Local Differential Privacy protects answer values, not full anonymity.
- The included final evidence pack is computationally constrained and should not be described as exhaustive full-preset evidence.

## Troubleshooting

### `streamlit` is not recognized

Install the dashboard dependencies:

```bash
pip install -e ".[streamlit,ai]"
```

or the full development environment:

```bash
pip install -e ".[dev,ai,streamlit,respondent]"
```

### Neural MRP is unavailable

Install the AI dependency:

```bash
pip install -e ".[ai,streamlit]"
```

### Uploaded JSONL has no rows

Submit at least one response through the respondent app first, then re-upload:

```text
respondent/data/responses.jsonl
```

### Post-stratification does not appear

Upload `app/data/population.csv` and select matching post-stratification key columns.

### Model fitting is slow

For demos, use:

```text
training steps = 50
batch size = 512
Neural model size = Small: 16
```

### Metrics differ between runs

Some methods use randomized initialization or simulated/randomized data. Set the seed where available and use the same uploaded files to make comparisons more repeatable.
