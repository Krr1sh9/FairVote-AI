# Dashboard Architecture

The Streamlit dashboard is split so that examiner-visible behaviour stays the same while parsing, inference support, plotting, export generation and run orchestration are testable without launching Streamlit.

## Entry point

- `app/streamlit_app.py` is now a thin page router. It sets the page config, creates tabs and calls `app.ui.*` render functions.

## UI modules

- `app/ui/upload.py` is a thin page wrapper; `app/controllers/upload_controller.py` renders the upload-and-estimate workflow and delegates pure logic to services.
- `app/ui/scenario.py` renders the one-click scenario simulator.
- `app/ui/runs.py` renders experiment-run browsing and command execution.
- `app/ui/recommendations.py` renders the optimisation/recommendation workflow.
- `app/ui/about.py` renders demo guidance.

These files may import Streamlit. They should not define core parsing, metric, plotting or export algorithms.

## Testable non-UI modules

- `app/parsing/upload.py` handles CSV/JSONL decoding, flattened respondent JSONL parsing, column detection and display-label loading.
- `app/services/category.py` handles category encoding, grouping, post-stratification utilities and feature-column safety helpers.
- `app/services/inference.py` handles optional estimator availability, method selection, baseline RR estimation and bootstrap CIs.
- `app/services/metrics.py` handles overall and subgroup metric summaries.
- `app/services/exports.py` builds CSVs, Markdown summaries, metadata and ZIP bundles.
- `app/services/scenario.py` contains synthetic scenario generation helpers.
- `app/services/runs.py` contains experiment-run listing, subprocess execution and CSV loading helpers.
- `app/plotting/charts.py` contains Matplotlib plot rendering helpers and optional Matplotlib handling.

## Regression tests

`tests/test_dashboard_modules.py` covers the extracted pure functions: CSV parsing, JSONL parsing, invalid input handling, method-selection fallback, result-summary generation and export-bundle generation.
