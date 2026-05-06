# Demo data policy

`app/data/` contains small dashboard-safe demo files only. These files do **not** include synthetic `true_choice` or `stated_choice` columns, because the dashboard upload path is intended to mirror the real respondent protocol where only privatized/reported answers are available.

Synthetic truth-labelled fixtures used for the dashboard Upload & Estimate evaluation/demo path live under `fixtures/synthetic_with_truth/` and are clearly separated from the sanitized app demo path.
