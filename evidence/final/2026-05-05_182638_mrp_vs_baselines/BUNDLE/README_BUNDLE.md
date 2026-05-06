# Results bundle
- Source run directory: `evidence/final/2026-05-05_182638_mrp_vs_baselines`
- Bundle directory: `evidence/final/2026-05-05_182638_mrp_vs_baselines/BUNDLE`

## Contents

- summary_with_ci.csv
- summary.csv
- raw_trials.csv
- results_trials.csv
- paired_comparisons.csv
- ablations.csv
- runtime_profile.csv
- manifest.json
- environment.json
- sha256sums.txt
- failures.csv
- README.md
- config.json
- config.json
- plots/ (png/pdf/md)

## How to reproduce

1) Run the experiment:
   - `python -m experiments.mrp_vs_baselines ...`

2) Build tables:
   - `python -m experiments.make_report_tables --summary_csv <run_dir>/summary_with_ci.csv --include_all ...`

3) Learned honesty:
   - `python -m experiments.summarise_learned_honesty --run_dir <run_dir>`

4) Recommendation / Pareto:
   - `python -m experiments.recommend_from_summary --summary_csv <run_dir>/summary_with_ci.csv --write_pareto`
