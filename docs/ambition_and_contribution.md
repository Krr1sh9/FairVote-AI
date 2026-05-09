# Ambition and contribution statement

The project should not be framed as “a privacy polling dashboard”. That undersells the work and makes it look like a routine implementation. The defensible ambitious contribution is:

> A reproducible research-engineering benchmark for evaluating locally-private polling estimators under sampling bias, nonresponse, privacy noise and sparse subgroup error, with a new RR-aware hierarchical partial-pooling MRP estimator and controlled comparisons against linear and neural baselines. The wider pipeline also contains oracle and misreport-aware extensions for supplementary analysis.

## What makes the work ambitious

| Ambition component | Required artefact in this repository |
|---|---|
| Locally private collection boundary | `respondent/static/rr.js`, `respondent/static/app.js`, `respondent/server.py`, `evidence/privacy/` |
| New estimator contribution | `fairvote/inference/mrp/hierarchical.py`, `tests/test_hierarchical_mrp.py` |
| Theory and implementation validation | `docs/theory_validation.md`, `experiments/theory_validation.py`, `evidence/final/theory/` |
| Controlled benchmark | `experiments/pipeline/`, `experiments/mrp_vs_baselines.py`, final `summary_with_ci.csv` |
| Robustness under hard conditions | `experiments/pipeline/scenarios.py`, `experiments/pipeline/presets.py` |
| Report-ready output | `paper/fairvote_ai_results.md`, generated `paper/generated/fairvote_ai_results.md` from the canonical final evidence run |
| Reproducibility | `requirements.lock.txt`, `Makefile`, manifests and hashes in evidence runs |
| Honest limitations | `docs/privacy_boundary.md`, `docs/evidence_interpretation.md`, generated `FINAL_RESULTS.md` |

## Main research claims that must be evidence-backed

No claim below should appear in the final report unless it is supported by the cited evidence files. Claims 1–3 are supported by the canonical 2026-05-06 run; misreporting/privacy-help claims require an explicitly cited supplementary run and should not be attributed to the canonical run.

1. **RR-aware MRP vs direct debiasing** — use `summary_with_ci.csv` and `ablations.csv`.
2. **Hierarchical partial pooling under sparse subgroups** — use sparse-scenario rows for `hierarchical_rr_mrp_poststrat`, especially worst-group error.
3. **Neural RR-MRP under nonlinear structure** — use `paired_comparisons.csv`; do not claim neural improvement from unpaired averages.
4. **Privacy/misreporting trade-off** — use only a supplementary run that actually includes `shy_privacy_helps`, `privacy_tradeoff` or `privacy_helps`; report it as supplementary rather than as the canonical final-run result.
5. **Theory-consistent RR behaviour** — use `docs/theory_validation.md` and `evidence/final/theory/theory_validation.json`.

## Strongest likely contribution

The most defensible report-ready conclusion is not “neural model always improves”. A stronger and more honest contribution is conditional:

> Hierarchical partial pooling and RR-aware likelihoods provide a principled baseline for locally-private polling; neural RR-MRP is only justified when repeated paired evidence shows stable gains under nonlinear demographic response surfaces that simpler RR-aware poststratification cannot represent.

If the final evidence shows that linear or hierarchical MRP beats neural methods, that is still a valid research contribution. Negative or mixed findings are acceptable if the pipeline is rigorous and the limitations are explicit.

## What would block an Outstanding ambition mark

- Smoke-only evidence under `evidence/final/`.
- Empty `paired_comparisons.csv` while making neural claims.
- No sparse-cell evidence for hierarchical partial pooling.
- No report-ready table/figure generated from committed results.
- Claims that demographics are private merely because answers are randomized.
- A final paper that describes expected results instead of actual results.
