# FairVote-AI: RR-aware MRP for locally private polling

This is the report-ready result artefact template. It is intentionally conservative: final numeric claims must be generated from a completed evidence run with:

- non-smoke `README.md` wording,
- empty `failures.csv`,
- at least 20 repeated trials per claimed condition,
- non-empty `paired_comparisons.csv` for any neural-vs-linear claim,
- committed `manifest.json`, `environment.json`, `sha256sums.txt`, plots and tables.

After running final evidence, regenerate the final paper draft and claim index with:

```bash
python -m experiments.write_publication_result \
  --run_dir evidence/final/<RUN_DIR> \
  --out_dir paper/generated
```

Do not replace this template with prose copied from a smoke run.

## Abstract

FairVote-AI evaluates locally private polling estimators under k-ary Randomized Response, sampling bias, sparse subgroup structure, nonlinear response patterns and strategic misreporting. The implementation includes a browser-side LDP respondent prototype, raw-answer rejection on the server, strict analyst upload validation, analytical RR debiasing, linear RR-aware MRP, hierarchical partial-pooling RR-aware MRP, optional neural RR-MRP, and a reproducible evidence pipeline.

## Research questions

1. Can RR-aware MRP reduce aggregate and subgroup error compared with direct randomized-response debiasing?
2. Does true hierarchical partial pooling improve sparse-cell robustness without observing raw answers?
3. Do high-capacity neural RR-MRP models justify their complexity under nonlinear response surfaces?
4. Can privacy help under strategic misreporting enough to offset randomized-response variance?
5. Are the privacy, unbiasedness and uncertainty claims supported by derivation and Monte Carlo checks?

## Method summary

The central modelling constraint is that training observes only randomized-response reports. RR-aware MRP estimators optimise the reported-label likelihood through the known RR transition matrix, then poststratify estimated latent choice probabilities over population cells. The hierarchical estimator adds feature-level varying effects with shrinkage so sparse subgroup estimates share information without storing raw answers.

## Result-generation rule

The final report-ready result must be generated from committed evidence files by `experiments.write_publication_result`. That script writes:

- `FINAL_RESULTS.md`,
- `CLAIM_TO_EVIDENCE_INDEX.md`,
- a generated `fairvote_ai_results.md` containing only evidence-backed status checks and tables.

The generated report explicitly marks claims as unsupported when the evidence is smoke-only, under-repeated, failed, skipped, or missing paired comparisons.

## Limitations to keep in the final generated version

Randomized response protects answer values only. Demographic combinations are not automatically private and require separate minimisation, rare-cell reporting and export controls. Synthetic-data validation does not replace an ethically approved human-participant study. Neural results require paired repeated-trial comparisons; without those, neural claims must be removed or labelled unsupported.
