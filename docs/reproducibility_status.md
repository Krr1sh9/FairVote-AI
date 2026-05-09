# Reproducibility status

The repository now uses one Python compatibility story: Python 3.14 is supported.

## Evidence provenance

The primary final report evidence run for this submitted archive is:

```text
evidence/final/2026-05-06_004647_mrp_vs_baselines/
```

This is a CPU-sized final-style run, not the full `final_evidence` preset. It is the canonical committed evidence because it is complete, repeated, includes paired neural-vs-linear comparisons, and has zero recorded failures. It contains:

- 20 trials per condition;
- sample sizes `500` and `1000`;
- epsilons `0.5`, `1.0` and `2.0`;
- scenarios `nonlinear_interaction`, `sparse_minority_curve`, `privacy_noise_sparse` and `simple_linear`;
- methods `baseline_rr_debias`, `mrp_rr_poststrat`, `hierarchical_rr_mrp_poststrat` and `neural_rr_mrp`;
- `raw_trials.csv`, `summary_with_ci.csv`, `paired_comparisons.csv`, `ablations.csv`, `runtime_profile.csv`, `failures.csv`, `config.json`, `manifest.json`, `environment.json`, `sha256sums.txt` and plots.

Older exploratory final-evidence directories are not included in this archive to avoid ambiguity. The canonical `2026-05-06_004647_mrp_vs_baselines` run is the only final evidence run used for report tables, generated summaries and viva claims.

## Generated report artefacts

Report-ready generated artefacts are included under:

```text
paper/generated/
```

They were generated from the canonical `2026-05-06_004647_mrp_vs_baselines` run. Regenerate them if the canonical evidence run changes.

## Known local verification limitation

This execution environment did not include every optional dependency used by CI, notably Flask, Hypothesis, Ruff and mypy. Tests that require Flask/Hypothesis are therefore skipped locally unless the full `.[dev]` environment is installed. The repository CI workflow installs `.[dev]` and runs those gates.

## Source-tree provenance

This uploaded archive is not a Git checkout, so generated evidence manifests record `git_sha: null`. To replace that missing Git provenance, the archive includes:

- `SOURCE_TREE_SHA256SUMS.txt`: per-file SHA-256 hashes for the submitted source tree.
- `SOURCE_TREE_SHA256.txt`: SHA-256 digest of the source-tree hash manifest.

These manifest files must be regenerated after any final source/documentation edit.
