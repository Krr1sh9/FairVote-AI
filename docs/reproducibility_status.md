# Reproducibility status

The repository now uses one Python compatibility story: Python 3.14 is supported.

## Evidence provenance

The primary final evidence run is `evidence/final/2026-05-05_182242_mrp_vs_baselines/`. Final evidence runs are under `evidence/final/` and include:

- `config.json`
- `manifest.json`
- `environment.json`
- `sha256sums.txt`
- raw and summarised CSVs
- generated plots
- clean `BUNDLE/` directories

Old smoke and superseded intermediate outputs were removed from the submission evidence folder and should not be cited as final evidence.

## Known local verification limitation

This execution environment did not include every optional dependency used by CI, notably Flask, Hypothesis, Ruff and mypy. Tests that require Flask/Hypothesis are therefore skipped locally unless the full `.[dev]` environment is installed. The repository CI workflow installs `.[dev]` and runs those gates.

## Source-tree provenance

This uploaded archive is not a Git checkout, so generated evidence manifests record `git_sha: null`. To replace that missing Git provenance, the remediated archive includes:

- `SOURCE_TREE_SHA256SUMS.txt`: per-file SHA-256 hashes for source, tests, docs and workflow files.
- `SOURCE_TREE_SHA256.txt`: SHA-256 digest of the source-tree hash manifest.
