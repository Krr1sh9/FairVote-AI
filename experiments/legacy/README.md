# Legacy experiment helpers

These modules are retained only for backwards compatibility with old notebooks,
manual checks and regression tests. They are **not** the canonical final-evidence
path. New evidence must use `python -m experiments.mrp_vs_baselines` and the
`experiments.pipeline` package.

Thin wrappers remain at the old module paths so older commands fail less
surprisingly while the assessment experiment architecture stays centred on
the canonical pipeline.
