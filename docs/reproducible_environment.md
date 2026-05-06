# Reproducible environment and final-evidence commands

This document defines the dependency and reproduction boundary for examiner use. It exists so the repository does not rely on vague “install whatever works” instructions.

## Locked install

The committed lockfile is:

```text
requirements.lock.txt
```

It is exact-pinned. The structural verifier rejects unpinned requirements and checks that every direct dependency declared in `pyproject.toml` appears in the lock:

```bash
python -m scripts.verify_lockfile
```

Install the locked environment with:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m pip install -e . --no-deps
```

The final `--no-deps` step is intentional: dependency versions should come from the lockfile, not from a fresh resolver pass.

## Regenerating the lockfile

When dependencies change, regenerate the lockfile with:

```bash
make lock
```

This runs:

```bash
uv pip compile pyproject.toml \
  --all-extras \
  --python-version 3.14 \
  --python-platform x86_64-manylinux_2_28 \
  --torch-backend cpu \
  --output-file requirements.lock.txt
```

If the evidence run is produced on another platform, regenerate the lock on that platform and preserve the regenerated lockfile inside the evidence bundle.

## One-command reproduction

For a fast but non-final check:

```bash
make reproduce-examiner OUT=evidence/check
```

This verifies the lockfile, runs the reproduction preset, runs theory validation, generates report-ready Markdown from the latest run, builds a bundle, and verifies the run structurally.

For the final evidence run:

```bash
make reproduce-final OUT=evidence/final
```

This is intentionally heavier. It runs the `final_evidence` preset, theory validation, and the standard bundle step. After the timestamped run directory is known, run `make report RUN=evidence/final/<RUN_DIR>` and `make verify-final RUN=evidence/final/<RUN_DIR>` to generate report-ready Markdown and apply the final-evidence verifier.

## Final-evidence verification

After manually running a final experiment, verify it with:

```bash
make verify-final RUN=evidence/final/<RUN_DIR>
```

The verifier rejects:

- smoke/sanity/non-final README wording;
- missing required files;
- non-empty `failures.csv`;
- fewer than 20 repeated rows per summary cell;
- skipped result cells;
- missing paired comparisons when neural methods are present;
- missing plots;
- invalid JSON provenance files.

The point is harsh by design: a weak run should fail before it reaches the report.

## Privacy-evidence verification

The privacy artefacts are also checked by CI:

```bash
make privacy-evidence
```

This checks the committed files under `evidence/privacy/`. It does not replace the executable respondent tests; it ensures the assessment evidence appendix is present.
