from __future__ import annotations

from pathlib import Path

from scripts.verify_lockfile import direct_dependency_names, parse_lock
from scripts.verify_privacy_evidence import REQUIRED as PRIVACY_REQUIRED

ROOT = Path(__file__).resolve().parents[1]


def test_lockfile_is_exact_pinned_and_covers_direct_dependencies() -> None:
    pins = parse_lock(ROOT / "requirements.lock.txt")
    required = direct_dependency_names(ROOT / "pyproject.toml")
    assert required <= set(pins)
    assert len(pins) >= len(required)


def test_committed_privacy_evidence_artifacts_exist() -> None:
    for path, needles in PRIVACY_REQUIRED.items():
        full = ROOT / path
        assert full.exists(), f"missing privacy evidence artefact: {path}"
        text = full.read_text(encoding="utf-8")
        for needle in needles:
            assert needle in text


def test_final_evidence_runs_are_not_smoke_or_sanity_runs() -> None:
    final_runs = sorted((ROOT / "evidence" / "final").glob("*_mrp_vs_baselines"))
    assert final_runs, "expected at least one verifiable evidence run under evidence/final"
    for run in final_runs:
        readme = run / "README.md"
        assert readme.exists(), f"missing README for final evidence run: {run}"
        lowered = readme.read_text(encoding="utf-8").lower()
        assert "smoke/sanity" not in lowered
        assert "do not use" not in lowered
        assert "non-final" not in lowered


def test_smoke_runs_are_quarantined_under_smoke_directory() -> None:
    smoke_runs = sorted((ROOT / "evidence" / "smoke").glob("*_mrp_vs_baselines"))
    assert smoke_runs, "expected retained smoke runs to be quarantined under evidence/smoke"
    for run in smoke_runs:
        readme = run / "README.md"
        assert readme.exists(), f"missing README for smoke evidence run: {run}"
        lowered = readme.read_text(encoding="utf-8").lower()
        assert "smoke" in lowered or "sanity" in lowered or "do not use" in lowered
