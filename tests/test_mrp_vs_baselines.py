from __future__ import annotations

from pathlib import Path
import csv
import importlib
import importlib.util
import types
import pytest


@pytest.mark.slow
def test_mrp_vs_baselines_smoke_run(tmp_path: Path, project_root: Path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("neural smoke test requires optional ai extra / PyTorch")
    # Run the module with very small settings so it finishes quickly.
    # Note: this still runs baseline + MRP + misreport + learned-misreport methods.
    out_dir = tmp_path / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    import subprocess, sys
    cmd = [
        sys.executable, "-m", "experiments.mrp_vs_baselines",
        "--trials", "1",
        "--eps", "1.0",
        "--scenarios", "no_bias",
        "--population_n", "800",
        "--n_sample", "120",
        "--mrp_steps", "5",
        "--mrp_batch_size", "128",
        "--neural_steps", "3",
        "--neural_batch_size", "128",
        "--neural_hidden_layers", "8",
        "--out_dir", str(out_dir),
    ]
    proc = subprocess.run(cmd, cwd=str(project_root), text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    # Should have created a run folder with summary.csv
    run_folders = sorted(out_dir.glob("*_mrp_vs_baselines*"))
    assert run_folders, f"No run folder created. stdout={proc.stdout} stderr={proc.stderr}"
    summary_path = run_folders[-1] / "summary.csv"
    results_path = run_folders[-1] / "results_trials.csv"
    assert summary_path.exists()
    assert results_path.exists()

    with summary_path.open(newline="", encoding="utf-8") as f:
        methods = {row["method"] for row in csv.DictReader(f)}
    assert "raw_reported_distribution" in methods
    assert "baseline_rr_debias" in methods
    assert "mrp_rr_poststrat" in methods
    assert "mrp_misreport_rr_poststrat" in methods
    assert "mrp_learned_misreport_rr_poststrat" in methods
    assert "neural_rr_mrp" in methods

    with results_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    neural_rows = [r for r in rows if r["method"] == "neural_rr_mrp"]
    assert neural_rows
    assert neural_rows[0]["neural_final_loss"]
    assert "runtime_sec" in rows[0]
    assert "winner_correct" in rows[0]


def test_ts_run_dir_unique(monkeypatch, tmp_path: Path, project_root: Path):
    # Import the module and force datetime.now() to return a fixed value,
    # then pre-create the first directory to ensure suffix behaviour works.
    mrp_mod = importlib.import_module("experiments.mrp_vs_baselines")

    class _FakeDT:
        @classmethod
        def now(cls):
            from datetime import datetime
            return datetime(2026, 1, 27, 17, 16, 40)

        @staticmethod
        def strftime(*args, **kwargs):
            raise RuntimeError("not used")

    # monkeypatch the datetime symbol in that module (it imported datetime from datetime)
    monkeypatch.setattr(mrp_mod, "datetime", _FakeDT)

    base = tmp_path / "outputs"
    base.mkdir(parents=True, exist_ok=True)

    # pre-create the first run folder expected
    first = base / "2026-01-27_171640_mrp_vs_baselines"
    first.mkdir(parents=True, exist_ok=False)
    (first / "plots").mkdir(parents=True, exist_ok=True)

    run_dir = mrp_mod._ts_run_dir(base, "mrp_vs_baselines")
    assert run_dir != first
    assert run_dir.name.startswith("2026-01-27_171640_mrp_vs_baselines_")
    assert run_dir.exists()
    assert (run_dir / "plots").exists()
