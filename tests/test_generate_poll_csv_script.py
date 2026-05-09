from __future__ import annotations

import csv
from pathlib import Path

import pytest

try:
    from ._helpers import read_csv_dicts, run_py
except ModuleNotFoundError:
    from ._helpers import read_csv_dicts, run_py


@pytest.mark.parametrize(
    "scenario,eps",
    [
        ("no_bias", "1.0"),
        ("nonresponse", "1.0"),
        ("shy_privacy_helps", "0.5"),
    ],
)
def test_generate_poll_csv_script(tmp_path: Path, scenario: str, eps: str, project_root: Path):
    candidates = [
        project_root / "experiments" / "generate_poll_csv.py",
        project_root / "generate_poll_csv.py",
        project_root / "app" / "generate_poll_csv.py",
        project_root / "scripts" / "generate_poll_csv.py",
    ]
    script = next((p for p in candidates if p.exists()), None)
    if script is None:
        pytest.skip("generate_poll_csv.py not found in repo (ok if you removed the script).")

    out_dir = tmp_path / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    proc = run_py(
        [
            str(script),
            "--out_dir",
            str(out_dir),
            "--n",
            "200",
            "--k",
            "5",
            "--epsilon",
            eps,
            "--scenario",
            scenario,
            "--include_truth",
        ],
        cwd=project_root,
        timeout_s=120,
    )
    assert "Wrote poll CSV" in proc.stdout

    # discover outputs
    poll_files = sorted(out_dir.glob("poll_*.csv"))
    assert poll_files, "Expected poll_*.csv in out_dir"
    pop_file = out_dir / "population.csv"
    assert pop_file.exists()

    rows = read_csv_dicts(poll_files[-1])
    assert len(rows) > 0

    # Required columns
    header = set(rows[0].keys())
    assert "reported_choice" in header
    assert "region" in header
    assert "age_band" in header
    assert "true_choice" in header

    # Validate reported_choice is int-like in range [0,k)
    for r in rows[:50]:
        v = int(float(r["reported_choice"]))
        assert 0 <= v < 5

    # Population CSV has counts
    pop_rows = read_csv_dicts(pop_file)
    assert len(pop_rows) > 0
    pop_header = set(pop_rows[0].keys())
    assert "count" in pop_header
