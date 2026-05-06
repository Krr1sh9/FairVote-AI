"""Helpers for running experiment modules and reading output folders."""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path


def list_runs(base: Path) -> list[Path]:
    if not base.exists():
        return []
    runs = [p for p in base.iterdir() if p.is_dir()]
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs


def run_module(module: str, argv: list[str], cwd: Path) -> tuple[int, str]:
    cmd = [sys.executable, "-m", module, *argv]
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, check=False)
    out = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode, out


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
