from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import pytest


def test_fairvote_imports():
    pytest.importorskip("fairvote")


def test_fairvote_main_help(project_root: Path):
    # This is a lightweight check that the module at least starts and prints help.
    # If you don't ship fairvote.main, this will skip.
    try:
        __import__("fairvote.main")
    except Exception:
        pytest.skip("fairvote.main not importable (ok if not used)")

    proc = subprocess.run([sys.executable, "-m", "fairvote.main", "--help"], cwd=str(project_root), text=True, capture_output=True)
    assert proc.returncode == 0
