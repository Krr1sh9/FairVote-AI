from __future__ import annotations

from pathlib import Path
import pytest

try:
    from ._helpers import py_compile
except ModuleNotFoundError:
    from ._helpers import py_compile


def test_streamlit_app_compiles(project_root: Path):
    # This lightweight check catches syntax errors without launching an
    # interactive Streamlit session in CI.
    app_path = project_root / "app" / "streamlit_app.py"
    if not app_path.exists():
        pytest.skip("Streamlit app not found at app/streamlit_app.py")
    py_compile(app_path)


def test_mrp_vs_baselines_compiles(project_root: Path):
    p = project_root / "experiments" / "mrp_vs_baselines.py"
    if not p.exists():
        pytest.skip("experiments/mrp_vs_baselines.py not found")
    py_compile(p)
