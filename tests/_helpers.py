from __future__ import annotations

import csv
import inspect
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def call_with_supported_kwargs(fn, /, **kwargs):
    """Call fn with only the keyword arguments it supports.

    Also bridges common naming differences across refactors, e.g.:
    - true_categories vs truth/categories
    - rng vs seed (int)
    """
    import inspect

    import numpy as np

    sig = inspect.signature(fn)
    params = sig.parameters

    # Map aliases for the main required categories argument.  The codebase has
    # undergone naming refactors (e.g. truth -> true_categories); this bridge
    # keeps older tests working with the current API.
    if "true_categories" in params and "true_categories" not in kwargs:
        if "truth" in kwargs:
            kwargs["true_categories"] = kwargs["truth"]
        elif "categories" in kwargs:
            kwargs["true_categories"] = kwargs["categories"]

    # Map aliases for reported categories argument (bootstrap CI / estimators)
    if "reported_categories" in params and "reported_categories" not in kwargs:
        if "reported" in kwargs:
            kwargs["reported_categories"] = kwargs["reported"]
        elif "reports" in kwargs:
            kwargs["reported_categories"] = kwargs["reports"]
        elif "y" in kwargs:
            kwargs["reported_categories"] = kwargs["y"]

    # Convert an integer seed to a Generator object when the function expects
    # an rng parameter.  This avoids polluting the global NumPy random state.
    if "rng" in params and "rng" not in kwargs and "seed" in kwargs and kwargs["seed"] is not None:
        try:
            kwargs["rng"] = np.random.default_rng(int(kwargs["seed"]))
        except Exception:
            kwargs["rng"] = np.random.default_rng(0)

    supported = {k: v for k, v in kwargs.items() if k in params}

    # If the function has a required positional-only arg (rare here), raise a clearer error
    return fn(**supported)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_py(cmd: list[str], cwd: Path | None = None, timeout_s: int = 120) -> subprocess.CompletedProcess[str]:
    """Run a python subprocess with text output."""
    return subprocess.run(
        [sys.executable, *cmd],
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )


def py_compile(path: Path) -> None:
    """Validate Python syntax without writing __pycache__ files.

    The standard ``python -m py_compile`` writes bytecode next to the source
    file, which can fail in read-only or archive-extracted test environments.
    Compiling the source string still catches SyntaxError while keeping tests
    side-effect free.
    """
    run_py(
        [
            "-c",
            (
                "import pathlib, sys; "
                "p = pathlib.Path(sys.argv[1]); "
                "compile(p.read_text(encoding='utf-8'), str(p), 'exec')"
            ),
            str(path),
        ],
        cwd=path.parent,
        timeout_s=120,
    )
