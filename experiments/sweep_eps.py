"""Compatibility wrapper for :mod:`experiments.legacy.sweep_eps`.

The canonical final-evidence path is now `experiments.mrp_vs_baselines` and
`experiments.pipeline`. This wrapper preserves old imports and commands while
quarantining legacy implementation code away from the main experiment package.
"""

from __future__ import annotations

from experiments.legacy.sweep_eps import *  # noqa: F403
from experiments.legacy.sweep_eps import main as _legacy_main

if __name__ == "__main__":  # pragma: no cover - command-line compatibility
    raise SystemExit(_legacy_main())
