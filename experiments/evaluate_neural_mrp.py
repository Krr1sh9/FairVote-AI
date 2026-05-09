"""Compatibility wrapper for :mod:`experiments.legacy.evaluate_neural_mrp`.

The canonical final-evidence path is now ``experiments.mrp_vs_baselines`` and
``experiments.pipeline``.  This module keeps old imports/commands working while
explicitly re-exporting the tested public compatibility API.  Do not use
``import *`` here: underscored helpers used by legacy tests are not exported by
star-imports and that caused a submission-breaking regression.
"""
from __future__ import annotations

from experiments.legacy.evaluate_neural_mrp import (
    COMPARISON_BASELINES,
    CORE_METHODS,
    DISPLAY_SCENARIO_NAMES,
    HIGHER_IS_BETTER_METRICS,
    LOWER_IS_BETTER_METRICS,
    PRESETS,
    SCENARIO_ALIASES,
    NeuralMRPPreset,
    _display_scenario,
    _normalise_scenarios,
    _parse_sample_sizes,
    _write_jsonl,
    build_method_rankings,
    build_neural_comparison,
    build_neural_verdict,
    main as _legacy_main,
)

__all__ = [
    "COMPARISON_BASELINES",
    "CORE_METHODS",
    "DISPLAY_SCENARIO_NAMES",
    "HIGHER_IS_BETTER_METRICS",
    "LOWER_IS_BETTER_METRICS",
    "PRESETS",
    "SCENARIO_ALIASES",
    "NeuralMRPPreset",
    "_display_scenario",
    "_normalise_scenarios",
    "_parse_sample_sizes",
    "_write_jsonl",
    "build_method_rankings",
    "build_neural_comparison",
    "build_neural_verdict",
]


if __name__ == "__main__":  # pragma: no cover - command-line compatibility
    raise SystemExit(_legacy_main())
