"""Evidence-run presets for reproducible final project experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class EvidencePreset:
    """Named run settings with a clear evidence role."""

    name: str
    description: str
    overrides: Dict[str, Any]


ROBUST_SCENARIOS = (
    "simple_linear,nonresponse,nonlinear_interaction,shy_fixed,"
    "sparse_minority_curve,nonlinear_response,privacy_noise_sparse,"
    "education_urbanicity_interaction,privacy_tradeoff,privacy_helps,shy_privacy_helps"
)

PRESETS: Dict[str, EvidencePreset] = {
    "smoke_test": EvidencePreset(
        name="smoke_test",
        description="Fast sanity check only. Uses minimal training settings and is not evidence for final claims.",
        overrides={
            "eps": "1.0",
            "sample_sizes": "120",
            "n_sample": 120,
            "scenarios": "simple_linear",
            "population_n": 1_500,
            "trials": 1,
            "methods": "default",
            "mrp_steps": 5,
            "neural_steps": 5,
            "neural_patience": 2,
        },
    ),
    "examiner_reproduce": EvidencePreset(
        name="examiner_reproduce",
        description="Minute-scale reproducibility check that exercises every deployable estimator, including hierarchical MRP.",
        overrides={
            "eps": "0.5,1.0",
            "sample_sizes": "180,260",
            "n_sample": 180,
            "scenarios": "simple_linear,sparse_minority_curve,privacy_helps",
            "population_n": 3_000,
            "trials": 2,
            "methods": "baseline_rr_debias,mrp_rr_poststrat,hierarchical_rr_mrp_poststrat,mrp_learned_misreport_rr_poststrat",
            "mrp_steps": 25,
            "neural_steps": 10,
            "neural_patience": 3,
            "fail_fast": True,
        },
    ),
    "medium_evidence": EvidencePreset(
        name="medium_evidence",
        description="Moderate repeated-run evidence for development and draft-report tables.",
        overrides={
            "eps": "0.5,1.0,2.0",
            "sample_sizes": "500,1000",
            "n_sample": 500,
            "scenarios": "simple_linear,nonresponse,nonlinear_interaction,shy_fixed,sparse_minority_curve",
            "population_n": 50_000,
            "trials": 10,
            "methods": "research",
            "mrp_steps": 800,
            "neural_steps": 250,
            "neural_patience": 30,
            "neural_hidden_layers": "32,16",
        },
    ),
    "robustness_evidence": EvidencePreset(
        name="robustness_evidence",
        description="Wide scenario/epsilon/sample-size robustness grid for the final report limitations and stress tests.",
        overrides={
            "eps": "0.1,0.2,0.5,1.0,2.0,4.0",
            "sample_sizes": "250,500,1000,2500,5000",
            "n_sample": 250,
            "scenarios": ROBUST_SCENARIOS,
            "population_n": 120_000,
            "trials": 25,
            "methods": "research",
            "mrp_steps": 1200,
            "neural_steps": 400,
            "neural_patience": 50,
            "neural_hidden_layers": "64,32",
            "fail_fast": True,
        },
    ),
    "final_evidence": EvidencePreset(
        name="final_evidence",
        description=(
            "Final-submission evidence. Includes hierarchical partial-pooling MRP, nonlinear/sparse/privacy scenarios, "
            "multiple epsilons and sample sizes, and fail-fast semantics so hidden partial failures cannot pass."
        ),
        overrides={
            "eps": "0.2,0.5,1.0,2.0,4.0",
            "sample_sizes": "500,1000,2500,5000",
            "n_sample": 500,
            "scenarios": ROBUST_SCENARIOS,
            "population_n": 100_000,
            "trials": 30,
            "methods": "research",
            "mrp_steps": 1500,
            "neural_steps": 500,
            "neural_patience": 50,
            "neural_hidden_layers": "64,32",
            "fail_fast": True,
        },
    ),
}


def preset_names() -> list[str]:
    return sorted(PRESETS)


def get_preset(name: str | None) -> EvidencePreset | None:
    if name is None or str(name).strip().lower() in {"", "custom", "none"}:
        return None
    key = str(name).strip().lower()
    try:
        return PRESETS[key]
    except KeyError as exc:
        raise ValueError(f"Unknown preset {name!r}. Valid presets: {preset_names()}") from exc
