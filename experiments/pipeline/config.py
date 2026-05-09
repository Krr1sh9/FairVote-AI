"""Typed configuration objects for the MRP-vs-baselines experiment pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .scenarios import validate_scenarios


def default_methods(enable_neural: bool) -> list[str]:
    """Return the backwards-compatible core method order used by default."""
    methods = [
        "raw_reported_distribution",
        "baseline_rr_debias",
        "mrp_rr_poststrat",
        "mrp_misreport_rr_poststrat",
        "mrp_learned_misreport_rr_poststrat",
    ]
    if enable_neural:
        methods.append("neural_rr_mrp")
    return methods


def research_methods(enable_neural: bool) -> list[str]:
    """Return the richer method set used for the neural-vs-linear research question."""
    methods = [
        "oracle_true_sample_distribution",
        "raw_reported_distribution",
        "baseline_rr_debias",
        "linear_rr_no_poststrat",
        "mrp_rr_poststrat",
        "hierarchical_rr_mrp_poststrat",
        "oracle_true_linear_mrp_poststrat",
        "mrp_misreport_rr_poststrat",
        "oracle_known_misreport_rr_mrp",
        "mrp_learned_misreport_rr_poststrat",
    ]
    if enable_neural:
        methods.extend(["neural_rr_mrp", "neural_naive_reported_mrp"])
    return methods


def resolve_methods(spec: str | list[str] | None, *, enable_neural: bool) -> list[str]:
    """Resolve a method preset or comma/list specification."""
    if spec is None or spec == []:
        return default_methods(enable_neural)
    if isinstance(spec, str):
        raw = spec.strip()
        if not raw or raw.lower() == "default":
            return default_methods(enable_neural)
        if raw.lower() == "research":
            return research_methods(enable_neural)
        if raw.lower() == "all":
            return research_methods(enable_neural)
        return [part.strip() for part in raw.split(",") if part.strip()]
    return [str(part).strip() for part in spec if str(part).strip()]


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete reproducible configuration for an experiment run.

    ``n_sample`` is retained for backwards-compatible single-size runs.  New
    evidence presets use ``sample_sizes`` to run the same scenario/epsilon/trial
    grid at several respondent sample sizes.  Every raw row still records the
    actual sample size used.
    """

    k: int
    eps_list: list[float]
    scenarios: list[str]
    population_n: int
    n_sample: int
    trials: int
    seed: int
    sampling: str
    strata: list[str]
    allocation: str
    min_per_stratum: int
    biased_feature: str
    biased_multipliers: dict[str, float]
    feature_order: list[str]
    shy_category: int
    shy_honesty: float
    mrp_steps: int
    mrp_lr: float
    mrp_l2: float
    mrp_batch_size: int
    verbose_every: int
    enable_neural: bool
    neural_hidden_layers: tuple[int, ...]
    neural_steps: int
    neural_lr: float
    neural_batch_size: int
    neural_seed: int
    neural_dropout: float
    neural_weight_decay: float
    major_mass: float
    neural_validation_fraction: float = 0.2
    neural_patience: int = 20
    methods: list[str] = field(default_factory=list)
    sample_sizes: list[int] = field(default_factory=list)
    preset: str = "custom"
    continue_on_error: bool = True
    hierarchical_global_l2: float = 0.01
    hierarchical_effect_l2: float = 1.0

    def __post_init__(self) -> None:
        if self.k < 2:
            raise ValueError("k must be >= 2")
        if not self.eps_list or any(float(e) <= 0 for e in self.eps_list):
            raise ValueError("eps_list must contain positive epsilons")
        if self.population_n <= 0:
            raise ValueError("population_n must be positive")
        if self.n_sample <= 0:
            raise ValueError("n_sample must be positive")
        sizes = [int(s) for s in (self.sample_sizes or [self.n_sample])]
        if not sizes or any(s <= 0 for s in sizes):
            raise ValueError("sample_sizes must contain positive integers")
        if any(s > self.population_n for s in sizes):
            raise ValueError("sample_sizes cannot exceed population_n")
        object.__setattr__(self, "sample_sizes", sizes)
        object.__setattr__(self, "n_sample", int(sizes[0] if self.sample_sizes else self.n_sample))
        if self.trials <= 0:
            raise ValueError("trials must be positive")
        if self.sampling not in {"srs", "stratified", "biased"}:
            raise ValueError("sampling must be one of: srs, stratified, biased")
        object.__setattr__(self, "scenarios", validate_scenarios(self.scenarios))
        if not self.feature_order:
            raise ValueError("at least one feature is required")
        if self.neural_validation_fraction < 0.0 or self.neural_validation_fraction >= 1.0:
            raise ValueError("neural_validation_fraction must satisfy 0 <= value < 1")
        if self.neural_patience < 1:
            raise ValueError("neural_patience must be >= 1")
        if self.mrp_steps < 1:
            raise ValueError("mrp_steps must be >= 1")
        if self.hierarchical_global_l2 < 0.0 or self.hierarchical_effect_l2 < 0.0:
            raise ValueError("hierarchical regularisation strengths must be non-negative")
        if self.neural_steps < 1:
            raise ValueError("neural_steps must be >= 1")
        if not self.methods:
            object.__setattr__(self, "methods", default_methods(self.enable_neural))

    @property
    def sample_size(self) -> int:
        """Alias used in output CSVs for readable reproducibility."""
        return int(self.n_sample)

    @property
    def sample_size_grid(self) -> list[int]:
        """All sample sizes included in this run."""
        return list(self.sample_sizes or [self.n_sample])

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["neural_hidden_layers"] = list(self.neural_hidden_layers)
        d["eps"] = list(self.eps_list)
        d["features"] = list(self.feature_order)
        d["multipliers"] = dict(self.biased_multipliers)
        d["sample_sizes"] = list(self.sample_size_grid)
        return d


@dataclass(frozen=True)
class TrialConfig:
    """Configuration for one sample-size/scenario/trial/epsilon cell."""

    scenario: str
    trial: int
    epsilon: float
    sample_seed: int
    privacy_seed: int
    sample_size: int

    @property
    def random_seed(self) -> int:
        """Seed used for RR perturbation in this specific result cell."""
        return int(self.privacy_seed)


@dataclass(frozen=True)
class MethodResult:
    """Output from one estimator runner before metric scoring."""

    method: str
    estimate_overall: Any
    by_feature: dict[str, dict[str, Any]]
    runtime_sec: float
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExperimentResult:
    """In-memory result bundle returned by the modular experiment pipeline."""

    rows: list[dict[str, Any]]
    summary: list[dict[str, Any]]
    paired_comparisons: list[dict[str, Any]]
    ablations: list[dict[str, Any]]
    runtime_profile: list[dict[str, Any]]
    failures: list[dict[str, Any]]
    config: ExperimentConfig
    manifest: dict[str, Any]
