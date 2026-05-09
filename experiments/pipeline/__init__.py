"""Modular experiment engine for FairVote-AI evidence generation."""

from .config import (
    ExperimentConfig,
    ExperimentResult,
    MethodResult,
    TrialConfig,
    default_methods,
    research_methods,
    resolve_methods,
)
from .runner import execute_experiment

__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "MethodResult",
    "TrialConfig",
    "default_methods",
    "research_methods",
    "resolve_methods",
    "execute_experiment",
]
