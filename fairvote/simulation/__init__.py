"""
Simulation components for FairVote-AI: population generation, sampling,
bias/nonresponse models, and evaluation metrics.
"""

from fairvote.simulation.bias_models import (
    FeatureNonresponseProfile,
    MisreportModel,
    PreferenceNonresponseProfile,
    apply_misreporting,
    apply_nonresponse,
    build_shy_model_from_epsilon,
    honesty_from_epsilon,
    make_default_feature_nonresponse_profile,
    make_general_misreport_model,
    make_identity_misreport_model,
    make_shy_supporter_model,
    participation_from_epsilon,
)
from fairvote.simulation.population import (
    Population,
    make_realistic_uk_like_population,
    overall_true_distribution,
    poststrat_table,
    subgroup_true_distribution,
)
from fairvote.simulation.sampling import (
    Sample,
    biased_frame_sample,
    nonresponse,
    simple_random_sample,
    stratified_sample,
)

__all__ = [
    # population
    "Population",
    "make_realistic_uk_like_population",
    "overall_true_distribution",
    "subgroup_true_distribution",
    "poststrat_table",
    # sampling
    "Sample",
    "simple_random_sample",
    "stratified_sample",
    "biased_frame_sample",
    "nonresponse",
    # bias models
    "FeatureNonresponseProfile",
    "PreferenceNonresponseProfile",
    "MisreportModel",
    "apply_nonresponse",
    "make_default_feature_nonresponse_profile",
    "make_identity_misreport_model",
    "make_shy_supporter_model",
    "make_general_misreport_model",
    "apply_misreporting",
    "honesty_from_epsilon",
    "participation_from_epsilon",
    "build_shy_model_from_epsilon",
]
