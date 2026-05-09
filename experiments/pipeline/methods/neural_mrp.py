"""Neural RR-aware MRP estimator runners."""
from __future__ import annotations

import time
import warnings

from ..config import ExperimentConfig, MethodResult, TrialConfig
from ..context import ExperimentContext
from .common import TrialData, _near_identity_epsilon
from .linear_mrp import _poststratify_model_predictions

def require_rr_neural_mrp_model():
    """Import the optional PyTorch neural MRP only when an enabled method needs it."""
    try:
        from fairvote.inference.mrp.rr_neural_mrp import RRNeuralMRPModel
    except Exception as exc:  # pragma: no cover - depends on optional torch environment
        raise RuntimeError(
            "Neural RR-MRP is enabled, but the PyTorch model could not be imported. "
            'Install the neural extra with `pip install -e ".[neural]"` or the full developer environment with '
            '`pip install -e ".[dev]"`, or rerun with --disable_neural.'
        ) from exc
    return RRNeuralMRPModel
def neural_rr_mrp(
    config: ExperimentConfig,
    context: ExperimentContext,
    trial: TrialConfig,
    data: TrialData,
) -> MethodResult:
    """PyTorch neural RR-aware MRP followed by poststratification."""
    start = time.perf_counter()
    RRNeuralMRPModel = require_rr_neural_mrp_model()
    model = RRNeuralMRPModel(
        k=config.k,
        epsilon=trial.epsilon,
        hidden_layers=config.neural_hidden_layers,
        dropout=config.neural_dropout,
        weight_decay=config.neural_weight_decay,
        seed=int(config.neural_seed) + 3999 + trial.trial,
    )
    fit_info = model.fit(
        data.X_train,
        data.perturbation.reported_categories,
        lr=config.neural_lr,
        steps=config.neural_steps,
        batch_size=config.neural_batch_size,
        verbose_every=config.verbose_every,
        validation_fraction=config.neural_validation_fraction,
        patience=config.neural_patience,
    )
    overall, by_feature = _poststratify_model_predictions(
        context=context,
        cell_theta=model.predict_true_proba(context.X_cells),
    )
    eval_summary = model.evaluation_summary(
        data.X_train,
        y_reported=data.perturbation.reported_categories,
        true_labels=data.perturbation.true_categories,
    )
    return MethodResult(
        "neural_rr_mrp",
        overall,
        by_feature,
        time.perf_counter() - start,
        {
            "neural_final_loss": fit_info.final_loss,
            "neural_validation_loss": fit_info.validation_loss,
            "neural_best_validation_loss": fit_info.best_validation_loss,
            "neural_steps_completed": fit_info.steps,
            "neural_early_stopped": int(fit_info.early_stopped),
            "neural_fit_runtime_sec": fit_info.runtime_sec,
            "neural_brier_score": eval_summary.get("brier_score"),
            "neural_reported_label_nll": eval_summary.get("reported_label_nll"),
            "neural_mean_normalized_entropy": eval_summary.get("mean_normalized_entropy"),
        },
    )


def neural_naive_reported_mrp(
    config: ExperimentConfig,
    context: ExperimentContext,
    trial: TrialConfig,
    data: TrialData,
) -> MethodResult:
    """Neural ablation that treats privatized reports as if they were true labels.

    This is intentionally *not* RR-aware. It asks whether the RR observation
    likelihood is doing useful work beyond a flexible neural architecture.
    """
    start = time.perf_counter()
    RRNeuralMRPModel = require_rr_neural_mrp_model()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = RRNeuralMRPModel(
            k=config.k,
            epsilon=_near_identity_epsilon(),
            hidden_layers=config.neural_hidden_layers,
            dropout=config.neural_dropout,
            weight_decay=config.neural_weight_decay,
            seed=int(config.neural_seed) + 4999 + trial.trial,
        )
    fit_info = model.fit(
        data.X_train,
        data.perturbation.reported_categories,
        lr=config.neural_lr,
        steps=config.neural_steps,
        batch_size=config.neural_batch_size,
        verbose_every=config.verbose_every,
        validation_fraction=config.neural_validation_fraction,
        patience=config.neural_patience,
    )
    overall, by_feature = _poststratify_model_predictions(
        context=context,
        cell_theta=model.predict_true_proba(context.X_cells),
    )
    eval_summary = model.evaluation_summary(
        data.X_train,
        y_reported=data.perturbation.reported_categories,
        true_labels=data.perturbation.true_categories,
    )
    return MethodResult(
        "neural_naive_reported_mrp",
        overall,
        by_feature,
        time.perf_counter() - start,
        {
            "ablation_no_rr_aware_loss": 1,
            "neural_final_loss": fit_info.final_loss,
            "neural_validation_loss": fit_info.validation_loss,
            "neural_best_validation_loss": fit_info.best_validation_loss,
            "neural_steps_completed": fit_info.steps,
            "neural_early_stopped": int(fit_info.early_stopped),
            "neural_fit_runtime_sec": fit_info.runtime_sec,
            "neural_brier_score": eval_summary.get("brier_score"),
            "neural_reported_label_nll": eval_summary.get("reported_label_nll"),
            "neural_mean_normalized_entropy": eval_summary.get("mean_normalized_entropy"),
        },
    )
