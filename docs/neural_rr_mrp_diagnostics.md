# RR-aware Neural MRP Diagnostics

The neural estimator is implemented in:

```text
`fairvote/inference/mrp/neural/` (facade: `fairvote/inference/mrp/rr_neural_mrp.py`)
```

It is not a generic classifier trained on true votes. The fitted target is the **reported-label likelihood after the Randomized Response channel**:

```text
P_theta(true=t | x)        = neural softmax output
A[t,r]                     = P_RR(reported=r | true=t)
P_theta(reported=r | x)    = sum_t P_theta(true=t | x) A[t,r]
loss                       = -mean log P_theta(reported_i | x_i)
```

The `fit` API accepts `X` and `y_reported`. It deliberately has no `y_true` argument. In synthetic experiments, true labels may be passed later to evaluation helpers such as `brier_score`, but they are not used for training.

## Training controls

`RRNeuralMRPModel.fit(...)` supports:

- deterministic seed control through the model `seed` parameter;
- CPU/GPU device handling through `device="cpu"`, `device="cuda"`, or `device="auto"`;
- validation reported-label NLL through either explicit `X_val` / `y_val_reported` or `validation_fraction`;
- early stopping with `patience` and `min_delta`;
- optional restoration of the best validation checkpoint with `keep_best=True`;
- training loss history with `keep_history=True`;
- validation loss history whenever validation data is provided;
- runtime tracking;
- optional model checkpoint saving through `checkpoint_path`.

The returned `RRNeuralMRPFitInfo` records:

```text
steps
final_loss
validation_loss
best_validation_loss
best_step
early_stopped
runtime_sec
device
history / validation_history
checkpoint_path
```

These fields are suitable for the final report's convergence and reproducibility discussion.

## Calibration and uncertainty helpers

The model exposes:

- `reported_label_nll(X, y_reported)` — real-data-compatible validation score;
- `brier_score(X, true_labels)` — synthetic-only calibration/error score;
- `entropy_summary(X)` — latent predictive uncertainty summary;
- `evaluation_summary(...)` — combines available reported-label and synthetic-only diagnostics;
- `fit_rr_neural_mrp_ensemble(...)` — small multi-seed ensemble for sensitivity checks.

Use reported-label NLL for real data because real true votes are not available. Use Brier score only in simulation, where the simulator generated true labels.

## When RR-aware Neural MRP is expected to help

The neural model is most defensible when:

- the demographic-to-vote relationship is nonlinear;
- there are interactions not captured well by a linear design matrix;
- the sample size is large enough for the extra parameters;
- epsilon is high enough that the RR signal is not completely drowned by privacy noise;
- validation NLL and held-out synthetic metrics improve over RR-aware linear poststratification/MRP.

## When RR-aware linear poststratification/MRP is preferable

The RR-aware linear poststratification/MRP model is usually preferable when:

- sample size is small;
- epsilon is very low;
- the signal is approximately linear/additive;
- the neural validation NLL is flat or worsening;
- early stopping triggers almost immediately;
- the neural model has worse subgroup error or winner correctness;
- runtime and interpretability matter more than capturing possible nonlinear interactions.

The final report should not claim that RR-aware Neural MRP is automatically better. The correct claim is narrower: RR-aware Neural MRP is a privacy-compatible learned estimator that can be evaluated against simpler baselines, and its usefulness depends on the scenario and metric.

## Example

```python
from fairvote.inference.mrp.rr_neural_mrp import RRNeuralMRPModel

model = RRNeuralMRPModel(k=5, epsilon=1.0, hidden_layers=(32, 16), seed=123, device="auto")
info = model.fit(
    X_train,
    y_reported_train,
    steps=500,
    batch_size=512,
    lr=0.01,
    validation_fraction=0.2,
    patience=20,
    keep_history=True,
    checkpoint_path="outputs/neural_rr_mrp.pt",
)

print(info.to_dict())
print(model.reported_label_nll(X_valid, y_reported_valid))
print(model.entropy_summary(X_valid))
model.save_metadata("outputs/neural_rr_mrp_metadata.json", include_history=True)
```
