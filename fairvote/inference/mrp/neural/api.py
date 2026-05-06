"""Main RR-aware neural MRP model API."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from fairvote.privacy.mechanisms.kary_rr import rr_transition_matrix
from .dependencies import torch
from .network import _MLP
from .types import ArrayLike, RRNeuralMRPFitInfo

class RRNeuralMRPModel:
    """RR-aware neural MRP model trained on privatized reported answers.

    The network learns ``P_theta(true=t | x)`` from demographic/features ``X``.
    Because the true category is not observed, fitting uses the Randomized
    Response observation model

    ``P(reported=r | x) = sum_t P_theta(true=t | x) * A[t, r]``.

    Here ``A`` is the k-ary RR transition matrix from the canonical privacy
    module. Training therefore uses only ``X`` and ``y_reported``; there is
    intentionally no API argument for true labels.
    """

    def __init__(
        self,
        *,
        k: int,
        epsilon: float,
        hidden_layers: Sequence[int] = (32,),
        dropout: float = 0.0,
        weight_decay: float = 0.0,
        seed: int = 0,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
        deterministic: bool = True,
    ):
        self.k = self._validate_k(k)
        self.epsilon = self._validate_epsilon(epsilon)
        self.hidden_layers = self._validate_hidden_layers(hidden_layers)
        self.dropout = self._validate_dropout(dropout)
        self.weight_decay = self._validate_nonnegative_float(weight_decay, "weight_decay")
        self.seed = int(seed)
        self.device = self._resolve_device(device)
        self.dtype = dtype
        self.deterministic = bool(deterministic)

        self.A_np = rr_transition_matrix(self.epsilon, self.k)
        self._A_tensor: Optional[torch.Tensor] = None
        self._network: Optional[_MLP] = None
        self._input_dim: Optional[int] = None
        self._last_fit_info: Optional[RRNeuralMRPFitInfo] = None

    @staticmethod
    def _validate_k(k: int) -> int:
        if not isinstance(k, (int, np.integer)):
            raise TypeError("k must be an integer")
        k = int(k)
        if k < 2:
            raise ValueError("k must be >= 2")
        return k

    @staticmethod
    def _validate_epsilon(epsilon: float) -> float:
        if not isinstance(epsilon, (int, float, np.floating)):
            raise TypeError("epsilon must be a number")
        eps = float(epsilon)
        if not np.isfinite(eps):
            raise ValueError("epsilon must be finite")
        if eps <= 0.0:
            raise ValueError("epsilon must be > 0")
        return eps

    @staticmethod
    def _validate_nonnegative_float(value: float, name: str) -> float:
        if not isinstance(value, (int, float, np.floating)):
            raise TypeError(f"{name} must be a number")
        out = float(value)
        if not np.isfinite(out):
            raise ValueError(f"{name} must be finite")
        if out < 0.0:
            raise ValueError(f"{name} must be >= 0")
        return out

    @staticmethod
    def _validate_dropout(dropout: float) -> float:
        out = RRNeuralMRPModel._validate_nonnegative_float(dropout, "dropout")
        if out >= 1.0:
            raise ValueError("dropout must be < 1")
        return out

    @staticmethod
    def _validate_hidden_layers(hidden_layers: Sequence[int]) -> Tuple[int, ...]:
        if isinstance(hidden_layers, (int, np.integer)):
            hidden_layers = (int(hidden_layers),)
        if hidden_layers is None:
            raise TypeError("hidden_layers must be a sequence of positive integers")
        widths = tuple(int(w) for w in hidden_layers)
        for w in widths:
            if w <= 0:
                raise ValueError("all hidden layer widths must be positive")
        return widths

    @staticmethod
    def _validate_steps(steps: int) -> int:
        if not isinstance(steps, (int, np.integer)):
            raise TypeError("steps must be an integer")
        steps = int(steps)
        if steps < 1:
            raise ValueError("steps must be >= 1")
        return steps

    @staticmethod
    def _validate_batch_size(batch_size: int) -> int:
        if not isinstance(batch_size, (int, np.integer)):
            raise TypeError("batch_size must be an integer")
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        return batch_size

    @staticmethod
    def _resolve_device(device: Union[str, torch.device]) -> torch.device:
        if str(device) == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        return resolved

    @staticmethod
    def _as_2d_float_array(X: ArrayLike, *, name: str = "X") -> np.ndarray:
        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError(f"{name} must be a 2D array")
        if arr.shape[0] == 0:
            raise ValueError(f"{name} must have at least one row")
        if arr.shape[1] == 0:
            raise ValueError(f"{name} must have at least one column")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values")
        return arr

    def _as_y_reported(self, y_reported: Sequence[int], *, expected_n: Optional[int] = None) -> np.ndarray:
        raw = np.asarray(y_reported)
        if raw.ndim != 1:
            raise ValueError("y_reported must be a 1D array")
        if raw.size == 0:
            raise ValueError("y_reported must not be empty")
        if expected_n is not None and raw.size != int(expected_n):
            raise ValueError("X and y_reported must have the same number of rows")
        try:
            numeric = raw.astype(float, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("y_reported must contain numeric integer category labels") from exc
        if not np.all(np.isfinite(numeric)):
            raise ValueError("y_reported must contain only finite values")
        y = numeric.astype(np.int64)
        if not np.allclose(numeric, y, atol=0.0, rtol=0.0):
            raise ValueError("y_reported must contain integer category labels")
        if np.any(y < 0) or np.any(y >= self.k):
            raise ValueError(f"y_reported labels must be in [0, {self.k - 1}]")
        return y

    def _set_seeds(self) -> None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():  # pragma: no cover - CI normally runs on CPU
            torch.cuda.manual_seed_all(self.seed)
        if self.deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

    def _build_network(self, input_dim: int) -> None:
        self._set_seeds()
        self._input_dim = int(input_dim)
        self._network = _MLP(
            input_dim=int(input_dim),
            output_dim=self.k,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
        ).to(device=self.device, dtype=self.dtype)

    def _require_network(self) -> _MLP:
        if self._network is None:
            raise RuntimeError("Model is not fitted")
        return self._network

    def _check_input_dim(self, X: np.ndarray) -> None:
        if self._input_dim is not None and X.shape[1] != self._input_dim:
            raise ValueError(f"X has {X.shape[1]} columns, expected {self._input_dim}")

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(X, dtype=self.dtype, device=self.device)

    def _rr_matrix_tensor(self) -> torch.Tensor:
        if self._A_tensor is None:
            self._A_tensor = torch.as_tensor(self.A_np, dtype=self.dtype, device=self.device)
        return self._A_tensor

    def _true_proba_tensor(self, X_tensor: torch.Tensor) -> torch.Tensor:
        logits = self._require_network()(X_tensor)
        return torch.softmax(logits, dim=1)

    def _reported_proba_tensor(self, X_tensor: torch.Tensor) -> torch.Tensor:
        theta = self._true_proba_tensor(X_tensor)
        return theta @ self._rr_matrix_tensor()

    def _weight_decay_penalty(self) -> torch.Tensor:
        network = self._require_network()
        if self.weight_decay <= 0.0:
            return torch.zeros((), dtype=self.dtype, device=self.device)
        params_vec = torch.nn.utils.parameters_to_vector(network.parameters())
        return 0.5 * float(self.weight_decay) * torch.sum(params_vec * params_vec)

    def _reported_nll_tensor(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor) -> torch.Tensor:
        q = self._reported_proba_tensor(X_tensor)
        q_y = q[torch.arange(q.shape[0], device=self.device), y_tensor]
        q_y_clamped = torch.clamp(q_y, min=1e-12)
        nll = -torch.mean(torch.log(q_y_clamped))
        if not torch.isfinite(nll).all():
            raise RuntimeError("Reported-label negative log-likelihood is not finite")
        return nll

    def _loss_tensor(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor) -> torch.Tensor:
        nll = self._reported_nll_tensor(X_tensor, y_tensor)
        total_loss = nll + self._weight_decay_penalty()
        if not torch.isfinite(total_loss).all():
            raise RuntimeError("Total loss (reported-label NLL + regularisation) is not finite")
        return total_loss

    @staticmethod
    def _validate_validation_fraction(validation_fraction: float) -> float:
        if not isinstance(validation_fraction, (int, float, np.floating)):
            raise TypeError("validation_fraction must be a number")
        frac = float(validation_fraction)
        if not np.isfinite(frac):
            raise ValueError("validation_fraction must be finite")
        if frac < 0.0 or frac >= 1.0:
            raise ValueError("validation_fraction must satisfy 0 <= validation_fraction < 1")
        return frac

    @staticmethod
    def _validate_patience(patience: Optional[int]) -> Optional[int]:
        if patience is None:
            return None
        if not isinstance(patience, (int, np.integer)):
            raise TypeError("patience must be an integer or None")
        patience = int(patience)
        if patience < 1:
            raise ValueError("patience must be >= 1")
        return patience

    def _split_train_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_fraction: float,
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        if validation_fraction <= 0.0:
            return X, y, None, None
        n = X.shape[0]
        n_val = int(round(n * validation_fraction))
        n_val = min(max(n_val, 1), n - 1)
        if n < 2:
            raise ValueError("validation_fraction requires at least two rows")
        rng = np.random.default_rng(self.seed)
        idx = rng.permutation(n)
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

    @property
    def is_fitted(self) -> bool:
        """Whether the underlying PyTorch network has been initialised/fitted."""
        return self._network is not None

    @property
    def last_fit_info(self) -> Optional[RRNeuralMRPFitInfo]:
        """Most recent fit summary, or ``None`` if the model has not been fitted."""
        return self._last_fit_info

    def fit(
        self,
        X: ArrayLike,
        y_reported: Sequence[int],
        *,
        lr: float = 1e-2,
        steps: int = 1000,
        batch_size: int = 512,
        keep_history: bool = False,
        verbose_every: int = 0,
        X_val: Optional[ArrayLike] = None,
        y_val_reported: Optional[Sequence[int]] = None,
        validation_fraction: float = 0.0,
        patience: Optional[int] = None,
        min_delta: float = 0.0,
        keep_best: bool = True,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> RRNeuralMRPFitInfo:
        """Fit using privatized RR reports only.

        Parameters ``X_val`` and ``y_val_reported`` are also reported-label data.
        They support validation reported-label NLL and early stopping without
        requiring any true labels. If no explicit validation set is supplied,
        ``validation_fraction`` can hold out part of the reported-label data.
        """

        X_arr = self._as_2d_float_array(X)
        y_arr = self._as_y_reported(y_reported, expected_n=X_arr.shape[0])
        lr = self._validate_nonnegative_float(lr, "lr")
        if lr <= 0.0:
            raise ValueError("lr must be > 0")
        steps = self._validate_steps(steps)
        batch_size = self._validate_batch_size(batch_size)
        if not isinstance(verbose_every, (int, np.integer)) or int(verbose_every) < 0:
            raise ValueError("verbose_every must be a non-negative integer")
        validation_fraction = self._validate_validation_fraction(validation_fraction)
        patience = self._validate_patience(patience)
        min_delta = self._validate_nonnegative_float(min_delta, "min_delta")

        if (X_val is None) != (y_val_reported is None):
            raise ValueError("X_val and y_val_reported must be provided together")
        if X_val is not None:
            if validation_fraction > 0.0:
                raise ValueError("Use either explicit validation data or validation_fraction, not both")
            X_train = X_arr
            y_train = y_arr
            X_val_arr = self._as_2d_float_array(X_val, name="X_val")
            y_val_arr = self._as_y_reported(y_val_reported, expected_n=X_val_arr.shape[0])
        else:
            X_train, y_train, X_val_arr, y_val_arr = self._split_train_validation(X_arr, y_arr, validation_fraction)

        if self._network is None:
            self._build_network(X_arr.shape[1])
        else:
            self._check_input_dim(X_arr)
        if X_val_arr is not None:
            self._check_input_dim(X_val_arr)

        network = self._require_network()
        network.train()
        self._set_seeds()

        X_tensor = self._to_tensor(X_train)
        y_tensor = torch.as_tensor(y_train, dtype=torch.long, device=self.device)
        X_val_tensor = None if X_val_arr is None else self._to_tensor(X_val_arr)
        y_val_tensor = None if y_val_arr is None else torch.as_tensor(y_val_arr, dtype=torch.long, device=self.device)

        optimiser = torch.optim.Adam(network.parameters(), lr=float(lr))
        rng = np.random.default_rng(self.seed)
        n = X_train.shape[0]
        train_hist: list[float] = []
        val_hist: list[float] = []
        best_val = float("inf")
        best_step: Optional[int] = None
        best_state: Optional[dict[str, torch.Tensor]] = None
        bad_steps = 0
        completed_steps = 0
        early_stopped = False
        start_time = time.perf_counter()

        full_batch_idx_tensor = None
        if batch_size >= n:
            full_batch_idx_tensor = torch.arange(n, dtype=torch.long, device=self.device)

        for step in range(1, steps + 1):
            if batch_size >= n:
                idx_tensor = full_batch_idx_tensor
            else:
                idx = rng.integers(0, n, size=batch_size)
                idx_tensor = torch.as_tensor(idx, dtype=torch.long, device=self.device)
            if idx_tensor is None:  # pragma: no cover - defensive guard
                raise RuntimeError("internal batching error")
            X_batch = X_tensor.index_select(0, idx_tensor)
            y_batch = y_tensor.index_select(0, idx_tensor)

            optimiser.zero_grad(set_to_none=True)
            batch_loss = self._loss_tensor(X_batch, y_batch)
            batch_loss.backward()
            optimiser.step()
            completed_steps = step

            should_eval_train = keep_history or (verbose_every and step % int(verbose_every) == 0)
            if should_eval_train:
                full_loss = self._loss_from_tensors(X_tensor, y_tensor)
                if keep_history:
                    train_hist.append(full_loss)
                if verbose_every and step % int(verbose_every) == 0:
                    print(f"[RR-Neural-MRP] step={step} loss={full_loss:.6f}")

            if X_val_tensor is not None and y_val_tensor is not None:
                val_loss = self._reported_nll_from_tensors(X_val_tensor, y_val_tensor)
                val_hist.append(val_loss)
                if val_loss < best_val - float(min_delta):
                    best_val = val_loss
                    best_step = step
                    bad_steps = 0
                    if keep_best:
                        best_state = {k: v.detach().clone().cpu() for k, v in network.state_dict().items()}
                else:
                    bad_steps += 1
                    if patience is not None and bad_steps >= patience:
                        early_stopped = True
                        break

        if best_state is not None and keep_best:
            network.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        final_loss = self.loss(X_train, y_train)
        final_val_loss = None
        if X_val_arr is not None and y_val_arr is not None:
            final_val_loss = self.reported_label_nll(X_val_arr, y_val_arr)
        runtime_sec = time.perf_counter() - start_time

        checkpoint_str: Optional[str] = None
        if checkpoint_path is not None:
            checkpoint_str = str(checkpoint_path)
            self.save_checkpoint(checkpoint_str)

        info = RRNeuralMRPFitInfo(
            steps=int(completed_steps),
            final_loss=float(final_loss),
            history=np.asarray(train_hist, dtype=float) if keep_history else None,
            validation_loss=None if final_val_loss is None else float(final_val_loss),
            validation_history=np.asarray(val_hist, dtype=float) if val_hist else None,
            best_validation_loss=None if best_step is None else float(best_val),
            best_step=best_step,
            early_stopped=early_stopped,
            runtime_sec=float(runtime_sec),
            device=str(self.device),
            checkpoint_path=checkpoint_str,
        )
        self._last_fit_info = info
        return info

    def _loss_from_tensors(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor) -> float:
        network = self._require_network()
        was_training = network.training
        network.eval()
        with torch.no_grad():
            out = float(self._loss_tensor(X_tensor, y_tensor).detach().cpu().item())
        if was_training:
            network.train()
        return out

    def _reported_nll_from_tensors(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor) -> float:
        network = self._require_network()
        was_training = network.training
        network.eval()
        with torch.no_grad():
            out = float(self._reported_nll_tensor(X_tensor, y_tensor).detach().cpu().item())
        if was_training:
            network.train()
        return out

    def loss(self, X: ArrayLike, y_reported: Sequence[int]) -> float:
        """Return full-data regularised RR-aware negative log-likelihood."""
        X_arr = self._as_2d_float_array(X)
        self._check_input_dim(X_arr)
        y_arr = self._as_y_reported(y_reported, expected_n=X_arr.shape[0])
        return self._loss_from_tensors(self._to_tensor(X_arr), torch.as_tensor(y_arr, dtype=torch.long, device=self.device))

    def reported_label_nll(self, X: ArrayLike, y_reported: Sequence[int]) -> float:
        """Return unregularised NLL of privatized reported labels.

        This is the validation/calibration score used for real data because it
        does not need true synthetic labels.
        """
        X_arr = self._as_2d_float_array(X)
        self._check_input_dim(X_arr)
        y_arr = self._as_y_reported(y_reported, expected_n=X_arr.shape[0])
        return self._reported_nll_from_tensors(
            self._to_tensor(X_arr),
            torch.as_tensor(y_arr, dtype=torch.long, device=self.device),
        )

    def predict_true_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict latent true-category probabilities ``P_theta(true=t | x)``."""
        X_arr = self._as_2d_float_array(X)
        self._check_input_dim(X_arr)
        network = self._require_network()
        was_training = network.training
        network.eval()
        with torch.no_grad():
            p = self._true_proba_tensor(self._to_tensor(X_arr)).detach().cpu().numpy()
        if was_training:
            network.train()
        return self._validate_probability_rows(p, name="predict_true_proba")

    def predict_reported_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict reported-label probabilities after the RR observation channel."""
        X_arr = self._as_2d_float_array(X)
        self._check_input_dim(X_arr)
        network = self._require_network()
        was_training = network.training
        network.eval()
        with torch.no_grad():
            q = self._reported_proba_tensor(self._to_tensor(X_arr)).detach().cpu().numpy()
        if was_training:
            network.train()
        return self._validate_probability_rows(q, name="predict_reported_proba")

    def brier_score(self, X: ArrayLike, true_labels: Sequence[int]) -> float:
        """Return multiclass Brier score against synthetic true labels.

        This helper is for synthetic experiments only. Real-data training and
        validation should use :meth:`reported_label_nll`, because real true
        labels are not observed by the server.
        """
        X_arr = self._as_2d_float_array(X)
        self._check_input_dim(X_arr)
        y_true = self._as_y_reported(true_labels, expected_n=X_arr.shape[0])
        p = self.predict_true_proba(X_arr)
        one_hot = np.eye(self.k, dtype=float)[y_true]
        return float(np.mean(np.sum((p - one_hot) ** 2, axis=1)))

    def entropy_summary(self, X: ArrayLike) -> dict[str, float]:
        """Summarise latent predictive entropy for uncertainty reporting."""
        p = np.clip(self.predict_true_proba(X), 1e-12, 1.0)
        entropy = -np.sum(p * np.log(p), axis=1)
        normaliser = float(np.log(self.k)) if self.k > 1 else 1.0
        return {
            "mean_entropy": float(np.mean(entropy)),
            "std_entropy": float(np.std(entropy)),
            "min_entropy": float(np.min(entropy)),
            "max_entropy": float(np.max(entropy)),
            "mean_normalized_entropy": float(np.mean(entropy / normaliser)),
        }

    def evaluation_summary(
        self,
        X: ArrayLike,
        *,
        y_reported: Optional[Sequence[int]] = None,
        true_labels: Optional[Sequence[int]] = None,
    ) -> dict[str, float]:
        """Return report-ready neural diagnostics for available labels."""
        out = self.entropy_summary(X)
        if y_reported is not None:
            out["reported_label_nll"] = self.reported_label_nll(X, y_reported)
        if true_labels is not None:
            out["brier_score"] = self.brier_score(X, true_labels)
        return out

    def poststratify(self, X_pop: ArrayLike, weights: Sequence[float]) -> np.ndarray:
        """Post-stratify latent true probabilities over population cells."""
        X_arr = self._as_2d_float_array(X_pop, name="X_pop")
        self._check_input_dim(X_arr)
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1:
            raise ValueError("weights must be a 1D array")
        if w.size != X_arr.shape[0]:
            raise ValueError("X_pop and weights must have the same number of rows")
        if not np.all(np.isfinite(w)):
            raise ValueError("weights must contain only finite values")
        if np.any(w < 0.0):
            raise ValueError("weights must be non-negative")
        total = float(np.sum(w))
        if total <= 0.0:
            raise ValueError("weights must have positive sum")
        w = w / total
        p = np.sum(self.predict_true_proba(X_arr) * w[:, None], axis=0)
        return self._validate_probability_vector(p, name="poststratify")

    def export_metadata(self, *, include_history: bool = False) -> dict[str, Any]:
        """Return JSON-serialisable fitted model metadata."""
        metadata: dict[str, Any] = {
            "model_type": "RR-aware neural MRP",
            "observation_model": "reported_label_nll_through_kary_randomized_response",
            "k": int(self.k),
            "epsilon": float(self.epsilon),
            "hidden_layers": list(self.hidden_layers),
            "dropout": float(self.dropout),
            "weight_decay": float(self.weight_decay),
            "seed": int(self.seed),
            "device": str(self.device),
            "dtype": str(self.dtype),
            "input_dim": self._input_dim,
            "is_fitted": self.is_fitted,
        }
        if self._network is not None:
            n_params = sum(int(p.numel()) for p in self._network.parameters())
            metadata["n_parameters"] = int(n_params)
        if self._last_fit_info is not None:
            metadata["fit_info"] = self._last_fit_info.to_dict(include_history=include_history)
        return metadata

    def save_metadata(self, path: Union[str, Path], *, include_history: bool = False) -> None:
        """Write fitted model metadata as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.export_metadata(include_history=include_history), indent=2), encoding="utf-8")

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save model state dict and metadata for reproducibility."""
        network = self._require_network()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": network.state_dict(),
                "metadata": self.export_metadata(include_history=False),
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: Union[str, Path], *, device: Union[str, torch.device] = "cpu") -> "RRNeuralMRPModel":
        """Load a checkpoint produced by :meth:`save_checkpoint`."""
        payload = torch.load(Path(path), map_location=device)
        metadata = dict(payload["metadata"])
        model = cls(
            k=int(metadata["k"]),
            epsilon=float(metadata["epsilon"]),
            hidden_layers=tuple(int(v) for v in metadata.get("hidden_layers", [])),
            dropout=float(metadata.get("dropout", 0.0)),
            weight_decay=float(metadata.get("weight_decay", 0.0)),
            seed=int(metadata.get("seed", 0)),
            device=device,
        )
        input_dim = metadata.get("input_dim")
        if input_dim is None:
            raise ValueError("checkpoint metadata is missing input_dim")
        model._build_network(int(input_dim))
        model._require_network().load_state_dict(payload["state_dict"])
        return model

    @staticmethod
    def _validate_probability_rows(p: np.ndarray, *, name: str) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if p.ndim != 2:
            raise RuntimeError(f"{name} returned a non-2D array")
        if not np.all(np.isfinite(p)):
            raise RuntimeError(f"{name} returned non-finite probabilities")
        if np.any(p < -1e-7):
            raise RuntimeError(f"{name} returned negative probabilities")
        p = np.clip(p, 0.0, 1.0)
        row_sums = p.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0.0):
            raise RuntimeError(f"{name} returned a row with zero probability mass")
        p = p / row_sums
        if not np.allclose(p.sum(axis=1), 1.0, atol=1e-5):
            raise RuntimeError(f"{name} probabilities are not normalised")
        return p

    @staticmethod
    def _validate_probability_vector(p: np.ndarray, *, name: str) -> np.ndarray:
        p = np.asarray(p, dtype=float).reshape(-1)
        if not np.all(np.isfinite(p)):
            raise RuntimeError(f"{name} returned non-finite probabilities")
        if np.any(p < -1e-7):
            raise RuntimeError(f"{name} returned negative probabilities")
        p = np.clip(p, 0.0, 1.0)
        total = float(np.sum(p))
        if total <= 0.0:
            raise RuntimeError(f"{name} returned zero probability mass")
        p = p / total
        if not np.isclose(float(p.sum()), 1.0, atol=1e-6):
            raise RuntimeError(f"{name} probabilities are not normalised")
        return p




# Backwards-compatible aliases so older imports continue to work.
NeuralRRMRPModel = RRNeuralMRPModel
NeuralRRMRPFitInfo = RRNeuralMRPFitInfo

__all__ = ["RRNeuralMRPModel", "RRNeuralMRPFitInfo", "NeuralRRMRPModel", "NeuralRRMRPFitInfo"]
