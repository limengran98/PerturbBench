from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np


ArrayDict = Dict[str, np.ndarray]


class MLPRegressor:
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        seed: int,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_dims = [int(value) for value in hidden_dims]
        self.rng = np.random.default_rng(int(seed))
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            scale = np.sqrt(2.0 / max(prev_dim, 1))
            self.weights.append((self.rng.standard_normal((prev_dim, hidden_dim)) * scale).astype(np.float32))
            self.biases.append(np.zeros(hidden_dim, dtype=np.float32))
            prev_dim = hidden_dim
        scale = np.sqrt(2.0 / max(prev_dim, 1))
        self.output_weight = (self.rng.standard_normal((prev_dim, self.output_dim)) * scale).astype(np.float32)
        self.output_bias = np.zeros(self.output_dim, dtype=np.float32)

    def _features(self, batch: ArrayDict) -> np.ndarray:
        return np.hstack([batch["baseline"], batch["intervention_onehot"], batch["context"]]).astype(np.float32)

    def _forward(self, batch: ArrayDict) -> Dict[str, np.ndarray]:
        features = self._features(batch)
        activations = [features]
        for weight, bias in zip(self.weights, self.biases):
            pre_activation = activations[-1] @ weight + bias
            activations.append(np.tanh(pre_activation))
        delta = activations[-1] @ self.output_weight + self.output_bias
        post = batch["baseline"] + delta
        return {
            "features": features,
            "activations": activations,
            "delta": delta.astype(np.float32),
            "post": post.astype(np.float32),
        }

    def _loss_and_gradient(
        self,
        batch: ArrayDict,
        forward: Dict[str, np.ndarray],
        delta_loss_weight: float,
    ) -> tuple[float, np.ndarray]:
        post_error = forward["post"] - batch["post"]
        delta_error = forward["delta"] - batch["delta"]
        loss = float(np.mean(np.square(post_error)) + float(delta_loss_weight) * np.mean(np.square(delta_error)))
        denom = batch["post"].shape[0] * batch["post"].shape[1]
        grad_delta = (2.0 / max(denom, 1)) * (post_error + float(delta_loss_weight) * delta_error)
        return loss, grad_delta.astype(np.float32)

    def fit(
        self,
        train_batch: ArrayDict,
        val_batch: ArrayDict,
        config: Dict[str, float],
        log_fn: Callable[[str], None],
    ) -> Dict[str, float]:
        epochs = int(config["epochs"])
        batch_size = int(config["batch_size"])
        learning_rate = float(config["learning_rate"])
        weight_decay = float(config.get("weight_decay", 0.0))
        patience = int(config["patience"])
        delta_loss_weight = float(config.get("delta_loss_weight", 1.0))

        best_state = None
        best_val = float("inf")
        best_epoch = -1
        stale_epochs = 0
        sample_count = train_batch["post"].shape[0]

        for epoch in range(epochs):
            order = self.rng.permutation(sample_count)
            batch_losses: List[float] = []
            for start in range(0, sample_count, batch_size):
                batch_indices = order[start : start + batch_size]
                batch = {key: value[batch_indices] for key, value in train_batch.items()}
                forward = self._forward(batch)
                loss, grad = self._loss_and_gradient(batch, forward, delta_loss_weight)
                batch_losses.append(loss)

                grad_hidden = grad @ self.output_weight.T
                grad_output_weight = forward["activations"][-1].T @ grad + weight_decay * self.output_weight
                grad_output_bias = np.sum(grad, axis=0)

                hidden_grads: List[np.ndarray] = []
                current_grad = grad_hidden
                for layer_idx in range(len(self.weights) - 1, -1, -1):
                    activation = forward["activations"][layer_idx + 1]
                    current_grad = current_grad * (1.0 - np.square(activation))
                    hidden_grads.append(current_grad)
                    if layer_idx > 0:
                        current_grad = current_grad @ self.weights[layer_idx].T
                hidden_grads.reverse()

                for layer_idx, layer_grad in enumerate(hidden_grads):
                    prev_activation = forward["activations"][layer_idx]
                    grad_weight = prev_activation.T @ layer_grad + weight_decay * self.weights[layer_idx]
                    grad_bias = np.sum(layer_grad, axis=0)
                    self.weights[layer_idx] -= learning_rate * grad_weight.astype(np.float32)
                    self.biases[layer_idx] -= learning_rate * grad_bias.astype(np.float32)

                self.output_weight -= learning_rate * grad_output_weight.astype(np.float32)
                self.output_bias -= learning_rate * grad_output_bias.astype(np.float32)

            val_forward = self._forward(val_batch)
            val_loss, _ = self._loss_and_gradient(val_batch, val_forward, delta_loss_weight)
            train_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
            log_fn(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                stale_epochs = 0
                best_state = self.state_dict()
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    log_fn(f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d}")
                    break

        if best_state is not None:
            self.load_state_dict(best_state)
        return {"best_val_loss": float(best_val), "best_epoch": int(best_epoch)}

    def predict(self, batch: ArrayDict) -> Dict[str, np.ndarray]:
        forward = self._forward(batch)
        return {"post": forward["post"], "delta": forward["delta"]}

    def state_dict(self) -> Dict[str, np.ndarray]:
        state: Dict[str, np.ndarray] = {
            "output_weight": self.output_weight.copy(),
            "output_bias": self.output_bias.copy(),
        }
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            state[f"weight_{idx}"] = weight.copy()
            state[f"bias_{idx}"] = bias.copy()
        return state

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.output_weight = state["output_weight"].copy()
        self.output_bias = state["output_bias"].copy()
        self.weights = []
        self.biases = []
        idx = 0
        while f"weight_{idx}" in state:
            self.weights.append(state[f"weight_{idx}"].copy())
            self.biases.append(state[f"bias_{idx}"].copy())
            idx += 1
