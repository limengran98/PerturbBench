from __future__ import annotations

from typing import Callable, Dict

import numpy as np


ArrayDict = Dict[str, np.ndarray]


class ConditionedResidualRegressor:
    def __init__(
        self,
        baseline_dim: int,
        intervention_dim: int,
        context_dim: int,
        output_dim: int,
        hidden_dim: int,
        conditioning_dim: int,
        seed: int,
    ) -> None:
        self.baseline_dim = int(baseline_dim)
        self.intervention_dim = int(intervention_dim)
        self.context_dim = int(context_dim)
        self.output_dim = int(output_dim)
        self.hidden_dim = int(hidden_dim)
        self.conditioning_dim = int(conditioning_dim)
        self.rng = np.random.default_rng(int(seed))

        cond_input_dim = self.intervention_dim + self.context_dim
        self.weight_cond = self._init_weight(cond_input_dim, self.conditioning_dim)
        self.bias_cond = np.zeros(self.conditioning_dim, dtype=np.float32)
        self.weight_base = self._init_weight(self.baseline_dim, self.hidden_dim)
        self.weight_gate = self._init_weight(self.conditioning_dim, self.hidden_dim)
        self.bias_hidden = np.zeros(self.hidden_dim, dtype=np.float32)
        self.weight_res = self._init_weight(self.hidden_dim, self.hidden_dim)
        self.weight_res_cond = self._init_weight(self.conditioning_dim, self.hidden_dim)
        self.bias_res = np.zeros(self.hidden_dim, dtype=np.float32)
        self.weight_out = self._init_weight(self.hidden_dim, self.output_dim)
        self.bias_out = np.zeros(self.output_dim, dtype=np.float32)

    def _init_weight(self, input_dim: int, output_dim: int) -> np.ndarray:
        scale = np.sqrt(2.0 / max(input_dim, 1))
        return (self.rng.standard_normal((input_dim, output_dim)) * scale).astype(np.float32)

    def _forward(self, batch: ArrayDict) -> Dict[str, np.ndarray]:
        cond_input = np.hstack([batch["intervention_onehot"], batch["context"]]).astype(np.float32)
        cond_pre = cond_input @ self.weight_cond + self.bias_cond
        cond = np.tanh(cond_pre)

        hidden_pre = batch["baseline"] @ self.weight_base + cond @ self.weight_gate + self.bias_hidden
        hidden = np.tanh(hidden_pre)

        residual_pre = hidden @ self.weight_res + cond @ self.weight_res_cond + self.bias_res
        residual = np.tanh(residual_pre)
        hidden_out = hidden + residual

        delta = hidden_out @ self.weight_out + self.bias_out
        post = batch["baseline"] + delta
        return {
            "cond_input": cond_input,
            "cond": cond,
            "hidden": hidden,
            "residual": residual,
            "hidden_out": hidden_out,
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
        sample_count = train_batch["post"].shape[0]

        best_state = None
        best_val = float("inf")
        best_epoch = -1
        stale_epochs = 0

        for epoch in range(epochs):
            order = self.rng.permutation(sample_count)
            batch_losses = []
            for start in range(0, sample_count, batch_size):
                batch_indices = order[start : start + batch_size]
                batch = {key: value[batch_indices] for key, value in train_batch.items()}
                forward = self._forward(batch)
                loss, grad_delta = self._loss_and_gradient(batch, forward, delta_loss_weight)
                batch_losses.append(loss)

                grad_weight_out = forward["hidden_out"].T @ grad_delta + weight_decay * self.weight_out
                grad_bias_out = np.sum(grad_delta, axis=0)
                grad_hidden_out = grad_delta @ self.weight_out.T

                grad_residual = grad_hidden_out * (1.0 - np.square(forward["residual"]))
                grad_weight_res = forward["hidden"].T @ grad_residual + weight_decay * self.weight_res
                grad_weight_res_cond = forward["cond"].T @ grad_residual + weight_decay * self.weight_res_cond
                grad_bias_res = np.sum(grad_residual, axis=0)

                grad_hidden = grad_hidden_out + grad_residual @ self.weight_res.T
                grad_cond = grad_residual @ self.weight_res_cond.T

                grad_hidden_pre = grad_hidden * (1.0 - np.square(forward["hidden"]))
                grad_weight_base = batch["baseline"].T @ grad_hidden_pre + weight_decay * self.weight_base
                grad_weight_gate = forward["cond"].T @ grad_hidden_pre + weight_decay * self.weight_gate
                grad_bias_hidden = np.sum(grad_hidden_pre, axis=0)
                grad_cond += grad_hidden_pre @ self.weight_gate.T

                grad_cond_pre = grad_cond * (1.0 - np.square(forward["cond"]))
                grad_weight_cond = forward["cond_input"].T @ grad_cond_pre + weight_decay * self.weight_cond
                grad_bias_cond = np.sum(grad_cond_pre, axis=0)

                self.weight_out -= learning_rate * grad_weight_out.astype(np.float32)
                self.bias_out -= learning_rate * grad_bias_out.astype(np.float32)
                self.weight_res -= learning_rate * grad_weight_res.astype(np.float32)
                self.weight_res_cond -= learning_rate * grad_weight_res_cond.astype(np.float32)
                self.bias_res -= learning_rate * grad_bias_res.astype(np.float32)
                self.weight_base -= learning_rate * grad_weight_base.astype(np.float32)
                self.weight_gate -= learning_rate * grad_weight_gate.astype(np.float32)
                self.bias_hidden -= learning_rate * grad_bias_hidden.astype(np.float32)
                self.weight_cond -= learning_rate * grad_weight_cond.astype(np.float32)
                self.bias_cond -= learning_rate * grad_bias_cond.astype(np.float32)

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
        return {
            "weight_cond": self.weight_cond.copy(),
            "bias_cond": self.bias_cond.copy(),
            "weight_base": self.weight_base.copy(),
            "weight_gate": self.weight_gate.copy(),
            "bias_hidden": self.bias_hidden.copy(),
            "weight_res": self.weight_res.copy(),
            "weight_res_cond": self.weight_res_cond.copy(),
            "bias_res": self.bias_res.copy(),
            "weight_out": self.weight_out.copy(),
            "bias_out": self.bias_out.copy(),
        }

    def load_state_dict(self, state: Dict[str, np.ndarray]) -> None:
        self.weight_cond = state["weight_cond"].copy()
        self.bias_cond = state["bias_cond"].copy()
        self.weight_base = state["weight_base"].copy()
        self.weight_gate = state["weight_gate"].copy()
        self.bias_hidden = state["bias_hidden"].copy()
        self.weight_res = state["weight_res"].copy()
        self.weight_res_cond = state["weight_res_cond"].copy()
        self.bias_res = state["bias_res"].copy()
        self.weight_out = state["weight_out"].copy()
        self.bias_out = state["bias_out"].copy()
