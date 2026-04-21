from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List

import numpy as np


ArrayDict = Dict[str, np.ndarray]


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("structured_hypothesis requires torch") from exc
    return torch, nn, DataLoader, TensorDataset


@dataclass(frozen=True)
class StructuredHypothesisConfig:
    hidden_dim: int
    trunk_depth: int
    residual_depth: int
    use_se_block: bool
    se_reduction: int
    conditioning_mode: str
    conditioning_dim: int
    predict_target: str
    baseline_skip: bool
    zero_init_head: bool
    response_loss_weight: float
    delta_loss_weight: float
    epochs: int
    batch_size: int
    eval_batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    max_grad_norm: float
    use_lr_scheduler: bool
    scheduler_factor: float
    scheduler_patience: int
    scheduler_min_lr: float
    min_delta: float
    divergence_ratio: float
    divergence_patience: int
    dropout: float = 0.0


class StructuredHypothesisRegressor:
    def __init__(
        self,
        *,
        baseline_dim: int,
        intervention_dim: int,
        context_dim: int,
        output_dim: int,
        config: StructuredHypothesisConfig,
        seed: int,
        requested_device: str = "cpu",
    ) -> None:
        self.baseline_dim = int(baseline_dim)
        self.intervention_dim = int(intervention_dim)
        self.context_dim = int(context_dim)
        self.output_dim = int(output_dim)
        self.config = config
        self.seed = int(seed)
        self.torch, self.nn, self.DataLoader, self.TensorDataset = _import_torch()
        self.device = self.torch.device(str(requested_device))
        self.network = self._build_network()
        self.network.to(self.device)

    def _build_network(self):
        torch, nn = self.torch, self.nn
        hidden_dim = int(self.config.hidden_dim)
        conditioning_mode = str(self.config.conditioning_mode)
        cond_input_dim = self.intervention_dim + self.context_dim
        if conditioning_mode == "concat":
            stem_input_dim = self.baseline_dim + cond_input_dim
            baseline_encoder = nn.Identity()
            cond_encoder = nn.Identity()
            cond_to_hidden = nn.Identity()
            gamma_head = nn.Identity()
            beta_head = nn.Identity()
        else:
            baseline_encoder = nn.Linear(self.baseline_dim, hidden_dim)
            cond_dim = max(1, int(self.config.conditioning_dim))
            cond_encoder = nn.Sequential(
                nn.Linear(max(cond_input_dim, 1), cond_dim),
                nn.ReLU(),
            )
            cond_to_hidden = nn.Linear(cond_dim, hidden_dim)
            if conditioning_mode == "film":
                gamma_head = nn.Linear(cond_dim, hidden_dim)
                beta_head = nn.Linear(cond_dim, hidden_dim)
            else:
                gamma_head = nn.Identity()
                beta_head = nn.Identity()
            stem_input_dim = hidden_dim

        trunk_layers: List[nn.Module] = []
        prev_dim = stem_input_dim
        for _ in range(max(1, int(self.config.trunk_depth))):
            trunk_layers.append(nn.Linear(prev_dim, hidden_dim))
            trunk_layers.append(nn.Tanh())
            if float(self.config.dropout) > 0.0:
                trunk_layers.append(nn.Dropout(float(self.config.dropout)))
            prev_dim = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)

        self.residual_blocks = nn.ModuleList()
        self.se_blocks = nn.ModuleList()
        for _ in range(max(0, int(self.config.residual_depth))):
            self.residual_blocks.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    *( [nn.Dropout(float(self.config.dropout))] if float(self.config.dropout) > 0.0 else [] ),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    *( [nn.Dropout(float(self.config.dropout))] if float(self.config.dropout) > 0.0 else [] ),
                )
            )
            if bool(self.config.use_se_block):
                se_hidden = max(1, hidden_dim // max(1, int(self.config.se_reduction)))
                self.se_blocks.append(
                    nn.Sequential(
                        nn.Linear(hidden_dim, se_hidden),
                        nn.ReLU(),
                        nn.Linear(se_hidden, hidden_dim),
                        nn.Sigmoid(),
                    )
                )
            else:
                self.se_blocks.append(nn.Identity())

        self.output_head = nn.Linear(hidden_dim, self.output_dim)
        if bool(self.config.zero_init_head):
            nn.init.zeros_(self.output_head.weight)
            nn.init.zeros_(self.output_head.bias)
        return nn.ModuleDict(
            {
                "baseline_encoder": baseline_encoder,
                "cond_encoder": cond_encoder,
                "cond_to_hidden": cond_to_hidden,
                "gamma_head": gamma_head,
                "beta_head": beta_head,
                "trunk": self.trunk,
                "residual_blocks": self.residual_blocks,
                "se_blocks": self.se_blocks,
                "output_head": self.output_head,
            }
        )

    def _to_tensor(self, array: Any) -> Any:
        if isinstance(array, self.torch.Tensor):
            return array.to(self.device, dtype=self.torch.float32)
        return self.torch.as_tensor(array, dtype=self.torch.float32, device=self.device)

    def _to_cpu_tensor(self, array: np.ndarray) -> Any:
        return self.torch.as_tensor(array, dtype=self.torch.float32)

    def _batch_to_device(self, batch: ArrayDict) -> ArrayDict:
        converted: ArrayDict = {}
        for key, value in batch.items():
            if isinstance(value, np.ndarray) and value.dtype.kind in {"O", "U", "S"}:
                converted[key] = value
            else:
                converted[key] = self._to_tensor(value)
        return converted

    def _resolve_eval_batch_size(self) -> int:
        configured = int(getattr(self.config, "eval_batch_size", 0))
        if configured > 0:
            return configured
        return max(1, min(int(self.config.batch_size), 256))

    def _iter_batch_slices(self, batch: ArrayDict, batch_size: int) -> Iterator[ArrayDict]:
        sample_count = int(batch["baseline"].shape[0])
        for start in range(0, sample_count, max(1, int(batch_size))):
            stop = min(sample_count, start + max(1, int(batch_size)))
            yield {key: value[start:stop] for key, value in batch.items()}

    def _compute_losses(self, outputs: Dict[str, Any], post_target: Any, delta_target: Any) -> Dict[str, Any]:
        mse = self.torch.nn.MSELoss()
        response_loss = mse(outputs["post"], post_target)
        delta_loss = mse(outputs["delta"], delta_target)
        total_loss = (
            float(self.config.response_loss_weight) * response_loss
            + float(self.config.delta_loss_weight) * delta_loss
        )
        return {
            "total": total_loss,
            "response": response_loss,
            "delta": delta_loss,
        }

    def _grad_global_norm(self) -> float:
        total = 0.0
        for parameter in self.network.parameters():
            if parameter.grad is None:
                continue
            grad_norm = float(parameter.grad.detach().data.norm(2).item())
            total += grad_norm ** 2
        return float(total ** 0.5)

    def _build_hidden(self, baseline, cond_input):
        torch = self.torch
        mode = str(self.config.conditioning_mode)
        if mode == "concat":
            hidden_input = torch.cat([baseline, cond_input], dim=1)
        else:
            cond_features = self.network["cond_encoder"](
                cond_input if cond_input.shape[1] > 0 else torch.zeros((baseline.shape[0], 1), device=self.device)
            )
            base_hidden = self.network["baseline_encoder"](baseline)
            if mode == "additive_gate":
                hidden_input = torch.tanh(base_hidden + self.network["cond_to_hidden"](cond_features))
            elif mode == "film":
                gamma = self.network["gamma_head"](cond_features)
                beta = self.network["beta_head"](cond_features)
                hidden_input = torch.tanh((1.0 + gamma) * base_hidden + beta)
            else:
                raise ValueError(f"Unsupported conditioning_mode: {mode}")
        hidden = self.network["trunk"](hidden_input)
        for residual_block, se_block in zip(self.network["residual_blocks"], self.network["se_blocks"]):
            residual = residual_block(hidden)
            residual = residual * se_block(residual)
            hidden = hidden + residual
        return hidden

    def _forward_tensor(self, batch: ArrayDict):
        baseline = self._to_tensor(batch["baseline"])
        intervention = self._to_tensor(batch["intervention_onehot"])
        context = self._to_tensor(batch["context"])
        cond_input = self.torch.cat([intervention, context], dim=1)
        hidden = self._build_hidden(baseline, cond_input)
        raw = self.network["output_head"](hidden)
        if str(self.config.predict_target) == "delta":
            delta = raw
            post = baseline + delta if bool(self.config.baseline_skip) else raw
        elif str(self.config.predict_target) == "response":
            post = raw + baseline if bool(self.config.baseline_skip) else raw
            delta = post - baseline
        else:
            raise ValueError(f"Unsupported predict_target: {self.config.predict_target}")
        return {
            "post": post,
            "delta": delta,
        }

    def _evaluate_loss_batched(self, batch: ArrayDict, batch_size: int) -> Dict[str, float]:
        total_weight = 0
        total_loss = 0.0
        response_loss = 0.0
        delta_loss = 0.0
        self.network.eval()
        with self.torch.no_grad():
            for batch_slice in self._iter_batch_slices(batch, batch_size):
                batch_tensors = self._batch_to_device(batch_slice)
                outputs = self._forward_tensor(batch_tensors)
                losses = self._compute_losses(outputs, batch_tensors["post"], batch_tensors["delta"])
                weight = int(batch_slice["baseline"].shape[0])
                total_weight += weight
                total_loss += float(losses["total"].detach().cpu().item()) * weight
                response_loss += float(losses["response"].detach().cpu().item()) * weight
                delta_loss += float(losses["delta"].detach().cpu().item()) * weight
        normalizer = float(max(total_weight, 1))
        return {
            "total": total_loss / normalizer,
            "response": response_loss / normalizer,
            "delta": delta_loss / normalizer,
        }

    def _predict_batched(self, batch: ArrayDict, batch_size: int) -> Dict[str, np.ndarray]:
        post_chunks: List[np.ndarray] = []
        delta_chunks: List[np.ndarray] = []
        self.network.eval()
        with self.torch.no_grad():
            for batch_slice in self._iter_batch_slices(batch, batch_size):
                outputs = self._forward_tensor(batch_slice)
                post_chunks.append(outputs["post"].detach().cpu().numpy().astype(np.float32))
                delta_chunks.append(outputs["delta"].detach().cpu().numpy().astype(np.float32))
        if not post_chunks:
            return {
                "post": np.zeros((0, self.output_dim), dtype=np.float32),
                "delta": np.zeros((0, self.output_dim), dtype=np.float32),
            }
        return {
            "post": np.concatenate(post_chunks, axis=0),
            "delta": np.concatenate(delta_chunks, axis=0),
        }

    def fit(
        self,
        train_batch: ArrayDict,
        val_batch: ArrayDict,
        config: Dict[str, float],
        log_fn: Callable[[str], None],
    ) -> Dict[str, float]:
        torch = self.torch
        torch.manual_seed(int(self.seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(self.seed))

        train_dataset = self.TensorDataset(
            self._to_cpu_tensor(train_batch["baseline"]),
            self._to_cpu_tensor(train_batch["post"]),
            self._to_cpu_tensor(train_batch["delta"]),
            self._to_cpu_tensor(train_batch["intervention_onehot"]),
            self._to_cpu_tensor(train_batch["context"]),
        )
        train_loader = self.DataLoader(train_dataset, batch_size=int(self.config.batch_size), shuffle=True)
        optimizer = torch.optim.Adam(self.network.parameters(), lr=float(self.config.learning_rate), weight_decay=float(self.config.weight_decay))
        scheduler = None
        if bool(self.config.use_lr_scheduler):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(self.config.scheduler_factor),
                patience=int(self.config.scheduler_patience),
                min_lr=float(self.config.scheduler_min_lr),
            )
        eval_batch_size = self._resolve_eval_batch_size()

        best_state = None
        best_val = float("inf")
        best_epoch = -1
        stale_epochs = 0
        divergence_stale = 0

        for epoch in range(int(self.config.epochs)):
            self.network.train()
            train_losses: List[float] = []
            train_response_losses: List[float] = []
            train_delta_losses: List[float] = []
            grad_norms: List[float] = []
            for baseline, post, delta, intervention_onehot, context in train_loader:
                batch = {
                    "baseline": baseline,
                    "post": post,
                    "delta": delta,
                    "intervention_onehot": intervention_onehot,
                    "context": context,
                }
                batch_tensors = self._batch_to_device(batch)
                outputs = self._forward_tensor(batch_tensors)
                losses = self._compute_losses(outputs, batch_tensors["post"], batch_tensors["delta"])
                loss = losses["total"]
                optimizer.zero_grad()
                loss.backward()
                if float(self.config.max_grad_norm) > 0.0:
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(self.network.parameters(), float(self.config.max_grad_norm)).detach().cpu().item()
                    )
                else:
                    grad_norm = self._grad_global_norm()
                optimizer.step()
                train_losses.append(float(loss.detach().cpu().item()))
                train_response_losses.append(float(losses["response"].detach().cpu().item()))
                train_delta_losses.append(float(losses["delta"].detach().cpu().item()))
                grad_norms.append(float(grad_norm))

            val_losses = self._evaluate_loss_batched(val_batch, eval_batch_size)
            val_value = float(val_losses["total"])
            val_response_value = float(val_losses["response"])
            val_delta_value = float(val_losses["delta"])
            if scheduler is not None:
                scheduler.step(val_value)
            current_lr = float(optimizer.param_groups[0]["lr"])
            train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
            train_response_loss = float(np.mean(train_response_losses)) if train_response_losses else float("nan")
            train_delta_loss = float(np.mean(train_delta_losses)) if train_delta_losses else float("nan")
            grad_norm_mean = float(np.mean(grad_norms)) if grad_norms else float("nan")
            log_fn(
                f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_value:.6f} "
                f"train_response_loss={train_response_loss:.6f} train_delta_loss={train_delta_loss:.6f} "
                f"val_response_loss={val_response_value:.6f} val_delta_loss={val_delta_value:.6f} "
                f"grad_norm={grad_norm_mean:.6f} lr={current_lr:.6e}"
            )
            if not np.isfinite(train_loss) or not np.isfinite(val_value):
                log_fn(f"divergence_stop epoch={epoch:03d} reason=non_finite_loss")
                break
            if val_value < best_val - float(self.config.min_delta):
                best_val = val_value
                best_epoch = epoch
                stale_epochs = 0
                divergence_stale = 0
                best_state = {key: value.detach().cpu().clone() for key, value in self.network.state_dict().items()}
            else:
                stale_epochs += 1
                if float(self.config.divergence_ratio) > 0.0 and best_val < float("inf"):
                    if val_value > best_val * float(self.config.divergence_ratio):
                        divergence_stale += 1
                    else:
                        divergence_stale = 0
                    if divergence_stale >= int(self.config.divergence_patience):
                        log_fn(
                            f"divergence_stop epoch={epoch:03d} best_epoch={best_epoch:03d} "
                            f"best_val={best_val:.6f} val_loss={val_value:.6f}"
                        )
                        break
                if stale_epochs >= int(self.config.patience):
                    log_fn(f"early_stop epoch={epoch:03d} best_epoch={best_epoch:03d}")
                    break

        if best_state is not None:
            self.network.load_state_dict(best_state)
        return {
            "best_val_loss": float(best_val),
            "best_epoch": int(best_epoch),
            "final_lr": float(optimizer.param_groups[0]["lr"]),
            "eval_batch_size": int(eval_batch_size),
        }

    def predict(self, batch: ArrayDict) -> Dict[str, np.ndarray]:
        return self._predict_batched(batch, self._resolve_eval_batch_size())
