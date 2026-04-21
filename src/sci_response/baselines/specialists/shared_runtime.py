from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from sci_response.data.schemas import PreparedDataset


COMPONENT_SPLIT_PATTERN = re.compile(r"[+|;&]")


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("shared-runtime specialist adaptations require torch") from exc
    return torch, nn, F


def split_intervention_components(label: str) -> List[str]:
    text = str(label).strip()
    if not text:
        return ["<empty>"]
    parts = [item.strip() for item in COMPONENT_SPLIT_PATTERN.split(text) if item.strip()]
    return parts or [text]


def build_component_vocabulary(labels: Sequence[str]) -> Dict[str, int]:
    vocabulary = {"<unk>": 0}
    for label in labels:
        for component in split_intervention_components(str(label)):
            if component not in vocabulary:
                vocabulary[component] = len(vocabulary)
    return vocabulary


def encode_component_bag(labels: np.ndarray, vocabulary: Dict[str, int]) -> np.ndarray:
    matrix = np.zeros((labels.shape[0], len(vocabulary)), dtype=np.float32)
    for row_idx, label in enumerate(labels.astype(str).tolist()):
        components = split_intervention_components(label)
        seen = False
        for component in components:
            if component in vocabulary:
                matrix[row_idx, vocabulary[component]] = 1.0
                seen = True
        if not seen:
            matrix[row_idx, vocabulary["<unk>"]] = 1.0
    return matrix


def _stable_bucket(text: str, bucket_count: int) -> int:
    import hashlib

    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % max(bucket_count, 1)


def hashed_text_features(values: np.ndarray, feature_dim: int, ngram_range: Tuple[int, int] = (2, 4)) -> np.ndarray:
    matrix = np.zeros((values.shape[0], feature_dim), dtype=np.float32)
    for row_idx, raw_value in enumerate(values.astype(str).tolist()):
        text = raw_value.strip().lower()
        tokens: List[str] = []
        if text:
            for n in range(ngram_range[0], ngram_range[1] + 1):
                if len(text) < n:
                    continue
                for start in range(0, len(text) - n + 1):
                    tokens.append(text[start : start + n])
        if not tokens:
            tokens = ["<empty>"]
        scale = 1.0 / float(len(tokens))
        for token in tokens:
            matrix[row_idx, _stable_bucket(token, feature_dim)] += scale
    return matrix


def build_auxiliary_matrix(dataset: PreparedDataset) -> np.ndarray:
    parts: List[np.ndarray] = []
    if dataset.context_matrix.shape[1] > 0:
        parts.append(dataset.context_matrix.astype(np.float32))
    dose = dataset.doses.astype(np.float32).reshape(-1, 1)
    time = dataset.times.astype(np.float32).reshape(-1, 1)
    parts.extend(
        [
            np.nan_to_num(dose, nan=0.0),
            np.isnan(dose).astype(np.float32),
            np.nan_to_num(time, nan=0.0),
            np.isnan(time).astype(np.float32),
        ]
    )
    return np.hstack(parts).astype(np.float32) if parts else np.zeros((dataset.sample_ids.shape[0], 0), dtype=np.float32)


def fit_standard_scaler(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(matrix, axis=0, keepdims=True).astype(np.float32)
    scale = np.std(matrix, axis=0, keepdims=True).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def apply_standard_scaler(matrix: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((matrix - mean) / scale).astype(np.float32)


def invert_standard_scaler(matrix: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (matrix * scale + mean).astype(np.float32)


@dataclass(frozen=True)
class AdaptationFeatureBundle:
    baseline_scaled: np.ndarray
    auxiliary_scaled: np.ndarray
    intervention_hash: np.ndarray
    component_bag: np.ndarray
    target_scaled: np.ndarray
    target_mean: np.ndarray
    target_scale: np.ndarray
    feature_manifest: Dict[str, Any]


def build_adaptation_features(
    dataset: PreparedDataset,
    train_indices: np.ndarray,
    *,
    hash_dim: int = 64,
    target_kind: str = "delta",
) -> AdaptationFeatureBundle:
    baseline_mean, baseline_scale = fit_standard_scaler(dataset.x_baseline[train_indices].astype(np.float32))
    auxiliary = build_auxiliary_matrix(dataset)
    if auxiliary.shape[1] > 0:
        auxiliary_mean, auxiliary_scale = fit_standard_scaler(auxiliary[train_indices].astype(np.float32))
        auxiliary_scaled = apply_standard_scaler(auxiliary, auxiliary_mean, auxiliary_scale)
    else:
        auxiliary_scaled = auxiliary.astype(np.float32)
        auxiliary_mean = np.zeros((1, 0), dtype=np.float32)
        auxiliary_scale = np.ones((1, 0), dtype=np.float32)
    if target_kind == "delta":
        target_matrix = dataset.delta_response.astype(np.float32)
    elif target_kind == "response":
        target_matrix = dataset.y_response.astype(np.float32)
    else:
        raise ValueError(f"Unsupported target_kind for adaptation features: {target_kind}")
    target_mean, target_scale = fit_standard_scaler(target_matrix[train_indices].astype(np.float32))
    baseline_scaled = apply_standard_scaler(dataset.x_baseline.astype(np.float32), baseline_mean, baseline_scale)
    target_scaled = apply_standard_scaler(target_matrix.astype(np.float32), target_mean, target_scale)
    intervention_hash = hashed_text_features(dataset.intervention_ids.astype(str), feature_dim=int(hash_dim)).astype(np.float32)
    component_vocab = build_component_vocabulary(dataset.intervention_ids[train_indices].astype(str).tolist())
    component_bag = encode_component_bag(dataset.intervention_ids.astype(str), component_vocab).astype(np.float32)
    feature_manifest = {
        "baseline_dim": int(baseline_scaled.shape[1]),
        "auxiliary_dim": int(auxiliary_scaled.shape[1]),
        "hash_dim": int(intervention_hash.shape[1]),
        "component_vocab_size": int(component_bag.shape[1]),
        "component_vocab_preview": list(component_vocab.keys())[:20],
        "target_kind": str(target_kind),
        "scalers": {
            "baseline_mean_shape": list(baseline_mean.shape),
            "auxiliary_mean_shape": list(auxiliary_mean.shape),
            "target_mean_shape": list(target_mean.shape),
        },
    }
    return AdaptationFeatureBundle(
        baseline_scaled=baseline_scaled,
        auxiliary_scaled=auxiliary_scaled,
        intervention_hash=intervention_hash,
        component_bag=component_bag,
        target_scaled=target_scaled,
        target_mean=target_mean,
        target_scale=target_scale,
        feature_manifest=feature_manifest,
    )


def resolve_torch_device(requested_device: str):
    torch, _, _ = _import_torch()
    if str(requested_device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested CUDA device {requested_device} for shared-runtime specialist adaptation, but CUDA is unavailable")
        return torch.device(str(requested_device))
    return torch.device("cpu")


class GEARSSharedAdapterModel:
    def __init__(
        self,
        *,
        baseline_dim: int,
        auxiliary_dim: int,
        hash_dim: int,
        component_dim: int,
        output_dim: int,
        hidden_dim: int,
        perturb_rank: int,
        dropout: float,
    ) -> None:
        torch, nn, _ = _import_torch()
        self.torch = torch
        self.nn = nn
        self.network = nn.ModuleDict(
            {
                "baseline_encoder": nn.Sequential(
                    nn.Linear(baseline_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ),
                "aux_encoder": nn.Sequential(
                    nn.Linear(max(auxiliary_dim, 1), hidden_dim),
                    nn.ReLU(),
                ),
                "perturb_encoder": nn.Sequential(
                    nn.Linear(hash_dim + component_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ),
                "shared_head": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                ),
                "gate_head": nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                ),
                "perturb_to_rank": nn.Linear(hidden_dim, perturb_rank),
            }
        )
        self.gene_basis = nn.Parameter(torch.randn(output_dim, perturb_rank) * 0.02)

    def parameters(self):
        return list(self.network.parameters()) + [self.gene_basis]

    def to(self, device):
        self.network.to(device)
        self.gene_basis = self.nn.Parameter(self.gene_basis.to(device))
        return self

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def state_dict(self):
        state = {"gene_basis": self.gene_basis.detach().clone()}
        for key, value in self.network.state_dict().items():
            state[f"network::{key}"] = value.detach().clone()
        return state

    def load_state_dict(self, state: Dict[str, Any]):
        network_state = {key.split("network::", 1)[1]: value for key, value in state.items() if key.startswith("network::")}
        self.network.load_state_dict(network_state)
        self.gene_basis.data.copy_(state["gene_basis"])

    def forward(self, baseline, auxiliary, perturb_hash, component_bag):
        torch = self.torch
        if auxiliary.shape[1] == 0:
            auxiliary = torch.zeros((baseline.shape[0], 1), device=baseline.device, dtype=baseline.dtype)
        base_hidden = self.network["baseline_encoder"](baseline)
        aux_hidden = self.network["aux_encoder"](auxiliary)
        perturb_hidden = self.network["perturb_encoder"](torch.cat([perturb_hash, component_bag], dim=1))
        mixed_hidden = base_hidden + aux_hidden + perturb_hidden
        shared_delta = self.network["shared_head"](mixed_hidden)
        perturb_rank = self.network["perturb_to_rank"](perturb_hidden)
        gene_specific = perturb_rank @ self.gene_basis.T
        gate = torch.sigmoid(self.network["gate_head"](torch.cat([base_hidden, perturb_hidden], dim=1)))
        delta = shared_delta + gate * gene_specific
        sparsity_penalty = gene_specific.abs().mean()
        return delta, {"gene_specific_l1": sparsity_penalty}


class CPASharedAdapterModel:
    def __init__(
        self,
        *,
        baseline_dim: int,
        auxiliary_dim: int,
        hash_dim: int,
        component_dim: int,
        output_dim: int,
        latent_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        torch, nn, _ = _import_torch()
        self.torch = torch
        self.nn = nn
        perturb_input_dim = hash_dim + component_dim + max(auxiliary_dim, 1)
        base_input_dim = baseline_dim + max(auxiliary_dim, 1)
        self.network = nn.ModuleDict(
            {
                "base_encoder": nn.Sequential(
                    nn.Linear(base_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, latent_dim),
                ),
                "perturb_encoder": nn.Sequential(
                    nn.Linear(perturb_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, latent_dim),
                ),
                "decoder": nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                ),
            }
        )

    def parameters(self):
        return list(self.network.parameters())

    def to(self, device):
        self.network.to(device)
        return self

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def state_dict(self):
        return {f"network::{key}": value.detach().clone() for key, value in self.network.state_dict().items()}

    def load_state_dict(self, state: Dict[str, Any]):
        network_state = {key.split("network::", 1)[1]: value for key, value in state.items()}
        self.network.load_state_dict(network_state)

    def forward(self, baseline, auxiliary, perturb_hash, component_bag):
        torch = self.torch
        if auxiliary.shape[1] == 0:
            auxiliary = torch.zeros((baseline.shape[0], 1), device=baseline.device, dtype=baseline.dtype)
        base_latent = self.network["base_encoder"](torch.cat([baseline, auxiliary], dim=1))
        perturb_latent = self.network["perturb_encoder"](torch.cat([perturb_hash, component_bag, auxiliary], dim=1))
        latent = base_latent + perturb_latent
        delta = self.network["decoder"](latent)
        latent_penalty = 0.5 * (base_latent.pow(2).mean() + perturb_latent.pow(2).mean())
        return delta, {"latent_l2": latent_penalty}


class XPertSharedAdapterModel:
    def __init__(
        self,
        *,
        baseline_dim: int,
        auxiliary_dim: int,
        hash_dim: int,
        component_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        perturb_rank: int,
    ) -> None:
        torch, nn, _ = _import_torch()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"XPert hidden_dim must be divisible by num_heads, got {hidden_dim} and {num_heads}")
        self.torch = torch
        self.nn = nn
        self.network = nn.ModuleDict(
            {
                "drug_encoder": nn.Sequential(
                    nn.Linear(hash_dim + component_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                "aux_encoder": nn.Sequential(
                    nn.Linear(max(auxiliary_dim, 1), hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                "summary_encoder": nn.Sequential(
                    nn.Linear(baseline_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                "query_token": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                ),
                "cross_attention": nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                ),
                "ffn": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                ),
                "ln1": nn.LayerNorm(hidden_dim),
                "ln2": nn.LayerNorm(hidden_dim),
                "shared_head": nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                ),
                "perturb_to_rank": nn.Linear(hidden_dim * 2, perturb_rank),
            }
        )
        self.gene_basis = nn.Parameter(torch.randn(output_dim, perturb_rank) * 0.02)

    def parameters(self):
        return list(self.network.parameters()) + [self.gene_basis]

    def to(self, device):
        self.network.to(device)
        self.gene_basis = self.nn.Parameter(self.gene_basis.to(device))
        return self

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def state_dict(self):
        state = {
            "gene_basis": self.gene_basis.detach().clone(),
        }
        for key, value in self.network.state_dict().items():
            state[f"network::{key}"] = value.detach().clone()
        return state

    def load_state_dict(self, state: Dict[str, Any]):
        network_state = {key.split("network::", 1)[1]: value for key, value in state.items() if key.startswith("network::")}
        self.network.load_state_dict(network_state)
        self.gene_basis.data.copy_(state["gene_basis"])

    def forward(self, baseline, auxiliary, perturb_hash, component_bag):
        torch = self.torch
        if auxiliary.shape[1] == 0:
            auxiliary = torch.zeros((baseline.shape[0], 1), device=baseline.device, dtype=baseline.dtype)
        drug_token = self.network["drug_encoder"](torch.cat([perturb_hash, component_bag], dim=1))
        aux_token = self.network["aux_encoder"](auxiliary)
        summary_token = self.network["summary_encoder"](baseline)
        key_value_tokens = torch.stack([drug_token, aux_token, summary_token], dim=1)
        query = self.network["query_token"](summary_token).unsqueeze(1)
        attn_output, _ = self.network["cross_attention"](
            query=query,
            key=key_value_tokens,
            value=key_value_tokens,
            need_weights=False,
        )
        hidden = self.network["ln1"](query + attn_output)
        hidden = self.network["ln2"](hidden + self.network["ffn"](hidden))
        pooled = hidden.squeeze(1)
        shared_delta = self.network["shared_head"](pooled)

        perturb_context = torch.cat([drug_token, aux_token], dim=1)
        perturb_rank = self.network["perturb_to_rank"](perturb_context)
        drug_specific = perturb_rank @ self.gene_basis.T
        delta = shared_delta + drug_specific
        regularizer = 0.5 * self.gene_basis.pow(2).mean()
        return delta, {"gene_basis_l2": regularizer}


def _gaussian_mmd(x, y, sigma: float = 1.0):
    torch, _, _ = _import_torch()
    if x.shape[0] == 0 or y.shape[0] == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    gamma = 1.0 / max(2.0 * sigma * sigma, 1e-6)
    x_norm = (x * x).sum(dim=1, keepdim=True)
    y_norm = (y * y).sum(dim=1, keepdim=True)
    k_xx = torch.exp(-gamma * (x_norm + x_norm.T - 2.0 * (x @ x.T)))
    k_yy = torch.exp(-gamma * (y_norm + y_norm.T - 2.0 * (y @ y.T)))
    k_xy = torch.exp(-gamma * (x_norm + y_norm.T - 2.0 * (x @ y.T)))
    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()


class CellOTSharedAdapterModel:
    def __init__(
        self,
        *,
        baseline_dim: int,
        auxiliary_dim: int,
        hash_dim: int,
        component_dim: int,
        output_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
    ) -> None:
        torch, nn, _ = _import_torch()
        self.torch = torch
        self.nn = nn
        base_input_dim = baseline_dim + max(auxiliary_dim, 1)
        cond_input_dim = hash_dim + component_dim + max(auxiliary_dim, 1)
        self.network = nn.ModuleDict(
            {
                "source_encoder": nn.Sequential(
                    nn.Linear(base_input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, latent_dim),
                ),
                "transport_encoder": nn.Sequential(
                    nn.Linear(cond_input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, latent_dim),
                ),
                "decoder": nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                ),
            }
        )

    def parameters(self):
        return list(self.network.parameters())

    def to(self, device):
        self.network.to(device)
        return self

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def state_dict(self):
        return {f"network::{key}": value.detach().clone() for key, value in self.network.state_dict().items()}

    def load_state_dict(self, state: Dict[str, Any]):
        network_state = {key.split("network::", 1)[1]: value for key, value in state.items()}
        self.network.load_state_dict(network_state)

    def forward(self, baseline, auxiliary, perturb_hash, component_bag):
        torch = self.torch
        if auxiliary.shape[1] == 0:
            auxiliary = torch.zeros((baseline.shape[0], 1), device=baseline.device, dtype=baseline.dtype)
        source_latent = self.network["source_encoder"](torch.cat([baseline, auxiliary], dim=1))
        transport_shift = self.network["transport_encoder"](torch.cat([perturb_hash, component_bag, auxiliary], dim=1))
        transported_latent = source_latent + transport_shift
        response = self.network["decoder"](transported_latent)
        transport_cost = transport_shift.pow(2).mean()
        return response, {
            "transport_cost": transport_cost,
            "source_latent": source_latent,
            "transported_latent": transported_latent,
        }


class TranSiGenSharedModel:
    def __init__(
        self,
        *,
        baseline_dim: int,
        auxiliary_dim: int,
        hash_dim: int,
        component_dim: int,
        output_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
    ) -> None:
        torch, nn, _ = _import_torch()
        self.torch = torch
        self.nn = nn
        base_input_dim = baseline_dim + max(auxiliary_dim, 1)
        cond_input_dim = hash_dim + component_dim + max(auxiliary_dim, 1)
        self.network = nn.ModuleDict(
            {
                "baseline_encoder": nn.Sequential(
                    nn.Linear(base_input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, latent_dim),
                ),
                "perturb_encoder": nn.Sequential(
                    nn.Linear(cond_input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, latent_dim),
                ),
                "baseline_decoder": nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, baseline_dim),
                ),
                "response_decoder": nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                ),
            }
        )

    def parameters(self):
        return list(self.network.parameters())

    def to(self, device):
        self.network.to(device)
        return self

    def train(self):
        self.network.train()

    def eval(self):
        self.network.eval()

    def state_dict(self):
        return {f"network::{key}": value.detach().clone() for key, value in self.network.state_dict().items()}

    def load_state_dict(self, state: Dict[str, Any]):
        network_state = {key.split("network::", 1)[1]: value for key, value in state.items()}
        self.network.load_state_dict(network_state)

    def forward(self, baseline, auxiliary, perturb_hash, component_bag):
        torch = self.torch
        if auxiliary.shape[1] == 0:
            auxiliary = torch.zeros((baseline.shape[0], 1), device=baseline.device, dtype=baseline.dtype)
        baseline_latent = self.network["baseline_encoder"](torch.cat([baseline, auxiliary], dim=1))
        perturb_latent = self.network["perturb_encoder"](torch.cat([perturb_hash, component_bag, auxiliary], dim=1))
        treated_latent = baseline_latent + perturb_latent
        baseline_recon = self.network["baseline_decoder"](baseline_latent)
        response = self.network["response_decoder"](treated_latent)
        latent_penalty = 0.5 * (baseline_latent.pow(2).mean() + perturb_latent.pow(2).mean())
        return response, {
            "baseline_recon": baseline_recon,
            "latent_l2": latent_penalty,
        }


def _make_tensor_bundle(bundle: AdaptationFeatureBundle, indices: np.ndarray, device) -> Dict[str, Any]:
    torch, _, _ = _import_torch()
    return {
        "baseline": torch.from_numpy(bundle.baseline_scaled[indices]).to(device=device, dtype=torch.float32),
        "auxiliary": torch.from_numpy(bundle.auxiliary_scaled[indices]).to(device=device, dtype=torch.float32),
        "perturb_hash": torch.from_numpy(bundle.intervention_hash[indices]).to(device=device, dtype=torch.float32),
        "component_bag": torch.from_numpy(bundle.component_bag[indices]).to(device=device, dtype=torch.float32),
        "target": torch.from_numpy(bundle.target_scaled[indices]).to(device=device, dtype=torch.float32),
    }


def _predict_model(model, bundle: AdaptationFeatureBundle, indices: np.ndarray, device) -> np.ndarray:
    torch, _, _ = _import_torch()
    model.eval()
    with torch.no_grad():
        tensors = _make_tensor_bundle(bundle, indices, device)
        delta_pred, _ = model.forward(
            tensors["baseline"],
            tensors["auxiliary"],
            tensors["perturb_hash"],
            tensors["component_bag"],
        )
    return delta_pred.detach().cpu().numpy().astype(np.float32)


def train_shared_runtime_specialist(
    *,
    model_kind: str,
    dataset: PreparedDataset,
    split,
    config: Dict[str, Any],
    seed: int,
    requested_device: str,
    log_fn: Callable[[str], None],
) -> Dict[str, Any]:
    torch, _, F = _import_torch()
    device = resolve_torch_device(str(requested_device))
    sample_id_to_index = {str(sample_id): idx for idx, sample_id in enumerate(dataset.sample_ids.astype(str).tolist())}
    train_idx_full = np.asarray([sample_id_to_index[str(sample_id)] for sample_id in split.train_ids], dtype=np.int64)
    val_idx_full = np.asarray([sample_id_to_index[str(sample_id)] for sample_id in split.val_ids], dtype=np.int64)
    test_idx_full = np.asarray([sample_id_to_index[str(sample_id)] for sample_id in split.test_ids], dtype=np.int64)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    max_train_samples = config.get("max_train_samples")
    max_train_samples = None if max_train_samples in {None, "", 0} else int(max_train_samples)
    max_val_samples = config.get("max_val_samples")
    max_val_samples = None if max_val_samples in {None, "", 0} else int(max_val_samples)

    train_idx_fit = train_idx_full
    val_idx_fit = val_idx_full
    if max_train_samples is not None and train_idx_fit.shape[0] > max_train_samples:
        train_idx_fit = np.random.default_rng(int(seed)).choice(train_idx_fit, size=max_train_samples, replace=False)
    if max_val_samples is not None and val_idx_fit.shape[0] > max_val_samples:
        val_idx_fit = np.random.default_rng(int(seed) + 1).choice(val_idx_fit, size=max_val_samples, replace=False)

    hash_dim = int(config.get("hash_dim", 64))
    target_kind = str(config.get("target_kind", "delta"))
    bundle = build_adaptation_features(dataset, train_idx_fit, hash_dim=hash_dim, target_kind=target_kind)

    hidden_dim = int(config.get("hidden_dim", 256))
    dropout = float(config.get("dropout", 0.1))
    if model_kind == "gears":
        model = GEARSSharedAdapterModel(
            baseline_dim=int(bundle.baseline_scaled.shape[1]),
            auxiliary_dim=int(bundle.auxiliary_scaled.shape[1]),
            hash_dim=int(bundle.intervention_hash.shape[1]),
            component_dim=int(bundle.component_bag.shape[1]),
            output_dim=int(dataset.output_dim),
            hidden_dim=hidden_dim,
            perturb_rank=int(config.get("perturb_rank", 32)),
            dropout=dropout,
        ).to(device)
        regularization_weight = float(config.get("gene_specific_l1_weight", 1e-4))
        regularization_key = "gene_specific_l1"
        execution_mode = "shared_runtime_adaptation_gears"
    elif model_kind == "cpa":
        model = CPASharedAdapterModel(
            baseline_dim=int(bundle.baseline_scaled.shape[1]),
            auxiliary_dim=int(bundle.auxiliary_scaled.shape[1]),
            hash_dim=int(bundle.intervention_hash.shape[1]),
            component_dim=int(bundle.component_bag.shape[1]),
            output_dim=int(dataset.output_dim),
            latent_dim=int(config.get("latent_dim", 128)),
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(device)
        regularization_weight = float(config.get("latent_l2_weight", 1e-4))
        regularization_key = "latent_l2"
        execution_mode = "shared_runtime_adaptation_cpa"
    elif model_kind == "xpert":
        model = XPertSharedAdapterModel(
            baseline_dim=int(bundle.baseline_scaled.shape[1]),
            auxiliary_dim=int(bundle.auxiliary_scaled.shape[1]),
            hash_dim=int(bundle.intervention_hash.shape[1]),
            component_dim=int(bundle.component_bag.shape[1]),
            output_dim=int(dataset.output_dim),
            hidden_dim=int(config.get("hidden_dim", 96)),
            num_heads=int(config.get("num_heads", 4)),
            dropout=dropout,
            perturb_rank=int(config.get("perturb_rank", 32)),
        ).to(device)
        regularization_weight = float(config.get("gene_basis_l2_weight", 1e-4))
        regularization_key = "gene_basis_l2"
        execution_mode = "shared_runtime_adaptation_xpert"
    elif model_kind == "cellot":
        model = CellOTSharedAdapterModel(
            baseline_dim=int(bundle.baseline_scaled.shape[1]),
            auxiliary_dim=int(bundle.auxiliary_scaled.shape[1]),
            hash_dim=int(bundle.intervention_hash.shape[1]),
            component_dim=int(bundle.component_bag.shape[1]),
            output_dim=int(dataset.output_dim),
            hidden_dim=hidden_dim,
            latent_dim=int(config.get("latent_dim", 128)),
            dropout=dropout,
        ).to(device)
        regularization_weight = float(config.get("transport_weight", 1e-3))
        regularization_key = "transport_cost"
        execution_mode = "in_framework_reimplementation_cellot"
        target_kind = "response"
        bundle = build_adaptation_features(dataset, train_idx_fit, hash_dim=hash_dim, target_kind=target_kind)
    elif model_kind == "transigen":
        model = TranSiGenSharedModel(
            baseline_dim=int(bundle.baseline_scaled.shape[1]),
            auxiliary_dim=int(bundle.auxiliary_scaled.shape[1]),
            hash_dim=int(bundle.intervention_hash.shape[1]),
            component_dim=int(bundle.component_bag.shape[1]),
            output_dim=int(dataset.output_dim),
            hidden_dim=hidden_dim,
            latent_dim=int(config.get("latent_dim", 128)),
            dropout=dropout,
        ).to(device)
        regularization_weight = float(config.get("latent_l2_weight", 1e-4))
        regularization_key = "latent_l2"
        execution_mode = "in_framework_reimplementation_transigen"
        target_kind = "response"
        bundle = build_adaptation_features(dataset, train_idx_fit, hash_dim=hash_dim, target_kind=target_kind)
    else:
        raise ValueError(f"Unsupported shared-runtime specialist model kind: {model_kind}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-5)),
    )
    batch_size = int(config.get("batch_size", 256))
    max_epochs = int(config.get("epochs", 30))
    patience = int(config.get("patience", 5))
    max_steps_per_epoch = config.get("max_steps_per_epoch")
    max_steps_per_epoch = None if max_steps_per_epoch in {None, "", 0} else int(max_steps_per_epoch)
    rng = np.random.default_rng(int(seed))

    best_state = None
    best_val_mse = float("inf")
    best_epoch = -1
    stale_epochs = 0
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(train_idx_fit.shape[0])
        epoch_losses: List[float] = []
        epoch_reg_losses: List[float] = []
        step_count = 0
        for start in range(0, train_idx_fit.shape[0], batch_size):
            batch_pick = train_idx_fit[order[start : start + batch_size]]
            tensors = _make_tensor_bundle(bundle, batch_pick, device)
            delta_pred, aux = model.forward(
                tensors["baseline"],
                tensors["auxiliary"],
                tensors["perturb_hash"],
                tensors["component_bag"],
            )
            mse_loss = F.mse_loss(delta_pred, tensors["target"])
            reg_loss = regularization_weight * aux[regularization_key]
            extra_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
            if model_kind == "cellot":
                extra_loss = float(config.get("mmd_weight", 1e-3)) * _gaussian_mmd(delta_pred, tensors["target"])
            elif model_kind == "transigen":
                baseline_recon_weight = float(config.get("baseline_recon_weight", 0.2))
                extra_loss = baseline_recon_weight * F.mse_loss(aux["baseline_recon"], tensors["baseline"])
            loss = mse_loss + reg_loss + extra_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(mse_loss.detach().cpu().item()))
            epoch_reg_losses.append(float((reg_loss + extra_loss).detach().cpu().item()))
            step_count += 1
            if max_steps_per_epoch is not None and step_count >= max_steps_per_epoch:
                break

        val_pred_scaled = _predict_model(model, bundle, val_idx_fit, device)
        val_pred = invert_standard_scaler(val_pred_scaled, bundle.target_mean, bundle.target_scale)
        if target_kind == "response":
            val_target = dataset.y_response[val_idx_fit].astype(np.float32)
        else:
            val_target = dataset.delta_response[val_idx_fit].astype(np.float32)
        val_mse = float(np.mean(np.square(val_target - val_pred)))
        train_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        train_reg = float(np.mean(epoch_reg_losses)) if epoch_reg_losses else 0.0
        log_fn(
            f"specialist={model_kind} epoch={epoch:03d} train_delta_mse={train_loss:.6f} "
            f"train_reg={train_reg:.6f} val_delta_mse={val_mse:.6f} device={device.type}"
        )
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            stale_epochs = 0
            best_state = model.state_dict()
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                log_fn(f"specialist={model_kind} early_stop epoch={epoch:03d} best_epoch={best_epoch:03d}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    split_indices = {"train": train_idx_full, "val": val_idx_full, "test": test_idx_full}
    predictions: Dict[str, Dict[str, np.ndarray]] = {}
    for split_name, indices in split_indices.items():
        delta_pred_scaled = _predict_model(model, bundle, indices, device)
        target_pred = invert_standard_scaler(delta_pred_scaled, bundle.target_mean, bundle.target_scale)
        if target_kind == "response":
            y_pred = target_pred.astype(np.float32)
            delta_pred = y_pred - dataset.x_baseline[indices].astype(np.float32)
        else:
            delta_pred = target_pred.astype(np.float32)
            y_pred = dataset.x_baseline[indices].astype(np.float32) + delta_pred.astype(np.float32)
        predictions[split_name] = {
            "delta_pred": delta_pred.astype(np.float32),
            "y_pred": y_pred.astype(np.float32),
        }

    return {
        "execution_mode": execution_mode,
        "feature_manifest": bundle.feature_manifest,
        "training_summary": {
            "best_val_mse": float(best_val_mse),
            "best_epoch": int(best_epoch),
            "device": str(device),
            "batch_size": batch_size,
            "epochs": max_epochs,
            "hash_dim": hash_dim,
            "max_steps_per_epoch": max_steps_per_epoch,
            "max_train_samples": max_train_samples,
            "max_val_samples": max_val_samples,
            "implementation_track": "shared_runtime_adaptation_target",
            "runtime_mode": "primary",
        },
        "predictions": predictions,
    }
