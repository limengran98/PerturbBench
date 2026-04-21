from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

from sci_response.data.schemas import PreparedDataset


def _import_sklearn():
    try:
        from sklearn.linear_model import MultiTaskElasticNet, Ridge  # type: ignore
        from sklearn.multioutput import MultiOutputRegressor  # type: ignore
        from sklearn.neural_network import MLPRegressor  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("universal baselines require scikit-learn") from exc
    return Ridge, MultiTaskElasticNet, MultiOutputRegressor, MLPRegressor


def _import_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ResNet-like MLP and FT-Transformer baselines require torch") from exc
    return torch, nn, DataLoader, TensorDataset


def _stable_bucket(text: str, bucket_count: int) -> int:
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


def categorical_onehot(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    categories = sorted(np.unique(values.astype(str)).tolist())
    lookup = {value: idx for idx, value in enumerate(categories)}
    matrix = np.zeros((values.shape[0], len(categories)), dtype=np.float32)
    for row_idx, value in enumerate(values.astype(str).tolist()):
        matrix[row_idx, lookup[value]] = 1.0
    feature_names = np.asarray([f"intervention_type={value}" for value in categories], dtype=object)
    return matrix, feature_names


@dataclass(frozen=True)
class FeatureBundle:
    matrix: np.ndarray
    feature_names: np.ndarray


def build_universal_features(
    dataset: PreparedDataset,
    intervention_hash_dim: int = 128,
) -> FeatureBundle:
    parts: List[np.ndarray] = [dataset.x_baseline.astype(np.float32)]
    names: List[str] = [f"baseline::{name}" for name in dataset.feature_names.astype(str).tolist()]

    if dataset.context_matrix.shape[1] > 0:
        parts.append(dataset.context_matrix.astype(np.float32))
        names.extend([f"context::{name}" for name in dataset.context_feature_names.astype(str).tolist()])

    if intervention_hash_dim > 0:
        intervention_hashed = hashed_text_features(
            dataset.intervention_ids.astype(str),
            feature_dim=int(intervention_hash_dim),
        )
        parts.append(intervention_hashed)
        names.extend([f"intervention_hash::{idx:03d}" for idx in range(intervention_hashed.shape[1])])

    if dataset.intervention_types.shape[0] > 0:
        intervention_type_matrix, type_names = categorical_onehot(dataset.intervention_types.astype(str))
        parts.append(intervention_type_matrix.astype(np.float32))
        names.extend(type_names.astype(str).tolist())

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
    names.extend(["dose", "dose_missing", "time", "time_missing"])
    return FeatureBundle(matrix=np.hstack(parts).astype(np.float32), feature_names=np.asarray(names, dtype=object))


def fit_feature_scaler(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(matrix, axis=0, keepdims=True).astype(np.float32)
    scale = np.std(matrix, axis=0, keepdims=True).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def apply_feature_scaler(matrix: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((matrix - mean) / scale).astype(np.float32)


def fit_target_scaler(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(matrix, axis=0, keepdims=True).astype(np.float32)
    scale = np.std(matrix, axis=0, keepdims=True).astype(np.float32)
    scale[scale < 1e-6] = 1.0
    return mean, scale


def apply_target_scaler(matrix: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return ((matrix - mean) / scale).astype(np.float32)


def invert_target_scaler(matrix: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (matrix * scale + mean).astype(np.float32)


class BaseBaselineRegressor:
    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        config: Dict[str, Any],
        seed: int,
        log_fn: Callable[[str], None],
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class RidgeRegressor(BaseBaselineRegressor):
    def __init__(self) -> None:
        Ridge, _, _, _ = _import_sklearn()
        self._model = Ridge(alpha=1.0)

    def fit(self, x_train, y_train, x_val, y_val, config, seed, log_fn):
        Ridge, _, _, _ = _import_sklearn()
        self._model = Ridge(alpha=float(config.get("alpha", 1.0)))
        self._model.fit(x_train, y_train)
        val_pred = self.predict(x_val)
        val_mse = float(np.mean(np.square(y_val - val_pred)))
        log_fn(f"baseline=ridge alpha={float(config.get('alpha', 1.0))} val_mse={val_mse:.6f}")
        return {"best_val_mse": val_mse}

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(x), dtype=np.float32)


class ElasticNetRegressor(BaseBaselineRegressor):
    def __init__(self) -> None:
        _, MultiTaskElasticNet, _, _ = _import_sklearn()
        self._model = MultiTaskElasticNet(alpha=0.001, l1_ratio=0.5, random_state=0, max_iter=2000)

    def fit(self, x_train, y_train, x_val, y_val, config, seed, log_fn):
        _, MultiTaskElasticNet, _, _ = _import_sklearn()
        self._model = MultiTaskElasticNet(
            alpha=float(config.get("alpha", 0.001)),
            l1_ratio=float(config.get("l1_ratio", 0.5)),
            random_state=int(seed),
            max_iter=int(config.get("max_iter", 2000)),
        )
        self._model.fit(x_train, y_train)
        val_pred = self.predict(x_val)
        val_mse = float(np.mean(np.square(y_val - val_pred)))
        log_fn(
            "baseline=elasticnet "
            f"alpha={float(config.get('alpha', 0.001))} "
            f"l1_ratio={float(config.get('l1_ratio', 0.5))} val_mse={val_mse:.6f}"
        )
        return {"best_val_mse": val_mse}

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(x), dtype=np.float32)


class XGBoostRegressor(BaseBaselineRegressor):
    def __init__(self) -> None:
        self._model = None

    def fit(self, x_train, y_train, x_val, y_val, config, seed, log_fn):
        try:
            from xgboost import XGBRegressor  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "XGBoost baseline requested but xgboost is not installed. "
                "Install it or skip the xgboost config."
            ) from exc
        _, _, MultiOutputRegressor, _ = _import_sklearn()
        base = XGBRegressor(
            n_estimators=int(config.get("n_estimators", 64)),
            max_depth=int(config.get("max_depth", 4)),
            learning_rate=float(config.get("learning_rate", 0.05)),
            subsample=float(config.get("subsample", 0.8)),
            colsample_bytree=float(config.get("colsample_bytree", 0.8)),
            reg_lambda=float(config.get("reg_lambda", 1.0)),
            objective="reg:squarederror",
            random_state=int(seed),
            tree_method=str(config.get("tree_method", "hist")),
            n_jobs=int(config.get("n_jobs", 4)),
        )
        self._model = MultiOutputRegressor(base)
        self._model.fit(x_train, y_train)
        val_pred = self.predict(x_val)
        val_mse = float(np.mean(np.square(y_val - val_pred)))
        log_fn(f"baseline=xgboost val_mse={val_mse:.6f}")
        return {"best_val_mse": val_mse}

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("XGBoost model is not fitted")
        return np.asarray(self._model.predict(x), dtype=np.float32)


class CatBoostRegressorWrapper(BaseBaselineRegressor):
    def __init__(self) -> None:
        self._model = None

    def fit(self, x_train, y_train, x_val, y_val, config, seed, log_fn):
        try:
            from catboost import CatBoostRegressor  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "CatBoost baseline requested but catboost is not installed. "
                "Install it or skip the catboost config."
            ) from exc
        runtime_device = str(config.get("_runtime_device", "cpu"))
        catboost_kwargs = {
            "loss_function": "MultiRMSE",
            "iterations": int(config.get("iterations", 400)),
            "depth": int(config.get("depth", 6)),
            "learning_rate": float(config.get("learning_rate", 0.05)),
            "random_seed": int(seed),
            "verbose": False,
        }
        if runtime_device.startswith("cuda"):
            logical_index = 0
            if ":" in runtime_device:
                logical_index = int(runtime_device.split(":", 1)[1])
            catboost_kwargs["task_type"] = str(config.get("task_type", "GPU"))
            catboost_kwargs["devices"] = str(config.get("devices", logical_index))
        elif "task_type" in config:
            catboost_kwargs["task_type"] = str(config["task_type"])
        self._model = CatBoostRegressor(**catboost_kwargs)
        self._model.fit(x_train, y_train, eval_set=(x_val, y_val), verbose=False)
        val_pred = self.predict(x_val)
        val_mse = float(np.mean(np.square(y_val - val_pred)))
        log_fn(
            f"baseline=catboost val_mse={val_mse:.6f} "
            f"task_type={catboost_kwargs.get('task_type', 'CPU')} "
            f"devices={catboost_kwargs.get('devices', 'cpu')}"
        )
        return {
            "best_val_mse": val_mse,
            "task_type": catboost_kwargs.get("task_type", "CPU"),
            "devices": catboost_kwargs.get("devices", "cpu"),
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("CatBoost model is not fitted")
        return np.asarray(self._model.predict(x), dtype=np.float32)


def _parse_hidden_layers(config: Dict[str, Any], default: Tuple[int, ...]) -> Tuple[int, ...]:
    raw = config.get("hidden_layers")
    if raw is None:
        if "hidden_dim" in config and "depth" in config:
            return tuple([int(config["hidden_dim"])] * int(config["depth"]))
        return default
    if isinstance(raw, int):
        return (int(raw),)
    if isinstance(raw, str):
        cleaned = [item.strip() for item in raw.split(",") if item.strip()]
        if not cleaned:
            raise ValueError("hidden_layers string must not be empty")
        return tuple(int(item) for item in cleaned)
    if isinstance(raw, Sequence):
        values = [int(item) for item in raw]
        if not values:
            raise ValueError("hidden_layers sequence must not be empty")
        return tuple(values)
    raise ValueError(f"Unsupported hidden_layers config: {raw!r}")


class MLPRegressorWrapper(BaseBaselineRegressor):
    def __init__(self) -> None:
        self._model = None

    def fit(self, x_train, y_train, x_val, y_val, config, seed, log_fn):
        _, _, _, MLPRegressor = _import_sklearn()
        hidden_layers = _parse_hidden_layers(config, default=(256, 128))
        self._model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=str(config.get("activation", "relu")),
            solver="adam",
            alpha=float(config.get("alpha", 1e-4)),
            batch_size=int(config.get("batch_size", 256)),
            learning_rate_init=float(config.get("learning_rate", 1e-3)),
            max_iter=int(config.get("max_iter", 80)),
            random_state=int(seed),
            early_stopping=False,
            shuffle=True,
            verbose=False,
        )
        self._model.fit(x_train, y_train)
        val_pred = self.predict(x_val)
        val_mse = float(np.mean(np.square(y_val - val_pred)))
        loss_curve = getattr(self._model, "loss_curve_", [])
        if loss_curve:
            log_fn(
                f"baseline=mlp hidden_layers={hidden_layers} "
                f"epochs={len(loss_curve)} final_train_loss={float(loss_curve[-1]):.6f} val_mse={val_mse:.6f}"
            )
        else:
            log_fn(f"baseline=mlp hidden_layers={hidden_layers} val_mse={val_mse:.6f}")
        return {
            "best_val_mse": val_mse,
            "hidden_layers": list(hidden_layers),
            "epochs_trained": int(len(loss_curve)),
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("MLP model is not fitted")
        return np.asarray(self._model.predict(x), dtype=np.float32)


class ResidualMLPRegressor(BaseBaselineRegressor):
    def __init__(self) -> None:
        self._skip_model = None
        self._residual_model = None

    def fit(self, x_train, y_train, x_val, y_val, config, seed, log_fn):
        Ridge, _, _, MLPRegressor = _import_sklearn()
        hidden_layers = _parse_hidden_layers(config, default=(256, 256))
        self._skip_model = Ridge(alpha=float(config.get("skip_alpha", 1.0)))
        self._skip_model.fit(x_train, y_train)
        train_skip = np.asarray(self._skip_model.predict(x_train), dtype=np.float32)
        val_skip = np.asarray(self._skip_model.predict(x_val), dtype=np.float32)
        residual_train = y_train - train_skip
        self._residual_model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=str(config.get("activation", "relu")),
            solver="adam",
            alpha=float(config.get("alpha", 1e-4)),
            batch_size=int(config.get("batch_size", 256)),
            learning_rate_init=float(config.get("learning_rate", 1e-3)),
            max_iter=int(config.get("max_iter", 100)),
            random_state=int(seed),
            early_stopping=False,
            shuffle=True,
            verbose=False,
        )
        self._residual_model.fit(x_train, residual_train)
        val_pred = self.predict(x_val)
        val_mse = float(np.mean(np.square(y_val - val_pred)))
        loss_curve = getattr(self._residual_model, "loss_curve_", [])
        log_fn(
            f"baseline=resnet_mlp hidden_layers={hidden_layers} "
            f"epochs={len(loss_curve)} skip_alpha={float(config.get('skip_alpha', 1.0))} "
            f"val_mse={val_mse:.6f}"
        )
        skip_val_mse = float(np.mean(np.square(y_val - val_skip)))
        return {
            "best_val_mse": val_mse,
            "skip_only_val_mse": skip_val_mse,
            "hidden_layers": list(hidden_layers),
            "epochs_trained": int(len(loss_curve)),
        }

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._skip_model is None or self._residual_model is None:
            raise RuntimeError("Residual MLP model is not fitted")
        skip_pred = np.asarray(self._skip_model.predict(x), dtype=np.float32)
        residual_pred = np.asarray(self._residual_model.predict(x), dtype=np.float32)
        return np.asarray(skip_pred + residual_pred, dtype=np.float32)


class FTTransformerLiteRegressor(BaseBaselineRegressor):
    """
    CPU-only FT-inspired baseline.

    This is intentionally a lightweight surrogate rather than a faithful
    PyTorch implementation. It keeps the benchmark interface runnable in the
    current environment by applying a deterministic feature-tokenizer +
    attention-style pooling transform before a shared linear readout.
    """

    def __init__(self) -> None:
        self._readout = None
        self._tokenizer: Dict[str, np.ndarray] | None = None
        self._input_dim = 0

    def _init_projection(self, rng: np.random.Generator, in_dim: int, out_dim: int) -> np.ndarray:
        scale = np.sqrt(2.0 / max(in_dim, 1))
        return (rng.standard_normal((in_dim, out_dim)) * scale).astype(np.float32)

    def _fit_tokenizer(self, input_dim: int, config: Dict[str, Any], seed: int) -> None:
        rng = np.random.default_rng(int(seed))
        d_token = int(config.get("d_token", 64))
        ff_dim = int(config.get("ff_dim", max(2 * d_token, 128)))
        n_layers = int(config.get("n_layers", 2))
        self._input_dim = int(input_dim)
        tokenizer: Dict[str, np.ndarray] = {
            "token_scale": self._init_projection(rng, 1, d_token).reshape(1, d_token),
            "token_bias": np.zeros((self._input_dim, d_token), dtype=np.float32),
            "cls_token": np.zeros((1, d_token), dtype=np.float32),
        }
        for layer_idx in range(n_layers):
            tokenizer[f"Wq_{layer_idx}"] = self._init_projection(rng, d_token, d_token)
            tokenizer[f"Wk_{layer_idx}"] = self._init_projection(rng, d_token, d_token)
            tokenizer[f"Wv_{layer_idx}"] = self._init_projection(rng, d_token, d_token)
            tokenizer[f"Wo_{layer_idx}"] = self._init_projection(rng, d_token, d_token)
            tokenizer[f"Wff1_{layer_idx}"] = self._init_projection(rng, d_token, ff_dim)
            tokenizer[f"Wff2_{layer_idx}"] = self._init_projection(rng, ff_dim, d_token)
            tokenizer[f"bff1_{layer_idx}"] = np.zeros((1, ff_dim), dtype=np.float32)
            tokenizer[f"bff2_{layer_idx}"] = np.zeros((1, d_token), dtype=np.float32)
        self._tokenizer = tokenizer

    def _softmax(self, values: np.ndarray) -> np.ndarray:
        shifted = values - np.max(values, axis=1, keepdims=True)
        weights = np.exp(np.clip(shifted, -30.0, 30.0))
        normalizer = np.sum(weights, axis=1, keepdims=True)
        normalizer[normalizer < 1e-8] = 1.0
        return (weights / normalizer).astype(np.float32)

    def _transform(self, x: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
        if self._tokenizer is None:
            raise RuntimeError("FT-transformer-lite tokenizer is not initialized")
        if x.shape[1] != self._input_dim:
            raise ValueError(f"FT-transformer-lite expected input_dim={self._input_dim}, got {x.shape[1]}")

        x = x.astype(np.float32, copy=False)
        token_scale = self._tokenizer["token_scale"]
        token_bias = self._tokenizer["token_bias"]
        tokens = x[:, :, None] * token_scale[None, :, :] + token_bias[None, :, :]
        cls = np.repeat(self._tokenizer["cls_token"], repeats=x.shape[0], axis=0)
        last_attn = np.full((x.shape[0], x.shape[1]), 1.0 / max(x.shape[1], 1), dtype=np.float32)
        n_layers = int(config.get("n_layers", 2))
        for layer_idx in range(n_layers):
            q = cls @ self._tokenizer[f"Wq_{layer_idx}"]
            k = np.einsum("nfd,dh->nfh", tokens, self._tokenizer[f"Wk_{layer_idx}"]).astype(np.float32)
            v = np.einsum("nfd,dh->nfh", tokens, self._tokenizer[f"Wv_{layer_idx}"]).astype(np.float32)
            scores = np.einsum("nd,nfd->nf", q, k) / np.sqrt(max(q.shape[1], 1))
            last_attn = self._softmax(scores)
            pooled = np.einsum("nf,nfd->nd", last_attn, v).astype(np.float32)
            cls = np.tanh(cls + pooled @ self._tokenizer[f"Wo_{layer_idx}"]).astype(np.float32)
            ff = np.tanh(cls @ self._tokenizer[f"Wff1_{layer_idx}"] + self._tokenizer[f"bff1_{layer_idx}"])
            ff = ff @ self._tokenizer[f"Wff2_{layer_idx}"] + self._tokenizer[f"bff2_{layer_idx}"]
            cls = np.tanh(cls + ff).astype(np.float32)

        token_mean = np.mean(tokens, axis=1).astype(np.float32)
        attn_weighted = np.einsum("nf,nfd->nd", last_attn, tokens).astype(np.float32)
        return np.hstack([x, cls, token_mean, attn_weighted]).astype(np.float32)

    def fit(self, x_train, y_train, x_val, y_val, config, seed, log_fn):
        Ridge, _, _, _ = _import_sklearn()
        self._fit_tokenizer(input_dim=int(x_train.shape[1]), config=config, seed=int(seed))
        z_train = self._transform(x_train, config)
        z_val = self._transform(x_val, config)
        self._readout = Ridge(alpha=float(config.get("alpha", 1.0)))
        self._readout.fit(z_train, y_train)
        val_pred = self.predict(x_val, config=config)
        val_mse = float(np.mean(np.square(y_val - val_pred)))
        transformed_dim = int(z_train.shape[1])
        log_fn(
            "baseline=ft_transformer "
            f"variant=cpu_ft_transformer_lite "
            f"d_token={int(config.get('d_token', 64))} "
            f"n_layers={int(config.get('n_layers', 2))} "
            f"transformed_dim={transformed_dim} val_mse={val_mse:.6f}"
        )
        return {
            "best_val_mse": val_mse,
            "implementation_variant": "cpu_ft_transformer_lite",
            "d_token": int(config.get("d_token", 64)),
            "n_layers": int(config.get("n_layers", 2)),
            "transformed_dim": transformed_dim,
        }

    def predict(self, x: np.ndarray, config: Dict[str, Any] | None = None) -> np.ndarray:
        if self._readout is None:
            raise RuntimeError("FT-transformer-lite model is not fitted")
        z = self._transform(x, config or {})
        return np.asarray(self._readout.predict(z), dtype=np.float32)


class SkeletonBaselineRegressor(BaseBaselineRegressor):
    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason

    def fit(self, x_train, y_train, x_val, y_val, config, seed, log_fn):
        raise RuntimeError(f"{self.name} baseline is currently a skeleton: {self.reason}")

    def predict(self, x: np.ndarray) -> np.ndarray:
        raise RuntimeError(f"{self.name} baseline is currently a skeleton: {self.reason}")


def build_baseline(name: str) -> BaseBaselineRegressor:
    normalized = str(name).lower()
    if normalized == "ridge":
        return RidgeRegressor()
    if normalized == "elasticnet":
        return ElasticNetRegressor()
    if normalized == "xgboost":
        return XGBoostRegressor()
    if normalized == "catboost":
        return CatBoostRegressorWrapper()
    if normalized == "mlp":
        return MLPRegressorWrapper()
    if normalized == "resnet_mlp":
        return ResidualMLPRegressor()
    if normalized == "ft_transformer":
        return FTTransformerLiteRegressor()
    raise ValueError(f"Unsupported baseline: {name}")
