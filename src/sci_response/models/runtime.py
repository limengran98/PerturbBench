from __future__ import annotations

import importlib.util
from typing import Any, Callable, Dict, Tuple

import numpy as np

from sci_response.models.conditioned_residual import ConditionedResidualRegressor
from sci_response.models.mlp import MLPRegressor
from sci_response.models.structured_hypothesis import StructuredHypothesisConfig, StructuredHypothesisRegressor


ArrayDict = Dict[str, np.ndarray]


def _load_generated_model_class(module_path: str, class_name: str):
    spec = importlib.util.spec_from_file_location(f"sci_response_generated_{abs(hash(module_path))}", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generated model module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise RuntimeError(f"Generated model class {class_name!r} not found in {module_path}") from exc


def fit_intervention_encoder(labels: np.ndarray) -> Dict[str, int]:
    vocabulary = sorted(np.unique(labels.astype(str)).tolist())
    return {label: idx for idx, label in enumerate(vocabulary)}


def encode_interventions(labels: np.ndarray, vocabulary: Dict[str, int]) -> np.ndarray:
    matrix = np.zeros((labels.shape[0], len(vocabulary)), dtype=np.float32)
    for row_idx, label in enumerate(labels.astype(str).tolist()):
        if label in vocabulary:
            matrix[row_idx, vocabulary[label]] = 1.0
    return matrix


def build_split_batch(
    baseline: np.ndarray,
    post: np.ndarray,
    delta: np.ndarray,
    context: np.ndarray,
    intervention_ids: np.ndarray,
    sample_ids: np.ndarray,
    indices: np.ndarray,
    vocabulary: Dict[str, int],
) -> ArrayDict:
    subset_interventions = intervention_ids[indices]
    return {
        "baseline": baseline[indices].astype(np.float32),
        "post": post[indices].astype(np.float32),
        "delta": delta[indices].astype(np.float32),
        "context": context[indices].astype(np.float32),
        "intervention_onehot": encode_interventions(subset_interventions, vocabulary),
        "intervention_ids": subset_interventions.astype(object),
        "sample_ids": sample_ids[indices].astype(object),
    }


def train_model(
    model_config: Dict[str, Any],
    data_splits: Dict[str, ArrayDict],
    seed: int,
    log_fn: Callable[[str], None],
    requested_device: str = "cpu",
) -> Tuple[object, Dict[str, Any]]:
    train_batch = data_splits["train"]
    val_batch = data_splits["val"]
    model_name = str(model_config["name"])
    baseline_dim = int(train_batch["baseline"].shape[1])
    intervention_dim = int(train_batch["intervention_onehot"].shape[1])
    context_dim = int(train_batch["context"].shape[1])
    output_dim = int(train_batch["post"].shape[1])

    if "factory" in model_config:
        factory_payload = dict(model_config["factory"])
        generated_cls = _load_generated_model_class(
            str(factory_payload["module_path"]),
            str(factory_payload["class_name"]),
        )
        model = generated_cls(
            baseline_dim=baseline_dim,
            intervention_dim=intervention_dim,
            context_dim=context_dim,
            output_dim=output_dim,
            config=model_config,
            seed=int(seed),
            requested_device=str(requested_device),
        )
        training_summary = model.fit(train_batch, val_batch, model_config, log_fn)
        training_summary["model_name"] = model_name
        training_summary["generated_factory_module_path"] = str(factory_payload["module_path"])
        training_summary["generated_factory_class_name"] = str(factory_payload["class_name"])
        return model, training_summary

    if model_name == "mlp":
        model = MLPRegressor(
            input_dim=baseline_dim + intervention_dim + context_dim,
            output_dim=output_dim,
            hidden_dims=[int(value) for value in model_config.get("hidden_dims", [64, 32])],
            seed=int(seed),
        )
    elif model_name == "conditioned_residual":
        model = ConditionedResidualRegressor(
            baseline_dim=baseline_dim,
            intervention_dim=intervention_dim,
            context_dim=context_dim,
            output_dim=output_dim,
            hidden_dim=int(model_config.get("hidden_dim", 64)),
            conditioning_dim=int(model_config.get("conditioning_dim", 32)),
            seed=int(seed),
        )
    elif model_name == "structured_hypothesis":
        model = StructuredHypothesisRegressor(
            baseline_dim=baseline_dim,
            intervention_dim=intervention_dim,
            context_dim=context_dim,
            output_dim=output_dim,
            config=StructuredHypothesisConfig(
                hidden_dim=int(model_config.get("hidden_dim", 64)),
                trunk_depth=int(model_config.get("trunk_depth", 2)),
                residual_depth=int(model_config.get("residual_depth", 0)),
                use_se_block=bool(model_config.get("use_se_block", False)),
                se_reduction=int(model_config.get("se_reduction", 4)),
                conditioning_mode=str(model_config.get("conditioning_mode", "concat")),
                conditioning_dim=int(model_config.get("conditioning_dim", 0)),
                predict_target=str(model_config.get("predict_target", "delta")),
                baseline_skip=bool(model_config.get("baseline_skip", True)),
                zero_init_head=bool(model_config.get("zero_init_head", False)),
                response_loss_weight=float(model_config.get("response_loss_weight", 1.0)),
                delta_loss_weight=float(model_config.get("delta_loss_weight", 1.0)),
                epochs=int(model_config.get("epochs", 80)),
                batch_size=int(model_config.get("batch_size", 256)),
                eval_batch_size=int(model_config.get("eval_batch_size", 0)),
                learning_rate=float(model_config.get("learning_rate", 1e-3)),
                weight_decay=float(model_config.get("weight_decay", 0.0)),
                patience=int(model_config.get("patience", 10)),
                max_grad_norm=float(model_config.get("max_grad_norm", 0.0)),
                use_lr_scheduler=bool(model_config.get("use_lr_scheduler", False)),
                scheduler_factor=float(model_config.get("scheduler_factor", 0.5)),
                scheduler_patience=int(model_config.get("scheduler_patience", 5)),
                scheduler_min_lr=float(model_config.get("scheduler_min_lr", 1e-5)),
                min_delta=float(model_config.get("min_delta", 0.0)),
                divergence_ratio=float(model_config.get("divergence_ratio", 0.0)),
                divergence_patience=int(model_config.get("divergence_patience", 0)),
                dropout=float(model_config.get("dropout", 0.0)),
            ),
            seed=int(seed),
            requested_device=str(requested_device),
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    training_summary = model.fit(train_batch, val_batch, model_config, log_fn)
    training_summary["model_name"] = model_name
    return model, training_summary
