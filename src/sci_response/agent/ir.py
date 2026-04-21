from __future__ import annotations

import copy
from typing import Any, Dict, List, Sequence


def _normalize_hidden_dims(raw: Any, *, default: Sequence[int]) -> List[int]:
    if raw is None:
        return [int(item) for item in default]
    if isinstance(raw, int):
        return [int(raw)]
    if isinstance(raw, str):
        tokens = [item.strip() for item in raw.split(",") if item.strip()]
        return [int(item) for item in tokens]
    if isinstance(raw, Sequence):
        return [int(item) for item in raw]
    raise ValueError(f"Unsupported hidden_dims payload: {raw!r}")


def extract_model_ir(model_config: Dict[str, Any]) -> Dict[str, Any]:
    model_name = str(model_config["name"])
    if model_name == "mlp":
        hidden_dims = _normalize_hidden_dims(model_config.get("hidden_dims"), default=[64, 32])
        return {
            "ir_version": "structured_hypothesis_v1",
            "seed_model_name": model_name,
            "compiled_model_name": "structured_hypothesis",
            "representation": {
                "trunk": {
                    "hidden_dim": int(hidden_dims[0]),
                    "trunk_depth": int(len(hidden_dims)),
                    "residual_depth": 0,
                    "use_se_block": False,
                    "se_reduction": 4,
                },
                "conditioning": {
                    "mode": "concat",
                    "conditioning_dim": 0,
                },
                "prediction": {
                    "target": "delta",
                    "baseline_skip": True,
                    "zero_init_head": False,
                },
                "loss": {
                    "delta_weight": float(model_config.get("delta_loss_weight", 1.0)),
                    "response_weight": float(model_config.get("response_loss_weight", 1.0)),
                },
                "optimizer": {
                    "epochs": int(model_config.get("epochs", 80)),
                    "batch_size": int(model_config.get("batch_size", 256)),
                    "learning_rate": float(model_config.get("learning_rate", 1e-3)),
                    "weight_decay": float(model_config.get("weight_decay", 0.0)),
                    "patience": int(model_config.get("patience", 10)),
                    "max_grad_norm": float(model_config.get("max_grad_norm", 0.0)),
                    "use_lr_scheduler": bool(model_config.get("use_lr_scheduler", False)),
                    "scheduler_factor": float(model_config.get("scheduler_factor", 0.5)),
                    "scheduler_patience": int(model_config.get("scheduler_patience", 5)),
                    "scheduler_min_lr": float(model_config.get("scheduler_min_lr", 1e-5)),
                    "min_delta": float(model_config.get("min_delta", 0.0)),
                    "divergence_ratio": float(model_config.get("divergence_ratio", 0.0)),
                    "divergence_patience": int(model_config.get("divergence_patience", 0)),
                },
            },
        }
    if model_name == "conditioned_residual":
        return {
            "ir_version": "structured_hypothesis_v1",
            "seed_model_name": model_name,
            "compiled_model_name": "structured_hypothesis",
            "representation": {
                "trunk": {
                    "hidden_dim": int(model_config.get("hidden_dim", 64)),
                    "trunk_depth": 1,
                    "residual_depth": 1,
                    "use_se_block": False,
                    "se_reduction": 4,
                },
                "conditioning": {
                    "mode": "additive_gate",
                    "conditioning_dim": int(model_config.get("conditioning_dim", 32)),
                },
                "prediction": {
                    "target": "delta",
                    "baseline_skip": True,
                    "zero_init_head": False,
                },
                "loss": {
                    "delta_weight": float(model_config.get("delta_loss_weight", 1.0)),
                    "response_weight": float(model_config.get("response_loss_weight", 1.0)),
                },
                "optimizer": {
                    "epochs": int(model_config.get("epochs", 80)),
                    "batch_size": int(model_config.get("batch_size", 256)),
                    "learning_rate": float(model_config.get("learning_rate", 1e-3)),
                    "weight_decay": float(model_config.get("weight_decay", 0.0)),
                    "patience": int(model_config.get("patience", 10)),
                    "max_grad_norm": float(model_config.get("max_grad_norm", 0.0)),
                    "use_lr_scheduler": bool(model_config.get("use_lr_scheduler", False)),
                    "scheduler_factor": float(model_config.get("scheduler_factor", 0.5)),
                    "scheduler_patience": int(model_config.get("scheduler_patience", 5)),
                    "scheduler_min_lr": float(model_config.get("scheduler_min_lr", 1e-5)),
                    "min_delta": float(model_config.get("min_delta", 0.0)),
                    "divergence_ratio": float(model_config.get("divergence_ratio", 0.0)),
                    "divergence_patience": int(model_config.get("divergence_patience", 0)),
                },
            },
        }
    if model_name == "structured_hypothesis":
        if "model_ir" in model_config:
            return copy.deepcopy(model_config["model_ir"])
        return {
            "ir_version": "structured_hypothesis_v1",
            "seed_model_name": model_name,
            "compiled_model_name": "structured_hypothesis",
            "representation": {
                "trunk": {
                    "hidden_dim": int(model_config.get("hidden_dim", 64)),
                    "trunk_depth": int(model_config.get("trunk_depth", 1)),
                    "residual_depth": int(model_config.get("residual_depth", 1)),
                    "use_se_block": bool(model_config.get("use_se_block", False)),
                    "se_reduction": int(model_config.get("se_reduction", 4)),
                    "dropout": float(model_config.get("dropout", 0.0)),
                },
                "conditioning": {
                    "mode": str(model_config.get("conditioning_mode", "additive_gate")),
                    "conditioning_dim": int(model_config.get("conditioning_dim", 32)),
                },
                "prediction": {
                    "target": str(model_config.get("predict_target", "delta")),
                    "baseline_skip": bool(model_config.get("baseline_skip", True)),
                    "zero_init_head": bool(model_config.get("zero_init_head", False)),
                },
                "loss": {
                    "delta_weight": float(model_config.get("delta_loss_weight", 1.0)),
                    "response_weight": float(model_config.get("response_loss_weight", 1.0)),
                },
                "optimizer": {
                    "epochs": int(model_config.get("epochs", 80)),
                    "batch_size": int(model_config.get("batch_size", 256)),
                    "eval_batch_size": int(model_config.get("eval_batch_size", 0)),
                    "learning_rate": float(model_config.get("learning_rate", 1e-3)),
                    "weight_decay": float(model_config.get("weight_decay", 0.0)),
                    "patience": int(model_config.get("patience", 10)),
                    "max_grad_norm": float(model_config.get("max_grad_norm", 0.0)),
                    "use_lr_scheduler": bool(model_config.get("use_lr_scheduler", False)),
                    "scheduler_factor": float(model_config.get("scheduler_factor", 0.5)),
                    "scheduler_patience": int(model_config.get("scheduler_patience", 5)),
                    "scheduler_min_lr": float(model_config.get("scheduler_min_lr", 1e-5)),
                    "min_delta": float(model_config.get("min_delta", 0.0)),
                    "divergence_ratio": float(model_config.get("divergence_ratio", 0.0)),
                    "divergence_patience": int(model_config.get("divergence_patience", 0)),
                },
            },
        }
    raise ValueError(f"Unsupported seed model for IR extraction: {model_name}")


def enumerate_editable_sites(model_ir: Dict[str, Any]) -> List[Dict[str, Any]]:
    rep = dict(model_ir["representation"])
    sites: List[Dict[str, Any]] = []
    sites.extend(
        [
            {
                "path": "representation.trunk.hidden_dim",
                "primitive_family": ["set_scalar", "scale_numeric"],
                "type": "int",
                "current_value": rep["trunk"]["hidden_dim"],
            },
            {
                "path": "representation.trunk.trunk_depth",
                "primitive_family": ["set_scalar", "increment"],
                "type": "int",
                "current_value": rep["trunk"]["trunk_depth"],
            },
            {
                "path": "representation.trunk.residual_depth",
                "primitive_family": ["set_scalar", "increment"],
                "type": "int",
                "current_value": rep["trunk"]["residual_depth"],
            },
            {
                "path": "representation.trunk.use_se_block",
                "primitive_family": ["toggle_boolean"],
                "type": "bool",
                "current_value": rep["trunk"]["use_se_block"],
            },
            {
                "path": "representation.prediction.target",
                "primitive_family": ["cycle_enum"],
                "type": "enum",
                "choices": ["delta", "response"],
                "current_value": rep["prediction"]["target"],
            },
            {
                "path": "representation.prediction.baseline_skip",
                "primitive_family": ["toggle_boolean"],
                "type": "bool",
                "current_value": rep["prediction"]["baseline_skip"],
            },
            {
                "path": "representation.prediction.zero_init_head",
                "primitive_family": ["toggle_boolean"],
                "type": "bool",
                "current_value": rep["prediction"]["zero_init_head"],
            },
            {
                "path": "representation.conditioning.mode",
                "primitive_family": ["cycle_enum"],
                "type": "enum",
                "choices": ["concat", "additive_gate", "film"],
                "current_value": rep["conditioning"]["mode"],
            },
            {
                "path": "representation.conditioning.conditioning_dim",
                "primitive_family": ["set_scalar", "scale_numeric"],
                "type": "int",
                "current_value": rep["conditioning"]["conditioning_dim"],
            },
            {
                "path": "representation.loss.delta_weight",
                "primitive_family": ["set_scalar", "scale_numeric"],
                "type": "float",
                "current_value": rep["loss"]["delta_weight"],
            },
            {
                "path": "representation.loss.response_weight",
                "primitive_family": ["set_scalar", "scale_numeric"],
                "type": "float",
                "current_value": rep["loss"]["response_weight"],
            },
            {
                "path": "representation.optimizer.learning_rate",
                "primitive_family": ["set_scalar", "scale_numeric"],
                "type": "float",
                "current_value": rep["optimizer"]["learning_rate"],
            },
            {
                "path": "representation.optimizer.weight_decay",
                "primitive_family": ["set_scalar", "scale_numeric"],
                "type": "float",
                "current_value": rep["optimizer"]["weight_decay"],
            },
        ]
    )
    registered_paths = {str(item["path"]) for item in sites}

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                _walk(next_prefix, child)
            return
        if prefix in registered_paths:
            return
        if isinstance(value, bool):
            sites.append(
                {
                    "path": prefix,
                    "primitive_family": ["toggle_boolean"],
                    "type": "bool",
                    "current_value": value,
                }
            )
            registered_paths.add(prefix)
            return
        if isinstance(value, int) and not isinstance(value, bool):
            sites.append(
                {
                    "path": prefix,
                    "primitive_family": ["set_scalar", "increment", "scale_numeric"],
                    "type": "int",
                    "current_value": int(value),
                }
            )
            registered_paths.add(prefix)
            return
        if isinstance(value, float):
            sites.append(
                {
                    "path": prefix,
                    "primitive_family": ["set_scalar", "scale_numeric"],
                    "type": "float",
                    "current_value": float(value),
                }
            )
            registered_paths.add(prefix)

    _walk("representation", rep)
    return sites
