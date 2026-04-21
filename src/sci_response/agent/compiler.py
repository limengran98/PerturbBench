from __future__ import annotations

import copy
import pprint
from pathlib import Path
from typing import Any, Dict


def _compiled_fields(model_ir: Dict[str, Any]) -> Dict[str, Any]:
    rep = copy.deepcopy(dict(model_ir["representation"]))
    trunk = dict(rep["trunk"])
    conditioning = dict(rep["conditioning"])
    prediction = dict(rep["prediction"])
    loss = dict(rep["loss"])
    optimizer = dict(rep["optimizer"])
    return {
        "seed_model_name": str(model_ir["seed_model_name"]),
        "model_ir": copy.deepcopy(model_ir),
        "hidden_dim": int(trunk["hidden_dim"]),
        "trunk_depth": int(trunk["trunk_depth"]),
        "residual_depth": int(trunk["residual_depth"]),
        "use_se_block": bool(trunk["use_se_block"]),
        "se_reduction": int(trunk.get("se_reduction", 4)),
        "conditioning_mode": str(conditioning["mode"]),
        "conditioning_dim": int(conditioning.get("conditioning_dim", 0)),
        "predict_target": str(prediction["target"]),
        "baseline_skip": bool(prediction["baseline_skip"]),
        "zero_init_head": bool(prediction["zero_init_head"]),
        "response_loss_weight": float(loss["response_weight"]),
        "delta_loss_weight": float(loss["delta_weight"]),
        "epochs": int(optimizer["epochs"]),
        "batch_size": int(optimizer["batch_size"]),
        "eval_batch_size": int(optimizer.get("eval_batch_size", 0)),
        "learning_rate": float(optimizer["learning_rate"]),
        "weight_decay": float(optimizer["weight_decay"]),
        "patience": int(optimizer["patience"]),
        "max_grad_norm": float(optimizer.get("max_grad_norm", 0.0)),
        "use_lr_scheduler": bool(optimizer.get("use_lr_scheduler", False)),
        "scheduler_factor": float(optimizer.get("scheduler_factor", 0.5)),
        "scheduler_patience": int(optimizer.get("scheduler_patience", 5)),
        "scheduler_min_lr": float(optimizer.get("scheduler_min_lr", 1e-5)),
        "min_delta": float(optimizer.get("min_delta", 0.0)),
        "divergence_ratio": float(optimizer.get("divergence_ratio", 0.0)),
        "divergence_patience": int(optimizer.get("divergence_patience", 0)),
        "dropout": float(trunk.get("dropout", 0.0)),
    }


def compile_model_ir(
    model_ir: Dict[str, Any],
    *,
    generated_module_path: Path | None = None,
    generated_class_name: str = "GeneratedStructuredHypothesisRegressor",
) -> Dict[str, Any]:
    payload = {
        "name": "structured_hypothesis",
        **_compiled_fields(model_ir),
    }
    if generated_module_path is not None:
        payload["name"] = "generated_structured_hypothesis"
        payload["factory"] = {
            "module_path": str(generated_module_path.resolve()),
            "class_name": str(generated_class_name),
            "factory_type": "agent_structured_codegen_v1",
        }
    return payload


def render_generated_model_source(
    model_ir: Dict[str, Any],
    *,
    class_name: str = "GeneratedStructuredHypothesisRegressor",
) -> str:
    compiled = compile_model_ir(model_ir)
    compiled_literal = pprint.pformat(compiled, width=100, sort_dicts=True)
    return f"""from __future__ import annotations

from typing import Any, Dict

from sci_response.models.structured_hypothesis import StructuredHypothesisConfig, StructuredHypothesisRegressor

GENERATED_MODEL_CONFIG = {compiled_literal}


class {class_name}:
    \"\"\"Code-generated structured hypothesis model produced by the agent compiler.\"\"\"

    GENERATED_MODEL_CONFIG = GENERATED_MODEL_CONFIG
    GENERATED_MODEL_IR = GENERATED_MODEL_CONFIG["model_ir"]

    def __init__(
        self,
        *,
        baseline_dim: int,
        intervention_dim: int,
        context_dim: int,
        output_dim: int,
        config: Dict[str, Any],
        seed: int,
        requested_device: str = "cpu",
    ) -> None:
        self.inner = StructuredHypothesisRegressor(
            baseline_dim=baseline_dim,
            intervention_dim=intervention_dim,
            context_dim=context_dim,
            output_dim=output_dim,
            config=StructuredHypothesisConfig(
                hidden_dim=int(GENERATED_MODEL_CONFIG["hidden_dim"]),
                trunk_depth=int(GENERATED_MODEL_CONFIG["trunk_depth"]),
                residual_depth=int(GENERATED_MODEL_CONFIG["residual_depth"]),
                use_se_block=bool(GENERATED_MODEL_CONFIG["use_se_block"]),
                se_reduction=int(GENERATED_MODEL_CONFIG["se_reduction"]),
                conditioning_mode=str(GENERATED_MODEL_CONFIG["conditioning_mode"]),
                conditioning_dim=int(GENERATED_MODEL_CONFIG["conditioning_dim"]),
                predict_target=str(GENERATED_MODEL_CONFIG["predict_target"]),
                baseline_skip=bool(GENERATED_MODEL_CONFIG["baseline_skip"]),
                zero_init_head=bool(GENERATED_MODEL_CONFIG["zero_init_head"]),
                response_loss_weight=float(GENERATED_MODEL_CONFIG["response_loss_weight"]),
                delta_loss_weight=float(GENERATED_MODEL_CONFIG["delta_loss_weight"]),
                epochs=int(GENERATED_MODEL_CONFIG["epochs"]),
                batch_size=int(GENERATED_MODEL_CONFIG["batch_size"]),
                eval_batch_size=int(GENERATED_MODEL_CONFIG.get("eval_batch_size", 0)),
                learning_rate=float(GENERATED_MODEL_CONFIG["learning_rate"]),
                weight_decay=float(GENERATED_MODEL_CONFIG["weight_decay"]),
                patience=int(GENERATED_MODEL_CONFIG["patience"]),
                max_grad_norm=float(GENERATED_MODEL_CONFIG.get("max_grad_norm", 0.0)),
                use_lr_scheduler=bool(GENERATED_MODEL_CONFIG.get("use_lr_scheduler", False)),
                scheduler_factor=float(GENERATED_MODEL_CONFIG.get("scheduler_factor", 0.5)),
                scheduler_patience=int(GENERATED_MODEL_CONFIG.get("scheduler_patience", 5)),
                scheduler_min_lr=float(GENERATED_MODEL_CONFIG.get("scheduler_min_lr", 1e-5)),
                min_delta=float(GENERATED_MODEL_CONFIG.get("min_delta", 0.0)),
                divergence_ratio=float(GENERATED_MODEL_CONFIG.get("divergence_ratio", 0.0)),
                divergence_patience=int(GENERATED_MODEL_CONFIG.get("divergence_patience", 0)),
                dropout=float(GENERATED_MODEL_CONFIG.get("dropout", 0.0)),
            ),
            seed=int(seed),
            requested_device=requested_device,
        )

    def fit(self, train_batch: Dict[str, Any], val_batch: Dict[str, Any], config: Dict[str, Any], log_fn) -> Dict[str, Any]:
        return self.inner.fit(train_batch, val_batch, config, log_fn)

    def predict(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return self.inner.predict(batch)
"""
