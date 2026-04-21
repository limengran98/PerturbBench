from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from sci_response.baselines.artifacts import beijing_timestamp_slug, build_manifest, utc_now, write_prediction_bundle
from sci_response.baselines.evaluate import evaluate_predictions
from sci_response.baselines.specialists import get_wrapper
from sci_response.baselines.specialists.base import collect_preflight_report, ensure_specialist_repo
from sci_response.baselines.universal import (
    apply_feature_scaler,
    apply_target_scaler,
    build_baseline,
    build_universal_features,
    fit_feature_scaler,
    fit_target_scaler,
    invert_target_scaler,
)
from sci_response.data.io import dump_json, dump_yaml, ensure_dir, load_json, load_yaml, resolve_path, stable_hash, write_text
from sci_response.data.registry import load_dataset_spec, load_prepared_dataset
from sci_response.data.schemas import DatasetSpec, PreparedDataset
from sci_response.data.splits import SplitSpec, indices_from_split, load_split
from sci_response.models.device import resolve_device
from sci_response.models.runtime import build_split_batch, fit_intervention_encoder, train_model
from sci_response.models.seed import set_global_seed


@dataclass(frozen=True)
class ResolvedBenchmarkConfig:
    config_path: Path
    run_name: str
    seed: int
    artifacts_root: Path
    dataset_config: Path
    split_path: Path
    split_summary_path: Path
    dataset_payload: Dict[str, Any]
    method: Dict[str, Any]
    metrics: Dict[str, Any]
    notes: Any
    runtime: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_name": self.run_name,
            "seed": self.seed,
            "artifacts_root": str(self.artifacts_root),
            "dataset_config": str(self.dataset_config),
            "split_path": str(self.split_path),
            "split_summary_path": str(self.split_summary_path),
            "dataset": self.dataset_payload,
            "method": self.method,
            "metrics": self.metrics,
            "notes": self.notes,
            "runtime": self.runtime,
        }


def resolve_benchmark_payload(payload: Dict[str, Any], config_path: Path) -> ResolvedBenchmarkConfig:
    dataset_config_path = resolve_path(str(payload["dataset_config"]), config_path.parent)
    split_path_value = payload.get("split_path")
    if split_path_value is None:
        raise ValueError(
            f"Unified benchmark interface requires explicit split_path in {config_path}"
        )
    split_path = resolve_path(str(split_path_value), config_path.parent)
    dataset_payload = load_yaml(dataset_config_path)

    if "baseline_name" in payload:
        default_baseline_root = (config_path.parents[2] / "baseline").resolve()
        method = {
            "family": "specialist_baseline",
            "name": str(payload["baseline_name"]),
            "spec": dict(payload.get("baseline", {})),
            "baseline_root": str(
                resolve_path(str(payload.get("baseline_root", default_baseline_root)), config_path.parent)
            ),
        }
        seed = int(payload.get("seed", dataset_payload.get("random_seed", 0)))
    elif "baseline" in payload:
        baseline_payload = dict(payload["baseline"])
        method = {
            "family": "universal_baseline",
            "name": str(baseline_payload["name"]),
            "spec": baseline_payload,
        }
        seed = int(payload["seed"])
    elif "model_config" in payload:
        model_config_path = resolve_path(str(payload["model_config"]), config_path.parent)
        model_payload = load_yaml(model_config_path)
        method = {
            "family": "trainable_model",
            "name": str(model_payload["name"]),
            "spec": model_payload,
            "model_config": str(model_config_path),
        }
        seed = int(payload["seed"])
    else:
        raise ValueError(
            f"Could not determine method family for config {config_path}; expected baseline, baseline_name, or model_config"
        )

    return ResolvedBenchmarkConfig(
        config_path=config_path,
        run_name=str(payload["run_name"]),
        seed=seed,
        artifacts_root=resolve_path(str(payload.get("artifacts_root", "artifacts")), config_path.parent),
        dataset_config=dataset_config_path,
        split_path=split_path,
        split_summary_path=split_path.parent / "split_summary.json",
        dataset_payload=dataset_payload,
        method=method,
        metrics=dict(payload.get("metrics", {"top_k": 20})),
        notes=payload.get("notes"),
        runtime={},
    )


def resolve_benchmark_config(config_path: Path) -> ResolvedBenchmarkConfig:
    payload = load_yaml(config_path)
    return resolve_benchmark_payload(payload, config_path)


def _method_supports_cuda(method: Dict[str, Any]) -> bool:
    family = str(method.get("family", ""))
    name = str(method.get("name", "")).lower()
    if family == "universal_baseline" and name == "catboost":
        return True
    if family == "specialist_baseline" and name in {"gperturb", "gears", "cpa", "xpert", "cellot", "transigen"}:
        return True
    if family == "trainable_model" and (name in {"structured_hypothesis", "generated_structured_hypothesis"} or name.startswith("generated_direct_code")):
        return True
    return False


def _load_dataset_inputs(resolved: ResolvedBenchmarkConfig) -> tuple[DatasetSpec, PreparedDataset, SplitSpec, Dict[str, Any] | None]:
    dataset_spec = load_dataset_spec(resolved.dataset_config)
    dataset = load_prepared_dataset(dataset_spec)
    split = load_split(resolved.split_path)
    if split.dataset_name != dataset.dataset_name:
        raise ValueError(f"Split dataset mismatch: {split.dataset_name} != {dataset.dataset_name}")
    split_summary = load_json(resolved.split_summary_path) if resolved.split_summary_path.exists() else None
    return dataset_spec, dataset, split, split_summary


def _prepare_run_dir(resolved: ResolvedBenchmarkConfig, run_id: Optional[str]) -> tuple[str, Path, list[str], Callable[[str], None], str]:
    start_time = utc_now()
    effective_run_id = run_id or f"{resolved.run_name}_{beijing_timestamp_slug()}"
    run_dir = ensure_dir(resolved.artifacts_root.resolve() / effective_run_id)
    log_lines: list[str] = []

    def log(message: Any) -> None:
        if isinstance(message, str):
            rendered = message
        else:
            try:
                rendered = json.dumps(message, ensure_ascii=True, sort_keys=True)
            except TypeError:
                rendered = repr(message)
        log_lines.append(rendered)
        print(f"[{effective_run_id}] {rendered}", flush=True)

    return effective_run_id, run_dir, log_lines, log, start_time


def _runtime_manifest_fields(resolved: ResolvedBenchmarkConfig) -> Dict[str, Any]:
    return {
        "runtime_mode": resolved.runtime.get("runtime_mode", "primary"),
        "runtime_execution_policy": resolved.runtime.get("runtime_execution_policy"),
        "runtime_implementation_track": resolved.runtime.get("runtime_implementation_track"),
        "runtime_primary_env_group": resolved.runtime.get("runtime_primary_env_group", "shared"),
        "runtime_fallback_env_group": resolved.runtime.get("runtime_fallback_env_group"),
        "runtime_env_group": resolved.runtime.get("runtime_env_group", "shared"),
        "runtime_env_config": resolved.runtime.get("runtime_env_config"),
        "runtime_python_executable": resolved.runtime.get("runtime_python_executable"),
    }


def _write_common_outputs(
    *,
    repo_root: Path,
    resolved: ResolvedBenchmarkConfig,
    run_dir: Path,
    log_lines: list[str],
    manifest: Dict[str, Any],
    metrics: Dict[str, Any],
) -> None:
    dump_yaml(run_dir / "resolved_config.yaml", resolved.to_dict())
    dump_json(run_dir / "manifest.json", manifest)
    dump_json(run_dir / "metrics.json", metrics)
    write_text(run_dir / "train.log", "\n".join(log_lines) + "\n")


def _run_universal_baseline(
    *,
    repo_root: Path,
    resolved: ResolvedBenchmarkConfig,
    dataset: PreparedDataset,
    split: SplitSpec,
    split_summary: Dict[str, Any] | None,
    run_dir: Path,
    log_lines: list[str],
    log: Callable[[str], None],
    start_time: str,
    device_ctx: Any,
) -> None:
    feature_cfg = dict(resolved.method["spec"].get("feature_builder", {}))
    feature_bundle = build_universal_features(
        dataset,
        intervention_hash_dim=int(feature_cfg.get("intervention_hash_dim", 128)),
    )
    split_indices = indices_from_split(dataset.sample_ids, split)
    x_train = feature_bundle.matrix[split_indices["train"]]
    x_val = feature_bundle.matrix[split_indices["val"]]
    x_test = feature_bundle.matrix[split_indices["test"]]

    y_train = dataset.delta_response[split_indices["train"]]
    y_val = dataset.delta_response[split_indices["val"]]
    y_test = dataset.delta_response[split_indices["test"]]

    x_mean, x_scale = fit_feature_scaler(x_train)
    y_mean, y_scale = fit_target_scaler(y_train)
    x_splits = {
        "train": apply_feature_scaler(x_train, x_mean, x_scale),
        "val": apply_feature_scaler(x_val, x_mean, x_scale),
        "test": apply_feature_scaler(x_test, x_mean, x_scale),
    }
    y_splits = {
        "train": apply_target_scaler(y_train, y_mean, y_scale),
        "val": apply_target_scaler(y_val, y_mean, y_scale),
        "test": apply_target_scaler(y_test, y_mean, y_scale),
    }

    baseline_name = str(resolved.method["name"])
    log(f"method_family=universal_baseline")
    log(f"method_name={baseline_name}")
    log(f"feature_builder={feature_cfg}")

    regressor = build_baseline(baseline_name)
    baseline_params = dict(resolved.method["spec"].get("params", {}))
    baseline_params.setdefault("_runtime_device", str(device_ctx.resolved_device))
    baseline_params.setdefault("_requested_device", str(device_ctx.requested_device))
    if device_ctx.cuda_visible_devices is not None:
        baseline_params.setdefault("_cuda_visible_devices", str(device_ctx.cuda_visible_devices))
    training_summary = regressor.fit(
        x_train=x_splits["train"],
        y_train=y_splits["train"],
        x_val=x_splits["val"],
        y_val=y_splits["val"],
        config=baseline_params,
        seed=int(resolved.seed),
        log_fn=log,
    )

    metrics: Dict[str, Any] = {
        "training": {
            **training_summary,
            "method_family": "universal_baseline",
            "method_name": baseline_name,
            "feature_dim": int(feature_bundle.matrix.shape[1]),
            "target_dim": int(dataset.output_dim),
        }
    }
    top_k = int(resolved.metrics.get("top_k", 20))
    test_prediction_dump: Dict[str, np.ndarray] = {}
    for split_name in ["train", "val", "test"]:
        delta_pred_scaled = regressor.predict(x_splits[split_name])
        delta_pred = invert_target_scaler(delta_pred_scaled, y_mean, y_scale)
        indices = split_indices[split_name]
        y_pred = dataset.x_baseline[indices] + delta_pred
        metrics[split_name] = evaluate_predictions(
            y_true=dataset.y_response[indices],
            delta_true=dataset.delta_response[indices],
            y_pred=y_pred,
            delta_pred=delta_pred,
            top_k=top_k,
        )
        if split_name == "test":
            test_prediction_dump = {
                "pred": y_pred.astype(np.float32),
                "target": dataset.y_response[indices].astype(np.float32),
                "delta_pred": delta_pred.astype(np.float32),
                "delta_target": dataset.delta_response[indices].astype(np.float32),
            }

    prediction_manifest = write_prediction_bundle(
        run_dir,
        sample_ids=dataset.sample_ids[split_indices["test"]],
        feature_names=dataset.feature_names,
        pred=test_prediction_dump["pred"],
        target=test_prediction_dump["target"],
        delta_pred=test_prediction_dump["delta_pred"],
        delta_target=test_prediction_dump["delta_target"],
        available=True,
        write_legacy_root=False,
    )

    end_time = utc_now()
    manifest = build_manifest(
        repo_root=repo_root,
        dataset_name=dataset.dataset_name,
        config_hash=stable_hash(resolved.to_dict()),
        split_path=resolved.split_path.resolve(),
        seed=int(resolved.seed),
        start_time=start_time,
        end_time=end_time,
        output_feature_dim=dataset.output_dim,
        baseline_mode=dataset.baseline_mode,
        device_record=device_ctx.to_manifest_fields(),
    )
    manifest.update(
        {
            "method_family": "universal_baseline",
            "method_name": baseline_name,
            "execution_status": "completed",
            "protocol": split.protocol,
            "group_fields": list(split.group_fields),
            "split_summary_path": str(resolved.split_summary_path) if resolved.split_summary_path.exists() else None,
            "required_zero_overlap_fields": (
                split_summary.get("required_zero_overlap_fields", []) if split_summary else []
            ),
            "required_zero_overlap_passed": (
                split_summary.get("required_zero_overlap_passed") if split_summary else None
            ),
            "prediction_bundle": prediction_manifest,
            "tuning_budget": resolved.method["spec"].get(
                "tuning_budget",
                {"definition": "fixed_default_v1", "num_trials": 1},
            ),
            "input_feature_dimension": int(feature_bundle.matrix.shape[1]),
            "feature_builder": feature_cfg,
            **_runtime_manifest_fields(resolved),
        }
    )
    _write_common_outputs(
        repo_root=repo_root,
        resolved=resolved,
        run_dir=run_dir,
        log_lines=log_lines,
        manifest=manifest,
        metrics=metrics,
    )


def _run_trainable_model(
    *,
    repo_root: Path,
    resolved: ResolvedBenchmarkConfig,
    dataset: PreparedDataset,
    split: SplitSpec,
    split_summary: Dict[str, Any] | None,
    run_dir: Path,
    log_lines: list[str],
    log: Callable[[str], None],
    start_time: str,
    device_ctx: Any,
) -> None:
    set_global_seed(int(resolved.seed))
    split_indices = indices_from_split(dataset.sample_ids, split)
    vocabulary = fit_intervention_encoder(dataset.intervention_ids[split_indices["train"]])
    data_splits = {
        split_name: build_split_batch(
            baseline=dataset.baseline,
            post=dataset.post,
            delta=dataset.delta,
            context=dataset.context,
            intervention_ids=dataset.intervention_ids,
            sample_ids=dataset.sample_ids,
            indices=indices,
            vocabulary=vocabulary,
        )
        for split_name, indices in split_indices.items()
    }

    model_name = str(resolved.method["name"])
    log("method_family=trainable_model")
    log(f"method_name={model_name}")
    log(
        f"split train={len(split.train_ids)} val={len(split.val_ids)} test={len(split.test_ids)} "
        f"interventions={len(vocabulary)}"
    )
    model, training_summary = train_model(
        model_config=resolved.method["spec"],
        data_splits=data_splits,
        seed=int(resolved.seed),
        log_fn=log,
        requested_device=str(device_ctx.resolved_device),
    )

    metrics: Dict[str, Any] = {
        "training": {
            **training_summary,
            "method_family": "trainable_model",
            "method_name": model_name,
            "target_dim": int(dataset.output_dim),
        }
    }
    top_k = int(resolved.metrics.get("top_k", 20))
    test_prediction_dump: Dict[str, np.ndarray] = {}
    for split_name in ["train", "val", "test"]:
        predictions = model.predict(data_splits[split_name])
        metrics[split_name] = evaluate_predictions(data_splits[split_name], predictions, top_k=top_k)
        if split_name == "test":
            test_prediction_dump = {
                "pred": predictions["post"].astype(np.float32),
                "target": data_splits[split_name]["post"].astype(np.float32),
                "delta_pred": predictions["delta"].astype(np.float32),
                "delta_target": data_splits[split_name]["delta"].astype(np.float32),
            }

    prediction_manifest = write_prediction_bundle(
        run_dir,
        sample_ids=dataset.sample_ids[split_indices["test"]],
        feature_names=dataset.feature_names,
        pred=test_prediction_dump["pred"],
        target=test_prediction_dump["target"],
        delta_pred=test_prediction_dump["delta_pred"],
        delta_target=test_prediction_dump["delta_target"],
        available=True,
        write_legacy_root=False,
    )

    end_time = utc_now()
    manifest = build_manifest(
        repo_root=repo_root,
        dataset_name=dataset.dataset_name,
        config_hash=stable_hash(resolved.to_dict()),
        split_path=resolved.split_path.resolve(),
        seed=int(resolved.seed),
        start_time=start_time,
        end_time=end_time,
        output_feature_dim=dataset.output_dim,
        baseline_mode=dataset.baseline_mode,
        device_record=device_ctx.to_manifest_fields(),
    )
    manifest.update(
        {
            "method_family": "trainable_model",
            "method_name": model_name,
            "execution_status": "completed",
            "protocol": split.protocol,
            "group_fields": list(split.group_fields),
            "split_summary_path": str(resolved.split_summary_path) if resolved.split_summary_path.exists() else None,
            "required_zero_overlap_fields": (
                split_summary.get("required_zero_overlap_fields", []) if split_summary else []
            ),
            "required_zero_overlap_passed": (
                split_summary.get("required_zero_overlap_passed") if split_summary else None
            ),
            "prediction_bundle": prediction_manifest,
            "model_config_path": resolved.method.get("model_config"),
            "top_k": top_k,
            **_runtime_manifest_fields(resolved),
        }
    )
    _write_common_outputs(
        repo_root=repo_root,
        resolved=resolved,
        run_dir=run_dir,
        log_lines=log_lines,
        manifest=manifest,
        metrics=metrics,
    )


def _run_specialist_baseline(
    *,
    repo_root: Path,
    resolved: ResolvedBenchmarkConfig,
    dataset: PreparedDataset,
    split: SplitSpec,
    split_summary: Dict[str, Any] | None,
    run_dir: Path,
    log_lines: list[str],
    log: Callable[[str], None],
    start_time: str,
    device_ctx: Any,
) -> None:
    wrapper_module = get_wrapper(str(resolved.method["name"]))
    wrapper = wrapper_module.SPEC
    baseline_root = Path(str(resolved.method["baseline_root"])).resolve()
    repo_state = ensure_specialist_repo(wrapper, baseline_root)
    preflight_report = collect_preflight_report(wrapper, baseline_root)
    native_input_manifest = None
    native_input_dir = run_dir / "native_input"
    if hasattr(wrapper_module, "export_native_inputs"):
        try:
            native_input_manifest = wrapper_module.export_native_inputs(dataset, split, native_input_dir)
        except Exception as exc:
            native_input_manifest = {
                "available": False,
                "directory": str(native_input_dir),
                "reason": f"native_input_export_failed: {type(exc).__name__}: {exc}",
            }
    split_indices = indices_from_split(dataset.sample_ids, split)

    specialist_result = None
    runtime_mode = str(resolved.runtime.get("runtime_mode", "primary"))
    execution_policy = str(resolved.runtime.get("runtime_execution_policy", "shared_runtime_first"))
    primary_shared_runtime = runtime_mode == "primary" and execution_policy in {"shared_only", "shared_runtime_first"}
    if primary_shared_runtime:
        can_execute_specialist = (
            hasattr(wrapper_module, "execute_specialist")
            and bool(preflight_report["status"]["repo_ready"])
        )
    else:
        can_execute_specialist = (
            hasattr(wrapper_module, "execute_specialist")
            and bool(preflight_report["status"]["repo_ready"])
            and bool(preflight_report["status"]["env_ready"])
            and bool(preflight_report["status"]["smoke_import_ready"])
        )
    if can_execute_specialist:
        try:
            specialist_result = wrapper_module.execute_specialist(
                dataset=dataset,
                split=split,
                baseline_root=baseline_root,
                config=dict(resolved.method["spec"]),
                seed=int(resolved.seed),
                requested_device=str(device_ctx.resolved_device),
                log_fn=log,
            )
        except Exception as exc:
            log(f"specialist_execution_failed={type(exc).__name__}: {exc}")
            specialist_result = None

    log("method_family=specialist_baseline")
    log(f"method_name={wrapper.name}")
    log(f"baseline_root={baseline_root}")
    log(
        f"repo_state={repo_state.get('status')} repo_ready={repo_state.get('repo_ready')} "
        f"extracted_this_run={repo_state.get('extracted_this_run')}"
    )
    log(
        f"preflight repo_ready={preflight_report['status']['repo_ready']} "
        f"env_ready={preflight_report['status']['env_ready']} "
        f"smoke_import_ready={preflight_report['status']['smoke_import_ready']} "
        f"config_parse_ready={preflight_report['status']['config_parse_ready']}"
    )
    if native_input_manifest is not None:
        log(
            f"native_input_export kind={native_input_manifest.get('native_export_kind')} "
            f"directory={native_input_manifest.get('directory')}"
        )
        if native_input_manifest.get("reason"):
            log(f"native_input_export_reason={native_input_manifest.get('reason')}")
    for blocker in preflight_report["blockers"]:
        log(f"blocker={blocker}")

    if specialist_result is not None:
        metrics: Dict[str, Any] = {
            "training": {
                **dict(specialist_result.get("training_summary", {})),
                "method_family": "specialist_baseline",
                "method_name": wrapper.name,
                "prediction_available": True,
            }
        }
        top_k = int(resolved.metrics.get("top_k", 20))
        test_prediction_dump: Dict[str, np.ndarray] = {}
        for split_name in ["train", "val", "test"]:
            split_prediction = specialist_result["predictions"][split_name]
            indices = split_indices[split_name]
            y_pred = np.asarray(split_prediction["y_pred"], dtype=np.float32)
            delta_pred = np.asarray(split_prediction["delta_pred"], dtype=np.float32)
            metrics[split_name] = evaluate_predictions(
                y_true=dataset.y_response[indices],
                delta_true=dataset.delta_response[indices],
                y_pred=y_pred,
                delta_pred=delta_pred,
                top_k=top_k,
            )
            if split_name == "test":
                test_prediction_dump = {
                    "pred": y_pred,
                    "target": dataset.y_response[indices].astype(np.float32),
                    "delta_pred": delta_pred,
                    "delta_target": dataset.delta_response[indices].astype(np.float32),
                }

        prediction_manifest = write_prediction_bundle(
            run_dir,
            sample_ids=dataset.sample_ids[split_indices["test"]],
            feature_names=dataset.feature_names,
            pred=test_prediction_dump["pred"],
            target=test_prediction_dump["target"],
            delta_pred=test_prediction_dump["delta_pred"],
            delta_target=test_prediction_dump["delta_target"],
            available=True,
            write_legacy_root=False,
        )
        end_time = utc_now()
        manifest = build_manifest(
            repo_root=repo_root,
            dataset_name=dataset.dataset_name,
            config_hash=stable_hash(resolved.to_dict()),
            split_path=resolved.split_path.resolve(),
            seed=int(resolved.seed),
            start_time=start_time,
            end_time=end_time,
            output_feature_dim=dataset.output_dim,
            baseline_mode=dataset.baseline_mode,
            device_record=device_ctx.to_manifest_fields(),
        )
        manifest.update(
            {
                "method_family": "specialist_baseline",
                "method_name": wrapper.name,
                "execution_status": "completed",
                "protocol": split.protocol,
                "group_fields": list(split.group_fields),
                "split_summary_path": str(resolved.split_summary_path) if resolved.split_summary_path.exists() else None,
                "required_zero_overlap_fields": (
                    split_summary.get("required_zero_overlap_fields", []) if split_summary else []
                ),
                "required_zero_overlap_passed": (
                    split_summary.get("required_zero_overlap_passed") if split_summary else None
                ),
                "prediction_bundle": prediction_manifest,
                "native_input_manifest": native_input_manifest,
                "specialist_preflight_summary": {
                    "asset_state": preflight_report["asset_state"],
                    "repo_ready": preflight_report["status"]["repo_ready"],
                    "env_ready": preflight_report["status"]["env_ready"],
                    "smoke_import_ready": preflight_report["status"]["smoke_import_ready"],
                    "config_parse_ready": preflight_report["status"]["config_parse_ready"],
                    "repo_extracted_this_run": bool(repo_state.get("extracted_this_run")),
                },
                "enhanced_information": bool(wrapper.enhanced_information),
                "best_fit_datasets": dict(wrapper.best_fit_datasets),
                "baseline_root": str(baseline_root),
                "specialist_execution_mode": specialist_result.get("execution_mode", "native_wrapper"),
                **_runtime_manifest_fields(resolved),
            }
        )
        _write_common_outputs(
            repo_root=repo_root,
            resolved=resolved,
            run_dir=run_dir,
            log_lines=log_lines,
            manifest=manifest,
            metrics=metrics,
        )
        return

    prediction_manifest = write_prediction_bundle(
        run_dir,
        sample_ids=dataset.sample_ids[split_indices["test"]],
        feature_names=dataset.feature_names,
        target=dataset.y_response[split_indices["test"]],
        delta_target=dataset.delta_response[split_indices["test"]],
        available=False,
        reason="specialist_execution_not_implemented_yet",
        write_legacy_root=False,
    )
    dump_json(run_dir / "preflight_report.json", preflight_report)

    metrics: Dict[str, Any] = {
        "training": {
            "status": "preflight_only",
            "method_family": "specialist_baseline",
            "method_name": wrapper.name,
            "repo_ready": bool(preflight_report["status"]["repo_ready"]),
            "env_ready": bool(preflight_report["status"]["env_ready"]),
            "smoke_import_ready": bool(preflight_report["status"]["smoke_import_ready"]),
            "config_parse_ready": bool(preflight_report["status"]["config_parse_ready"]),
            "prediction_available": False,
            "native_input_exported": native_input_manifest is not None,
        },
        "train": {"status": "not_executed"},
        "val": {"status": "not_executed"},
        "test": {"status": "not_executed"},
    }

    end_time = utc_now()
    manifest = build_manifest(
        repo_root=repo_root,
        dataset_name=dataset.dataset_name,
        config_hash=stable_hash(resolved.to_dict()),
        split_path=resolved.split_path.resolve(),
        seed=int(resolved.seed),
        start_time=start_time,
        end_time=end_time,
        output_feature_dim=dataset.output_dim,
        baseline_mode=dataset.baseline_mode,
        device_record=device_ctx.to_manifest_fields(),
    )
    manifest.update(
        {
            "method_family": "specialist_baseline",
            "method_name": wrapper.name,
            "execution_status": "preflight_only",
            "protocol": split.protocol,
            "group_fields": list(split.group_fields),
            "split_summary_path": str(resolved.split_summary_path) if resolved.split_summary_path.exists() else None,
            "required_zero_overlap_fields": (
                split_summary.get("required_zero_overlap_fields", []) if split_summary else []
            ),
            "required_zero_overlap_passed": (
                split_summary.get("required_zero_overlap_passed") if split_summary else None
            ),
            "prediction_bundle": prediction_manifest,
            "native_input_manifest": native_input_manifest,
            "specialist_preflight_summary": {
                "asset_state": preflight_report["asset_state"],
                "repo_ready": preflight_report["status"]["repo_ready"],
                "env_ready": preflight_report["status"]["env_ready"],
                "smoke_import_ready": preflight_report["status"]["smoke_import_ready"],
                "config_parse_ready": preflight_report["status"]["config_parse_ready"],
                "repo_extracted_this_run": bool(repo_state.get("extracted_this_run")),
            },
            "enhanced_information": bool(wrapper.enhanced_information),
            "best_fit_datasets": dict(wrapper.best_fit_datasets),
            "baseline_root": str(baseline_root),
            **_runtime_manifest_fields(resolved),
        }
    )
    _write_common_outputs(
        repo_root=repo_root,
        resolved=resolved,
        run_dir=run_dir,
        log_lines=log_lines,
        manifest=manifest,
        metrics=metrics,
    )


def run_benchmark(
    *,
    repo_root: Path,
    config_path: Optional[Path],
    inline_payload: Optional[Dict[str, Any]],
    run_id: Optional[str],
    requested_device: str,
    cuda_visible_devices: Optional[str],
) -> Path:
    if inline_payload is not None:
        synthetic_config_path = config_path or (repo_root / "configs" / "experiments" / "_inline_benchmark.yaml")
        resolved = resolve_benchmark_payload(inline_payload, synthetic_config_path)
    else:
        if config_path is None:
            raise ValueError("run_benchmark requires either config_path or inline_payload")
        resolved = resolve_benchmark_config(config_path)
    dataset_spec, dataset, split, split_summary = _load_dataset_inputs(resolved)
    effective_requested_device = str(requested_device)
    if str(resolved.method.get("family", "")) == "specialist_baseline":
        runtime_device_override = str(resolved.method.get("spec", {}).get("runtime_device_override", "")).strip()
        if runtime_device_override:
            effective_requested_device = runtime_device_override
    device_ctx = resolve_device(
        requested_device=effective_requested_device,
        cuda_visible_devices=cuda_visible_devices,
        model_supports_cuda=_method_supports_cuda(resolved.method),
    )
    effective_run_id, run_dir, log_lines, log, start_time = _prepare_run_dir(resolved, run_id)

    log(f"dataset_name={dataset.dataset_name}")
    log(f"dataset_config={resolved.dataset_config}")
    log(f"prepared_path={dataset_spec.prepared_path}")
    log(f"split_path={resolved.split_path}")
    log(
        f"runtime_mode={os.environ.get('SCI_RESPONSE_RUNTIME_MODE', 'primary')} "
        f"runtime_env_group={os.environ.get('SCI_RESPONSE_RUNTIME_ENV_GROUP', 'shared')} "
        f"runtime_execution_policy={os.environ.get('SCI_RESPONSE_RUNTIME_EXECUTION_POLICY')} "
        f"runtime_implementation_track={os.environ.get('SCI_RESPONSE_RUNTIME_IMPLEMENTATION_TRACK')}"
    )
    log(
        f"requested_device={device_ctx.requested_device} resolved_device={device_ctx.resolved_device} "
        f"cuda_visible_devices={device_ctx.cuda_visible_devices} "
        f"torch_cuda_available={device_ctx.torch_cuda_available} gpu_count={device_ctx.gpu_count} "
        f"gpu_name={device_ctx.gpu_name} model_uses_gpu={device_ctx.model_uses_gpu} "
        f"device_note={device_ctx.device_note}"
    )

    resolved = ResolvedBenchmarkConfig(
        config_path=resolved.config_path,
        run_name=resolved.run_name,
        seed=resolved.seed,
        artifacts_root=resolved.artifacts_root,
        dataset_config=resolved.dataset_config,
        split_path=resolved.split_path,
        split_summary_path=resolved.split_summary_path,
        dataset_payload=resolved.dataset_payload,
        method=resolved.method,
        metrics=resolved.metrics,
        notes=resolved.notes,
        runtime={
            "requested_device": str(requested_device),
            "effective_requested_device": str(effective_requested_device),
            "cuda_visible_devices": cuda_visible_devices,
            "resolved_device": device_ctx.resolved_device,
            "run_id": effective_run_id,
            "runtime_mode": os.environ.get("SCI_RESPONSE_RUNTIME_MODE", "primary"),
            "runtime_execution_policy": os.environ.get("SCI_RESPONSE_RUNTIME_EXECUTION_POLICY"),
            "runtime_implementation_track": os.environ.get("SCI_RESPONSE_RUNTIME_IMPLEMENTATION_TRACK"),
            "runtime_primary_env_group": os.environ.get("SCI_RESPONSE_RUNTIME_PRIMARY_ENV_GROUP", "shared"),
            "runtime_fallback_env_group": os.environ.get("SCI_RESPONSE_RUNTIME_FALLBACK_ENV_GROUP"),
            "runtime_env_group": os.environ.get("SCI_RESPONSE_RUNTIME_ENV_GROUP", "shared"),
            "runtime_env_config": os.environ.get("SCI_RESPONSE_RUNTIME_ENV_CONFIG"),
            "runtime_python_executable": os.environ.get("SCI_RESPONSE_RUNTIME_PYTHON", sys.executable),
        },
    )

    family = str(resolved.method["family"])
    if family == "universal_baseline":
        _run_universal_baseline(
            repo_root=repo_root,
            resolved=resolved,
            dataset=dataset,
            split=split,
            split_summary=split_summary,
            run_dir=run_dir,
            log_lines=log_lines,
            log=log,
            start_time=start_time,
            device_ctx=device_ctx,
        )
    elif family == "trainable_model":
        _run_trainable_model(
            repo_root=repo_root,
            resolved=resolved,
            dataset=dataset,
            split=split,
            split_summary=split_summary,
            run_dir=run_dir,
            log_lines=log_lines,
            log=log,
            start_time=start_time,
            device_ctx=device_ctx,
        )
    elif family == "specialist_baseline":
        _run_specialist_baseline(
            repo_root=repo_root,
            resolved=resolved,
            dataset=dataset,
            split=split,
            split_summary=split_summary,
            run_dir=run_dir,
            log_lines=log_lines,
            log=log,
            start_time=start_time,
            device_ctx=device_ctx,
        )
    else:
        raise ValueError(f"Unsupported benchmark family: {family}")

    return run_dir
