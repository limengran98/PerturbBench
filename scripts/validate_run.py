#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sci_response.data.splits import load_split
from sci_response.data.io import load_json, load_yaml


REQUIRED_ARTIFACT_FILES = (
    "manifest.json",
    "metrics.json",
    "resolved_config.yaml",
    "train.log",
)

REQUIRED_PREDICTION_FILES = (
    "target_test.npy",
    "sample_ids_test.json",
    "feature_names.json",
)

REQUIRED_METRIC_FIELDS = ("mse", "mae", "pearson", "spearman", "r2")
REQUIRED_MANIFEST_FIELDS = (
    "dataset_name",
    "split_file_path",
    "seed",
    "requested_device",
    "resolved_device",
    "cuda_visible_devices",
    "torch_cuda_available",
    "gpu_name",
    "gpu_count",
)


def _check_metrics(metrics: Dict[str, Any], execution_status: str) -> List[str]:
    errors: List[str] = []
    if execution_status == "preflight_only":
        training = metrics.get("training", {})
        if training.get("status") != "preflight_only":
            errors.append("metrics[training].status must be preflight_only for specialist preflight runs")
        for split_name in ("train", "val", "test"):
            if metrics.get(split_name, {}).get("status") != "not_executed":
                errors.append(f"metrics[{split_name}].status must be not_executed for preflight-only runs")
        return errors

    for split_name in ("train", "val", "test"):
        if split_name not in metrics:
            errors.append(f"metrics missing split section: {split_name}")
            continue
        split_payload = metrics[split_name]
        for target_name in ("response", "delta"):
            if target_name not in split_payload:
                errors.append(f"metrics[{split_name}] missing subsection: {target_name}")
                continue
            target_metrics = split_payload[target_name]
            for field_name in REQUIRED_METRIC_FIELDS:
                if field_name not in target_metrics:
                    errors.append(
                        f"metrics[{split_name}][{target_name}] missing field: {field_name}"
                    )
    return errors


def validate_run(run_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    errors: List[str] = []
    warnings: List[str] = []

    if not run_dir.exists():
        raise FileNotFoundError(f"Artifact directory not found: {run_dir}")

    existing_files = {path.name for path in run_dir.iterdir() if path.is_file()}
    for file_name in REQUIRED_ARTIFACT_FILES:
        if file_name not in existing_files:
            errors.append(f"Missing required artifact file: {file_name}")

    if errors:
        return False, {"artifact_path": str(run_dir), "errors": errors, "warnings": warnings}

    manifest = load_json(run_dir / "manifest.json")
    metrics = load_json(run_dir / "metrics.json")
    resolved_config = load_yaml(run_dir / "resolved_config.yaml")
    execution_status = str(manifest.get("execution_status", "completed"))
    prediction_dir = run_dir / "predictions"
    prediction_manifest_path = prediction_dir / "manifest.json"

    for field_name in REQUIRED_MANIFEST_FIELDS:
        if field_name not in manifest:
            errors.append(f"manifest missing required field: {field_name}")

    if "method_family" not in manifest:
        errors.append("manifest missing required field: method_family")
    if "method_name" not in manifest:
        errors.append("manifest missing required field: method_name")

    if not prediction_manifest_path.exists():
        errors.append(f"Missing unified prediction manifest: {prediction_manifest_path}")
        prediction_manifest = None
    else:
        prediction_manifest = load_json(prediction_manifest_path)
        for file_name in REQUIRED_PREDICTION_FILES:
            if prediction_manifest.get("files", {}).get(file_name.replace(".npy", "").replace(".json", "")) is None:
                errors.append(f"prediction manifest missing required file entry: {file_name}")
            if not (prediction_dir / file_name).exists():
                errors.append(f"Missing unified prediction file: {prediction_dir / file_name}")

    split_path = Path(str(manifest.get("split_file_path", "")))
    if not split_path.exists():
        errors.append(f"manifest split_file_path does not exist: {split_path}")
        split = None
    else:
        split = load_split(split_path)
        if split.dataset_name != manifest.get("dataset_name"):
            errors.append(
                f"manifest dataset_name mismatch: manifest={manifest.get('dataset_name')} split={split.dataset_name}"
            )

    resolved_split_path = Path(str(resolved_config.get("split_path", ""))).resolve()
    if split_path and str(split_path.resolve()) != str(resolved_split_path):
        errors.append(
            f"resolved_config split_path mismatch: manifest={split_path.resolve()} resolved_config={resolved_split_path}"
        )

    if int(manifest.get("seed", -1)) != int(resolved_config.get("seed", -2)):
        errors.append(
            f"seed mismatch: manifest={manifest.get('seed')} resolved_config={resolved_config.get('seed')}"
        )

    if not manifest.get("dataset_name"):
        errors.append("manifest dataset_name is empty")

    if not manifest.get("method_name"):
        errors.append("run is missing method_name information")

    errors.extend(_check_metrics(metrics, execution_status))

    pred = None
    target = None
    sample_ids: List[str] = []
    feature_names: List[str] = []
    if prediction_manifest is not None:
        target = np.load(prediction_dir / "target_test.npy", allow_pickle=False)
        sample_ids = json.loads((prediction_dir / "sample_ids_test.json").read_text(encoding="utf-8"))
        feature_names = json.loads((prediction_dir / "feature_names.json").read_text(encoding="utf-8"))
        pred_path = prediction_dir / "pred_test.npy"
        if pred_path.exists():
            pred = np.load(pred_path, allow_pickle=False)

        if target.shape[0] != len(sample_ids):
            errors.append(
                f"sample_ids_test.json length {len(sample_ids)} does not match target_test rows {target.shape[0]}"
            )
        if target.ndim == 2 and len(feature_names) != target.shape[1]:
            errors.append(
                f"feature_names.json length {len(feature_names)} does not match target width {target.shape[1]}"
            )
        if bool(prediction_manifest.get("available")):
            if pred is None:
                errors.append("prediction manifest marks available=true but pred_test.npy is missing")
            elif pred.shape != target.shape:
                errors.append(f"pred_test.npy shape {pred.shape} != target_test.npy shape {target.shape}")
        elif pred is not None:
            warnings.append("pred_test.npy exists even though prediction manifest marks available=false")

    root_feature_names_path = run_dir / "feature_names.json"
    if root_feature_names_path.exists() and pred is not None:
        root_feature_names = json.loads(root_feature_names_path.read_text(encoding="utf-8"))
        if pred.ndim == 2 and len(root_feature_names) != pred.shape[1]:
            errors.append(
                f"root feature_names.json length {len(root_feature_names)} does not match prediction width {pred.shape[1]}"
            )

    summary = {
        "artifact_path": str(run_dir),
        "dataset_name": manifest.get("dataset_name"),
        "method_family": manifest.get("method_family"),
        "method_name": manifest.get("method_name"),
        "seed": manifest.get("seed"),
        "requested_device": manifest.get("requested_device"),
        "resolved_device": manifest.get("resolved_device"),
        "cuda_visible_devices": manifest.get("cuda_visible_devices"),
        "torch_cuda_available": manifest.get("torch_cuda_available"),
        "gpu_name": manifest.get("gpu_name"),
        "gpu_count": manifest.get("gpu_count"),
        "execution_status": execution_status,
        "split_file_path": str(split_path) if split_path else None,
        "test_prediction_shape": list(target.shape) if target is not None else None,
        "required_files_checked": list(REQUIRED_ARTIFACT_FILES),
        "errors": errors,
        "warnings": warnings,
    }
    return len(errors) == 0, summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate a baseline artifact directory.")
    parser.add_argument("--artifact-path", required=True, help="Path to artifacts/<run_id>/")
    args = parser.parse_args()

    artifact_path = Path(args.artifact_path).resolve()
    passed, summary = validate_run(artifact_path)

    print(f"artifact_path: {summary['artifact_path']}")
    print(f"dataset_name: {summary.get('dataset_name')}")
    print(f"method_family: {summary.get('method_family')}")
    print(f"method_name: {summary.get('method_name')}")
    print(f"seed: {summary.get('seed')}")
    print(f"execution_status: {summary.get('execution_status')}")
    print(
        "device: "
        f"requested={summary.get('requested_device')} resolved={summary.get('resolved_device')} "
        f"cuda_visible_devices={summary.get('cuda_visible_devices')} "
        f"torch_cuda_available={summary.get('torch_cuda_available')} "
        f"gpu_count={summary.get('gpu_count')} gpu_name={summary.get('gpu_name')}"
    )
    print(f"split_file_path: {summary.get('split_file_path')}")
    print(f"test_prediction_shape: {summary.get('test_prediction_shape')}")

    if summary["warnings"]:
        print("warnings:")
        for item in summary["warnings"]:
            print(f"  - {item}")

    if summary["errors"]:
        print("errors:")
        for item in summary["errors"]:
            print(f"  - {item}")
        raise SystemExit(1)

    print("validation_status: PASS")


if __name__ == "__main__":
    main()
