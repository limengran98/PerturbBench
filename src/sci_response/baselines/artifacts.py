from __future__ import annotations

import json
import platform
import socket
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from sci_response.data.io import dump_json, ensure_dir, write_text


BEIJING_TZ = timezone(timedelta(hours=8), name="Asia/Shanghai")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def beijing_now() -> str:
    return datetime.now(BEIJING_TZ).replace(microsecond=0).isoformat()


def beijing_timestamp_slug() -> str:
    return datetime.now(BEIJING_TZ).strftime("%Y%m%d_%H%M%S_BJT")


def git_commit_hash(repo_root: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = completed.stdout.strip()
    return commit or None


def device_info() -> Dict[str, Any]:
    return {
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def build_manifest(
    repo_root: Path,
    dataset_name: str,
    config_hash: str,
    split_path: Path,
    seed: int,
    start_time: str,
    end_time: str,
    output_feature_dim: int,
    baseline_mode: str,
    device_record: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    start_time_beijing = datetime.fromisoformat(start_time).astimezone(BEIJING_TZ).replace(microsecond=0).isoformat()
    end_time_beijing = datetime.fromisoformat(end_time).astimezone(BEIJING_TZ).replace(microsecond=0).isoformat()
    manifest = {
        "git_commit_hash": git_commit_hash(repo_root),
        "dataset_name": dataset_name,
        "config_hash": config_hash,
        "split_file_path": str(split_path),
        "seed": int(seed),
        "start_time": start_time,
        "end_time": end_time,
        "start_time_beijing": start_time_beijing,
        "end_time_beijing": end_time_beijing,
        "hostname": socket.gethostname(),
        "device_info": device_info(),
        "output_feature_dimension": int(output_feature_dim),
        "baseline_mode": baseline_mode,
    }
    if device_record:
        manifest.update(device_record)
    return manifest


def write_prediction_bundle(
    run_dir: Path,
    *,
    sample_ids: np.ndarray,
    feature_names: np.ndarray,
    target: np.ndarray,
    delta_target: np.ndarray | None = None,
    pred: np.ndarray | None = None,
    delta_pred: np.ndarray | None = None,
    available: bool,
    reason: Optional[str] = None,
    write_legacy_root: bool = False,
) -> Dict[str, Any]:
    prediction_dir = ensure_dir(run_dir / "predictions")
    files: Dict[str, Optional[str]] = {
        "pred_test": None,
        "target_test": "target_test.npy",
        "delta_pred_test": None,
        "delta_target_test": "delta_target_test.npy" if delta_target is not None else None,
        "sample_ids_test": "sample_ids_test.json",
        "feature_names": "feature_names.json",
    }

    np.save(prediction_dir / "target_test.npy", target.astype(np.float32), allow_pickle=False)
    if delta_target is not None:
        np.save(prediction_dir / "delta_target_test.npy", delta_target.astype(np.float32), allow_pickle=False)
    write_text(
        prediction_dir / "sample_ids_test.json",
        json.dumps(sample_ids.astype(str).tolist(), indent=2, ensure_ascii=True) + "\n",
    )
    write_text(
        prediction_dir / "feature_names.json",
        json.dumps(feature_names.astype(str).tolist(), indent=2, ensure_ascii=True) + "\n",
    )

    if pred is not None:
        np.save(prediction_dir / "pred_test.npy", pred.astype(np.float32), allow_pickle=False)
        files["pred_test"] = "pred_test.npy"
    if delta_pred is not None:
        np.save(prediction_dir / "delta_pred_test.npy", delta_pred.astype(np.float32), allow_pickle=False)
        files["delta_pred_test"] = "delta_pred_test.npy"

    if write_legacy_root and available:
        if pred is not None:
            np.save(run_dir / "pred_test.npy", pred.astype(np.float32), allow_pickle=False)
        np.save(run_dir / "target_test.npy", target.astype(np.float32), allow_pickle=False)
        if delta_pred is not None:
            np.save(run_dir / "delta_pred_test.npy", delta_pred.astype(np.float32), allow_pickle=False)
        if delta_target is not None:
            np.save(run_dir / "delta_target_test.npy", delta_target.astype(np.float32), allow_pickle=False)
        write_text(
            run_dir / "sample_ids_test.json",
            json.dumps(sample_ids.astype(str).tolist(), indent=2, ensure_ascii=True) + "\n",
        )
        write_text(
            run_dir / "feature_names.json",
            json.dumps(feature_names.astype(str).tolist(), indent=2, ensure_ascii=True) + "\n",
        )

    manifest = {
        "available": bool(available),
        "reason": reason,
        "directory": str(prediction_dir),
        "files": files,
        "legacy_root_files_written": bool(write_legacy_root and available),
    }
    dump_json(prediction_dir / "manifest.json", manifest)
    return manifest
