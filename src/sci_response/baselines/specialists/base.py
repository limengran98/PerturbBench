from __future__ import annotations

import csv
import hashlib
import importlib
import json
import os
import platform
import pickle
import subprocess
import sys
import textwrap
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import yaml

from sci_response.baselines.artifacts import utc_now
from sci_response.data.io import dump_json, dump_yaml, ensure_dir, write_text
from sci_response.data.schemas import PreparedDataset
from sci_response.data.splits import SplitSpec, indices_from_split


@dataclass(frozen=True)
class SpecialistBaselineSpec:
    name: str
    slug: str
    archive_name: str
    repo_dir_name: str
    version_hint: str
    upstream_hint: Optional[str]
    source_classification: str
    source_note: str
    license_name: str
    key_files: Sequence[str]
    entrypoints: Sequence[str]
    dependency_files: Sequence[str]
    critical_packages: Sequence[str]
    smoke_import_modules: Sequence[str]
    help_script: Optional[str]
    config_files: Sequence[str]
    python_requirement: str
    torch_requirement: str
    cuda_requirement: str
    native_input_format: str
    native_output_format: str
    extra_priors: Sequence[str]
    enhanced_information: bool
    enhanced_information_note: str
    best_fit_datasets: Mapping[str, str]
    best_fit_notes: Mapping[str, str]
    current_blockers: Sequence[str] = field(default_factory=tuple)

    def archive_path(self, baseline_root: Path) -> Path:
        return baseline_root / self.archive_name

    def repo_dir_path(self, baseline_root: Path) -> Path:
        return baseline_root / self.repo_dir_name

    def member_path(self, relative_path: str) -> str:
        return f"{self.repo_dir_name}/{relative_path}"


def ensure_specialist_repo(spec: SpecialistBaselineSpec, baseline_root: Path) -> Dict[str, Any]:
    baseline_root = baseline_root.resolve()
    archive_path = spec.archive_path(baseline_root)
    repo_dir = spec.repo_dir_path(baseline_root)
    if repo_dir.exists() and repo_dir.is_dir():
        return {
            "repo_dir": str(repo_dir),
            "archive_path": str(archive_path) if archive_path.exists() else None,
            "repo_ready": True,
            "extracted_this_run": False,
            "status": "repo_dir",
        }
    if not archive_path.exists():
        return {
            "repo_dir": None,
            "archive_path": None,
            "repo_ready": False,
            "extracted_this_run": False,
            "status": "missing",
            "reason": "archive_missing",
        }
    with zipfile.ZipFile(archive_path) as handle:
        handle.extractall(baseline_root)
    repo_ready = repo_dir.exists() and repo_dir.is_dir()
    return {
        "repo_dir": str(repo_dir) if repo_ready else None,
        "archive_path": str(archive_path),
        "repo_ready": repo_ready,
        "extracted_this_run": True,
        "status": "repo_dir" if repo_ready else "extract_failed",
    }


def _sha256_prefix(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _load_archive_members(archive_path: Optional[Path]) -> set[str]:
    if archive_path is None or not archive_path.exists():
        return set()
    with zipfile.ZipFile(archive_path) as handle:
        return set(handle.namelist())


def _read_text_from_archive(archive_path: Optional[Path], member_path: str) -> Optional[str]:
    if archive_path is None or not archive_path.exists():
        return None
    with zipfile.ZipFile(archive_path) as handle:
        try:
            payload = handle.read(member_path)
        except KeyError:
            return None
    return payload.decode("utf-8", errors="replace")


def _read_text_from_asset(
    spec: SpecialistBaselineSpec,
    repo_dir: Optional[Path],
    archive_path: Optional[Path],
    relative_path: str,
) -> Optional[str]:
    if repo_dir is not None:
        candidate = repo_dir / relative_path
        if candidate.exists():
            return candidate.read_text(encoding="utf-8", errors="replace")
    return _read_text_from_archive(archive_path, spec.member_path(relative_path))


def _file_presence(
    spec: SpecialistBaselineSpec,
    repo_dir: Optional[Path],
    archive_members: set[str],
    relative_paths: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for relative_path in relative_paths:
        repo_present = bool(repo_dir and (repo_dir / relative_path).exists())
        archive_present = spec.member_path(relative_path) in archive_members
        result[relative_path] = {
            "repo_present": repo_present,
            "archive_present": archive_present,
            "present": repo_present or archive_present,
        }
    return result


def _git_info(repo_dir: Optional[Path]) -> Dict[str, Optional[str]]:
    if repo_dir is None or not repo_dir.exists() or not (repo_dir / ".git").exists():
        return {
            "remote_url": None,
            "branch": None,
            "commit_hash": None,
            "available": False,
        }

    def _run_git(args: Sequence[str]) -> Optional[str]:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repo_dir), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
        except Exception:
            return None
        value = completed.stdout.strip()
        return value or None

    return {
        "remote_url": _run_git(["remote", "get-url", "origin"]),
        "branch": _run_git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "commit_hash": _run_git(["rev-parse", "HEAD"]),
        "available": True,
    }


def _probe_dependency_imports(packages: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    for package_name in packages:
        try:
            module = importlib.import_module(package_name)
            results[package_name] = {
                "ok": True,
                "version": getattr(module, "__version__", None),
                "error": None,
            }
        except Exception as exc:  # pragma: no cover - environment dependent
            results[package_name] = {
                "ok": False,
                "version": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
    return results


def _probe_import_modules(repo_dir: Optional[Path], modules: Sequence[str]) -> Dict[str, Any]:
    if not modules:
        return {"status": "not_applicable", "reason": "no_import_modules_declared", "details": {}}
    if repo_dir is None or not repo_dir.exists():
        return {"status": "unavailable", "reason": "repo_not_extracted", "details": {}}

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_dir) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    code = textwrap.dedent(
        """
        import importlib
        import json
        modules = json.loads(%r)
        results = {}
        all_ok = True
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
                results[module_name] = {
                    "ok": True,
                    "version": getattr(module, "__version__", None),
                    "error": None,
                }
            except Exception as exc:
                all_ok = False
                results[module_name] = {
                    "ok": False,
                    "version": None,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        print(json.dumps({"all_ok": all_ok, "results": results}))
        """
        % json.dumps(list(modules))
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(repo_dir),
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # pragma: no cover - subprocess failure path
        return {"status": "failed", "reason": f"subprocess_error: {exc}", "details": {}}

    stdout = completed.stdout.strip()
    if not stdout:
        return {
            "status": "failed",
            "reason": f"empty_stdout exit_code={completed.returncode}",
            "stderr": completed.stderr.strip(),
            "details": {},
        }
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return {
            "status": "failed",
            "reason": "invalid_json",
            "stdout": stdout,
            "stderr": completed.stderr.strip(),
            "details": {},
        }
    return {
        "status": "ok" if payload.get("all_ok") else "failed",
        "reason": None if payload.get("all_ok") else "one_or_more_imports_failed",
        "details": payload.get("results", {}),
        "stderr": completed.stderr.strip() or None,
        "exit_code": completed.returncode,
    }


def _probe_help(repo_dir: Optional[Path], help_script: Optional[str]) -> Dict[str, Any]:
    if help_script is None:
        return {"status": "not_applicable", "reason": "no_help_script_declared"}
    if repo_dir is None or not repo_dir.exists():
        return {"status": "unavailable", "reason": "repo_not_extracted"}
    script_path = repo_dir / help_script
    if not script_path.exists():
        return {"status": "failed", "reason": f"missing_script: {help_script}"}
    try:
        completed = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:  # pragma: no cover - subprocess failure path
        return {"status": "failed", "reason": f"subprocess_error: {exc}"}

    output = (completed.stdout or "") + (completed.stderr or "")
    status = "ok" if completed.returncode == 0 else "failed"
    return {
        "status": status,
        "reason": None if status == "ok" else f"exit_code={completed.returncode}",
        "exit_code": completed.returncode,
        "output_excerpt": output[:800].strip() or None,
    }


def _probe_configs(
    spec: SpecialistBaselineSpec,
    repo_dir: Optional[Path],
    archive_path: Optional[Path],
) -> Dict[str, Any]:
    if not spec.config_files:
        return {"status": "not_applicable", "details": {}}
    details: Dict[str, Any] = {}
    all_ok = True
    for relative_path in spec.config_files:
        text = _read_text_from_asset(spec, repo_dir, archive_path, relative_path)
        if text is None:
            details[relative_path] = {"ok": False, "reason": "missing"}
            all_ok = False
            continue
        suffix = Path(relative_path).suffix.lower()
        try:
            if suffix in {".yaml", ".yml"}:
                payload = yaml.safe_load(text)
            elif suffix == ".json":
                payload = json.loads(text)
            else:
                payload = None
        except Exception as exc:
            details[relative_path] = {"ok": False, "reason": f"parse_error: {type(exc).__name__}: {exc}"}
            all_ok = False
            continue
        details[relative_path] = {
            "ok": True,
            "top_level_type": type(payload).__name__ if payload is not None else "NoneType",
        }
    return {"status": "ok" if all_ok else "failed", "details": details}


def _dependency_file_summary(
    spec: SpecialistBaselineSpec,
    repo_dir: Optional[Path],
    archive_path: Optional[Path],
) -> Dict[str, List[str]]:
    summary: Dict[str, List[str]] = {}
    for relative_path in spec.dependency_files:
        text = _read_text_from_asset(spec, repo_dir, archive_path, relative_path)
        if text is None:
            summary[relative_path] = ["<missing>"]
            continue
        lines: List[str] = []
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)
            if len(lines) >= 20:
                break
        summary[relative_path] = lines or ["<empty>"]
    return summary


def collect_preflight_report(spec: SpecialistBaselineSpec, baseline_root: Path) -> Dict[str, Any]:
    baseline_root = baseline_root.resolve()
    archive_path = spec.archive_path(baseline_root)
    repo_dir_candidate = spec.repo_dir_path(baseline_root)
    repo_dir = repo_dir_candidate if repo_dir_candidate.exists() and repo_dir_candidate.is_dir() else None
    archive_members = _load_archive_members(archive_path if archive_path.exists() else None)

    asset_state = "missing"
    if repo_dir is not None:
        asset_state = "repo_dir"
    elif archive_path.exists():
        asset_state = "archive_only"

    key_files = _file_presence(spec, repo_dir, archive_members, spec.key_files)
    entrypoints = _file_presence(spec, repo_dir, archive_members, spec.entrypoints)
    dependency_files = _file_presence(spec, repo_dir, archive_members, spec.dependency_files)
    config_files = _file_presence(spec, repo_dir, archive_members, spec.config_files)
    dependency_imports = _probe_dependency_imports(spec.critical_packages)
    baseline_import_probe = _probe_import_modules(repo_dir, spec.smoke_import_modules)
    help_probe = _probe_help(repo_dir, spec.help_script)
    config_probe = _probe_configs(spec, repo_dir, archive_path if archive_path.exists() else None)
    git_info = _git_info(repo_dir)

    repo_ready = asset_state == "repo_dir" and all(item["present"] for item in key_files.values())
    env_ready = all(item["ok"] for item in dependency_imports.values())
    smoke_import_ready = baseline_import_probe["status"] == "ok"
    help_ready = help_probe["status"] == "ok"
    config_ready = config_probe["status"] == "ok"

    blockers: List[str] = []
    if asset_state != "repo_dir":
        blockers.append("baseline repo is not extracted; only zip archive is available")
    if not env_ready:
        blockers.append("current shared Python environment is missing one or more critical dependencies")
    if not smoke_import_ready:
        blockers.append("minimal baseline import smoke test is not passing")
    if not help_ready and spec.help_script is not None:
        blockers.append("CLI help smoke test is not passing")
    if not config_ready and spec.config_files:
        blockers.append("declared config files cannot all be parsed")
    blockers.extend(str(item) for item in spec.current_blockers)

    report = {
        "baseline_name": spec.name,
        "baseline_slug": spec.slug,
        "baseline_root": str(baseline_root),
        "asset_state": asset_state,
        "archive_path": str(archive_path) if archive_path.exists() else None,
        "repo_dir_path": str(repo_dir) if repo_dir is not None else None,
        "archive_size_bytes": archive_path.stat().st_size if archive_path.exists() else None,
        "archive_sha256_prefix": _sha256_prefix(archive_path) if archive_path.exists() else None,
        "source": {
            "classification": spec.source_classification,
            "source_note": spec.source_note,
            "upstream_hint": spec.upstream_hint,
            "git": git_info,
        },
        "version_hint": spec.version_hint,
        "license": spec.license_name,
        "python_requirement": spec.python_requirement,
        "torch_requirement": spec.torch_requirement,
        "cuda_requirement": spec.cuda_requirement,
        "key_files": key_files,
        "entrypoints": entrypoints,
        "dependency_files": dependency_files,
        "config_files": config_files,
        "dependency_file_summary": _dependency_file_summary(
            spec,
            repo_dir,
            archive_path if archive_path.exists() else None,
        ),
        "native_input_format": spec.native_input_format,
        "native_output_format": spec.native_output_format,
        "extra_priors": list(spec.extra_priors),
        "enhanced_information": spec.enhanced_information,
        "enhanced_information_note": spec.enhanced_information_note,
        "best_fit_datasets": dict(spec.best_fit_datasets),
        "best_fit_notes": dict(spec.best_fit_notes),
        "critical_dependency_imports": dependency_imports,
        "baseline_import_probe": baseline_import_probe,
        "help_probe": help_probe,
        "config_probe": config_probe,
        "status": {
            "repo_ready": repo_ready,
            "env_ready": env_ready,
            "smoke_import_ready": smoke_import_ready,
            "help_ready": help_ready,
            "config_parse_ready": config_ready,
            "wrapper_ready": True,
            "training_ready": repo_ready and env_ready and (smoke_import_ready or help_ready),
        },
        "current_machine": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
        },
        "blockers": blockers,
    }
    return report


def write_preflight_artifact(
    report: Dict[str, Any],
    artifacts_root: Path,
    run_id: str,
    cli_args: Mapping[str, Any],
) -> Path:
    run_dir = ensure_dir(artifacts_root.resolve() / run_id)
    manifest = {
        "status": "preflight_only",
        "baseline_name": report["baseline_name"],
        "baseline_slug": report["baseline_slug"],
        "asset_state": report["asset_state"],
        "repo_ready": report["status"]["repo_ready"],
        "env_ready": report["status"]["env_ready"],
        "smoke_import_ready": report["status"]["smoke_import_ready"],
        "help_ready": report["status"]["help_ready"],
        "config_parse_ready": report["status"]["config_parse_ready"],
        "start_time": cli_args["start_time"],
        "end_time": cli_args["end_time"],
    }
    dump_json(run_dir / "manifest.json", manifest)
    dump_json(run_dir / "preflight_report.json", report)
    dump_yaml(run_dir / "resolved_config.yaml", dict(cli_args))
    log_lines = [
        f"baseline_name={report['baseline_name']}",
        f"asset_state={report['asset_state']}",
        f"repo_ready={report['status']['repo_ready']}",
        f"env_ready={report['status']['env_ready']}",
        f"smoke_import_ready={report['status']['smoke_import_ready']}",
        f"help_ready={report['status']['help_ready']}",
        f"config_parse_ready={report['status']['config_parse_ready']}",
    ]
    for blocker in report["blockers"]:
        log_lines.append(f"blocker={blocker}")
    write_text(run_dir / "train.log", "\n".join(log_lines) + "\n")
    return run_dir


def run_specialist_preflight(
    spec: SpecialistBaselineSpec,
    baseline_root: Path,
    artifacts_root: Path,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    start_time = utc_now()
    report = collect_preflight_report(spec, baseline_root)
    end_time = utc_now()
    report["start_time"] = start_time
    report["end_time"] = end_time
    resolved_run_id = run_id or f"preflight_{spec.slug}_{start_time.replace(':', '').replace('+00:00', 'Z')}"
    run_dir = write_preflight_artifact(
        report=report,
        artifacts_root=artifacts_root,
        run_id=resolved_run_id,
        cli_args={
            "baseline_name": spec.name,
            "baseline_slug": spec.slug,
            "baseline_root": str(baseline_root.resolve()),
            "artifacts_root": str(artifacts_root.resolve()),
            "run_id": resolved_run_id,
            "start_time": start_time,
            "end_time": end_time,
        },
    )
    report["artifact_run_dir"] = str(run_dir)
    return report


def _dump_json_text(path: Path, payload: Any) -> None:
    write_text(path, json.dumps(payload, indent=2, ensure_ascii=True) + "\n")


def _metadata_lookup(dataset: PreparedDataset, indices: np.ndarray) -> Dict[str, np.ndarray]:
    names = dataset.metadata_column_names.astype(str).tolist()
    lookup: Dict[str, np.ndarray] = {}
    for idx, name in enumerate(names):
        lookup[name] = dataset.metadata_values[indices, idx].astype(str)
    return lookup


def _write_common_split_bundle(
    dataset: PreparedDataset,
    split: SplitSpec,
    export_root: Path,
) -> Dict[str, Any]:
    export_root = ensure_dir(export_root)
    split_indices = indices_from_split(dataset.sample_ids, split)
    _dump_json_text(export_root / "feature_names.json", dataset.feature_names.astype(str).tolist())
    _dump_json_text(export_root / "context_feature_names.json", dataset.context_feature_names.astype(str).tolist())
    _dump_json_text(export_root / "metadata_column_names.json", dataset.metadata_column_names.astype(str).tolist())

    split_summaries: Dict[str, Any] = {}
    for split_name, indices in split_indices.items():
        split_dir = ensure_dir(export_root / split_name)
        np.save(split_dir / "x_baseline.npy", dataset.x_baseline[indices].astype(np.float32), allow_pickle=False)
        np.save(split_dir / "y_response.npy", dataset.y_response[indices].astype(np.float32), allow_pickle=False)
        np.save(split_dir / "delta_response.npy", dataset.delta_response[indices].astype(np.float32), allow_pickle=False)
        np.save(split_dir / "context_matrix.npy", dataset.context_matrix[indices].astype(np.float32), allow_pickle=False)
        _dump_json_text(split_dir / "sample_ids.json", dataset.sample_ids[indices].astype(str).tolist())
        _dump_json_text(split_dir / "intervention_ids.json", dataset.intervention_ids[indices].astype(str).tolist())
        _dump_json_text(split_dir / "intervention_types.json", dataset.intervention_types[indices].astype(str).tolist())

        metadata_lookup = _metadata_lookup(dataset, indices)
        sample_csv = split_dir / "samples.csv"
        fieldnames = [
            "sample_id",
            "split_name",
            "intervention_id",
            "intervention_type",
            "dose",
            "time",
            "group_id",
        ]
        fieldnames.extend([f"context::{name}" for name in dataset.context_feature_names.astype(str).tolist()])
        fieldnames.extend(dataset.metadata_column_names.astype(str).tolist())
        fieldnames.append("sample_metadata_json")
        with sample_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            context_names = dataset.context_feature_names.astype(str).tolist()
            for row_idx, sample_idx in enumerate(indices.tolist()):
                row: Dict[str, Any] = {
                    "sample_id": str(dataset.sample_ids[sample_idx]),
                    "split_name": split_name,
                    "intervention_id": str(dataset.intervention_ids[sample_idx]),
                    "intervention_type": str(dataset.intervention_types[sample_idx]),
                    "dose": None if np.isnan(dataset.doses[sample_idx]) else float(dataset.doses[sample_idx]),
                    "time": None if np.isnan(dataset.times[sample_idx]) else float(dataset.times[sample_idx]),
                    "group_id": str(dataset.group_ids[sample_idx]),
                    "sample_metadata_json": str(dataset.sample_metadata_json[sample_idx]),
                }
                for context_col, context_name in enumerate(context_names):
                    row[f"context::{context_name}"] = float(dataset.context_matrix[sample_idx, context_col])
                for metadata_name, values in metadata_lookup.items():
                    row[metadata_name] = str(values[row_idx])
                writer.writerow(row)

        split_summaries[split_name] = {
            "sample_count": int(indices.shape[0]),
            "directory": str(split_dir),
            "files": {
                "samples_csv": "samples.csv",
                "x_baseline": "x_baseline.npy",
                "y_response": "y_response.npy",
                "delta_response": "delta_response.npy",
                "context_matrix": "context_matrix.npy",
                "sample_ids": "sample_ids.json",
                "intervention_ids": "intervention_ids.json",
                "intervention_types": "intervention_types.json",
            },
        }

    manifest = {
        "native_export_kind": "canonical_split_bundle",
        "directory": str(export_root),
        "protocol": split.protocol,
        "dataset_name": dataset.dataset_name,
        "feature_count": int(dataset.feature_names.shape[0]),
        "context_feature_count": int(dataset.context_feature_names.shape[0]),
        "metadata_column_count": int(dataset.metadata_column_names.shape[0]),
        "splits": split_summaries,
    }
    dump_json(export_root / "manifest.json", manifest)
    return manifest


def use_lightweight_shared_export() -> bool:
    runtime_mode = str(os.environ.get("SCI_RESPONSE_RUNTIME_MODE", "primary"))
    execution_policy = str(os.environ.get("SCI_RESPONSE_RUNTIME_EXECUTION_POLICY", "shared_runtime_first"))
    return runtime_mode == "primary" and execution_policy in {"shared_only", "shared_runtime_first"}


def export_lightweight_native_manifest(
    spec: SpecialistBaselineSpec,
    dataset: PreparedDataset,
    split: SplitSpec,
    export_root: Path,
    *,
    native_export_kind: str,
    notes: Sequence[str] = (),
) -> Dict[str, Any]:
    export_root = ensure_dir(export_root)
    _dump_json_text(export_root / "feature_names.json", dataset.feature_names.astype(str).tolist())
    _dump_json_text(export_root / "context_feature_names.json", dataset.context_feature_names.astype(str).tolist())
    _dump_json_text(export_root / "metadata_column_names.json", dataset.metadata_column_names.astype(str).tolist())
    manifest = {
        "native_export_kind": native_export_kind,
        "baseline_name": spec.name,
        "directory": str(export_root),
        "dataset_name": dataset.dataset_name,
        "protocol": split.protocol,
        "available": True,
        "lightweight_shared_runtime_export": True,
        "sample_count": int(dataset.sample_count),
        "feature_count": int(dataset.feature_names.shape[0]),
        "context_feature_count": int(dataset.context_feature_names.shape[0]),
        "metadata_column_count": int(dataset.metadata_column_names.shape[0]),
        "notes": list(notes),
    }
    dump_json(export_root / "manifest.json", manifest)
    return manifest


def export_ann_like_bundle(
    spec: SpecialistBaselineSpec,
    dataset: PreparedDataset,
    split: SplitSpec,
    export_root: Path,
    *,
    extra_notes: Sequence[str] = (),
) -> Dict[str, Any]:
    common_manifest = _write_common_split_bundle(dataset, split, export_root)
    manifest = {
        **common_manifest,
        "native_export_kind": "ann_like_directory",
        "baseline_name": spec.name,
        "write_h5ad_supported_in_current_env": False,
        "blockers": [
            "current shared environment does not provide anndata/scanpy, so a true h5ad export is not written here",
            "use the emitted split-wise matrices + samples.csv bundle to reconstruct AnnData in a dedicated specialist env",
        ],
        "notes": list(extra_notes),
    }
    dump_json(export_root / "manifest.json", manifest)
    return manifest


def export_gperturb_bundle(
    spec: SpecialistBaselineSpec,
    dataset: PreparedDataset,
    split: SplitSpec,
    export_root: Path,
) -> Dict[str, Any]:
    common_manifest = _write_common_split_bundle(dataset, split, export_root)
    vocabulary = sorted(np.unique(dataset.intervention_ids.astype(str)).tolist())
    vocab_index = {label: idx for idx, label in enumerate(vocabulary)}
    split_indices = indices_from_split(dataset.sample_ids, split)
    split_exports: Dict[str, Any] = {}
    for split_name, indices in split_indices.items():
        split_dir = ensure_dir(export_root / split_name)
        perturbation_matrix = np.zeros((indices.shape[0], len(vocabulary)), dtype=np.float32)
        for row_idx, label in enumerate(dataset.intervention_ids[indices].astype(str).tolist()):
            perturbation_matrix[row_idx, vocab_index[label]] = 1.0
        np.save(split_dir / "X_expression.npy", dataset.y_response[indices].astype(np.float32), allow_pickle=False)
        np.save(split_dir / "C_covariates.npy", dataset.context_matrix[indices].astype(np.float32), allow_pickle=False)
        np.save(split_dir / "P_perturbation.npy", perturbation_matrix.astype(np.float32), allow_pickle=False)
        split_exports[split_name] = {
            "X_expression": "X_expression.npy",
            "C_covariates": "C_covariates.npy",
            "P_perturbation": "P_perturbation.npy",
        }

    _dump_json_text(export_root / "perturbation_names.json", vocabulary)
    manifest = {
        **common_manifest,
        "native_export_kind": "gperturb_xcp_bundle",
        "baseline_name": spec.name,
        "perturbation_names_file": "perturbation_names.json",
        "split_exports": split_exports,
        "notes": [
            "X_expression stores post-treatment response profiles",
            "C_covariates stores canonical context_matrix",
            "P_perturbation is a one-hot intervention design matrix over the full dataset vocabulary",
        ],
    }
    dump_json(export_root / "manifest.json", manifest)
    return manifest


def export_transigen_bundle(
    spec: SpecialistBaselineSpec,
    dataset: PreparedDataset,
    split: SplitSpec,
    export_root: Path,
) -> Dict[str, Any]:
    export_root = ensure_dir(export_root)
    common_manifest = _write_common_split_bundle(dataset, split, export_root)
    metadata_names = dataset.metadata_column_names.astype(str).tolist()
    if "cid" not in metadata_names:
        manifest = {
            **common_manifest,
            "native_export_kind": "transigen_hdf_bundle",
            "baseline_name": spec.name,
            "available": False,
            "reason": "metadata column `cid` is required for TranSiGen export",
        }
        dump_json(export_root / "manifest.json", manifest)
        return manifest

    import h5py  # local import to keep preflight lightweight

    split_indices = indices_from_split(dataset.sample_ids, split)
    cid_values = dataset.metadata_column("cid").astype(str)
    hdf_files: Dict[str, str] = {}
    for split_name, indices in split_indices.items():
        hdf_path = export_root / f"{split_name}.h5"
        with h5py.File(hdf_path, "w") as handle:
            handle["x1"] = dataset.x_baseline[indices].astype(np.float32)
            handle["x2"] = dataset.y_response[indices].astype(np.float32)
            handle["canonical_smiles"] = dataset.intervention_ids[indices].astype("S")
            handle["cid"] = cid_values[indices].astype("S")
            handle["sig"] = dataset.sample_ids[indices].astype("S")
            handle["LINCS_index"] = np.arange(indices.shape[0], dtype=np.int64)
        hdf_files[split_name] = hdf_path.name

    idx2smi = {key: key for key in np.unique(dataset.intervention_ids.astype(str)).tolist()}
    with (export_root / "idx2smi.pickle").open("wb") as handle:
        pickle.dump(idx2smi, handle)

    manifest = {
        **common_manifest,
        "native_export_kind": "transigen_hdf_bundle",
        "baseline_name": spec.name,
        "available": True,
        "hdf_files": hdf_files,
        "idx2smi_file": "idx2smi.pickle",
        "notes": [
            "Each split HDF contains x1/x2/canonical_smiles/cid/sig/LINCS_index",
            "This export is aligned with TranSiGen's load_from_HDF utility rather than this repo's canonical NPZ schema",
        ],
    }
    dump_json(export_root / "manifest.json", manifest)
    return manifest
