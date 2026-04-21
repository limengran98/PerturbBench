#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sci_response.baselines.artifacts import beijing_timestamp_slug
from sci_response.data.io import dump_json, dump_yaml, ensure_dir, load_json, load_yaml, write_text
from sci_response.data.registry import load_dataset_spec


PRIMARY_METRIC_PATHS = (
    "test.delta.mse",
    "test.delta.mae",
    "test.delta.pearson",
    "test.delta.spearman",
    "test.delta.r2",
    "test.delta.topk_overlap",
    "test.response.mse",
    "test.response.mae",
    "test.response.pearson",
    "test.response.spearman",
    "test.response.r2",
    "test.response.topk_overlap",
)


def _parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _flatten_numeric(payload: Any, prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_numeric(value, next_prefix))
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        flat[prefix] = float(payload)
    return flat


def _aggregate_metric_tables(metric_tables: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = sorted(set().union(*(table.keys() for table in metric_tables)))
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = [table[key] for table in metric_tables if key in table]
        if not values:
            continue
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        summary[key] = {
            "mean": float(mean_value),
            "std": float(variance ** 0.5),
            "count": int(len(values)),
        }
    return summary


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _resolve_protocol_specs(
    dataset_key: str,
    dataset_entry: Dict[str, Any],
    *,
    include_supplementary: bool,
) -> List[Dict[str, str]]:
    protocol_specs = [
        {
            "protocol": str(dataset_entry["primary_protocol"]),
            "split_path": str(dataset_entry["primary_split"]),
            "kind": "primary",
        }
    ]
    if include_supplementary:
        for item in dataset_entry.get("supplementary_protocols", []):
            protocol_specs.append(
                {
                    "protocol": str(item["protocol"]),
                    "split_path": str(item["split_path"]),
                    "kind": "supplementary",
                }
            )
    deduped: List[Dict[str, str]] = []
    seen = set()
    for item in protocol_specs:
        key = (item["protocol"], item["split_path"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    if not deduped:
        raise ValueError(f"No protocol specs resolved for dataset {dataset_key}")
    return deduped


def _build_inline_payload(
    *,
    dataset_config: Path,
    split_path: Path,
    run_name: str,
    artifacts_root: Path,
    seed: int,
    top_k: int,
    method_name: str,
    method_defaults: Dict[str, Any],
    baseline_root: Path,
) -> Dict[str, Any]:
    family = str(method_defaults["family"])
    payload: Dict[str, Any] = {
        "run_name": run_name,
        "seed": int(seed),
        "artifacts_root": str(artifacts_root.resolve()),
        "dataset_config": str(dataset_config.resolve()),
        "split_path": str(split_path.resolve()),
        "metrics": {"top_k": int(top_k)},
        "notes": "dataset pack invocation",
    }
    if family == "universal":
        payload["baseline"] = {
            "name": method_name,
            "feature_builder": {"intervention_hash_dim": 64},
            "tuning_budget": {"definition": "fixed_default_v1", "num_trials": 1},
            "params": dict(method_defaults.get("params", {})),
        }
        return payload
    if family == "specialist":
        payload["baseline_name"] = method_name
        payload["baseline_root"] = str(baseline_root.resolve())
        payload["baseline"] = dict(method_defaults.get("params", {}))
        return payload
    raise ValueError(f"Unsupported method family for {method_name}: {family}")


def _run_or_reuse(
    *,
    inline_payload: Dict[str, Any],
    run_id: str,
    requested_device: str,
    cuda_visible_devices: str | None,
    runtime_env_config: Path,
    runtime_mode: str,
    runtime_env_group: str | None,
    skip_existing: bool,
) -> Path:
    run_dir = Path(inline_payload["artifacts_root"]).resolve() / run_id
    if skip_existing and (run_dir / "manifest.json").exists() and (run_dir / "metrics.json").exists():
        return run_dir
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_benchmark.py"),
        "--dataset-config",
        str(Path(inline_payload["dataset_config"]).resolve()),
        "--split-path",
        str(Path(inline_payload["split_path"]).resolve()),
        "--run-name",
        str(inline_payload["run_name"]),
        "--artifacts-root",
        str(Path(inline_payload["artifacts_root"]).resolve()),
        "--seed",
        str(int(inline_payload["seed"])),
        "--top-k",
        str(int(dict(inline_payload.get("metrics", {})).get("top_k", 20))),
        "--run-id",
        run_id,
        "--device",
        str(requested_device),
        "--runtime-env-config",
        str(runtime_env_config.resolve()),
        "--runtime-mode",
        str(runtime_mode),
    ]
    if runtime_env_group is not None:
        cmd.extend(["--runtime-env-group", str(runtime_env_group)])
    if cuda_visible_devices is not None:
        cmd.extend(["--cuda-visible-devices", str(cuda_visible_devices)])
    if "baseline_name" in inline_payload:
        cmd.extend(
            [
                "--specialist",
                str(inline_payload["baseline_name"]),
                "--baseline-root",
                str(Path(inline_payload["baseline_root"]).resolve()),
            ]
        )
        for key, value in dict(inline_payload.get("baseline", {})).items():
            cmd.extend(["--param", f"{key}={json.dumps(value) if isinstance(value, (dict, list)) else value}"])
    elif "baseline" in inline_payload:
        baseline_payload = dict(inline_payload["baseline"])
        cmd.extend(
            [
                "--baseline",
                str(baseline_payload["name"]),
                "--feature-builder-intervention-hash-dim",
                str(int(dict(baseline_payload.get("feature_builder", {})).get("intervention_hash_dim", 64))),
            ]
        )
        for key, value in dict(baseline_payload.get("params", {})).items():
            cmd.extend(["--param", f"{key}={json.dumps(value) if isinstance(value, (dict, list)) else value}"])
    elif "model_config" in inline_payload:
        cmd.extend(["--model-config", str(Path(inline_payload["model_config"]).resolve())])
    else:
        raise ValueError("Unsupported inline payload for dataset pack dispatch")
    subprocess.run(cmd, check=True)
    return run_dir


def _summary_row(
    *,
    dataset_key: str,
    dataset_entry: Dict[str, Any],
    protocol_name: str,
    split_path: Path,
    method_name: str,
    method_family: str,
    artifact_paths: Sequence[Path],
    aggregate_metrics: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "dataset_key": dataset_key,
        "dataset_display_name": str(dataset_entry["display_name"]),
        "benchmark_role": str(dataset_entry["benchmark_role"]),
        "task_group": str(dataset_entry["task_group"]),
        "protocol": protocol_name,
        "split_path": str(split_path.resolve()),
        "method_name": method_name,
        "method_family": method_family,
        "seed_count": int(len(artifact_paths)),
        "artifact_paths": json.dumps([str(path.resolve()) for path in artifact_paths], ensure_ascii=True),
    }
    for metric_key in PRIMARY_METRIC_PATHS:
        metric = aggregate_metrics.get(metric_key)
        row[f"{metric_key}_mean"] = metric["mean"] if metric else None
        row[f"{metric_key}_std"] = metric["std"] if metric else None
    return row


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _progress(message: str) -> None:
    print(f"[dataset-pack] {message}", flush=True)


def _method_supports_cuda(method_name: str, method_defaults: Dict[str, Any]) -> bool:
    family = str(method_defaults.get("family", ""))
    normalized = str(method_name).lower()
    if family == "universal" and normalized == "catboost":
        return True
    if family == "specialist" and normalized in {"gperturb", "gears", "cpa", "xpert", "cellot", "transigen"}:
        return True
    return False


def _order_methods(method_names: Sequence[str], baseline_defaults: Dict[str, Any]) -> List[str]:
    decorated = []
    for index, method_name in enumerate(method_names):
        method_defaults = dict(baseline_defaults[method_name])
        gpu_priority = 0 if _method_supports_cuda(method_name, method_defaults) else 1
        decorated.append((gpu_priority, index, method_name))
    decorated.sort()
    return [method_name for _, _, method_name in decorated]


def _copy_latest(path: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run all runnable baselines for one dataset and aggregate 3-seed mean±std.")
    parser.add_argument("--matrix-config", default=str(ROOT / "configs" / "benchmark_matrix.yaml"))
    parser.add_argument("--runtime-env-config", default=str(ROOT / "configs" / "runtime_envs.yaml"))
    parser.add_argument(
        "--runtime-mode",
        default="primary",
        choices=["primary", "fallback", "upstream"],
        help="Primary shared-runtime mode by default; fallback/upstream is only for strict reproduction.",
    )
    parser.add_argument(
        "--runtime-env-group",
        default=None,
        help="Optional explicit env-group override for debugging.",
    )
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--methods", default=None, help="Optional comma-separated subset of runnable methods.")
    parser.add_argument("--seeds", default="11,12,13", help="Comma-separated training seeds.")
    parser.add_argument("--device", default="cuda:0", help="Requested device, typically cuda:0 for a single 4090.")
    parser.add_argument("--cuda-visible-devices", default=None, help="CUDA visibility mask, e.g. 0")
    parser.add_argument("--artifacts-root", default=str(ROOT / "artifacts"))
    parser.add_argument("--baseline-root", default=str(ROOT / "baseline"))
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--include-supplementary", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    matrix_path = Path(args.matrix_config).resolve()
    runtime_env_config = Path(args.runtime_env_config).resolve()
    matrix = load_yaml(matrix_path)
    datasets = dict(matrix.get("datasets", {}))
    if args.dataset_key not in datasets:
        raise KeyError(f"Unknown dataset_key {args.dataset_key!r} in {matrix_path}")

    dataset_entry = dict(datasets[args.dataset_key])
    dataset_config = (ROOT / str(dataset_entry["dataset_config"])).resolve()
    dataset_spec = load_dataset_spec(dataset_config)
    baseline_defaults = dict(matrix.get("baseline_defaults", {}))
    runnable_methods = [str(item) for item in dataset_entry.get("runnable_methods", [])]
    if args.methods:
        requested_methods = _parse_csv_list(args.methods)
        unsupported = [item for item in requested_methods if item not in runnable_methods]
        if unsupported:
            raise ValueError(
                f"Requested methods are not in the runnable matrix for {args.dataset_key}: {unsupported}"
            )
        method_names = _order_methods(requested_methods, baseline_defaults)
    else:
        method_names = _order_methods(runnable_methods, baseline_defaults)

    seeds = [int(item) for item in _parse_csv_list(args.seeds)]
    artifacts_root = Path(args.artifacts_root).resolve()
    baseline_root = Path(args.baseline_root).resolve()
    session_timestamp = beijing_timestamp_slug()
    dataset_root = ensure_dir(artifacts_root / "datasets" / args.dataset_key)
    runs_root = ensure_dir(dataset_root / "methods")
    pack_root = ensure_dir(dataset_root / "pack_runs" / session_timestamp)
    global_root = ensure_dir(artifacts_root / "global")
    protocol_specs = _resolve_protocol_specs(
        args.dataset_key,
        dataset_entry,
        include_supplementary=bool(args.include_supplementary),
    )

    per_method_rows: List[Dict[str, Any]] = []
    pack_summary: Dict[str, Any] = {
        "dataset_key": args.dataset_key,
        "dataset_name": dataset_spec.dataset_name,
        "dataset_config": str(dataset_config),
        "display_name": str(dataset_entry["display_name"]),
        "benchmark_role": str(dataset_entry["benchmark_role"]),
        "task_group": str(dataset_entry["task_group"]),
        "session_timestamp_beijing": session_timestamp,
        "dataset_root": str(dataset_root),
        "runs_root": str(runs_root),
        "pack_root": str(pack_root),
        "device_request": str(args.device),
        "cuda_visible_devices": args.cuda_visible_devices,
        "seeds": seeds,
        "include_supplementary": bool(args.include_supplementary),
        "methods": method_names,
        "protocol_runs": [],
        "blocked_intake_only_methods": list(dataset_entry.get("intake_only_methods", [])),
        "biological_validation_notes": list(dataset_entry.get("biological_validation_notes", [])),
    }

    _progress(
        f"dataset={args.dataset_key} display_name={dataset_entry['display_name']} "
        f"session={session_timestamp} device={args.device} "
        f"cuda_visible_devices={args.cuda_visible_devices} "
        f"seeds={seeds} methods={method_names}"
    )

    protocol_total = len(protocol_specs)
    for protocol_index, protocol_spec in enumerate(protocol_specs, start=1):
        protocol_name = str(protocol_spec["protocol"])
        split_path = (ROOT / str(protocol_spec["split_path"])).resolve()
        _progress(
            f"protocol[{protocol_index}/{protocol_total}] start "
            f"dataset={args.dataset_key} protocol={protocol_name} split={split_path}"
        )
        protocol_summary: Dict[str, Any] = {
            "protocol": protocol_name,
            "kind": str(protocol_spec["kind"]),
            "split_path": str(split_path),
            "methods": {},
        }
        method_total = len(method_names)
        for method_index, method_name in enumerate(method_names, start=1):
            if method_name not in baseline_defaults:
                raise KeyError(f"Method {method_name!r} is missing from baseline_defaults in {matrix_path}")
            method_defaults = dict(baseline_defaults[method_name])
            dataset_method_overrides = dict(dataset_entry.get("method_overrides", {})).get(method_name)
            if isinstance(dataset_method_overrides, dict):
                method_defaults = _deep_update(method_defaults, dataset_method_overrides)
            artifact_paths: List[Path] = []
            metric_tables: List[Dict[str, float]] = []
            seed_total = len(seeds)
            _progress(
                f"method[{method_index}/{method_total}] start "
                f"dataset={args.dataset_key} protocol={protocol_name} method={method_name} "
                f"family={method_defaults['family']}"
            )
            for seed_index, seed in enumerate(seeds, start=1):
                run_name = f"{args.dataset_key}_{protocol_name}_{method_name}"
                run_id = f"{method_name}/{protocol_name}/{session_timestamp}__seed{seed}"
                _progress(
                    f"seed[{seed_index}/{seed_total}] dispatch "
                    f"run_id={run_id} device={args.device} cuda_visible_devices={args.cuda_visible_devices}"
                )
                inline_payload = _build_inline_payload(
                    dataset_config=dataset_config,
                    split_path=split_path,
                    run_name=run_name,
                    artifacts_root=runs_root,
                    seed=int(seed),
                    top_k=int(args.top_k),
                    method_name=method_name,
                    method_defaults=method_defaults,
                    baseline_root=baseline_root,
                )
                run_dir = _run_or_reuse(
                    inline_payload=inline_payload,
                    run_id=run_id,
                    requested_device=str(args.device),
                    cuda_visible_devices=args.cuda_visible_devices,
                    runtime_env_config=runtime_env_config,
                    runtime_mode=str(args.runtime_mode),
                    runtime_env_group=args.runtime_env_group,
                    skip_existing=bool(args.skip_existing),
                )
                manifest = load_json(run_dir / "manifest.json")
                execution_status = str(manifest.get("execution_status"))
                if execution_status != "completed":
                    raise RuntimeError(
                        f"Run {run_dir} did not complete successfully: execution_status={execution_status}"
                    )
                _progress(
                    f"seed[{seed_index}/{seed_total}] completed "
                    f"run_id={run_id} resolved_device={manifest.get('resolved_device')} "
                    f"model_uses_gpu={manifest.get('model_uses_gpu')} metrics={run_dir / 'metrics.json'}"
                )
                artifact_paths.append(run_dir)
                metrics = load_json(run_dir / "metrics.json")
                metric_tables.append(_flatten_numeric(metrics))

            aggregate_metrics = _aggregate_metric_tables(metric_tables)
            method_summary = {
                "method_name": method_name,
                "method_family": str(method_defaults["family"]),
                "artifact_paths": [str(path) for path in artifact_paths],
                "aggregate_metrics": aggregate_metrics,
            }
            protocol_summary["methods"][method_name] = method_summary

            method_summary_path = pack_root / "method_summaries" / f"{protocol_name}__{method_name}.json"
            dump_json(method_summary_path, method_summary)
            primary_delta = aggregate_metrics.get("test.delta.mse")
            primary_note = (
                f"test.delta.mse.mean={primary_delta['mean']:.6f} std={primary_delta['std']:.6f}"
                if primary_delta
                else "test.delta.mse.mean=NA"
            )
            _progress(
                f"method[{method_index}/{method_total}] completed "
                f"dataset={args.dataset_key} protocol={protocol_name} method={method_name} {primary_note}"
            )
            per_method_rows.append(
                _summary_row(
                    dataset_key=args.dataset_key,
                    dataset_entry=dataset_entry,
                    protocol_name=protocol_name,
                    split_path=split_path,
                    method_name=method_name,
                    method_family=str(method_defaults["family"]),
                    artifact_paths=artifact_paths,
                    aggregate_metrics=aggregate_metrics,
                )
            )

        pack_summary["protocol_runs"].append(protocol_summary)
        _progress(
            f"protocol[{protocol_index}/{protocol_total}] completed "
            f"dataset={args.dataset_key} protocol={protocol_name}"
        )

    dump_json(pack_root / "benchmark_pack_summary.json", pack_summary)
    _write_csv(pack_root / "benchmark_pack_summary.csv", per_method_rows)
    dump_yaml(
        pack_root / "resolved_pack_config.yaml",
        {
                    "dataset_key": args.dataset_key,
                    "dataset_config": str(dataset_config),
                    "methods": method_names,
                    "seeds": seeds,
                    "session_timestamp_beijing": session_timestamp,
                    "dataset_root": str(dataset_root),
                    "runs_root": str(runs_root),
                    "pack_root": str(pack_root),
                    "device": str(args.device),
                    "cuda_visible_devices": args.cuda_visible_devices,
                    "include_supplementary": bool(args.include_supplementary),
                    "artifacts_root": str(artifacts_root),
                    "baseline_root": str(baseline_root),
                    "matrix_config": str(matrix_path),
                    "runtime_env_config": str(runtime_env_config),
                    "runtime_mode": str(args.runtime_mode),
                    "runtime_env_group": args.runtime_env_group,
        },
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "aggregate_results.py"),
            "--artifacts-root",
            str(artifacts_root),
        ],
        check=True,
    )

    latest_prefix = dataset_root / "latest"
    _copy_latest(pack_root / "benchmark_pack_summary.json", latest_prefix / "benchmark_pack_summary.json")
    _copy_latest(pack_root / "benchmark_pack_summary.csv", latest_prefix / "benchmark_pack_summary.csv")
    _copy_latest(pack_root / "resolved_pack_config.yaml", latest_prefix / "resolved_pack_config.yaml")
    write_text(dataset_root / "LATEST_PACK_RUN.txt", str(pack_root.resolve()) + "\n")

    _progress(f"dataset={args.dataset_key} pack complete summary_json={pack_root / 'benchmark_pack_summary.json'}")
    _progress(f"dataset={args.dataset_key} pack complete summary_csv={pack_root / 'benchmark_pack_summary.csv'}")
    _progress(f"dataset={args.dataset_key} latest summary csv={latest_prefix / 'benchmark_pack_summary.csv'}")
    _progress(f"global benchmark summary csv={global_root / 'benchmark_summary.csv'}")
    print(f"dataset_pack_summary_json={pack_root / 'benchmark_pack_summary.json'}")
    print(f"dataset_pack_summary_csv={pack_root / 'benchmark_pack_summary.csv'}")
    print(f"dataset_latest_summary_csv={latest_prefix / 'benchmark_pack_summary.csv'}")
    print(f"benchmark_summary_csv={global_root / 'benchmark_summary.csv'}")
    print(f"benchmark_summary_json={global_root / 'benchmark_summary.json'}")


if __name__ == "__main__":
    main()
