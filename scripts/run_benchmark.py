#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sci_response.baselines.benchmark import resolve_benchmark_config, resolve_benchmark_payload, run_benchmark
from sci_response.baselines.runtime_envs import resolve_runtime_env


def _coerce_cli_value(raw: str) -> Any:
    stripped = raw.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    lowered = raw.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def _strip_flag_with_value(argv: list[str], flag: str) -> list[str]:
    stripped: list[str] = []
    skip_next = False
    for index, item in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if item == flag:
            if index + 1 < len(argv):
                skip_next = True
            continue
        if item.startswith(f"{flag}="):
            continue
        stripped.append(item)
    return stripped


def _parse_kv_pairs(items: list[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected key=value, got: {item}")
        key, value = item.split("=", 1)
        params[str(key).strip()] = _coerce_cli_value(value)
    return params


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Unified benchmark runner for universal baselines, specialist baselines, and trainable models.")
    parser.add_argument("--config", default=None, help="Path to a benchmark config YAML file.")
    parser.add_argument("--run-id", default=None, help="Optional fixed run id.")
    parser.add_argument("--device", default="cpu", help="Device request: cpu, cuda, cuda:0, cuda:1, ...")
    parser.add_argument(
        "--cuda-visible-devices",
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override, e.g. 0 or 0,1",
    )
    parser.add_argument("--dataset-config", default=None, help="Inline mode: dataset config YAML path.")
    parser.add_argument("--split-path", default=None, help="Inline mode: explicit split JSON path.")
    parser.add_argument("--run-name", default=None, help="Inline mode: logical run name.")
    parser.add_argument("--artifacts-root", default="artifacts", help="Artifacts root for inline mode.")
    parser.add_argument("--seed", type=int, default=11, help="Seed for inline mode.")
    parser.add_argument("--top-k", type=int, default=20, help="Evaluation top-k for inline mode.")
    parser.add_argument("--baseline", default=None, help="Inline mode: universal baseline name.")
    parser.add_argument("--specialist", default=None, help="Inline mode: specialist baseline name.")
    parser.add_argument("--model-config", default=None, help="Inline mode: trainable model config YAML.")
    parser.add_argument("--baseline-root", default=str(ROOT / "baseline"), help="Inline mode: specialist baseline root.")
    parser.add_argument(
        "--runtime-env-config",
        default=str(ROOT / "configs" / "runtime_envs.yaml"),
        help="Runtime environment grouping config for automatic env dispatch.",
    )
    parser.add_argument(
        "--runtime-mode",
        default="primary",
        choices=["primary", "fallback", "upstream"],
        help="Primary shared-runtime mode by default; fallback/upstream is only for strict reproduction.",
    )
    parser.add_argument(
        "--runtime-env-group",
        default=None,
        help="Optional explicit env-group override. Use sparingly for debugging or strict-upstream reproduction.",
    )
    parser.add_argument(
        "--feature-builder-intervention-hash-dim",
        type=int,
        default=64,
        help="Inline universal-baseline mode: hashed intervention feature dimension.",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Inline universal-baseline mode: baseline hyperparameter override, repeatable key=value.",
    )
    parser.add_argument("--dispatch-depth", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    inline_payload = None
    config_path = Path(args.config).resolve() if args.config else None
    if config_path is None:
        if not args.dataset_config or not args.split_path:
            raise SystemExit("Either --config or both --dataset-config and --split-path are required.")
        requested_methods = [bool(args.baseline), bool(args.specialist), bool(args.model_config)]
        if sum(int(item) for item in requested_methods) != 1:
            raise SystemExit("Inline mode requires exactly one of --baseline, --specialist, or --model-config.")

        dataset_config_path = Path(args.dataset_config).resolve()
        split_path = Path(args.split_path).resolve()
        run_name = args.run_name
        if not run_name:
            method_name = args.baseline or args.specialist or Path(args.model_config).stem
            run_name = f"{dataset_config_path.stem}_{split_path.parent.name}_{method_name}"
        inline_payload = {
            "run_name": str(run_name),
            "seed": int(args.seed),
            "artifacts_root": str(Path(args.artifacts_root).resolve()),
            "dataset_config": str(dataset_config_path),
            "split_path": str(split_path),
            "metrics": {"top_k": int(args.top_k)},
            "notes": "inline benchmark invocation",
        }
        if args.baseline:
            inline_payload["baseline"] = {
                "name": str(args.baseline),
                "feature_builder": {
                    "intervention_hash_dim": int(args.feature_builder_intervention_hash_dim),
                },
                "tuning_budget": {"definition": "fixed_default_v1", "num_trials": 1},
                "params": _parse_kv_pairs(list(args.param)),
            }
        elif args.specialist:
            inline_payload["baseline_name"] = str(args.specialist)
            inline_payload["baseline_root"] = str(Path(args.baseline_root).resolve())
            inline_payload["baseline"] = _parse_kv_pairs(list(args.param))
        else:
            inline_payload["model_config"] = str(Path(args.model_config).resolve())

    preview_resolved = (
        resolve_benchmark_payload(inline_payload, config_path or (ROOT / "configs" / "experiments" / "_inline_benchmark.yaml"))
        if inline_payload is not None
        else resolve_benchmark_config(config_path)
    )
    runtime_env = resolve_runtime_env(
        config_path=Path(args.runtime_env_config).resolve(),
        method_name=str(preview_resolved.method["name"]),
        method_family=str(preview_resolved.method["family"]),
        current_python=sys.executable,
        runtime_mode=str(args.runtime_mode),
        forced_env_group=args.runtime_env_group,
    )
    os.environ["SCI_RESPONSE_RUNTIME_ENV_GROUP"] = runtime_env.env_group
    os.environ["SCI_RESPONSE_RUNTIME_ENV_CONFIG"] = str(Path(args.runtime_env_config).resolve())
    os.environ["SCI_RESPONSE_RUNTIME_PYTHON"] = runtime_env.python_executable or sys.executable
    os.environ["SCI_RESPONSE_RUNTIME_MODE"] = str(args.runtime_mode)
    os.environ["SCI_RESPONSE_RUNTIME_EXECUTION_POLICY"] = runtime_env.execution_policy
    os.environ["SCI_RESPONSE_RUNTIME_IMPLEMENTATION_TRACK"] = runtime_env.implementation_track
    os.environ["SCI_RESPONSE_RUNTIME_PRIMARY_ENV_GROUP"] = runtime_env.primary_env_group
    if runtime_env.fallback_env_group:
        os.environ["SCI_RESPONSE_RUNTIME_FALLBACK_ENV_GROUP"] = runtime_env.fallback_env_group
    else:
        os.environ.pop("SCI_RESPONSE_RUNTIME_FALLBACK_ENV_GROUP", None)
    if args.runtime_env_group:
        os.environ["SCI_RESPONSE_RUNTIME_FORCE_ENV_GROUP"] = str(args.runtime_env_group)
    else:
        os.environ.pop("SCI_RESPONSE_RUNTIME_FORCE_ENV_GROUP", None)
    if args.dispatch_depth == 0 and runtime_env.should_dispatch:
        if not runtime_env.available or runtime_env.python_executable is None:
            raise SystemExit(
                f"Method {preview_resolved.method['name']} requires env group {runtime_env.env_group}, "
                f"but no runnable Python executable is configured. {runtime_env.reason}"
            )
        forwarded_args = _strip_flag_with_value(list(sys.argv[1:]), "--dispatch-depth")
        child_env = os.environ.copy()
        child_env["SCI_RESPONSE_RUNTIME_ENV_GROUP"] = runtime_env.env_group
        child_env["SCI_RESPONSE_RUNTIME_ENV_CONFIG"] = str(Path(args.runtime_env_config).resolve())
        child_env["SCI_RESPONSE_RUNTIME_PYTHON"] = runtime_env.python_executable
        child_env["SCI_RESPONSE_RUNTIME_MODE"] = str(args.runtime_mode)
        child_env["SCI_RESPONSE_RUNTIME_EXECUTION_POLICY"] = runtime_env.execution_policy
        child_env["SCI_RESPONSE_RUNTIME_IMPLEMENTATION_TRACK"] = runtime_env.implementation_track
        child_env["SCI_RESPONSE_RUNTIME_PRIMARY_ENV_GROUP"] = runtime_env.primary_env_group
        if runtime_env.fallback_env_group:
            child_env["SCI_RESPONSE_RUNTIME_FALLBACK_ENV_GROUP"] = runtime_env.fallback_env_group
        if args.runtime_env_group:
            child_env["SCI_RESPONSE_RUNTIME_FORCE_ENV_GROUP"] = str(args.runtime_env_group)
        completed = subprocess.run(
            [runtime_env.python_executable, str(Path(__file__).resolve()), *forwarded_args, "--dispatch-depth", "1"],
            env=child_env,
            check=False,
        )
        raise SystemExit(completed.returncode)

    run_dir = run_benchmark(
        repo_root=ROOT,
        config_path=config_path,
        inline_payload=inline_payload,
        run_id=args.run_id,
        requested_device=str(args.device),
        cuda_visible_devices=args.cuda_visible_devices,
    )
    print(f"run_dir={run_dir}")
    print(f"metrics_json={run_dir / 'metrics.json'}")
    print(f"manifest_json={run_dir / 'manifest.json'}")
    prediction_manifest = run_dir / "predictions" / "manifest.json"
    if prediction_manifest.exists():
        print(f"prediction_manifest_json={prediction_manifest}")


if __name__ == "__main__":
    main()
