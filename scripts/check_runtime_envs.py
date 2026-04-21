#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sci_response.baselines.runtime_envs import resolve_runtime_env
from sci_response.data.io import load_yaml


def _assigned_methods(method_runtime: Dict[str, Dict[str, Any]], *, key: str) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for method_name, payload in method_runtime.items():
        env_group = payload.get(key)
        if not env_group:
            continue
        result.setdefault(str(env_group), []).append(str(method_name))
    for env_group in result:
        result[env_group] = sorted(result[env_group])
    return result


def _run_preflight(method_name: str, runtime_env_config: Path) -> int:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "preflight_specialist.py"),
        "--baseline",
        method_name,
        "--runtime-env-config",
        str(runtime_env_config),
        "--run-id",
        f"preflight_{method_name.lower()}_runtime_check",
    ]
    completed = subprocess.run(cmd, check=False)
    return int(completed.returncode)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Check runtime-environment assignments for benchmark methods.")
    parser.add_argument("--runtime-env-config", default=str(ROOT / "configs" / "runtime_envs.yaml"))
    parser.add_argument("--run-preflight", action="store_true", help="Run specialist preflight through each assigned env.")
    args = parser.parse_args()

    config_path = Path(args.runtime_env_config).resolve()
    payload = load_yaml(config_path)
    env_groups = dict(payload.get("env_groups", {}))
    method_runtime = {str(key): dict(value) for key, value in dict(payload.get("method_runtime", {})).items()}
    primary_assigned = _assigned_methods(method_runtime, key="primary_env_group")
    fallback_assigned = _assigned_methods(method_runtime, key="fallback_env_group")

    print(f"runtime_env_config={config_path}")
    for env_group, group_payload in env_groups.items():
        primary_methods = primary_assigned.get(env_group, [])
        fallback_methods = fallback_assigned.get(env_group, [])
        representative = primary_methods[0] if primary_methods else (fallback_methods[0] if fallback_methods else "ridge")
        resolved_mode = "fallback" if group_payload.get("role") == "fallback_upstream" else "primary"
        resolution = resolve_runtime_env(
            config_path=config_path,
            method_name=representative,
            method_family="specialist_baseline" if representative in {"gears", "cpa", "cellot", "gperturb", "xpert", "transigen"} else "universal_baseline",
            current_python=sys.executable,
            runtime_mode=resolved_mode,
            forced_env_group=env_group,
        )
        print(f"[{env_group}]")
        print(f"  display_name={group_payload.get('display_name')}")
        print(f"  role={group_payload.get('role')}")
        print(f"  runtime_mode={resolved_mode}")
        print(f"  python={resolution.python_executable}")
        print(f"  available={resolution.available}")
        print(f"  should_dispatch={resolution.should_dispatch}")
        print(f"  gpu_status={group_payload.get('gpu_status')}")
        print(f"  primary_methods={','.join(primary_methods) if primary_methods else '<none>'}")
        print(f"  fallback_methods={','.join(fallback_methods) if fallback_methods else '<none>'}")
        if resolution.reason:
            print(f"  reason={resolution.reason}")

    if args.run_preflight:
        print("preflight_results:")
        for method_name in ["GEARS", "CPA", "CellOT", "GPerturb", "XPert", "TranSiGen"]:
            code = _run_preflight(method_name, config_path)
            print(f"  {method_name}: exit_code={code}")


if __name__ == "__main__":
    main()
