#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sci_response.baselines.runtime_envs import resolve_runtime_env
from sci_response.baselines.specialists import get_wrapper


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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a non-invasive preflight check for a specialist baseline.")
    parser.add_argument(
        "--baseline",
        required=True,
        help="One of: GEARS, CPA, CellOT, GPerturb, XPert, TranSiGen",
    )
    parser.add_argument(
        "--baseline-root",
        default=str(ROOT / "baseline"),
        help="Directory containing baseline archives or extracted repos.",
    )
    parser.add_argument(
        "--artifacts-root",
        default=str(ROOT / "artifacts"),
        help="Artifact root for preflight reports.",
    )
    parser.add_argument("--run-id", default=None, help="Optional fixed artifact run id.")
    parser.add_argument(
        "--runtime-env-config",
        default=str(ROOT / "configs" / "runtime_envs.yaml"),
        help="Runtime environment grouping config for automatic specialist-env dispatch.",
    )
    parser.add_argument(
        "--runtime-mode",
        default="primary",
        choices=["primary", "fallback", "upstream"],
        help="Check the primary shared-runtime target or the strict-upstream fallback env when configured.",
    )
    parser.add_argument(
        "--runtime-env-group",
        default=None,
        help="Optional explicit env-group override. Use sparingly for debugging.",
    )
    parser.add_argument("--dispatch-depth", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    runtime_env = resolve_runtime_env(
        config_path=Path(args.runtime_env_config).resolve(),
        method_name=str(args.baseline),
        method_family="specialist_baseline",
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
        if runtime_env.available and runtime_env.python_executable is not None:
            forwarded_args = _strip_flag_with_value(list(sys.argv[1:]), "--dispatch-depth")
            completed = subprocess.run(
                [runtime_env.python_executable, str(Path(__file__).resolve()), *forwarded_args, "--dispatch-depth", "1"],
                env={
                    **os.environ,
                    "SCI_RESPONSE_RUNTIME_ENV_GROUP": runtime_env.env_group,
                    "SCI_RESPONSE_RUNTIME_ENV_CONFIG": str(Path(args.runtime_env_config).resolve()),
                    "SCI_RESPONSE_RUNTIME_PYTHON": runtime_env.python_executable,
                    "SCI_RESPONSE_RUNTIME_MODE": str(args.runtime_mode),
                    "SCI_RESPONSE_RUNTIME_EXECUTION_POLICY": runtime_env.execution_policy,
                    "SCI_RESPONSE_RUNTIME_IMPLEMENTATION_TRACK": runtime_env.implementation_track,
                    "SCI_RESPONSE_RUNTIME_PRIMARY_ENV_GROUP": runtime_env.primary_env_group,
                    **({"SCI_RESPONSE_RUNTIME_FALLBACK_ENV_GROUP": runtime_env.fallback_env_group} if runtime_env.fallback_env_group else {}),
                    **({"SCI_RESPONSE_RUNTIME_FORCE_ENV_GROUP": str(args.runtime_env_group)} if args.runtime_env_group else {}),
                },
                check=False,
            )
            raise SystemExit(completed.returncode)

    wrapper = get_wrapper(args.baseline)
    report = wrapper.run_preflight(
        baseline_root=Path(args.baseline_root).resolve(),
        artifacts_root=Path(args.artifacts_root).resolve(),
        run_id=args.run_id,
    )

    print(f"baseline_name: {report['baseline_name']}")
    print(f"execution_policy: {runtime_env.execution_policy}")
    print(f"implementation_track: {runtime_env.implementation_track}")
    print(f"runtime_mode: {runtime_env.runtime_mode}")
    print(f"primary_env_group: {runtime_env.primary_env_group}")
    print(f"fallback_env_group: {runtime_env.fallback_env_group}")
    print(f"runtime_env_group: {runtime_env.env_group}")
    print(f"runtime_env_python: {runtime_env.python_executable}")
    print(f"runtime_env_available: {runtime_env.available}")
    print(f"asset_state: {report['asset_state']}")
    print(f"repo_ready: {report['status']['repo_ready']}")
    print(f"env_ready: {report['status']['env_ready']}")
    print(f"smoke_import_ready: {report['status']['smoke_import_ready']}")
    print(f"help_ready: {report['status']['help_ready']}")
    print(f"config_parse_ready: {report['status']['config_parse_ready']}")
    print(f"artifact_run_dir: {report['artifact_run_dir']}")
    print("blockers:")
    for blocker in report["blockers"]:
        print(f"  - {blocker}")


if __name__ == "__main__":
    main()
