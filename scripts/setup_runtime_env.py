#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sci_response.data.io import load_yaml


def _load_config(path: Path) -> Dict[str, Any]:
    return load_yaml(path)


def _print_cmd(cmd: List[str]) -> None:
    print("$ " + " ".join(cmd))


def _run(cmd: List[str], *, execute: bool) -> None:
    _print_cmd(cmd)
    if execute:
        subprocess.run(cmd, check=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare a fallback runtime environment plan for specialist baselines.")
    parser.add_argument("--runtime-env-config", default=str(ROOT / "configs" / "runtime_envs.yaml"))
    parser.add_argument("--env-group", required=True)
    parser.add_argument("--python-executable", default=None, help="Interpreter to use for venv-style env groups.")
    parser.add_argument("--venv-root", default=str(ROOT / ".venvs"))
    parser.add_argument("--execute", action="store_true", help="Actually create the environment instead of printing commands.")
    args = parser.parse_args()

    config_path = Path(args.runtime_env_config).resolve()
    payload = _load_config(config_path)
    env_groups = dict(payload.get("env_groups", {}))
    if args.env_group not in env_groups:
        raise SystemExit(f"Unknown env group {args.env_group!r} in {config_path}")

    group = dict(env_groups[args.env_group])
    setup = dict(group.get("setup", {}))
    kind = str(setup.get("kind", "existing"))

    print(f"env_group={args.env_group}")
    print(f"display_name={group.get('display_name')}")
    print(f"description={group.get('description')}")
    print(f"role={group.get('role')}")
    print(f"gpu_status={group.get('gpu_status')}")
    print(f"gpu_note={group.get('gpu_note')}")

    if kind == "existing":
        print("setup_kind=existing")
        print("No setup action is required; this env group reuses the current/shared environment.")
        return

    if kind == "venv":
        python_executable = args.python_executable
        if not python_executable:
            raise SystemExit(
                f"{args.env_group} requires --python-executable. "
                f"Recommended interpreter: Python {setup.get('recommended_python')}"
            )
        venv_root = Path(args.venv_root).resolve()
        venv_dir = venv_root / args.env_group
        requirements_file = ROOT / str(setup["requirements_file"])
        _run([python_executable, "-m", "venv", str(venv_dir)], execute=bool(args.execute))
        _run([str(venv_dir / "bin" / "python"), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], execute=bool(args.execute))
        _run([str(venv_dir / "bin" / "python"), "-m", "pip", "install", "-r", str(requirements_file)], execute=bool(args.execute))
        print("notes:")
        for note in setup.get("notes", []):
            print(f"  - {note}")
        print(f"export_hint=export {group['selection']['python_executable_env']}={venv_dir / 'bin' / 'python'}")
        return

    if kind == "conda":
        environment_file = ROOT / str(setup["environment_file"])
        cmd = ["conda", "env", "create", "-f", str(environment_file)]
        _run(cmd, execute=bool(args.execute))
        print("notes:")
        for note in setup.get("notes", []):
            print(f"  - {note}")
        print(
            f"export_hint=export {group['selection']['python_executable_env']}=$CONDA_PREFIX/bin/python"
        )
        return

    raise SystemExit(f"Unsupported setup kind: {kind}")


if __name__ == "__main__":
    main()
