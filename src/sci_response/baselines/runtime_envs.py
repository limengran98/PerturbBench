from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from sci_response.data.io import load_yaml


@dataclass(frozen=True)
class RuntimeEnvResolution:
    method_name: str
    method_family: str
    execution_policy: str
    implementation_track: str
    primary_env_group: str
    fallback_env_group: str | None
    runtime_mode: str
    env_group: str
    display_name: str
    python_executable: str | None
    available: bool
    should_dispatch: bool
    reason: str | None
    source: str
    config_path: Path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method_name": self.method_name,
            "method_family": self.method_family,
            "execution_policy": self.execution_policy,
            "implementation_track": self.implementation_track,
            "primary_env_group": self.primary_env_group,
            "fallback_env_group": self.fallback_env_group,
            "runtime_mode": self.runtime_mode,
            "env_group": self.env_group,
            "display_name": self.display_name,
            "python_executable": self.python_executable,
            "available": self.available,
            "should_dispatch": self.should_dispatch,
            "reason": self.reason,
            "source": self.source,
            "config_path": str(self.config_path),
        }


def _safe_resolve_executable(raw: str | None) -> str | None:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    candidate = Path(value).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    located = shutil.which(value)
    if located:
        return str(Path(located).resolve())
    return value


def resolve_runtime_env(
    *,
    config_path: Path,
    method_name: str,
    method_family: str,
    current_python: str | None = None,
    runtime_mode: str | None = None,
    forced_env_group: str | None = None,
) -> RuntimeEnvResolution:
    current_python_value = _safe_resolve_executable(current_python or sys.executable) or sys.executable
    config = load_yaml(config_path)
    env_groups = dict(config.get("env_groups", {}))
    method_runtime = {
        str(key).lower(): dict(value) for key, value in dict(config.get("method_runtime", {})).items()
    }
    default_env_group = str(config.get("default_env_group", "shared"))
    default_execution_policy = str(config.get("default_execution_policy", "shared_runtime_first"))
    normalized_method = str(method_name).strip().lower()
    method_payload = dict(method_runtime.get(normalized_method, {}))
    execution_policy = str(method_payload.get("execution_policy", default_execution_policy))
    implementation_track = str(method_payload.get("implementation_track", "shared_runtime_native"))
    primary_env_group = str(method_payload.get("primary_env_group", default_env_group))
    fallback_env_group_value = method_payload.get("fallback_env_group")
    fallback_env_group = str(fallback_env_group_value) if fallback_env_group_value else None
    requested_runtime_mode = str(runtime_mode or os.environ.get("SCI_RESPONSE_RUNTIME_MODE", "primary")).strip().lower()
    forced_env_group_value = forced_env_group or os.environ.get("SCI_RESPONSE_RUNTIME_FORCE_ENV_GROUP")

    if forced_env_group_value:
        env_group = str(forced_env_group_value)
    elif requested_runtime_mode in {"fallback", "upstream"} and fallback_env_group:
        env_group = fallback_env_group
    else:
        env_group = primary_env_group

    group_payload = dict(env_groups.get(env_group, {}))
    selection = dict(group_payload.get("selection", {}))
    mode = str(selection.get("mode", "current_or_env"))
    env_var_name = selection.get("python_executable_env")

    python_executable: str | None = None
    source = "current_process"
    if isinstance(env_var_name, str) and env_var_name:
        python_executable = _safe_resolve_executable(os.environ.get(env_var_name))
        if python_executable:
            source = f"env_var:{env_var_name}"

    if python_executable is None and mode == "current_or_env":
        python_executable = current_python_value
        source = "current_process"

    available = python_executable is not None
    reason = None
    if not available:
        if mode == "env_only" and env_var_name:
            reason = (
                f"runtime env group {env_group!r} requires ${env_var_name} to point to a Python executable"
            )
        else:
            reason = f"could not resolve a Python executable for runtime env group {env_group!r}"
    elif Path(python_executable).exists():
        available = True
    elif Path(python_executable).anchor:
        available = False
        reason = f"configured Python executable does not exist: {python_executable}"

    current_path = Path(current_python_value).resolve()
    target_path = Path(python_executable).resolve() if available and python_executable else current_path
    should_dispatch = bool(available and target_path != current_path)

    return RuntimeEnvResolution(
        method_name=normalized_method,
        method_family=str(method_family),
        execution_policy=execution_policy,
        implementation_track=implementation_track,
        primary_env_group=primary_env_group,
        fallback_env_group=fallback_env_group,
        runtime_mode=requested_runtime_mode,
        env_group=env_group,
        display_name=str(group_payload.get("display_name", env_group)),
        python_executable=python_executable,
        available=bool(available),
        should_dispatch=should_dispatch,
        reason=reason,
        source=source,
        config_path=config_path.resolve(),
    )
