from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def resolve_codeevo_root(repo_root: Path) -> Path:
    path = (repo_root / "legacy" / "CodeEvo").resolve()
    if not path.exists():
        raise FileNotFoundError(f"CodeEvo root not found: {path}")
    return path


def list_codeevo_configs(repo_root: Path) -> List[Path]:
    codeevo_root = resolve_codeevo_root(repo_root)
    return sorted(codeevo_root.glob("config*.json"))


def build_codeevo_command(repo_root: Path, *, config_path: Path | None = None, extra_args: Iterable[str] = ()) -> List[str]:
    codeevo_root = resolve_codeevo_root(repo_root)
    command = ["python3", str(codeevo_root / "main_agent.py")]
    if config_path is not None:
        command.extend(["--config", str(config_path.resolve())])
    command.extend(str(arg) for arg in extra_args)
    return command
