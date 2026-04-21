from __future__ import annotations

from pathlib import Path
from typing import Mapping


REPO_ROOT = Path(__file__).resolve().parents[2]


def repo_root() -> Path:
    return REPO_ROOT


def repo_relative_str(path: Path | str) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        normalized = candidate.resolve()
    else:
        normalized = (REPO_ROOT / candidate).resolve()
    try:
        return str(normalized.relative_to(REPO_ROOT))
    except ValueError:
        return str(candidate)


def repo_relative_mapping(paths: Mapping[str, Path | str]) -> dict[str, str]:
    return {str(key): repo_relative_str(value) for key, value in paths.items()}


def source_description_for(prefix: str, path: Path | str) -> str:
    return f"{prefix}::{repo_relative_str(path)}"
