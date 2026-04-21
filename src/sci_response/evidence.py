from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

from sci_response.baselines.artifacts import beijing_timestamp_slug
from sci_response.data.io import dump_json, ensure_dir, write_text
from sci_response.pathing import repo_relative_str, repo_root as default_repo_root


ANALYSIS_ROOT_NAME = "analysis_runs"


def repo_root(root: Path | None = None) -> Path:
    return (root or default_repo_root()).resolve()


def analysis_root(root: Path | None = None) -> Path:
    return ensure_dir(repo_root(root) / ANALYSIS_ROOT_NAME)


def create_analysis_run(
    analysis_name: str,
    *,
    root: Path | None = None,
    dataset_key: str | None = None,
    latest_marker_name: str | None = None,
) -> tuple[Path, Path, str]:
    base_root = analysis_root(root) / analysis_name
    if dataset_key:
        base_root = base_root / dataset_key
    base_root = ensure_dir(base_root)
    run_id = beijing_timestamp_slug()
    history_root = ensure_dir(base_root / "history" / run_id)
    latest_root = ensure_dir(base_root / "latest")
    marker_name = latest_marker_name or f"LATEST_{analysis_name.upper()}.txt"
    write_text(base_root / marker_name, repo_relative_str(history_root) + "\n")
    return history_root, latest_root, run_id


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], preferred_order: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = list({key for row in rows for key in row.keys()})
    if preferred_order:
        extras = [key for key in sorted(existing_keys) if key not in preferred_order]
        fieldnames = [key for key in preferred_order if key in existing_keys] + extras
    else:
        fieldnames = sorted(existing_keys)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_csv_to_run(
    history_root: Path,
    latest_root: Path,
    filename: str,
    rows: Sequence[Dict[str, Any]],
    preferred_order: Sequence[str] | None = None,
) -> None:
    write_csv(history_root / filename, rows, preferred_order)
    write_csv(latest_root / filename, rows, preferred_order)


def write_json_to_run(history_root: Path, latest_root: Path, filename: str, payload: Dict[str, Any]) -> None:
    dump_json(history_root / filename, payload)
    dump_json(latest_root / filename, payload)


def write_text_to_run(history_root: Path, latest_root: Path, filename: str, text: str) -> None:
    write_text(history_root / filename, text)
    write_text(latest_root / filename, text)


def copy_if_exists(history_root: Path, latest_root: Path, src: Path, dst_name: str | None = None) -> None:
    if not src.exists():
        return
    dst_filename = dst_name or src.name
    history_dst = history_root / dst_filename
    latest_dst = latest_root / dst_filename
    history_dst.parent.mkdir(parents=True, exist_ok=True)
    latest_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, history_dst)
    shutil.copy2(src, latest_dst)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_seed_from_session_id(session_id: str | None) -> int | None:
    if not session_id:
        return None
    marker = "seed"
    if marker not in session_id:
        return None
    suffix = session_id.split(marker)[-1]
    digits = []
    for char in suffix:
        if char.isdigit():
            digits.append(char)
        else:
            break
    if not digits:
        return None
    return int("".join(digits))


def safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return float(sum(values) / len(values))


def format_float(value: float | None, digits: int = 6) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}g}"
