from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from sci_response.data.io import load_json


def load_history_rows(history_root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not history_root.exists():
        return rows
    for path in sorted(history_root.glob("iter_*.json")):
        payload = load_json(path)
        if isinstance(payload, dict):
            rows.append(dict(payload))
    rows.sort(key=lambda item: int(item.get("iteration", 0) or 0))
    return rows


def load_rows_payload(path: Path, *, key: str = "rows") -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    payload = load_json(path)
    rows = payload.get(key, []) if isinstance(payload, dict) else []
    return [dict(item) for item in rows if isinstance(item, dict)]


def load_previous_log_lines(log_path: Path, default_lines: Sequence[str]) -> List[str]:
    if not log_path.exists():
        return list(default_lines)
    return [line.rstrip("\n") for line in log_path.read_text(encoding="utf-8").splitlines()]


def load_previous_session_wall_clock_seconds(session_root: Path) -> float:
    manifest_path = session_root / "agent_session.json"
    if not manifest_path.exists():
        return 0.0
    payload = load_json(manifest_path)
    try:
        return float(payload.get("session_wall_clock_seconds", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def next_iteration_index(history: Sequence[Dict[str, Any]]) -> int:
    if not history:
        return 0
    return 1 + max(int(item.get("iteration", 0) or 0) for item in history)


def baseline_objective_from_history(history: Sequence[Dict[str, Any]]) -> float | None:
    if not history:
        return None
    value = history[0].get("objective_value")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
