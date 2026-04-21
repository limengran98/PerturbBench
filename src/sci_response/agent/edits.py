from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from sci_response.agent.ir import enumerate_editable_sites


def _resolve_parent(payload: Dict[str, Any], path: str) -> Tuple[Dict[str, Any], str]:
    parts = path.split(".")
    current: Dict[str, Any] = payload
    for part in parts[:-1]:
        next_value = current.get(part)
        if not isinstance(next_value, dict):
            raise KeyError(f"Path is not editable or does not resolve to a mapping: {path}")
        current = next_value
    return current, parts[-1]


def get_value(payload: Dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"Path not found in payload: {path}")
        current = current[part]
    return current


def apply_edit(model_ir: Dict[str, Any], edit: Dict[str, Any]) -> Dict[str, Any]:
    updated = copy.deepcopy(model_ir)
    primitive = str(edit["primitive"])
    path = str(edit["path"])
    parent, key = _resolve_parent(updated, path)
    current_value = parent[key]
    if primitive == "set_scalar":
        parent[key] = edit["value"]
    elif primitive == "toggle_boolean":
        if not isinstance(current_value, bool):
            raise TypeError(f"toggle_boolean requires a bool target: {path}")
        parent[key] = not current_value
    elif primitive == "scale_numeric":
        factor = float(edit["factor"])
        if isinstance(current_value, int) and not isinstance(current_value, bool):
            parent[key] = max(1, int(round(int(current_value) * factor)))
        elif isinstance(current_value, float):
            parent[key] = float(current_value) * factor
        else:
            raise TypeError(f"scale_numeric requires an int or float target: {path}")
    elif primitive == "increment":
        delta = int(edit.get("delta", 1))
        if not isinstance(current_value, int) or isinstance(current_value, bool):
            raise TypeError(f"increment requires an int target: {path}")
        parent[key] = max(0, int(current_value) + delta)
    elif primitive == "cycle_enum":
        choices = list(edit["choices"])
        if current_value not in choices:
            raise ValueError(f"Current value {current_value!r} not present in enum choices for {path}")
        current_index = choices.index(current_value)
        parent[key] = choices[(current_index + int(edit.get("step", 1))) % len(choices)]
    else:
        raise ValueError(f"Unsupported edit primitive: {primitive}")
    return updated


def apply_edit_sequence(model_ir: Dict[str, Any], edits: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    updated = copy.deepcopy(model_ir)
    for edit in edits:
        updated = apply_edit(updated, edit)
    return updated


def describe_edit(edit: Dict[str, Any]) -> str:
    primitive = str(edit["primitive"])
    path = str(edit["path"])
    if primitive == "set_scalar":
        return f"{path}={edit['value']}"
    if primitive == "toggle_boolean":
        return f"toggle({path})"
    if primitive == "scale_numeric":
        return f"{path}*= {edit['factor']}"
    if primitive == "increment":
        return f"{path}+= {edit.get('delta', 1)}"
    if primitive == "cycle_enum":
        return f"cycle({path})"
    return f"{primitive}({path})"


def enumerate_candidate_edits(model_ir: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for site in enumerate_editable_sites(model_ir):
        path = str(site["path"])
        current_value = site["current_value"]
        site_type = str(site["type"])
        primitive_family = set(str(item) for item in site.get("primitive_family", []))
        if site_type == "bool":
            if "toggle_boolean" in primitive_family:
                candidates.append({"primitive": "toggle_boolean", "path": path})
        elif site_type == "enum":
            if "cycle_enum" in primitive_family:
                candidates.append({"primitive": "cycle_enum", "path": path, "choices": list(site["choices"]), "step": 1})
        elif site_type == "int":
            if "scale_numeric" in primitive_family:
                candidates.extend(
                    [
                        {"primitive": "scale_numeric", "path": path, "factor": 0.75},
                        {"primitive": "scale_numeric", "path": path, "factor": 1.25},
                    ]
                )
            if "increment" in primitive_family:
                candidates.append({"primitive": "increment", "path": path, "delta": 1})
                if int(current_value) > 0:
                    candidates.append({"primitive": "increment", "path": path, "delta": -1})
        elif site_type == "float":
            if "scale_numeric" in primitive_family:
                candidates.extend(
                    [
                        {"primitive": "scale_numeric", "path": path, "factor": 0.5},
                        {"primitive": "scale_numeric", "path": path, "factor": 0.8},
                        {"primitive": "scale_numeric", "path": path, "factor": 1.25},
                        {"primitive": "scale_numeric", "path": path, "factor": 2.0},
                    ]
                )
    return candidates


def dedupe_edit_sequences(edit_sequences: Iterable[Sequence[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    deduped: List[List[Dict[str, Any]]] = []
    seen = set()
    for sequence in edit_sequences:
        marker = tuple((item["primitive"], item["path"], repr(item.get("value")), repr(item.get("factor")), repr(item.get("delta")), repr(item.get("choices"))) for item in sequence)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append([dict(item) for item in sequence])
    return deduped
