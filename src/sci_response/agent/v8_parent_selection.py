from __future__ import annotations

import json
import re
import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from sci_response.data.io import load_json
from sci_response.data.io import load_yaml


BRANCH_ROOTS_FOR_V8_PARENT = (
    "agent_runs_v8",
    "agent_runs_v7",
    "agent_runs_v6",
    "agent_runs_v5",
    "agent_runs_v4",
    "agent_runs_v3_msf",
    "agent_runs_v3",
    "agent_runs_v2",
    "agent_runs",
    "direct_code_runs",
    "random_edit_runs",
)

BASELINE_TABLE_SPECS = (
    (
        "paper/benchmark/latest/tables/example_baseline_universal_mse_table.csv",
        {
            "Ridge": "ridge",
            "ElasticNet": "elasticnet",
            "XGBoost": "xgboost",
            "CatBoost": "catboost",
            "MLP": "mlp",
            "ResNetMLP": "resnet_mlp",
            "FTTransformer": "ft_transformer",
        },
    ),
    (
        "paper/benchmark/latest/tables/example_baseline_specialist_mse_table.csv",
        {
            "GPerturb": "gperturb",
            "GEARS": "gears",
            "CPA": "cpa",
            "CellOT": "cellot",
            "XPert": "xpert",
            "TranSiGen": "transigen",
        },
    ),
)


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _infer_generated_class_name(generated_code_path: Path) -> str | None:
    try:
        text = generated_code_path.read_text(encoding="utf-8")
    except Exception:
        return None
    generated_matches = re.findall(r"^class\s+(Generated[A-Za-z0-9_]+)\b", text, flags=re.MULTILINE)
    if generated_matches:
        return str(generated_matches[0])
    generic_matches = re.findall(r"^class\s+([A-Za-z_][A-Za-z0-9_]*)\b", text, flags=re.MULTILINE)
    if generic_matches:
        return str(generic_matches[0])
    return None


def _candidate_roots(repo_root: Path) -> list[Path]:
    roots: list[Path] = []
    for name in BRANCH_ROOTS_FOR_V8_PARENT:
        path = repo_root / name
        if path.exists():
            roots.append(path)
    return roots


def _session_manifest_matches(
    manifest: Mapping[str, Any],
    *,
    dataset_key: str,
    seed_model_name: str,
) -> bool:
    if str(manifest.get("dataset_key", "")).strip() != dataset_key:
        return False
    manifest_seed_model_name = str(manifest.get("seed_model_name", "")).strip()
    return manifest_seed_model_name == seed_model_name


def _best_record_candidate(
    best_iteration_path: Path,
    *,
    branch_root_name: str,
    dataset_key: str,
    seed_model_name: str,
    current_session_id: str | None,
) -> Dict[str, Any] | None:
    try:
        best_record = load_json(best_iteration_path)
    except Exception:
        return None
    generated_code_path = Path(str(best_record.get("generated_code_path") or ""))
    if not generated_code_path.exists():
        return None
    generated_class_name = _infer_generated_class_name(generated_code_path)
    if not generated_class_name:
        return None
    session_root = best_iteration_path.parent
    agent_session_path = session_root / "agent_session.json"
    if not agent_session_path.exists():
        return None
    try:
        manifest = load_json(agent_session_path)
    except Exception:
        return None
    if current_session_id and str(manifest.get("session_id", "")).strip() == current_session_id:
        return None
    if not _session_manifest_matches(manifest, dataset_key=dataset_key, seed_model_name=seed_model_name):
        return None
    objective_value = _safe_float(best_record.get("objective_value"))
    if objective_value is None:
        return None
    return {
        "branch_root": branch_root_name,
        "session_root": str(session_root.resolve()),
        "session_id": str(manifest.get("session_id", "")),
        "agent_line": str(manifest.get("agent_line", "")),
        "agent_mode": str(manifest.get("agent_mode", "") or manifest.get("agent_variant", "")),
        "generated_code_path": str(generated_code_path.resolve()),
        "generated_class_name": str(generated_class_name),
        "objective_value": float(objective_value),
        "iteration": _safe_int(best_record.get("iteration")) or 0,
        "structural_signature": best_record.get("structural_signature"),
        "mechanism_title": best_record.get("mechanism_title"),
        "payload": best_record,
    }


def practical_parent_candidates_v8(
    *,
    repo_root: Path,
    dataset_key: str,
    seed_model_name: str,
    current_session_id: str | None,
) -> List[Dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for branch_root in _candidate_roots(repo_root):
        for best_iteration_path in branch_root.rglob("best_iteration.json"):
            candidate = _best_record_candidate(
                best_iteration_path,
                branch_root_name=branch_root.name,
                dataset_key=dataset_key,
                seed_model_name=seed_model_name,
                current_session_id=current_session_id,
            )
            if candidate is not None:
                candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            float(item["objective_value"]),
            0 if str(item["branch_root"]).startswith("direct_code_runs") else 1,
            str(item["branch_root"]),
            str(item["session_id"]),
        )
    )
    return candidates


def choose_practical_parent_v8(
    *,
    repo_root: Path,
    dataset_key: str,
    seed_model_name: str,
    current_session_id: str | None,
    top_k: int = 5,
) -> Dict[str, Any]:
    candidates = practical_parent_candidates_v8(
        repo_root=repo_root,
        dataset_key=dataset_key,
        seed_model_name=seed_model_name,
        current_session_id=current_session_id,
    )
    if not candidates:
        return {
            "selected": None,
            "top_candidates": [],
            "selection_reason": "no_historical_codegen_parent_available",
        }
    selected = dict(candidates[0])
    return {
        "selected": selected,
        "top_candidates": candidates[: int(top_k)],
        "selection_reason": "best_historical_codegen_parent",
    }


def parent_candidates_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "selection_reason": payload.get("selection_reason"),
            "selected": payload.get("selected"),
            "top_candidates": payload.get("top_candidates", []),
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def _deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def choose_champion_baseline_v8(
    *,
    repo_root: Path,
    dataset_key: str,
) -> Dict[str, Any]:
    matrix_path = repo_root / "configs" / "benchmark_matrix.yaml"
    if not matrix_path.exists():
        return {
            "selected": None,
            "top_candidates": [],
            "selection_reason": "benchmark_matrix_missing",
        }
    matrix = load_yaml(matrix_path)
    dataset_entry = dict(dict(matrix.get("datasets", {})).get(dataset_key, {}))
    if not dataset_entry:
        return {
            "selected": None,
            "top_candidates": [],
            "selection_reason": "dataset_not_registered_in_benchmark_matrix",
        }
    runnable_methods = {str(item) for item in dataset_entry.get("runnable_methods", [])}
    baseline_defaults = dict(matrix.get("baseline_defaults", {}))
    method_overrides = dict(dataset_entry.get("method_overrides", {}))
    candidates: list[dict[str, Any]] = []
    for relative_path, column_mapping in BASELINE_TABLE_SPECS:
        table_path = repo_root / relative_path
        if not table_path.exists():
            continue
        with table_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            dataset_row = next((dict(row) for row in reader if str(row.get("dataset_key", "")).strip() == dataset_key), None)
        if dataset_row is None:
            continue
        for column_name, method_name in column_mapping.items():
            if method_name not in runnable_methods:
                continue
            raw_value = str(dataset_row.get(column_name, "")).strip()
            if not raw_value:
                continue
            metric_value = _safe_float(raw_value)
            if metric_value is None:
                continue
            defaults = dict(baseline_defaults.get(method_name, {}))
            if not defaults:
                continue
            merged_defaults = _deep_update(defaults, dict(method_overrides.get(method_name, {})))
            candidates.append(
                {
                    "method_name": method_name,
                    "display_name": column_name,
                    "family": str(merged_defaults.get("family", "")),
                    "mse_mean": float(metric_value),
                    "params": dict(merged_defaults.get("params", {})),
                    "source_table": str(table_path.resolve()),
                }
            )
    candidates.sort(key=lambda item: (float(item["mse_mean"]), str(item["method_name"])))
    selected = dict(candidates[0]) if candidates else None
    return {
        "selected": selected,
        "top_candidates": candidates[:5],
        "selection_reason": "best_example_baseline_mse" if selected is not None else "no_completed_baseline_candidate_available",
    }
