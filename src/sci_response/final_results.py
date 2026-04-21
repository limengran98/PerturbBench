from __future__ import annotations

import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from sci_response.data.io import ensure_dir, load_json, load_yaml
from sci_response.pathing import repo_root as default_repo_root


DEFAULT_SEEDS: tuple[int, ...] = (11, 12, 13)
METRIC_FIELDS: list[tuple[str, str]] = [
    ("mae", "test.delta.mae"),
    ("mse", "test.delta.mse"),
    ("pearson", "test.delta.pearson"),
    ("r2", "test.delta.r2"),
    ("spearman", "test.delta.spearman"),
    ("topk_overlap", "test.delta.topk_overlap"),
]
FINAL_METRIC_FILENAME = "final_metric_result.csv"
FINAL_BUDGET_FILENAME = "final_budget_result.csv"
STATIC_BRANCH_ROOT_NAMES = ("artifacts", "agent_runs", "direct_code_runs", "random_edit_runs", "hpo_runs")
VERSIONED_AGENT_ROOT_PATTERN = re.compile(r"^agent_runs_v(?P<version>\d+)(?:_(?P<suffix>[A-Za-z0-9_]+))?$")


@dataclass(frozen=True)
class IterativeBranchSpec:
    root_name: str
    comparison_group: str
    expected_modes: tuple[str, ...]
    default_seed_model_name: str


def _repo_root(root: Path | None = None) -> Path:
    return (root or default_repo_root()).resolve()


def _ensure_large_csv_field_limit() -> None:
    limit = int(sys.maxsize)
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = max(131072, limit // 10)


def _candidate_attempt_budget_limit(max_completed_evaluations: int) -> int:
    return int(max(1, int(max_completed_evaluations)) * 2)


def _load_defaults(repo_root: Path) -> Dict[str, Any]:
    return load_yaml(repo_root / "configs" / "agent" / "defaults.yaml")


def _load_matrix(repo_root: Path) -> Dict[str, Any]:
    return load_yaml(repo_root / "configs" / "benchmark_matrix.yaml")


def _public_main_dataset_keys(matrix: Mapping[str, Any]) -> list[str]:
    scope = dict(matrix.get("benchmark_scope", {}))
    return [str(item) for item in scope.get("public_main_datasets", [])]


def _dataset_entry(matrix: Mapping[str, Any], dataset_key: str) -> Dict[str, Any]:
    return dict(dict(matrix.get("datasets", {})).get(dataset_key, {}))


def _primary_and_supplementary_protocols(dataset_entry: Mapping[str, Any]) -> list[str]:
    protocols = [str(dataset_entry.get("primary_protocol", ""))]
    protocols.extend(str(item.get("protocol")) for item in dataset_entry.get("supplementary_protocols", []) if item.get("protocol"))
    return [item for item in protocols if item]


def _expected_seed_model_name_from_config(config_value: str) -> str:
    return Path(str(config_value)).stem


def _versioned_agent_root_sort_key(root_name: str) -> tuple[int, str]:
    match = VERSIONED_AGENT_ROOT_PATTERN.match(str(root_name))
    if not match:
        return (10_000, str(root_name))
    return (int(match.group("version")), str(root_name))


def _is_supported_versioned_agent_root(root_name: str) -> bool:
    match = VERSIONED_AGENT_ROOT_PATTERN.match(str(root_name))
    if match is None:
        return False
    suffix = str(match.group("suffix") or "").lower()
    if "smoke" in suffix or "pilot" in suffix or "pack" in suffix:
        return False
    return True


def _discover_branch_root_names(repo_root: Path) -> list[str]:
    branch_names: list[str] = [name for name in STATIC_BRANCH_ROOT_NAMES if (repo_root / name).exists()]
    versioned_agent_roots = sorted(
        [
            path.name
            for path in repo_root.iterdir()
            if path.is_dir() and _is_supported_versioned_agent_root(path.name)
        ],
        key=_versioned_agent_root_sort_key,
    )
    branch_names.extend(versioned_agent_roots)
    return branch_names


def _comparison_group_for_agent_root(root_name: str) -> str:
    if root_name == "agent_runs":
        return "main_agent"
    match = VERSIONED_AGENT_ROOT_PATTERN.match(str(root_name))
    if match is None:
        return "main_agent"
    return f"main_agent_v{int(match.group('version'))}"


def _discover_modes_for_agent_branch(branch_root: Path, comparison_group: str) -> tuple[str, ...]:
    modes: set[str] = set()
    for agent_session_path in branch_root.rglob("agent_session.json"):
        session_manifest = load_json(agent_session_path)
        if str(session_manifest.get("agent_line") or "") != comparison_group:
            continue
        mode = str(session_manifest.get("agent_mode") or session_manifest.get("agent_variant") or "").strip()
        if mode:
            modes.add(mode)
    return tuple(sorted(modes))


def _iterative_branch_specs(repo_root: Path) -> list[IterativeBranchSpec]:
    defaults = _load_defaults(repo_root)
    default_seed_model_name = _expected_seed_model_name_from_config(
        str(defaults.get("seed_model_config", "configs/models/conditioned_residual.yaml"))
    )
    specs: list[IterativeBranchSpec] = []

    for branch_root_name in _discover_branch_root_names(repo_root):
        if branch_root_name == "artifacts":
            continue
        branch_root = repo_root / branch_root_name
        if branch_root_name == "agent_runs" or VERSIONED_AGENT_ROOT_PATTERN.match(branch_root_name):
            comparison_group = _comparison_group_for_agent_root(branch_root_name)
            discovered_modes = _discover_modes_for_agent_branch(branch_root, comparison_group)
            fallback_modes: tuple[str, ...]
            if branch_root_name == "agent_runs":
                fallback_modes = (str(defaults.get("agent_mode", "structured_llm_search")),)
            else:
                version = int(VERSIONED_AGENT_ROOT_PATTERN.match(branch_root_name).group("version"))  # type: ignore[union-attr]
                fallback_modes = (f"structured_llm_search_v{version}",)
            specs.append(
                IterativeBranchSpec(
                    root_name=branch_root_name,
                    comparison_group=comparison_group,
                    expected_modes=discovered_modes or fallback_modes,
                    default_seed_model_name=default_seed_model_name,
                )
            )
            continue
        if branch_root_name == "direct_code_runs":
            specs.append(
                IterativeBranchSpec(
                    root_name=branch_root_name,
                    comparison_group="direct_code_llm",
                    expected_modes=("direct_code_llm_singleshot", "direct_code_llm_repairloop"),
                    default_seed_model_name=default_seed_model_name,
                )
            )
            continue
        if branch_root_name == "random_edit_runs":
            specs.append(
                IterativeBranchSpec(
                    root_name=branch_root_name,
                    comparison_group="random_edit_search",
                    expected_modes=("random_edit_uniform", "random_edit_stratified"),
                    default_seed_model_name=default_seed_model_name,
                )
            )
            continue
        if branch_root_name == "hpo_runs":
            specs.append(
                IterativeBranchSpec(
                    root_name=branch_root_name,
                    comparison_group="hpo_search",
                    expected_modes=("hpo_optuna_tpe", "hpo_flaml_cfo"),
                    default_seed_model_name=_expected_seed_model_name_from_config(
                        str(defaults.get("hpo_seed_model_config", "configs/models/structured_hypothesis_hpo.yaml"))
                    ),
                )
            )
    return specs


def _iterative_spec_map(repo_root: Path) -> dict[str, IterativeBranchSpec]:
    return {spec.root_name: spec for spec in _iterative_branch_specs(repo_root)}


def _fallback_iterative_spec_for_custom_branch(
    *,
    repo_root: Path,
    branch_root_name: str,
    comparison_group: str,
) -> IterativeBranchSpec | None:
    defaults = _load_defaults(repo_root)
    default_seed_model_name = _expected_seed_model_name_from_config(
        str(defaults.get("seed_model_config", "configs/models/conditioned_residual.yaml"))
    )
    if comparison_group == "direct_code_llm" and str(branch_root_name).startswith("direct_code_runs"):
        return IterativeBranchSpec(
            root_name=str(branch_root_name),
            comparison_group="direct_code_llm",
            expected_modes=("direct_code_llm_singleshot", "direct_code_llm_repairloop"),
            default_seed_model_name=default_seed_model_name,
        )
    if comparison_group == "random_edit_search" and str(branch_root_name).startswith("random_edit_runs"):
        return IterativeBranchSpec(
            root_name=str(branch_root_name),
            comparison_group="random_edit_search",
            expected_modes=("random_edit_uniform", "random_edit_stratified"),
            default_seed_model_name=default_seed_model_name,
        )
    if comparison_group == "hpo_search" and str(branch_root_name).startswith("hpo_runs"):
        return IterativeBranchSpec(
            root_name=str(branch_root_name),
            comparison_group="hpo_search",
            expected_modes=("hpo_optuna_tpe", "hpo_flaml_cfo"),
            default_seed_model_name=_expected_seed_model_name_from_config(
                str(defaults.get("hpo_seed_model_config", "configs/models/structured_hypothesis_hpo.yaml"))
            ),
        )
    return None


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    return numeric


def _mean_std(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean_value = sum(values) / len(values)
    variance = sum((value - mean_value) ** 2 for value in values) / len(values)
    return float(mean_value), float(math.sqrt(variance))


def _format_mean_std(mean_value: float | None, std_value: float | None) -> str:
    if mean_value is None:
        return ""
    if std_value is None:
        return f"{mean_value:.6g}"
    return f"{mean_value:.6g} ± {std_value:.6g}"


def _parse_seed_from_session_id(session_id: str | None) -> int | None:
    if not session_id:
        return None
    match = re.search(r"(?:^|__)seed(\d+)$", str(session_id))
    if match:
        return int(match.group(1))
    fallback = re.search(r"seed(\d+)", str(session_id))
    if fallback:
        return int(fallback.group(1))
    return None


def _parse_iso_duration_seconds(start_time: Any, end_time: Any) -> float | None:
    if not start_time or not end_time:
        return None
    try:
        start_dt = datetime.fromisoformat(str(start_time))
        end_dt = datetime.fromisoformat(str(end_time))
    except ValueError:
        return None
    return float((end_dt - start_dt).total_seconds())


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    _ensure_large_csv_field_limit()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _attempted_iteration_count(iteration_rows: Mapping[int, Mapping[str, Any]]) -> int:
    if not iteration_rows:
        return 0
    return int(max(int(raw_iteration) for raw_iteration in iteration_rows) + 1)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], preferred_order: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_keys = list({key for row in rows for key in row.keys()})
    extras = [key for key in sorted(existing_keys) if key not in preferred_order]
    fieldnames = [key for key in preferred_order if key in existing_keys] + extras
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric_table_column_order() -> list[str]:
    ordered = [
        "comparison_group",
        "comparison_method",
        "branch_root",
        "dataset_key",
        "dataset_display_name",
        "protocol",
        "method_name",
        "seed_model_name",
        "iteration",
        "expected_seed_count",
        "completed_seed_count",
        "row_status",
        "available_seeds_json",
        "missing_seeds_json",
    ]
    for metric_name, _ in METRIC_FIELDS:
        ordered.extend([metric_name, f"{metric_name}_mean", f"{metric_name}_std"])
    return ordered


def _budget_table_column_order() -> list[str]:
    return [
        "comparison_group",
        "comparison_method",
        "branch_root",
        "dataset_key",
        "dataset_display_name",
        "protocol",
        "method_name",
        "seed_model_name",
        "iteration",
        "expected_seed_count",
        "completed_seed_count",
        "row_status",
        "available_seeds_json",
        "missing_seeds_json",
        "budget_limit_completed_evaluations",
        "candidate_attempt_budget_limit",
        "stopping_rule_values",
        "stopping_reason_values",
        "evaluation_wall_clock_seconds",
        "evaluation_wall_clock_seconds_mean",
        "evaluation_wall_clock_seconds_std",
        "session_wall_clock_seconds",
        "session_wall_clock_seconds_mean",
        "session_wall_clock_seconds_std",
        "cumulative_completed_evaluations",
        "cumulative_completed_evaluations_mean",
        "cumulative_completed_evaluations_std",
        "cumulative_failed_candidates",
        "cumulative_failed_candidates_mean",
        "cumulative_failed_candidates_std",
        "iteration_llm_request_count",
        "iteration_llm_request_count_mean",
        "iteration_llm_request_count_std",
        "iteration_llm_total_tokens",
        "iteration_llm_total_tokens_mean",
        "iteration_llm_total_tokens_std",
        "iteration_llm_repair_request_count",
        "iteration_llm_repair_request_count_mean",
        "iteration_llm_repair_request_count_std",
        "cumulative_llm_request_count",
        "cumulative_llm_request_count_mean",
        "cumulative_llm_request_count_std",
        "cumulative_llm_total_tokens",
        "cumulative_llm_total_tokens_mean",
        "cumulative_llm_total_tokens_std",
        "cumulative_llm_repair_request_count",
        "cumulative_llm_repair_request_count_mean",
        "cumulative_llm_repair_request_count_std",
    ]


def _row_status(completed_seed_count: int, expected_seed_count: int) -> str:
    if completed_seed_count >= expected_seed_count:
        return "completed"
    if completed_seed_count > 0:
        return "partial"
    return "missing"


def _aggregate_metric_record(
    *,
    comparison_group: str,
    comparison_method: str,
    branch_root_name: str,
    dataset_key: str,
    dataset_display_name: str,
    protocol: str,
    method_name: str | None,
    seed_model_name: str | None,
    iteration: int,
    observed_by_seed: Mapping[int, Mapping[str, float | None]],
    expected_seeds: Sequence[int],
) -> Dict[str, Any]:
    completed_seeds = [seed for seed in expected_seeds if seed in observed_by_seed]
    row: Dict[str, Any] = {
        "comparison_group": comparison_group,
        "comparison_method": comparison_method,
        "branch_root": branch_root_name,
        "dataset_key": dataset_key,
        "dataset_display_name": dataset_display_name,
        "protocol": protocol,
        "method_name": method_name,
        "seed_model_name": seed_model_name,
        "iteration": int(iteration),
        "expected_seed_count": int(len(expected_seeds)),
        "completed_seed_count": int(len(completed_seeds)),
        "row_status": _row_status(len(completed_seeds), len(expected_seeds)),
        "available_seeds_json": json.dumps(completed_seeds, ensure_ascii=True),
        "missing_seeds_json": json.dumps([seed for seed in expected_seeds if seed not in observed_by_seed], ensure_ascii=True),
    }
    for metric_name, _metric_path in METRIC_FIELDS:
        values = [
            float(payload[metric_name])
            for seed, payload in observed_by_seed.items()
            if seed in expected_seeds and payload.get(metric_name) is not None
        ]
        mean_value, std_value = _mean_std(values)
        row[metric_name] = _format_mean_std(mean_value, std_value)
        row[f"{metric_name}_mean"] = mean_value
        row[f"{metric_name}_std"] = std_value
    return row


def _aggregate_budget_record(
    *,
    comparison_group: str,
    comparison_method: str,
    branch_root_name: str,
    dataset_key: str,
    dataset_display_name: str,
    protocol: str,
    method_name: str | None,
    seed_model_name: str | None,
    iteration: int,
    observed_by_seed: Mapping[int, Mapping[str, Any]],
    expected_seeds: Sequence[int],
) -> Dict[str, Any]:
    completed_seeds = [seed for seed in expected_seeds if seed in observed_by_seed]
    row: Dict[str, Any] = {
        "comparison_group": comparison_group,
        "comparison_method": comparison_method,
        "branch_root": branch_root_name,
        "dataset_key": dataset_key,
        "dataset_display_name": dataset_display_name,
        "protocol": protocol,
        "method_name": method_name,
        "seed_model_name": seed_model_name,
        "iteration": int(iteration),
        "expected_seed_count": int(len(expected_seeds)),
        "completed_seed_count": int(len(completed_seeds)),
        "row_status": _row_status(len(completed_seeds), len(expected_seeds)),
        "available_seeds_json": json.dumps(completed_seeds, ensure_ascii=True),
        "missing_seeds_json": json.dumps([seed for seed in expected_seeds if seed not in observed_by_seed], ensure_ascii=True),
        "stopping_rule_values": "|".join(
            sorted(
                {
                    str(payload.get("stopping_rule"))
                    for payload in observed_by_seed.values()
                    if payload.get("stopping_rule")
                }
            )
        ),
        "stopping_reason_values": "|".join(
            sorted(
                {
                    str(payload.get("stopping_reason"))
                    for payload in observed_by_seed.values()
                    if payload.get("stopping_reason")
                }
            )
        ),
    }
    scalar_specs = [
        "budget_limit_completed_evaluations",
        "candidate_attempt_budget_limit",
        "evaluation_wall_clock_seconds",
        "session_wall_clock_seconds",
        "cumulative_completed_evaluations",
        "cumulative_failed_candidates",
        "iteration_llm_request_count",
        "iteration_llm_total_tokens",
        "iteration_llm_repair_request_count",
        "cumulative_llm_request_count",
        "cumulative_llm_total_tokens",
        "cumulative_llm_repair_request_count",
    ]
    for field in scalar_specs:
        values = [
            float(payload[field])
            for seed, payload in observed_by_seed.items()
            if seed in expected_seeds and payload.get(field) is not None
        ]
        mean_value, std_value = _mean_std(values)
        row[field] = _format_mean_std(mean_value, std_value)
        row[f"{field}_mean"] = mean_value
        row[f"{field}_std"] = std_value
    return row


def _extract_nested_metric(metrics_payload: Mapping[str, Any], dotted_path: str) -> float | None:
    current: Any = metrics_payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return _safe_float(current)


def _baseline_observed_runs(repo_root: Path, branch_root: Path) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    summary_path = branch_root / "global" / "benchmark_summary.csv"
    if not summary_path.exists():
        return {}
    latest: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    pattern = re.compile(r"^datasets/(?P<dataset>[^/]+)/methods/(?P<method>[^/]+)/(?P<protocol>[^/]+)/(?P<stamp>[^/]+)$")
    for row in _load_csv_rows(summary_path):
        run_id = str(row.get("run_id") or "")
        match = pattern.match(run_id)
        if not match:
            continue
        seed = _safe_int(row.get("seed"))
        if seed not in DEFAULT_SEEDS:
            continue
        key = (match.group("dataset"), match.group("method"), match.group("protocol"), int(seed))
        existing = latest.get(key)
        if existing is None or run_id > str(existing.get("run_id") or ""):
            latest[key] = dict(row)
    return latest


def _baseline_metric_budget_rows(repo_root: Path, branch_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matrix = _load_matrix(repo_root)
    observed_runs = _baseline_observed_runs(repo_root, branch_root)
    metric_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    for dataset_key in _public_main_dataset_keys(matrix):
        entry = _dataset_entry(matrix, dataset_key)
        dataset_display_name = str(entry.get("display_name", dataset_key))
        protocols = _primary_and_supplementary_protocols(entry)
        methods = [str(item) for item in entry.get("runnable_methods", [])]
        for protocol in protocols:
            for method_name in methods:
                metric_by_seed: dict[int, dict[str, float | None]] = {}
                budget_by_seed: dict[int, dict[str, Any]] = {}
                for seed in DEFAULT_SEEDS:
                    run_row = observed_runs.get((dataset_key, method_name, protocol, seed))
                    if run_row is None:
                        continue
                    metric_payload = {
                        metric_name: _safe_float(run_row.get(metric_path))
                        for metric_name, metric_path in METRIC_FIELDS
                    }
                    metric_by_seed[int(seed)] = metric_payload
                    manifest_path = branch_root / str(run_row["run_id"]) / "manifest.json"
                    manifest_payload = load_json(manifest_path) if manifest_path.exists() else {}
                    duration_seconds = _parse_iso_duration_seconds(
                        manifest_payload.get("start_time"),
                        manifest_payload.get("end_time"),
                    )
                    budget_by_seed[int(seed)] = {
                        "budget_limit_completed_evaluations": 1.0,
                        "candidate_attempt_budget_limit": 1.0,
                        "evaluation_wall_clock_seconds": duration_seconds,
                        "session_wall_clock_seconds": duration_seconds,
                        "cumulative_completed_evaluations": 1.0,
                        "cumulative_failed_candidates": 0.0,
                        "iteration_llm_request_count": 0.0,
                        "iteration_llm_total_tokens": 0.0,
                        "iteration_llm_repair_request_count": 0.0,
                        "cumulative_llm_request_count": 0.0,
                        "cumulative_llm_total_tokens": 0.0,
                        "cumulative_llm_repair_request_count": 0.0,
                        "stopping_rule": "single_evaluation_baseline",
                        "stopping_reason": "single_evaluation_completed",
                    }
                metric_rows.append(
                    _aggregate_metric_record(
                        comparison_group="baseline",
                        comparison_method=method_name,
                        branch_root_name=branch_root.name,
                        dataset_key=dataset_key,
                        dataset_display_name=dataset_display_name,
                        protocol=protocol,
                        method_name=method_name,
                        seed_model_name=None,
                        iteration=0,
                        observed_by_seed=metric_by_seed,
                        expected_seeds=DEFAULT_SEEDS,
                    )
                )
                budget_rows.append(
                    _aggregate_budget_record(
                        comparison_group="baseline",
                        comparison_method=method_name,
                        branch_root_name=branch_root.name,
                        dataset_key=dataset_key,
                        dataset_display_name=dataset_display_name,
                        protocol=protocol,
                        method_name=method_name,
                        seed_model_name=None,
                        iteration=0,
                        observed_by_seed=budget_by_seed,
                        expected_seeds=DEFAULT_SEEDS,
                    )
                )
    return metric_rows, budget_rows


def _load_llm_usage_by_iteration(trace_root: Path) -> dict[int, dict[str, float]]:
    usage_path = trace_root / "llm_usage.csv"
    if not usage_path.exists():
        return {}
    by_iteration: dict[int, dict[str, float]] = {}
    for row in _load_csv_rows(usage_path):
        iteration = _safe_int(row.get("iteration"))
        if iteration is None:
            continue
        bucket = by_iteration.setdefault(
            int(iteration),
            {
                "llm_request_count": 0.0,
                "llm_total_tokens": 0.0,
                "llm_repair_request_count": 0.0,
            },
        )
        bucket["llm_request_count"] += 1.0
        bucket["llm_total_tokens"] += float(_safe_int(row.get("total_tokens")) or 0)
        if str(row.get("is_repair_request")).lower() == "true":
            bucket["llm_repair_request_count"] += 1.0
    return by_iteration


def _load_iteration_rows_by_index(session_root: Path) -> dict[int, dict[str, Any]]:
    rows_by_index: dict[int, dict[str, Any]] = {}
    for row in _load_csv_rows(session_root / "iterations.csv"):
        iteration = _safe_int(row.get("iteration"))
        if iteration is None:
            continue
        if not row.get("test.delta.topk_overlap"):
            metrics_path = row.get("metrics_path")
            if metrics_path:
                metrics_file = Path(str(metrics_path))
                if metrics_file.exists():
                    metrics_payload = load_json(metrics_file)
                    row["test.delta.topk_overlap"] = _extract_nested_metric(metrics_payload, "test.delta.topk_overlap")
        rows_by_index[int(iteration)] = row
    return rows_by_index


def _select_latest_iterative_sessions(
    repo_root: Path,
    branch_root: Path,
    branch_line: str,
    expected_modes: Sequence[str],
) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    latest: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    expected_mode_set = set(expected_modes)
    for agent_session_path in branch_root.rglob("agent_session.json"):
        session_root = agent_session_path.parent
        session_manifest = load_json(agent_session_path)
        if str(session_manifest.get("agent_line")) != branch_line:
            continue
        mode = str(session_manifest.get("agent_mode") or session_manifest.get("agent_variant") or "")
        if mode not in expected_mode_set:
            continue
        dataset_key = str(session_manifest.get("dataset_key") or "")
        if not dataset_key:
            continue
        session_id = str(session_manifest.get("session_id") or session_root.name)
        seed = _parse_seed_from_session_id(session_id)
        if seed not in DEFAULT_SEEDS:
            continue
        seed_model_name = str(session_manifest.get("seed_model_name") or "")
        key = (dataset_key, mode, seed_model_name, int(seed))
        iteration_rows = _load_iteration_rows_by_index(session_root)
        attempted_iteration_count = _attempted_iteration_count(iteration_rows)
        existing = latest.get(key)
        existing_attempted_iteration_count = int(existing["attempted_iteration_count"]) if existing is not None else -1
        if (
            existing is None
            or attempted_iteration_count > existing_attempted_iteration_count
            or (
                attempted_iteration_count == existing_attempted_iteration_count
                and session_id > str(existing["session_id"])
            )
        ):
            latest[key] = {
                "session_id": session_id,
                "session_root": session_root,
                "session_manifest": session_manifest,
                "iteration_rows": iteration_rows,
                "attempted_iteration_count": int(attempted_iteration_count),
                "llm_usage_by_iteration": _load_llm_usage_by_iteration(session_root / "trace"),
            }
    return latest


def latest_iterative_session_info(
    *,
    repo_root: Path | None,
    branch_root: Path,
    comparison_group: str,
    comparison_method: str,
    dataset_key: str,
    seed_model_name: str,
    seed: int,
) -> dict[str, Any] | None:
    repo_root = _repo_root(repo_root)
    branch_root = Path(branch_root).resolve()
    spec_map = _iterative_spec_map(repo_root)
    spec = spec_map.get(branch_root.name)
    if spec is None or spec.comparison_group != comparison_group:
        spec = _fallback_iterative_spec_for_custom_branch(
            repo_root=repo_root,
            branch_root_name=branch_root.name,
            comparison_group=comparison_group,
        )
    if spec is None or spec.comparison_group != comparison_group:
        return None
    latest_sessions = _select_latest_iterative_sessions(
        repo_root=repo_root,
        branch_root=branch_root,
        branch_line=comparison_group,
        expected_modes=(comparison_method,),
    )
    bundle = latest_sessions.get((dataset_key, comparison_method, seed_model_name, int(seed)))
    if bundle is None:
        return None
    session_manifest = dict(bundle["session_manifest"])
    attempted_iteration_count = int(bundle.get("attempted_iteration_count") or _attempted_iteration_count(dict(bundle["iteration_rows"])))
    completed_count = _safe_int(session_manifest.get("completed_evaluation_count"))
    if completed_count is None:
        completed_count = sum(
            1
            for row in dict(bundle["iteration_rows"]).values()
            if str(row.get("execution_status")) == "completed" and _safe_float(row.get("objective_value")) is not None
        )
    return {
        "session_id": str(bundle["session_id"]),
        "session_root": str(bundle["session_root"]),
        "attempted_iteration_count": int(attempted_iteration_count),
        "completed_evaluation_count": int(completed_count),
        "candidate_attempt_count": _safe_int(session_manifest.get("candidate_attempt_count")) or int(attempted_iteration_count),
    }


def iterative_seed_run_is_complete(
    *,
    repo_root: Path | None,
    branch_root: Path,
    comparison_group: str,
    comparison_method: str,
    dataset_key: str,
    seed_model_name: str,
    seed: int,
    budget_limit: int,
) -> dict[str, Any] | None:
    info = latest_iterative_session_info(
        repo_root=repo_root,
        branch_root=branch_root,
        comparison_group=comparison_group,
        comparison_method=comparison_method,
        dataset_key=dataset_key,
        seed_model_name=seed_model_name,
        seed=seed,
    )
    if info is None:
        return None
    if int(info.get("attempted_iteration_count", 0)) >= int(budget_limit):
        return info
    return None


def _completed_evaluation_records(
    *,
    iteration_rows: Mapping[int, Mapping[str, Any]],
    llm_usage_by_iteration: Mapping[int, Mapping[str, float]],
    session_manifest: Mapping[str, Any],
    budget_limit: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    cumulative_completed = 0
    cumulative_failed = 0
    cumulative_llm_requests = 0.0
    cumulative_llm_tokens = 0.0
    cumulative_llm_repairs = 0.0

    for raw_iteration in sorted(iteration_rows):
        if int(raw_iteration) >= int(budget_limit):
            break
        iteration_row = dict(iteration_rows[raw_iteration])
        usage_payload = dict(llm_usage_by_iteration.get(raw_iteration, {}))
        cumulative_llm_requests += float(usage_payload.get("llm_request_count", 0.0))
        cumulative_llm_tokens += float(usage_payload.get("llm_total_tokens", 0.0))
        cumulative_llm_repairs += float(usage_payload.get("llm_repair_request_count", 0.0))

        if str(iteration_row.get("execution_status")) == "completed":
            cumulative_completed += 1
            records.append(
                {
                    "completed_eval_index": cumulative_completed - 1,
                    "raw_iteration": int(raw_iteration),
                    "metric_payload": {
                        metric_name: _safe_float(iteration_row.get(metric_path))
                        for metric_name, metric_path in METRIC_FIELDS
                    },
                    "budget_payload": {
                        "budget_limit_completed_evaluations": float(
                            _safe_int(session_manifest.get("budget_limit_completed_evaluations")) or budget_limit
                        ),
                        "candidate_attempt_budget_limit": float(
                            _safe_int(session_manifest.get("candidate_attempt_budget_limit"))
                            or _candidate_attempt_budget_limit(budget_limit)
                        ),
                        "evaluation_wall_clock_seconds": _safe_float(iteration_row.get("evaluation_wall_clock_seconds")),
                        "session_wall_clock_seconds": _safe_float(session_manifest.get("session_wall_clock_seconds")),
                        "cumulative_completed_evaluations": float(cumulative_completed),
                        "cumulative_failed_candidates": float(cumulative_failed),
                        "iteration_llm_request_count": float(usage_payload.get("llm_request_count", 0.0)),
                        "iteration_llm_total_tokens": float(usage_payload.get("llm_total_tokens", 0.0)),
                        "iteration_llm_repair_request_count": float(usage_payload.get("llm_repair_request_count", 0.0)),
                        "cumulative_llm_request_count": float(cumulative_llm_requests),
                        "cumulative_llm_total_tokens": float(cumulative_llm_tokens),
                        "cumulative_llm_repair_request_count": float(cumulative_llm_repairs),
                        "stopping_rule": session_manifest.get("stopping_rule"),
                        "stopping_reason": session_manifest.get("stopping_reason"),
                    },
                }
            )
            if len(records) >= int(budget_limit):
                break
        else:
            cumulative_failed += 1
    return records


def _iterative_metric_budget_rows(
    repo_root: Path,
    branch_root: Path,
    spec: IterativeBranchSpec,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matrix = _load_matrix(repo_root)
    defaults = _load_defaults(repo_root)
    budget_limit = int(defaults.get("max_iteration", 10))
    latest_sessions = _select_latest_iterative_sessions(
        repo_root=repo_root,
        branch_root=branch_root,
        branch_line=spec.comparison_group,
        expected_modes=spec.expected_modes,
    )
    completed_records_by_key: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for key, session_bundle in latest_sessions.items():
        completed_records_by_key[key] = _completed_evaluation_records(
            iteration_rows=dict(session_bundle["iteration_rows"]),
            llm_usage_by_iteration=dict(session_bundle["llm_usage_by_iteration"]),
            session_manifest=dict(session_bundle["session_manifest"]),
            budget_limit=budget_limit,
        )
    metric_rows: list[dict[str, Any]] = []
    budget_rows: list[dict[str, Any]] = []
    for dataset_key in _public_main_dataset_keys(matrix):
        entry = _dataset_entry(matrix, dataset_key)
        dataset_display_name = str(entry.get("display_name", dataset_key))
        protocol = str(entry.get("primary_protocol", ""))
        for mode in spec.expected_modes:
            observed_seed_models = sorted(
                {
                    seed_model_name
                    for (observed_dataset, observed_mode, seed_model_name, _seed) in latest_sessions.keys()
                    if observed_dataset == dataset_key and observed_mode == mode and seed_model_name
                }
            )
            seed_models = observed_seed_models or [spec.default_seed_model_name]
            for seed_model_name in seed_models:
                for iteration in range(int(budget_limit)):
                    metric_by_seed: dict[int, dict[str, float | None]] = {}
                    budget_by_seed: dict[int, dict[str, Any]] = {}
                    for seed in DEFAULT_SEEDS:
                        session_bundle = latest_sessions.get((dataset_key, mode, seed_model_name, int(seed)))
                        if session_bundle is None:
                            continue
                        completed_records = completed_records_by_key.get((dataset_key, mode, seed_model_name, int(seed)), [])
                        if int(iteration) < len(completed_records):
                            completed_record = dict(completed_records[int(iteration)])
                            metric_by_seed[int(seed)] = dict(completed_record.get("metric_payload", {}))
                            budget_by_seed[int(seed)] = dict(completed_record.get("budget_payload", {}))
                    method_name = None
                    for seed in DEFAULT_SEEDS:
                        session_bundle = latest_sessions.get((dataset_key, mode, seed_model_name, int(seed)))
                        if session_bundle is not None:
                            method_name = session_bundle["session_manifest"].get("method_name")
                            break
                    metric_rows.append(
                        _aggregate_metric_record(
                            comparison_group=spec.comparison_group,
                            comparison_method=mode,
                            branch_root_name=branch_root.name,
                            dataset_key=dataset_key,
                            dataset_display_name=dataset_display_name,
                            protocol=protocol,
                            method_name=str(method_name) if method_name else None,
                            seed_model_name=seed_model_name,
                            iteration=int(iteration),
                            observed_by_seed=metric_by_seed,
                            expected_seeds=DEFAULT_SEEDS,
                        )
                    )
                    budget_rows.append(
                        _aggregate_budget_record(
                            comparison_group=spec.comparison_group,
                            comparison_method=mode,
                            branch_root_name=branch_root.name,
                            dataset_key=dataset_key,
                            dataset_display_name=dataset_display_name,
                            protocol=protocol,
                            method_name=str(method_name) if method_name else None,
                            seed_model_name=seed_model_name,
                            iteration=int(iteration),
                            observed_by_seed=budget_by_seed,
                            expected_seeds=DEFAULT_SEEDS,
                        )
                    )
    return metric_rows, budget_rows


def _sort_rows(rows: Sequence[Dict[str, Any]]) -> list[Dict[str, Any]]:
    def sort_key(row: Dict[str, Any]) -> tuple[Any, ...]:
        return (
            str(row.get("dataset_key") or ""),
            str(row.get("protocol") or ""),
            str(row.get("comparison_group") or ""),
            str(row.get("comparison_method") or ""),
            str(row.get("seed_model_name") or ""),
            int(_safe_int(row.get("iteration")) or 0),
        )

    return sorted((dict(row) for row in rows), key=sort_key)


def _write_branch_tables(branch_root: Path, metric_rows: Sequence[Dict[str, Any]], budget_rows: Sequence[Dict[str, Any]]) -> None:
    branch_root = ensure_dir(branch_root)
    _write_csv(branch_root / FINAL_METRIC_FILENAME, _sort_rows(metric_rows), _metric_table_column_order())
    _write_csv(branch_root / FINAL_BUDGET_FILENAME, _sort_rows(budget_rows), _budget_table_column_order())


def _read_existing_branch_table(branch_root: Path, filename: str) -> list[dict[str, Any]]:
    path = branch_root / filename
    if not path.exists():
        return []
    return _load_csv_rows(path)


def refresh_branch_final_tables(branch_root: Path, *, repo_root: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    repo_root = _repo_root(repo_root)
    branch_root = branch_root.resolve()
    if branch_root.name == "artifacts":
        metric_rows, budget_rows = _baseline_metric_budget_rows(repo_root, branch_root)
    else:
        spec_map = {spec.root_name: spec for spec in _iterative_branch_specs(repo_root)}
        spec = spec_map.get(branch_root.name)
        if spec is None:
            return [], []
        metric_rows, budget_rows = _iterative_metric_budget_rows(repo_root, branch_root, spec)
    _write_branch_tables(branch_root, metric_rows, budget_rows)
    return metric_rows, budget_rows


def _write_combined_tables(repo_root: Path) -> None:
    combined_metric_rows: list[dict[str, Any]] = []
    combined_budget_rows: list[dict[str, Any]] = []
    for branch_root_name in _discover_branch_root_names(repo_root):
        branch_root = repo_root / branch_root_name
        if not branch_root.exists():
            continue
        combined_metric_rows.extend(_read_existing_branch_table(branch_root, FINAL_METRIC_FILENAME))
        combined_budget_rows.extend(_read_existing_branch_table(branch_root, FINAL_BUDGET_FILENAME))
    _write_csv(repo_root / FINAL_METRIC_FILENAME, _sort_rows(combined_metric_rows), _metric_table_column_order())
    _write_csv(repo_root / FINAL_BUDGET_FILENAME, _sort_rows(combined_budget_rows), _budget_table_column_order())


def refresh_all_final_results(*, repo_root: Path | None = None, updated_branch_root: Path | None = None) -> dict[str, str]:
    repo_root = _repo_root(repo_root)
    if updated_branch_root is not None:
        refresh_branch_final_tables(Path(updated_branch_root).resolve(), repo_root=repo_root)
    else:
        for branch_root_name in _discover_branch_root_names(repo_root):
            branch_root = repo_root / branch_root_name
            if branch_root.exists():
                refresh_branch_final_tables(branch_root, repo_root=repo_root)
    _write_combined_tables(repo_root)
    return {
        "final_metric_result": str(repo_root / FINAL_METRIC_FILENAME),
        "final_budget_result": str(repo_root / FINAL_BUDGET_FILENAME),
    }
