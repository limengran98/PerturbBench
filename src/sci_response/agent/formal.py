from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from sci_response.agent.search import params_signature, propose_next_params
from sci_response.baselines.artifacts import beijing_timestamp_slug
from sci_response.data.io import dump_json, dump_yaml, ensure_dir, load_json, load_yaml, stable_hash, write_text
from sci_response.final_results import refresh_all_final_results
from sci_response.pathing import repo_relative_str


OBJECTIVE_PATH_DEFAULT = "val.delta.mse"
RAW_ITERATION_BUDGET_SEMANTICS = "raw_iteration_budget_v2_attempt_limit"
REPORT_METRIC_PATHS = [
    "val.delta.mse",
    "val.delta.mae",
    "val.delta.pearson",
    "val.delta.spearman",
    "val.delta.r2",
    "test.delta.mse",
    "test.delta.mae",
    "test.delta.pearson",
    "test.delta.spearman",
    "test.delta.r2",
    "test.response.mse",
    "test.response.mae",
]


def _parse_csv_list(raw: str | None) -> List[str]:
    if raw is None:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _get_nested(payload: Dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _flatten_selected(metrics: Dict[str, Any], metric_paths: Sequence[str]) -> Dict[str, float | None]:
    flattened: Dict[str, float | None] = {}
    for path in metric_paths:
        value = _get_nested(metrics, path)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            flattened[path] = float(value)
        else:
            flattened[path] = None
    return flattened


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _aggregate_llm_usage_rows(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    repair_request_count = 0
    for row in rows:
        prompt_tokens += _safe_int(row.get("prompt_tokens"), 0)
        completion_tokens += _safe_int(row.get("completion_tokens"), 0)
        total_tokens += _safe_int(row.get("total_tokens"), 0)
        if bool(row.get("is_repair_request")):
            repair_request_count += 1
    return {
        "llm_request_count": int(len(rows)),
        "llm_prompt_tokens": int(prompt_tokens),
        "llm_completion_tokens": int(completion_tokens),
        "llm_total_tokens": int(total_tokens),
        "llm_repair_request_count": int(repair_request_count),
    }


def normalize_agent_result_row(row: Dict[str, Any], *, max_iteration: int | None = None) -> Dict[str, Any]:
    normalized = dict(row)
    session_root_value = normalized.get("session_root")
    if not session_root_value:
        normalized["budget_semantics"] = RAW_ITERATION_BUDGET_SEMANTICS
        return normalized

    session_root = Path(str(session_root_value))
    iterations_path = session_root / "iterations.json"
    if not iterations_path.exists():
        normalized["budget_semantics"] = RAW_ITERATION_BUDGET_SEMANTICS
        return normalized

    budget_limit = int(
        max(
            1,
            _safe_int(
                max_iteration if max_iteration is not None else normalized.get("max_iteration", normalized.get("budget_limit_completed_evaluations", 10)),
                10,
            ),
        )
    )
    iterations_payload = load_json(iterations_path)
    raw_iterations = list(iterations_payload.get("iterations", []))
    if not raw_iterations:
        normalized["budget_semantics"] = RAW_ITERATION_BUDGET_SEMANTICS
        return normalized

    truncated_iterations = [dict(item) for item in raw_iterations[:budget_limit]]
    baseline_row = dict(truncated_iterations[0])
    completed_rows = [
        dict(item)
        for item in truncated_iterations
        if str(item.get("execution_status")) == "completed" and item.get("objective_value") is not None
    ]
    best_row = (
        min(completed_rows, key=lambda item: float(item["objective_value"]))
        if completed_rows
        else baseline_row
    )

    normalized["budget_semantics"] = RAW_ITERATION_BUDGET_SEMANTICS
    normalized["budget_limit_completed_evaluations"] = int(budget_limit)
    normalized["candidate_attempt_budget_limit"] = int(budget_limit)
    normalized["candidate_attempt_count"] = int(len(truncated_iterations))
    normalized["completed_evaluation_count"] = int(len(completed_rows))
    normalized["failed_candidate_count"] = int(len(truncated_iterations) - len(completed_rows))
    normalized["stopping_rule"] = RAW_ITERATION_BUDGET_SEMANTICS
    normalized["stopping_reason"] = (
        "raw_iteration_budget_exhausted"
        if len(truncated_iterations) >= budget_limit
        else "search_stopped_without_budget_exhaustion"
    )

    normalized["baseline_iteration"] = int(baseline_row.get("iteration", 0) or 0)
    normalized["baseline_objective"] = baseline_row.get("objective_value")
    normalized["baseline_run_dir"] = baseline_row.get("run_dir")
    normalized["best_iteration"] = int(best_row.get("iteration", 0) or 0)
    normalized["best_objective"] = best_row.get("objective_value")
    normalized["best_run_dir"] = best_row.get("run_dir")
    normalized["best_resolved_device"] = best_row.get("resolved_device")
    normalized["best_model_uses_gpu"] = best_row.get("model_uses_gpu")

    baseline_objective = _safe_float(baseline_row.get("objective_value"))
    best_objective = _safe_float(best_row.get("objective_value"))
    normalized["objective_improvement"] = (
        float(baseline_objective - best_objective)
        if baseline_objective is not None and best_objective is not None
        else None
    )

    for metric_path in REPORT_METRIC_PATHS:
        normalized[f"baseline.{metric_path}"] = baseline_row.get(metric_path)
        normalized[f"best.{metric_path}"] = best_row.get(metric_path)

    llm_usage_path = session_root / "trace" / "llm_usage.json"
    if llm_usage_path.exists():
        llm_usage_payload = load_json(llm_usage_path)
        llm_rows = [
            dict(item)
            for item in list(llm_usage_payload.get("rows", []))
            if _safe_int(item.get("iteration"), 0) < budget_limit
        ]
        llm_summary = _aggregate_llm_usage_rows(llm_rows)
        normalized.update(llm_summary)

    return normalized


def _load_matrix(matrix_config: Path) -> Dict[str, Any]:
    return load_yaml(matrix_config)


def resolve_dataset_keys(matrix_config: Path, *, dataset_keys: str | None = None, all_main_datasets: bool = False) -> List[str]:
    matrix = _load_matrix(matrix_config)
    available = dict(matrix.get("datasets", {}))
    if all_main_datasets:
        requested = list(dict(matrix.get("benchmark_scope", {})).get("public_main_datasets", []))
        resolved = [str(item) for item in requested if str(item) in available]
        if not resolved:
            raise ValueError("No public_main_datasets found in benchmark matrix")
        return resolved
    requested = _parse_csv_list(dataset_keys)
    if not requested:
        raise ValueError("Agent mode requires --dataset-keys or --all-main-datasets")
    missing = [item for item in requested if item not in available]
    if missing:
        raise KeyError(f"Unknown dataset keys in benchmark matrix: {missing}")
    return requested


def resolve_method_names(matrix_config: Path, dataset_key: str, *, methods: str | None = None, all_methods: bool = False) -> List[str]:
    matrix = _load_matrix(matrix_config)
    dataset_entry = dict(dict(matrix.get("datasets", {})).get(dataset_key, {}))
    runnable_methods = [str(item) for item in dataset_entry.get("runnable_methods", [])]
    if all_methods or not methods:
        if not runnable_methods:
            raise ValueError(f"No runnable methods registered for dataset {dataset_key}")
        return runnable_methods
    requested = _parse_csv_list(methods)
    unsupported = [item for item in requested if item not in runnable_methods]
    if unsupported:
        raise ValueError(f"Requested methods are not runnable for dataset {dataset_key}: {unsupported}")
    return requested


def _build_method_payload(
    *,
    method_name: str,
    method_defaults: Dict[str, Any],
    baseline_root: Path,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    family = str(method_defaults["family"])
    if family == "universal":
        return {
            "baseline": {
                "name": method_name,
                "feature_builder": {"intervention_hash_dim": 64},
                "tuning_budget": {"definition": "agent_iterative_search_v1", "num_trials": 1},
                "params": params,
            }
        }
    if family == "specialist":
        return {
            "baseline_name": method_name,
            "baseline_root": str(baseline_root.resolve()),
            "baseline": params,
        }
    raise ValueError(f"Unsupported method family for agent runner: {family}")


def _write_iteration_config(
    *,
    config_path: Path,
    dataset_config: Path,
    split_path: Path,
    run_name: str,
    benchmark_runs_root: Path,
    seed: int,
    top_k: int,
    notes: str,
    method_payload: Dict[str, Any],
) -> None:
    payload: Dict[str, Any] = {
        "run_name": run_name,
        "seed": int(seed),
        "artifacts_root": str(benchmark_runs_root.resolve()),
        "dataset_config": str(dataset_config.resolve()),
        "split_path": str(split_path.resolve()),
        "metrics": {"top_k": int(top_k)},
        "notes": notes,
    }
    payload.update(method_payload)
    dump_yaml(config_path, payload)


def _run_iteration_benchmark(
    *,
    repo_root: Path,
    config_path: Path,
    run_id: str,
    requested_device: str,
    cuda_visible_devices: str | None,
    runtime_env_config: Path,
    runtime_mode: str,
    runtime_env_group: str | None,
) -> Path:
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_benchmark.py"),
        "--config",
        str(config_path.resolve()),
        "--run-id",
        run_id,
        "--device",
        requested_device,
        "--runtime-env-config",
        str(runtime_env_config.resolve()),
        "--runtime-mode",
        runtime_mode,
    ]
    if cuda_visible_devices is not None:
        cmd.extend(["--cuda-visible-devices", str(cuda_visible_devices)])
    if runtime_env_group is not None:
        cmd.extend(["--runtime-env-group", str(runtime_env_group)])
    subprocess.run(cmd, check=True, cwd=str(repo_root))
    return Path(load_yaml(config_path)["artifacts_root"]).resolve() / run_id


def _iteration_record(
    *,
    iteration: int,
    phase: str,
    proposal_note: str,
    params: Dict[str, Any],
    config_path: Path,
    run_dir: Path,
    objective_path: str,
    baseline_objective: float | None,
) -> Dict[str, Any]:
    manifest = load_json(run_dir / "manifest.json")
    metrics = load_json(run_dir / "metrics.json")
    execution_status = str(manifest.get("execution_status"))
    objective_value_raw = _get_nested(metrics, objective_path)
    objective_value = float(objective_value_raw) if isinstance(objective_value_raw, (int, float)) else None
    record: Dict[str, Any] = {
        "iteration": int(iteration),
        "phase": phase,
        "proposal_note": proposal_note,
        "params": params,
        "params_signature": params_signature(params),
        "config_path": str(config_path.resolve()),
        "run_dir": str(run_dir.resolve()),
        "manifest_path": str((run_dir / "manifest.json").resolve()),
        "metrics_path": str((run_dir / "metrics.json").resolve()),
        "execution_status": execution_status,
        "objective_path": objective_path,
        "objective_value": objective_value,
        "requested_device": manifest.get("requested_device"),
        "resolved_device": manifest.get("resolved_device"),
        "model_uses_gpu": manifest.get("model_uses_gpu"),
        "protocol": manifest.get("protocol"),
        "start_time_beijing": manifest.get("start_time_beijing"),
        "end_time_beijing": manifest.get("end_time_beijing"),
    }
    record.update(_flatten_selected(metrics, REPORT_METRIC_PATHS))
    if baseline_objective is not None and objective_value is not None:
        record["delta_vs_iteration0"] = float(baseline_objective - objective_value)
    else:
        record["delta_vs_iteration0"] = None
    return record


def _iteration_row(record: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "iteration": record["iteration"],
        "phase": record["phase"],
        "proposal_note": record["proposal_note"],
        "execution_status": record["execution_status"],
        "objective_path": record["objective_path"],
        "objective_value": record["objective_value"],
        "delta_vs_iteration0": record.get("delta_vs_iteration0"),
        "requested_device": record.get("requested_device"),
        "resolved_device": record.get("resolved_device"),
        "model_uses_gpu": record.get("model_uses_gpu"),
        "run_dir": record["run_dir"],
    }
    for metric_path in REPORT_METRIC_PATHS:
        row[metric_path] = record.get(metric_path)
    row["params_json"] = json.dumps(record["params"], ensure_ascii=True, sort_keys=True)
    return row


def _best_record(history: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    completed = [item for item in history if item.get("execution_status") == "completed" and item.get("objective_value") is not None]
    if not completed:
        return None
    return min(completed, key=lambda item: float(item["objective_value"]))


def run_method_session(
    *,
    repo_root: Path,
    matrix_config: Path,
    runtime_env_config: Path,
    dataset_key: str,
    method_name: str,
    session_id: str,
    max_iteration: int,
    seed: int,
    top_k: int,
    requested_device: str,
    cuda_visible_devices: str | None,
    runtime_mode: str,
    runtime_env_group: str | None,
    agent_root: Path,
    baseline_root: Path,
    agent_mode: str = "ablation_method_search",
) -> Dict[str, Any]:
    matrix = _load_matrix(matrix_config)
    dataset_entry = dict(dict(matrix.get("datasets", {}))[dataset_key])
    baseline_defaults = dict(dict(matrix.get("baseline_defaults", {}))[method_name])
    dataset_config = (repo_root / str(dataset_entry["dataset_config"])).resolve()
    split_path = (repo_root / str(dataset_entry["primary_split"])).resolve()
    dataset_root = ensure_dir(agent_root / "datasets" / dataset_key)
    method_session_root = ensure_dir(dataset_root / "methods" / method_name / session_id)
    configs_root = ensure_dir(method_session_root / "configs")
    benchmark_runs_root = ensure_dir(method_session_root / "benchmark_runs")
    history_root = ensure_dir(method_session_root / "history")

    base_params = dict(baseline_defaults.get("params", {}))
    history: List[Dict[str, Any]] = []
    log_lines = [
        f"dataset_key={dataset_key}",
        f"method_name={method_name}",
        f"session_id={session_id}",
        f"dataset_config={dataset_config}",
        f"split_path={split_path}",
        f"requested_device={requested_device}",
        f"cuda_visible_devices={cuda_visible_devices}",
    ]

    baseline_objective: float | None = None
    for iteration in range(int(max_iteration) + 1):
        phase = "baseline" if iteration == 0 else "agent"
        if iteration == 0:
            params = dict(base_params)
            proposal_note = "iteration_0_baseline"
        else:
            params, proposal_note = propose_next_params(base_params=base_params, history=history)
        config_path = configs_root / f"iter_{iteration:03d}.yaml"
        run_name = f"{dataset_key}_{method_name}_agent_iter_{iteration:03d}"
        run_id = f"iter_{iteration:03d}"
        method_payload = _build_method_payload(
            method_name=method_name,
            method_defaults=baseline_defaults,
            baseline_root=baseline_root,
            params=params,
        )
        _write_iteration_config(
            config_path=config_path,
            dataset_config=dataset_config,
            split_path=split_path,
            run_name=run_name,
            benchmark_runs_root=benchmark_runs_root,
            seed=int(seed),
            top_k=int(top_k),
            notes=f"agent iteration {iteration} for {dataset_key}/{method_name}",
            method_payload=method_payload,
        )
        log_lines.append(
            f"dispatch iteration={iteration} phase={phase} proposal_note={proposal_note} config={config_path.name}"
        )
        run_dir = _run_iteration_benchmark(
            repo_root=repo_root,
            config_path=config_path,
            run_id=run_id,
            requested_device=requested_device,
            cuda_visible_devices=cuda_visible_devices,
            runtime_env_config=runtime_env_config,
            runtime_mode=runtime_mode,
            runtime_env_group=runtime_env_group,
        )
        record = _iteration_record(
            iteration=int(iteration),
            phase=phase,
            proposal_note=proposal_note,
            params=params,
            config_path=config_path,
            run_dir=run_dir,
            objective_path=OBJECTIVE_PATH_DEFAULT,
            baseline_objective=baseline_objective,
        )
        if iteration == 0:
            baseline_objective = record.get("objective_value")
            record["delta_vs_iteration0"] = 0.0 if baseline_objective is not None else None
        history.append(record)
        best_record = _best_record(history)
        if best_record is not None:
            record["best_so_far_iteration"] = int(best_record["iteration"])
            record["best_so_far_objective"] = float(best_record["objective_value"])
            record["accepted_as_best"] = bool(best_record["iteration"] == record["iteration"])
        else:
            record["best_so_far_iteration"] = None
            record["best_so_far_objective"] = None
            record["accepted_as_best"] = False
        dump_json(history_root / f"iter_{iteration:03d}.json", record)
        log_lines.append(
            f"completed iteration={iteration} objective={record.get('objective_value')} "
            f"best_so_far_iteration={record.get('best_so_far_iteration')}"
        )

    iteration_rows = [_iteration_row(record) for record in history]
    dump_json(method_session_root / "iterations.json", {"iterations": history})
    _write_csv(method_session_root / "iterations.csv", iteration_rows)
    best_record = _best_record(history)
    if best_record is None:
        raise RuntimeError(f"No completed iterations found for {dataset_key}/{method_name}")
    dump_json(method_session_root / "best_iteration.json", best_record)
    session_manifest = {
        "dataset_key": dataset_key,
        "method_name": method_name,
        "method_family": baseline_defaults["family"],
        "agent_line": "ablation_method_search",
        "agent_mode": agent_mode,
        "agent_variant": agent_mode,
        "session_id": session_id,
        "session_root": str(method_session_root.resolve()),
        "benchmark_runs_root": str(benchmark_runs_root.resolve()),
        "iterations_requested": int(max_iteration),
        "seed": int(seed),
        "requested_device": requested_device,
        "cuda_visible_devices": cuda_visible_devices,
        "dataset_config": str(dataset_config),
        "split_path": str(split_path),
        "base_params": base_params,
        "best_iteration": int(best_record["iteration"]),
        "best_objective": best_record["objective_value"],
        "best_run_dir": best_record["run_dir"],
        "objective_path": OBJECTIVE_PATH_DEFAULT,
        "session_hash": stable_hash(
            {
                "dataset_key": dataset_key,
                "method_name": method_name,
                "session_id": session_id,
                "seed": seed,
                "base_params": base_params,
            }
        ),
    }
    dump_json(method_session_root / "agent_session.json", session_manifest)
    write_text(method_session_root / "agent.log", "\n".join(log_lines) + "\n")

    baseline_record = history[0]
    return {
        "dataset_key": dataset_key,
        "method_name": method_name,
        "method_family": baseline_defaults["family"],
        "agent_line": "ablation_method_search",
        "agent_mode": agent_mode,
        "agent_variant": agent_mode,
        "session_id": session_id,
        "session_root": str(method_session_root.resolve()),
        "baseline_iteration": 0,
        "baseline_objective": baseline_record.get("objective_value"),
        "best_iteration": int(best_record["iteration"]),
        "best_objective": best_record.get("objective_value"),
        "objective_improvement": (
            float(baseline_record["objective_value"]) - float(best_record["objective_value"])
            if baseline_record.get("objective_value") is not None and best_record.get("objective_value") is not None
            else None
        ),
        "baseline_run_dir": baseline_record["run_dir"],
        "best_run_dir": best_record["run_dir"],
        "requested_device": requested_device,
        "best_resolved_device": best_record.get("resolved_device"),
        "best_model_uses_gpu": best_record.get("model_uses_gpu"),
        **{f"baseline.{metric}": baseline_record.get(metric) for metric in REPORT_METRIC_PATHS},
        **{f"best.{metric}": best_record.get(metric) for metric in REPORT_METRIC_PATHS},
    }


def write_dataset_pack_summary(
    *,
    agent_root: Path,
    dataset_key: str,
    session_id: str,
    rows: Sequence[Dict[str, Any]],
    resolved_config: Dict[str, Any],
) -> Path:
    dataset_root = ensure_dir(agent_root / "datasets" / dataset_key)
    pack_root = ensure_dir(dataset_root / "pack_runs" / session_id)
    normalized_rows = [normalize_agent_result_row(dict(row)) for row in rows]
    dump_json(pack_root / "agent_pack_summary.json", {"rows": list(normalized_rows)})
    _write_csv(pack_root / "agent_pack_summary.csv", list(normalized_rows))
    dump_yaml(pack_root / "resolved_agent_pack_config.yaml", resolved_config)
    latest_root = ensure_dir(dataset_root / "latest")
    dump_json(latest_root / "agent_pack_summary.json", {"rows": list(normalized_rows)})
    _write_csv(latest_root / "agent_pack_summary.csv", list(normalized_rows))
    dump_yaml(latest_root / "resolved_agent_pack_config.yaml", resolved_config)
    write_text(dataset_root / "LATEST_AGENT_PACK_RUN.txt", repo_relative_str(pack_root) + "\n")
    return pack_root


def update_global_agent_summary(agent_root: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for best_iteration_path in sorted(agent_root.rglob("best_iteration.json")):
        session_root = best_iteration_path.parent
        agent_session_path = session_root / "agent_session.json"
        iterations_path = session_root / "iterations.json"
        if not agent_session_path.exists() or not iterations_path.exists():
            continue
        session_manifest = load_json(agent_session_path)
        best_iteration = load_json(best_iteration_path)
        iterations_payload = load_json(iterations_path)
        baseline_iteration = dict(iterations_payload.get("iterations", [])[0]) if iterations_payload.get("iterations") else {}
        rows.append(
            normalize_agent_result_row(
                {
                "dataset_key": session_manifest.get("dataset_key"),
                "method_name": session_manifest.get("method_name"),
                "method_family": session_manifest.get("method_family"),
                "agent_line": session_manifest.get("agent_line"),
                "agent_mode": session_manifest.get("agent_mode"),
                "agent_variant": session_manifest.get("agent_variant"),
                "seed_model_name": session_manifest.get("seed_model_name"),
                "session_id": session_manifest.get("session_id"),
                "session_root": session_manifest.get("session_root"),
                "best_iteration": session_manifest.get("best_iteration"),
                "best_objective": session_manifest.get("best_objective"),
                "baseline_objective": baseline_iteration.get("objective_value"),
                "objective_improvement": (
                    float(baseline_iteration["objective_value"]) - float(best_iteration["objective_value"])
                    if baseline_iteration.get("objective_value") is not None and best_iteration.get("objective_value") is not None
                    else None
                ),
                "best_run_dir": session_manifest.get("best_run_dir"),
                "best_resolved_device": best_iteration.get("resolved_device"),
                "best_model_uses_gpu": best_iteration.get("model_uses_gpu"),
                "budget_limit_completed_evaluations": session_manifest.get("budget_limit_completed_evaluations"),
                "candidate_attempt_budget_limit": session_manifest.get("candidate_attempt_budget_limit"),
                "completed_evaluation_count": session_manifest.get("completed_evaluation_count"),
                "failed_candidate_count": session_manifest.get("failed_candidate_count"),
                "candidate_attempt_count": session_manifest.get("candidate_attempt_count"),
                "stopping_rule": session_manifest.get("stopping_rule"),
                "stopping_reason": session_manifest.get("stopping_reason"),
                "session_wall_clock_seconds": session_manifest.get("session_wall_clock_seconds"),
                "llm_request_count": session_manifest.get("llm_request_count"),
                "llm_prompt_tokens": session_manifest.get("llm_prompt_tokens"),
                "llm_completion_tokens": session_manifest.get("llm_completion_tokens"),
                "llm_total_tokens": session_manifest.get("llm_total_tokens"),
                "llm_repair_request_count": session_manifest.get("llm_repair_request_count"),
                **{f"best.{metric}": best_iteration.get(metric) for metric in REPORT_METRIC_PATHS},
                }
            )
        )
    if not rows:
        return
    global_root = ensure_dir(agent_root / "global")
    dump_json(global_root / "agent_summary.json", {"rows": rows})
    _write_csv(global_root / "agent_summary.csv", rows)
    history_root = ensure_dir(global_root / "history" / beijing_timestamp_slug())
    dump_json(history_root / "agent_summary.json", {"rows": rows})
    _write_csv(history_root / "agent_summary.csv", rows)
    dump_json(
        history_root / "summary_manifest.json",
        {
            "row_count": int(len(rows)),
            "latest_csv": repo_relative_str(global_root / "agent_summary.csv"),
            "latest_json": repo_relative_str(global_root / "agent_summary.json"),
            "history_csv": repo_relative_str(history_root / "agent_summary.csv"),
            "history_json": repo_relative_str(history_root / "agent_summary.json"),
        },
    )
    write_text(global_root / "LATEST_AGENT_SUMMARY.txt", repo_relative_str(history_root) + "\n")
    refresh_all_final_results(updated_branch_root=agent_root)


def default_agent_root(repo_root: Path) -> Path:
    return (repo_root / "agent_runs").resolve()


def default_session_id() -> str:
    return beijing_timestamp_slug()
