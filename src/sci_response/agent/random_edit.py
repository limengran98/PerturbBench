from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from sci_response.agent.compiler import compile_model_ir, render_generated_model_source
from sci_response.agent.edits import apply_edit_sequence, dedupe_edit_sequences, describe_edit, enumerate_candidate_edits
from sci_response.agent.harness import changed_ir_paths, infer_hypothesis_axes
from sci_response.agent.ir import extract_model_ir
from sci_response.agent.matched_budget import (
    MATCHED_BUDGET_STOPPING_RULE,
    candidate_attempt_budget_limit,
    completed_evaluation_count,
    default_stopping_reason,
    failed_history_count,
    should_continue_search,
)
from sci_response.agent.resume import (
    baseline_objective_from_history,
    load_history_rows,
    load_previous_log_lines,
    load_previous_session_wall_clock_seconds,
    load_rows_payload,
    next_iteration_index,
)
from sci_response.agent.structured import OBJECTIVE_PATH, REPORT_METRIC_PATHS
from sci_response.data.io import dump_json, dump_yaml, ensure_dir, load_json, load_yaml, stable_hash, write_text


def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text.strip().lower()).strip("_")


def _get_nested(payload: Dict[str, Any], path: str) -> Any:
    current: Any = payload
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _flatten_selected(metrics: Dict[str, Any], metric_paths: Sequence[str]) -> Dict[str, float | None]:
    flat: Dict[str, float | None] = {}
    for path in metric_paths:
        value = _get_nested(metrics, path)
        flat[path] = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    return flat


def _write_trace_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_benchmark_payload(
    *,
    dataset_config: Path,
    split_path: Path,
    run_name: str,
    artifacts_root: Path,
    seed: int,
    top_k: int,
    model_config: Path,
) -> Dict[str, Any]:
    return {
        "run_name": run_name,
        "seed": int(seed),
        "artifacts_root": str(artifacts_root.resolve()),
        "dataset_config": str(dataset_config.resolve()),
        "split_path": str(split_path.resolve()),
        "model_config": str(model_config.resolve()),
        "metrics": {"top_k": int(top_k)},
        "notes": "random edit search control iteration",
    }


def _run_benchmark_with_model(
    *,
    repo_root: Path,
    benchmark_config_path: Path,
    run_id: str,
    requested_device: str,
    cuda_visible_devices: str | None,
    runtime_env_config: Path,
    runtime_mode: str,
    runtime_env_group: str | None,
) -> tuple[Path, float]:
    import subprocess

    started = perf_counter()
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_benchmark.py"),
        "--config",
        str(benchmark_config_path.resolve()),
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
    benchmark_payload = load_yaml(benchmark_config_path)
    return Path(str(benchmark_payload["artifacts_root"])).resolve() / run_id, float(perf_counter() - started)


def _candidate_sequences(model_ir: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
    primitive_edits = enumerate_candidate_edits(model_ir)
    single_edits = [[edit] for edit in primitive_edits]
    paired_edits: List[List[Dict[str, Any]]] = []
    for first in primitive_edits:
        for second in primitive_edits:
            if first["path"] == second["path"]:
                continue
            paired_edits.append([first, second])
    return dedupe_edit_sequences(single_edits + paired_edits)


def _enumerate_candidates(model_ir: Dict[str, Any], seen_ir_hashes: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set(str(item) for item in seen_ir_hashes)
    for sequence in _candidate_sequences(model_ir):
        candidate_ir = apply_edit_sequence(model_ir, sequence)
        ir_hash = stable_hash(candidate_ir)
        if ir_hash in seen:
            continue
        changed_paths = changed_ir_paths(model_ir, candidate_ir)
        axes = infer_hypothesis_axes(changed_paths)
        rows.append(
            {
                "edits": sequence,
                "candidate_ir": candidate_ir,
                "ir_hash": ir_hash,
                "changed_paths": changed_paths,
                "edit_families": axes or ["misc"],
                "proposal_note": ", ".join(describe_edit(edit) for edit in sequence),
            }
        )
    return rows


def _sample_candidate(
    *,
    rng: np.random.Generator,
    model_ir: Dict[str, Any],
    seen_ir_hashes: Sequence[str],
    strategy: str,
) -> Dict[str, Any] | None:
    candidates = _enumerate_candidates(model_ir, seen_ir_hashes)
    if not candidates:
        return None
    if strategy == "uniform_random_edit":
        index = int(rng.integers(0, len(candidates)))
        chosen = dict(candidates[index])
        chosen["sampling_group"] = "all_candidates"
        return chosen
    if strategy == "stratified_random_edit":
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for row in candidates:
            family = str(row["edit_families"][0]) if row["edit_families"] else "misc"
            grouped.setdefault(family, []).append(row)
        families = sorted(grouped)
        family_index = int(rng.integers(0, len(families)))
        family_name = families[family_index]
        bucket = grouped[family_name]
        chosen = dict(bucket[int(rng.integers(0, len(bucket)))])
        chosen["sampling_group"] = family_name
        return chosen
    raise ValueError(f"Unsupported random edit strategy: {strategy}")


def run_random_edit_model_session(
    *,
    repo_root: Path,
    dataset_key: str,
    dataset_config: Path,
    split_path: Path,
    seed_model_config_path: Path,
    runtime_env_config: Path,
    requested_device: str,
    cuda_visible_devices: str | None,
    runtime_mode: str,
    runtime_env_group: str | None,
    agent_root: Path,
    seed: int,
    top_k: int,
    max_iteration: int,
    session_id: str | None,
    agent_mode: str,
) -> Dict[str, Any]:
    if agent_mode == "random_edit_uniform":
        strategy = "uniform_random_edit"
    elif agent_mode == "random_edit_stratified":
        strategy = "stratified_random_edit"
    else:
        raise ValueError(f"Unsupported random edit mode: {agent_mode}")

    seed_model_config = load_yaml(seed_model_config_path)
    seed_model_name = str(seed_model_config["name"])
    session_slug = session_id or f"{_safe_slug(agent_mode)}_{seed}"
    dataset_root = ensure_dir(agent_root / "datasets" / dataset_key)
    session_root = ensure_dir(dataset_root / "random_edit" / _safe_slug(agent_mode) / _safe_slug(seed_model_name) / session_slug)
    ir_root = ensure_dir(session_root / "ir")
    edit_root = ensure_dir(session_root / "edits")
    compiled_root = ensure_dir(session_root / "compiled_models")
    generated_code_root = ensure_dir(session_root / "generated_code")
    benchmark_configs_root = ensure_dir(session_root / "benchmark_configs")
    benchmark_runs_root = ensure_dir(session_root / "benchmark_runs")
    history_root = ensure_dir(session_root / "history")
    trace_root = ensure_dir(session_root / "trace")
    prior_session_wall_clock_seconds = load_previous_session_wall_clock_seconds(session_root)

    rng = np.random.default_rng(int(seed))
    budget_limit = int(max_iteration)
    session_started_at = perf_counter()
    log_lines = [
        f"dataset_key={dataset_key}",
        f"agent_mode={agent_mode}",
        f"sampling_strategy={strategy}",
        f"seed_model_name={seed_model_name}",
        f"seed_model_config={seed_model_config_path}",
        f"session_id={session_slug}",
        f"requested_device={requested_device}",
        f"cuda_visible_devices={cuda_visible_devices}",
        "mechanism_constraints=disabled",
        "semantic_checker=disabled",
        "hypothesis_memory=disabled",
    ]
    log_lines = load_previous_log_lines(session_root / "agent.log", log_lines)

    def log(message: str) -> None:
        log_lines.append(message)
        print(f"[agent-random-edit] {message}", flush=True)

    history: List[Dict[str, Any]] = load_history_rows(history_root)
    trace_rows: List[Dict[str, Any]] = load_rows_payload(trace_root / "random_edit_trace.json")
    seed_ir = extract_model_ir(seed_model_config)
    current_reference_ir = seed_ir
    baseline_objective: float | None = baseline_objective_from_history(history)
    stopping_reason: str | None = None
    iteration = next_iteration_index(history)
    if history:
        log(
            f"resume session_id={session_slug} next_iteration={iteration} "
            f"completed_evaluations={completed_evaluation_count(history)} candidate_attempts={len(history)}"
        )

    while should_continue_search(history, budget_limit):
        if iteration == 0:
            proposal = {
                "proposal_source": "seed_model_ir",
                "proposal_note": f"iteration_0_seed_ir::{seed_model_name}",
                "candidate_ir": seed_ir,
                "ir_hash": stable_hash(seed_ir),
                "edits": [],
                "changed_paths": [],
                "edit_families": [],
                "sampling_group": "baseline",
            }
        else:
            best_completed = [item for item in history if item.get("execution_status") == "completed" and item.get("objective_value") is not None]
            if best_completed:
                best_reference_record = min(best_completed, key=lambda item: float(item["objective_value"]))
                current_reference_ir = dict(best_reference_record["model_ir"])
            seen_ir_hashes = [str(item.get("ir_hash")) for item in history]
            sampled = _sample_candidate(
                rng=rng,
                model_ir=current_reference_ir,
                seen_ir_hashes=seen_ir_hashes,
                strategy=strategy,
            )
            if sampled is None:
                log(f"search_frontier_exhausted iteration={iteration}")
                stopping_reason = "search_frontier_exhausted"
                break
            proposal = {
                "proposal_source": strategy,
                "proposal_note": sampled["proposal_note"],
                "candidate_ir": sampled["candidate_ir"],
                "ir_hash": sampled["ir_hash"],
                "edits": sampled["edits"],
                "changed_paths": sampled["changed_paths"],
                "edit_families": sampled["edit_families"],
                "sampling_group": sampled["sampling_group"],
            }

        candidate_ir = proposal["candidate_ir"]
        ir_path = ir_root / f"iter_{iteration:03d}.json"
        edit_path = edit_root / f"iter_{iteration:03d}.json"
        compiled_model_path = compiled_root / f"iter_{iteration:03d}.yaml"
        generated_code_path = generated_code_root / f"iter_{iteration:03d}_model.py"
        benchmark_config_path = benchmark_configs_root / f"iter_{iteration:03d}.yaml"

        dump_json(ir_path, candidate_ir)
        dump_json(
            edit_path,
            {
                "iteration": int(iteration),
                "proposal_source": proposal["proposal_source"],
                "proposal_note": proposal["proposal_note"],
                "sampling_group": proposal.get("sampling_group"),
                "edit_families": list(proposal.get("edit_families", [])),
                "changed_paths": list(proposal.get("changed_paths", [])),
                "edits": list(proposal["edits"]),
                "ir_hash": proposal["ir_hash"],
            },
        )

        compiled_model_config = compile_model_ir(
            candidate_ir,
            generated_module_path=generated_code_path,
            generated_class_name="GeneratedStructuredHypothesisRegressor",
        )
        write_text(generated_code_path, render_generated_model_source(candidate_ir, class_name="GeneratedStructuredHypothesisRegressor"))
        dump_yaml(compiled_model_path, compiled_model_config)
        dump_yaml(
            benchmark_config_path,
            _build_benchmark_payload(
                dataset_config=dataset_config,
                split_path=split_path,
                run_name=f"{dataset_key}_{seed_model_name}_{agent_mode}_iter_{iteration:03d}",
                artifacts_root=benchmark_runs_root,
                seed=int(seed),
                top_k=int(top_k),
                model_config=compiled_model_path,
            ),
        )
        log(
            f"dispatch iteration={iteration} proposal_source={proposal['proposal_source']} "
            f"proposal_note={proposal['proposal_note']}"
        )
        try:
            run_dir, evaluation_wall_clock_seconds = _run_benchmark_with_model(
                repo_root=repo_root,
                benchmark_config_path=benchmark_config_path,
                run_id=f"iter_{iteration:03d}",
                requested_device=requested_device,
                cuda_visible_devices=cuda_visible_devices,
                runtime_env_config=runtime_env_config,
                runtime_mode=runtime_mode,
                runtime_env_group=runtime_env_group,
            )
            manifest = load_json(run_dir / "manifest.json")
            metrics = load_json(run_dir / "metrics.json")
            objective_raw = _get_nested(metrics, OBJECTIVE_PATH)
            objective_value = float(objective_raw) if isinstance(objective_raw, (int, float)) else None
            if iteration == 0:
                baseline_objective = objective_value
            record = {
                "iteration": int(iteration),
                "phase": "baseline" if iteration == 0 else "agent",
                "proposal_source": proposal["proposal_source"],
                "proposal_note": proposal["proposal_note"],
                "sampling_group": proposal.get("sampling_group"),
                "edit_families": list(proposal.get("edit_families", [])),
                "changed_paths": list(proposal.get("changed_paths", [])),
                "edits": list(proposal["edits"]),
                "edit_preview": [describe_edit(edit) for edit in proposal["edits"]],
                "ir_hash": proposal["ir_hash"],
                "model_ir": candidate_ir,
                "compiled_model_config_path": str(compiled_model_path.resolve()),
                "generated_code_path": str(generated_code_path.resolve()),
                "benchmark_config_path": str(benchmark_config_path.resolve()),
                "run_dir": str(run_dir.resolve()),
                "manifest_path": str((run_dir / "manifest.json").resolve()),
                "metrics_path": str((run_dir / "metrics.json").resolve()),
                "execution_status": str(manifest.get("execution_status")),
                "requested_device": manifest.get("requested_device"),
                "resolved_device": manifest.get("resolved_device"),
                "model_uses_gpu": manifest.get("model_uses_gpu"),
                "objective_path": OBJECTIVE_PATH,
                "objective_value": objective_value,
                "delta_vs_iteration0": (
                    float(baseline_objective) - float(objective_value)
                    if baseline_objective is not None and objective_value is not None
                    else None
                ),
                "seed_model_name": seed_model_name,
                "agent_mode": agent_mode,
                "sampling_strategy": strategy,
                "protocol": manifest.get("protocol"),
                "start_time_beijing": manifest.get("start_time_beijing"),
                "end_time_beijing": manifest.get("end_time_beijing"),
                "evaluation_wall_clock_seconds": float(evaluation_wall_clock_seconds),
            }
            record.update(_flatten_selected(metrics, REPORT_METRIC_PATHS))
        except Exception as exc:
            if iteration == 0:
                raise
            record = {
                "iteration": int(iteration),
                "phase": "agent",
                "proposal_source": proposal["proposal_source"],
                "proposal_note": proposal["proposal_note"],
                "sampling_group": proposal.get("sampling_group"),
                "edit_families": list(proposal.get("edit_families", [])),
                "changed_paths": list(proposal.get("changed_paths", [])),
                "edits": list(proposal["edits"]),
                "edit_preview": [describe_edit(edit) for edit in proposal["edits"]],
                "ir_hash": proposal["ir_hash"],
                "model_ir": candidate_ir,
                "compiled_model_config_path": str(compiled_model_path.resolve()),
                "generated_code_path": str(generated_code_path.resolve()),
                "benchmark_config_path": str(benchmark_config_path.resolve()),
                "run_dir": None,
                "manifest_path": None,
                "metrics_path": None,
                "execution_status": "failed",
                "requested_device": requested_device,
                "resolved_device": None,
                "model_uses_gpu": None,
                "objective_path": OBJECTIVE_PATH,
                "objective_value": None,
                "delta_vs_iteration0": None,
                "seed_model_name": seed_model_name,
                "agent_mode": agent_mode,
                "sampling_strategy": strategy,
                "protocol": None,
                "start_time_beijing": None,
                "end_time_beijing": None,
                "evaluation_wall_clock_seconds": None,
                "error_tail": f"{type(exc).__name__}: {exc}",
            }
            for metric_path in REPORT_METRIC_PATHS:
                record[metric_path] = None
            log(f"candidate_failed iteration={iteration} reason={type(exc).__name__}: {exc}")
        history.append(record)
        trace_rows.append(
            {
                "iteration": int(iteration),
                "proposal_source": proposal["proposal_source"],
                "proposal_note": proposal["proposal_note"],
                "sampling_group": proposal.get("sampling_group"),
                "edit_families_json": json.dumps(list(proposal.get("edit_families", [])), ensure_ascii=True),
                "changed_paths_json": json.dumps(list(proposal.get("changed_paths", [])), ensure_ascii=True),
                "objective_value": record.get("objective_value"),
                "benchmark_run_dir": str(record.get("run_dir")),
                "generated_code_path": str(generated_code_path.resolve()),
            }
        )
        dump_json(history_root / f"iter_{iteration:03d}.json", record)
        log(f"completed iteration={iteration} objective={record.get('objective_value')}")
        iteration += 1

    completed = [item for item in history if item.get("execution_status") == "completed" and item.get("objective_value") is not None]
    if not completed:
        raise RuntimeError(f"No completed random-edit iterations for {dataset_key}/{seed_model_name}/{agent_mode}")
    best_record = min(completed, key=lambda item: float(item["objective_value"]))
    if stopping_reason is None:
        stopping_reason = default_stopping_reason(history, budget_limit)
    completed_count = completed_evaluation_count(history)
    failed_candidate_count = int(failed_history_count(history))
    candidate_attempt_count = int(len(history))
    session_wall_clock_seconds = prior_session_wall_clock_seconds + float(perf_counter() - session_started_at)
    for record in history:
        record["best_so_far_iteration"] = int(best_record["iteration"])
        record["best_so_far_objective"] = float(best_record["objective_value"])
        record["accepted_as_best"] = bool(record["iteration"] == best_record["iteration"])

    dump_json(session_root / "iterations.json", {"iterations": history})
    fieldnames = sorted({key for row in history for key in row.keys() if key not in {"model_ir", "edits"}} | {"model_ir_json", "edits_json"})
    with (session_root / "iterations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            flattened = {key: value for key, value in row.items() if key not in {"model_ir", "edits"}}
            flattened["model_ir_json"] = json.dumps(row["model_ir"], ensure_ascii=True, sort_keys=True)
            flattened["edits_json"] = json.dumps(row["edits"], ensure_ascii=True, sort_keys=True)
            writer.writerow(flattened)
    dump_json(session_root / "best_iteration.json", best_record)
    dump_json(trace_root / "random_edit_trace.json", {"rows": trace_rows})
    if trace_rows:
        _write_trace_csv(trace_root / "random_edit_trace.csv", trace_rows)
    dump_json(
        session_root / "agent_session.json",
        {
            "dataset_key": dataset_key,
            "method_family": "agent_random_edit_model",
            "method_name": f"{agent_mode}::{seed_model_name}",
            "agent_line": "random_edit_search",
            "agent_mode": agent_mode,
            "agent_variant": agent_mode,
            "seed_model_name": seed_model_name,
            "seed_model_config_path": str(seed_model_config_path.resolve()),
            "session_id": session_slug,
            "session_root": str(session_root.resolve()),
            "generated_code_root": str(generated_code_root.resolve()),
            "requested_device": requested_device,
            "cuda_visible_devices": cuda_visible_devices,
            "sampling_strategy": strategy,
            "random_constraints": {
                "biological_constraints": False,
                "semantic_checker": False,
                "hypothesis_memory": False,
                "template_priority_scoring": False,
            },
            "max_iteration": int(max_iteration),
            "budget_limit_completed_evaluations": int(budget_limit),
            "candidate_attempt_budget_limit": int(candidate_attempt_budget_limit(budget_limit)),
            "completed_evaluation_count": int(completed_count),
            "failed_candidate_count": int(failed_candidate_count),
            "candidate_attempt_count": int(candidate_attempt_count),
            "stopping_rule": MATCHED_BUDGET_STOPPING_RULE,
            "stopping_reason": stopping_reason,
            "session_wall_clock_seconds": float(session_wall_clock_seconds),
            "best_iteration": int(best_record["iteration"]),
            "best_objective": best_record["objective_value"],
            "best_run_dir": best_record["run_dir"],
            "objective_path": OBJECTIVE_PATH,
        },
    )
    write_text(session_root / "agent.log", "\n".join(log_lines) + "\n")

    baseline_record = history[0]
    return {
        "dataset_key": dataset_key,
        "method_name": f"{agent_mode}::{seed_model_name}",
        "method_family": "agent_random_edit_model",
        "agent_line": "random_edit_search",
        "agent_mode": agent_mode,
        "agent_variant": agent_mode,
        "seed_model_name": seed_model_name,
        "session_id": session_slug,
        "session_root": str(session_root.resolve()),
        "baseline_iteration": 0,
        "baseline_objective": baseline_record.get("objective_value"),
        "best_iteration": int(best_record["iteration"]),
        "best_objective": best_record.get("objective_value"),
        "objective_improvement": (
            float(baseline_record["objective_value"]) - float(best_record["objective_value"])
            if baseline_record.get("objective_value") is not None and best_record.get("objective_value") is not None
            else None
        ),
        "baseline_run_dir": baseline_record.get("run_dir"),
        "best_run_dir": best_record.get("run_dir"),
        "requested_device": requested_device,
        "best_resolved_device": best_record.get("resolved_device"),
        "best_model_uses_gpu": best_record.get("model_uses_gpu"),
        "sampling_strategy": strategy,
        "llm_enabled": False,
        "budget_limit_completed_evaluations": int(budget_limit),
        "candidate_attempt_budget_limit": int(candidate_attempt_budget_limit(budget_limit)),
        "completed_evaluation_count": int(completed_count),
        "failed_candidate_count": int(failed_candidate_count),
        "candidate_attempt_count": int(candidate_attempt_count),
        "stopping_rule": MATCHED_BUDGET_STOPPING_RULE,
        "stopping_reason": stopping_reason,
        "session_wall_clock_seconds": float(session_wall_clock_seconds),
        **{f"baseline.{metric}": baseline_record.get(metric) for metric in REPORT_METRIC_PATHS},
        **{f"best.{metric}": best_record.get(metric) for metric in REPORT_METRIC_PATHS},
    }
