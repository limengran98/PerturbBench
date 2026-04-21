from __future__ import annotations

import csv
import importlib.util
import json
import re
import subprocess
import sys
import traceback
from pathlib import Path
from string import Template
from time import perf_counter
from typing import Any, Dict, List, Sequence

from sci_response.agent.config import (
    default_agent_defaults_path,
    default_direct_code_prompt_config_path,
    default_direct_code_skill_card_path,
)
from sci_response.agent.llm import StructuredLLMClient, load_llm_settings
from sci_response.agent.matched_budget import (
    MATCHED_BUDGET_STOPPING_RULE,
    aggregate_llm_usage,
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


DIRECT_CODE_CLASS_NAME = "GeneratedDirectCodeRegressor"


def _safe_slug(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text.strip().lower()).strip("_")


def _coerce_scalar_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{field_name} must be a scalar value, got {value!r}")
        value = value[0]
    return int(value)


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


def _seed_model_source_path(repo_root: Path, seed_model_name: str) -> Path:
    candidate = repo_root / "src" / "sci_response" / "models" / f"{seed_model_name}.py"
    if not candidate.exists():
        raise FileNotFoundError(f"Could not resolve seed model source for {seed_model_name}: {candidate}")
    return candidate


def _recent_history(history: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in history[-5:]:
        rows.append(
            {
                "iteration": item.get("iteration"),
                "execution_status": item.get("execution_status"),
                "objective_value": item.get("objective_value"),
                "proposal_note": item.get("proposal_note"),
                "best_so_far_iteration": item.get("best_so_far_iteration"),
                "best_so_far_objective": item.get("best_so_far_objective"),
                "delta_vs_iteration0": item.get("delta_vs_iteration0"),
            }
        )
    return rows


def _extract_python_code(text: str) -> str:
    stripped = str(text).strip()
    matches = re.findall(r"```(?:python)?\s*(.*?)```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return max((match.strip() for match in matches), key=len)
    return stripped


def _load_generated_class(module_path: Path, class_name: str):
    spec = importlib.util.spec_from_file_location(f"sci_response_direct_code_{stable_hash(str(module_path))[:10]}", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load generated module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        return getattr(module, class_name)
    except AttributeError as exc:
        raise RuntimeError(f"Generated class {class_name!r} not found in {module_path}") from exc


def _build_direct_code_prompts(
    *,
    dataset_key: str,
    agent_mode: str,
    iteration: int,
    attempt_index: int,
    seed_model_name: str,
    seed_model_config: Dict[str, Any],
    seed_model_source: str,
    history: Sequence[Dict[str, Any]],
    previous_code: str | None,
    previous_error: str | None,
) -> Dict[str, str]:
    prompt_config = load_yaml(default_direct_code_prompt_config_path())
    skill_card = default_direct_code_skill_card_path().read_text(encoding="utf-8")
    previous_code_text = previous_code.strip() if previous_code and previous_code.strip() else "<none>"
    previous_error_text = previous_error.strip() if previous_error and previous_error.strip() else "<none>"
    user_prompt = Template(str(prompt_config["user_template"])).safe_substitute(
        dataset_key=dataset_key,
        agent_variant=agent_mode,
        iteration=int(iteration),
        attempt_index=int(attempt_index),
        seed_model_name=seed_model_name,
        recent_history_json=json.dumps(_recent_history(history), indent=2, ensure_ascii=True),
        previous_code_text=previous_code_text,
        previous_error_text=previous_error_text,
        seed_model_config_json=json.dumps(seed_model_config, indent=2, ensure_ascii=True),
        seed_model_source=seed_model_source,
    )
    user_prompt = user_prompt.rstrip() + "\n\nDIRECT CODE SKILL CARD:\n" + skill_card.strip() + "\n"
    return {
        "system_prompt": str(prompt_config["system"]).strip(),
        "user_prompt": user_prompt,
    }


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
        "notes": "direct code llm control iteration",
    }


def _run_benchmark_with_capture(
    *,
    repo_root: Path,
    benchmark_config_path: Path,
    run_id: str,
    requested_device: str,
    cuda_visible_devices: str | None,
    runtime_env_config: Path,
    runtime_mode: str,
    runtime_env_group: str | None,
) -> tuple[bool, Path, str, str, float]:
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
    completed = subprocess.run(cmd, cwd=str(repo_root), text=True, capture_output=True, check=False)
    benchmark_payload = load_yaml(benchmark_config_path)
    run_dir = Path(str(benchmark_payload["artifacts_root"])).resolve() / run_id
    return int(completed.returncode) == 0, run_dir, str(completed.stdout), str(completed.stderr), float(perf_counter() - started)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_direct_code_model_session(
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
    llm_config_path: Path,
    agent_mode: str,
) -> Dict[str, Any]:
    if agent_mode not in {"direct_code_llm_singleshot", "direct_code_llm_repairloop"}:
        raise ValueError(f"Unsupported direct code agent_mode: {agent_mode}")
    agent_defaults = load_yaml(default_agent_defaults_path())
    repair_rounds = (
        1
        if agent_mode == "direct_code_llm_singleshot"
        else max(2, _coerce_scalar_int(agent_defaults.get("direct_code_repair_rounds", 2), field_name="direct_code_repair_rounds"))
    )
    llm_client = StructuredLLMClient(load_llm_settings(llm_config_path))
    seed_model_config = load_yaml(seed_model_config_path)
    seed_model_name = str(seed_model_config["name"])
    seed_model_source_path = _seed_model_source_path(repo_root, seed_model_name)
    seed_model_source = seed_model_source_path.read_text(encoding="utf-8")
    session_slug = session_id or beijing_timestamp_slug()
    seed_value = _coerce_scalar_int(seed, field_name="seed")
    top_k_value = _coerce_scalar_int(top_k, field_name="top_k")
    budget_limit = _coerce_scalar_int(max_iteration, field_name="max_iteration")
    session_started_at = perf_counter()

    dataset_root = ensure_dir(agent_root / "datasets" / dataset_key)
    session_root = ensure_dir(dataset_root / "direct_code" / _safe_slug(agent_mode) / _safe_slug(seed_model_name) / session_slug)
    generated_code_root = ensure_dir(session_root / "generated_code")
    raw_response_root = ensure_dir(session_root / "raw_responses")
    benchmark_configs_root = ensure_dir(session_root / "benchmark_configs")
    benchmark_runs_root = ensure_dir(session_root / "benchmark_runs")
    model_configs_root = ensure_dir(session_root / "model_configs")
    history_root = ensure_dir(session_root / "history")
    trace_root = ensure_dir(session_root / "trace")
    prompts_root = ensure_dir(session_root / "prompts")
    attempt_logs_root = ensure_dir(session_root / "attempt_logs")
    prior_session_wall_clock_seconds = load_previous_session_wall_clock_seconds(session_root)

    log_lines = [
        f"dataset_key={dataset_key}",
        f"agent_mode={agent_mode}",
        f"seed_model_name={seed_model_name}",
        f"seed_model_config={seed_model_config_path}",
        f"seed_model_source={seed_model_source_path}",
        f"session_id={session_slug}",
        f"requested_device={requested_device}",
        f"cuda_visible_devices={cuda_visible_devices}",
        f"llm_config_path={llm_config_path}",
        f"repair_rounds={repair_rounds}",
    ]
    log_lines = load_previous_log_lines(session_root / "agent.log", log_lines)

    def log(message: str) -> None:
        log_lines.append(message)
        print(f"[agent-direct-code] {message}", flush=True)

    history: List[Dict[str, Any]] = load_history_rows(history_root)
    attempt_rows: List[Dict[str, Any]] = load_rows_payload(trace_root / "direct_code_attempts.json")
    llm_usage_rows: List[Dict[str, Any]] = load_rows_payload(trace_root / "llm_usage.json")
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
            model_config_path = seed_model_config_path.resolve()
            benchmark_config_path = benchmark_configs_root / f"iter_{iteration:03d}.yaml"
            dump_yaml(
                benchmark_config_path,
                _build_benchmark_payload(
                    dataset_config=dataset_config,
                    split_path=split_path,
                    run_name=f"{dataset_key}_{seed_model_name}_{agent_mode}_iter_{iteration:03d}",
                    artifacts_root=benchmark_runs_root,
                    seed=seed_value,
                    top_k=top_k_value,
                    model_config=model_config_path,
                ),
            )
            log(f"dispatch iteration=0 proposal_source=seed_model proposal_note=seed_model_baseline::{seed_model_name}")
            success, run_dir, stdout_text, stderr_text, evaluation_wall_clock_seconds = _run_benchmark_with_capture(
                repo_root=repo_root,
                benchmark_config_path=benchmark_config_path,
                run_id=f"iter_{iteration:03d}",
                requested_device=requested_device,
                cuda_visible_devices=cuda_visible_devices,
                runtime_env_config=runtime_env_config,
                runtime_mode=runtime_mode,
                runtime_env_group=runtime_env_group,
            )
            write_text(attempt_logs_root / f"iter_{iteration:03d}_stdout.txt", stdout_text)
            write_text(attempt_logs_root / f"iter_{iteration:03d}_stderr.txt", stderr_text)
            if not success:
                raise RuntimeError(f"Seed model baseline failed for direct_code session: {stderr_text or stdout_text}")
            manifest = load_json(run_dir / "manifest.json")
            metrics = load_json(run_dir / "metrics.json")
            objective_raw = _get_nested(metrics, OBJECTIVE_PATH)
            objective_value = float(objective_raw) if isinstance(objective_raw, (int, float)) else None
            baseline_objective = objective_value
            record = {
                "iteration": 0,
                "phase": "baseline",
                "proposal_source": "seed_model",
                "proposal_note": f"seed_model_baseline::{seed_model_name}",
                "attempt_count": 0,
                "execution_status": str(manifest.get("execution_status")),
                "objective_path": OBJECTIVE_PATH,
                "objective_value": objective_value,
                "delta_vs_iteration0": 0.0 if objective_value is not None else None,
                "generated_code_path": None,
                "model_config_path": str(model_config_path),
                "benchmark_config_path": str(benchmark_config_path.resolve()),
                "run_dir": str(run_dir.resolve()),
                "manifest_path": str((run_dir / "manifest.json").resolve()),
                "metrics_path": str((run_dir / "metrics.json").resolve()),
                "requested_device": manifest.get("requested_device"),
                "resolved_device": manifest.get("resolved_device"),
                "model_uses_gpu": manifest.get("model_uses_gpu"),
                "seed_model_name": seed_model_name,
                "agent_mode": agent_mode,
                "repair_rounds": repair_rounds,
                "protocol": manifest.get("protocol"),
                "start_time_beijing": manifest.get("start_time_beijing"),
                "end_time_beijing": manifest.get("end_time_beijing"),
                "evaluation_wall_clock_seconds": float(evaluation_wall_clock_seconds),
            }
            record.update(_flatten_selected(metrics, REPORT_METRIC_PATHS))
            history.append(record)
            dump_json(history_root / f"iter_{iteration:03d}.json", record)
            log(f"completed iteration=0 objective={objective_value}")
            iteration += 1
            continue

        proposal_note = f"{agent_mode}::{seed_model_name}"
        previous_code = ""
        previous_error = ""
        completed_record: Dict[str, Any] | None = None
        for attempt_index in range(1, repair_rounds + 1):
            prompts = _build_direct_code_prompts(
                dataset_key=dataset_key,
                agent_mode=agent_mode,
                iteration=int(iteration),
                attempt_index=int(attempt_index),
                seed_model_name=seed_model_name,
                seed_model_config=seed_model_config,
                seed_model_source=seed_model_source,
                history=history,
                previous_code=previous_code,
                previous_error=previous_error,
            )
            write_text(prompts_root / f"iter_{iteration:03d}_attempt_{attempt_index:02d}_system.txt", prompts["system_prompt"] + "\n")
            write_text(prompts_root / f"iter_{iteration:03d}_attempt_{attempt_index:02d}_user.txt", prompts["user_prompt"] + "\n")
            raw_response_text = ""
            generated_code_path = generated_code_root / f"iter_{iteration:03d}_attempt_{attempt_index:02d}_model.py"
            model_config_path = model_configs_root / f"iter_{iteration:03d}_attempt_{attempt_index:02d}.yaml"
            benchmark_config_path = benchmark_configs_root / f"iter_{iteration:03d}_attempt_{attempt_index:02d}.yaml"

            try:
                raw_response_text = llm_client.chat_text(
                    system_prompt=prompts["system_prompt"],
                    user_prompt=prompts["user_prompt"],
                )
                for usage_event in llm_client.drain_usage_events():
                    llm_usage_rows.append(
                        {
                            "iteration": int(iteration),
                            "attempt_index": int(attempt_index),
                            **usage_event,
                        }
                    )
                code_text = _extract_python_code(raw_response_text)
                write_text(raw_response_root / f"iter_{iteration:03d}_attempt_{attempt_index:02d}.txt", raw_response_text + "\n")
                write_text(generated_code_path, code_text.rstrip() + "\n")
                _load_generated_class(generated_code_path, DIRECT_CODE_CLASS_NAME)

                model_payload = dict(seed_model_config)
                model_payload["name"] = f"generated_direct_code::{_safe_slug(agent_mode)}"
                model_payload["factory"] = {
                    "module_path": str(generated_code_path.resolve()),
                    "class_name": DIRECT_CODE_CLASS_NAME,
                }
                model_payload["seed_model_name"] = seed_model_name
                model_payload["agent_mode"] = agent_mode
                model_payload["proposal_note"] = proposal_note
                dump_yaml(model_config_path, model_payload)
                dump_yaml(
                    benchmark_config_path,
                    _build_benchmark_payload(
                        dataset_config=dataset_config,
                        split_path=split_path,
                        run_name=f"{dataset_key}_{seed_model_name}_{agent_mode}_iter_{iteration:03d}",
                        artifacts_root=benchmark_runs_root,
                        seed=seed_value,
                        top_k=top_k_value,
                        model_config=model_config_path,
                    ),
                )
                log(
                    f"dispatch iteration={iteration} attempt={attempt_index} proposal_source=direct_code_llm "
                    f"proposal_note={proposal_note}"
                )
                success, run_dir, stdout_text, stderr_text, evaluation_wall_clock_seconds = _run_benchmark_with_capture(
                    repo_root=repo_root,
                    benchmark_config_path=benchmark_config_path,
                    run_id=f"iter_{iteration:03d}_attempt_{attempt_index:02d}",
                    requested_device=requested_device,
                    cuda_visible_devices=cuda_visible_devices,
                    runtime_env_config=runtime_env_config,
                    runtime_mode=runtime_mode,
                    runtime_env_group=runtime_env_group,
                )
                write_text(attempt_logs_root / f"iter_{iteration:03d}_attempt_{attempt_index:02d}_stdout.txt", stdout_text)
                write_text(attempt_logs_root / f"iter_{iteration:03d}_attempt_{attempt_index:02d}_stderr.txt", stderr_text)
                attempt_rows.append(
                    {
                        "iteration": int(iteration),
                        "attempt_index": int(attempt_index),
                        "proposal_source": "direct_code_llm",
                        "execution_status": "completed" if success else "failed",
                        "generated_code_path": str(generated_code_path.resolve()),
                        "model_config_path": str(model_config_path.resolve()),
                        "benchmark_config_path": str(benchmark_config_path.resolve()),
                        "evaluation_wall_clock_seconds": float(evaluation_wall_clock_seconds),
                    }
                )
                if success:
                    manifest = load_json(run_dir / "manifest.json")
                    metrics = load_json(run_dir / "metrics.json")
                    objective_raw = _get_nested(metrics, OBJECTIVE_PATH)
                    objective_value = float(objective_raw) if isinstance(objective_raw, (int, float)) else None
                    canonical_code_path = generated_code_root / f"iter_{iteration:03d}_model.py"
                    canonical_model_config_path = model_configs_root / f"iter_{iteration:03d}.yaml"
                    write_text(canonical_code_path, generated_code_path.read_text(encoding="utf-8"))
                    dump_yaml(canonical_model_config_path, model_payload)
                    completed_record = {
                        "iteration": int(iteration),
                        "phase": "agent",
                        "proposal_source": "direct_code_llm",
                        "proposal_note": proposal_note,
                        "attempt_count": int(attempt_index),
                        "execution_status": str(manifest.get("execution_status")),
                        "objective_path": OBJECTIVE_PATH,
                        "objective_value": objective_value,
                        "delta_vs_iteration0": (
                            float(baseline_objective) - float(objective_value)
                            if baseline_objective is not None and objective_value is not None
                            else None
                        ),
                        "generated_code_path": str(canonical_code_path.resolve()),
                        "model_config_path": str(canonical_model_config_path.resolve()),
                        "benchmark_config_path": str(benchmark_config_path.resolve()),
                        "run_dir": str(run_dir.resolve()),
                        "manifest_path": str((run_dir / "manifest.json").resolve()),
                        "metrics_path": str((run_dir / "metrics.json").resolve()),
                        "requested_device": manifest.get("requested_device"),
                        "resolved_device": manifest.get("resolved_device"),
                        "model_uses_gpu": manifest.get("model_uses_gpu"),
                        "seed_model_name": seed_model_name,
                        "agent_mode": agent_mode,
                        "repair_rounds": repair_rounds,
                        "protocol": manifest.get("protocol"),
                        "start_time_beijing": manifest.get("start_time_beijing"),
                        "end_time_beijing": manifest.get("end_time_beijing"),
                        "evaluation_wall_clock_seconds": float(evaluation_wall_clock_seconds),
                    }
                    completed_record.update(_flatten_selected(metrics, REPORT_METRIC_PATHS))
                    break

                previous_code = generated_code_path.read_text(encoding="utf-8")
                previous_error = (stderr_text or stdout_text or "benchmark_failed_without_error_output")[-12000:]
                log(
                    f"iteration={iteration} attempt={attempt_index} benchmark_failed "
                    f"repair_possible={attempt_index < repair_rounds}"
                )
            except Exception:
                for usage_event in llm_client.drain_usage_events():
                    llm_usage_rows.append(
                        {
                            "iteration": int(iteration),
                            "attempt_index": int(attempt_index),
                            **usage_event,
                        }
                    )
                previous_error = traceback.format_exc()[-12000:]
                if raw_response_text:
                    write_text(raw_response_root / f"iter_{iteration:03d}_attempt_{attempt_index:02d}.txt", raw_response_text + "\n")
                attempt_rows.append(
                    {
                        "iteration": int(iteration),
                        "attempt_index": int(attempt_index),
                        "proposal_source": "direct_code_llm",
                        "execution_status": "failed_pre_benchmark",
                        "generated_code_path": str(generated_code_path.resolve()),
                        "model_config_path": str(model_config_path.resolve()),
                        "benchmark_config_path": str(benchmark_config_path.resolve()),
                        "evaluation_wall_clock_seconds": None,
                    }
                )
                write_text(attempt_logs_root / f"iter_{iteration:03d}_attempt_{attempt_index:02d}_stderr.txt", previous_error)
                error_preview = " | ".join(
                    line.strip()
                    for line in previous_error.strip().splitlines()[-2:]
                    if line.strip()
                )[:400]
                log(
                    f"iteration={iteration} attempt={attempt_index} code_generation_failed "
                    f"repair_possible={attempt_index < repair_rounds} "
                    f"error_preview={error_preview or 'unknown_error'}"
                )
            previous_code = generated_code_path.read_text(encoding="utf-8") if generated_code_path.exists() else previous_code

        if completed_record is None:
            failed_record = {
                "iteration": int(iteration),
                "phase": "agent",
                "proposal_source": "direct_code_llm",
                "proposal_note": proposal_note,
                "attempt_count": int(repair_rounds),
                "execution_status": "failed",
                "objective_path": OBJECTIVE_PATH,
                "objective_value": None,
                "delta_vs_iteration0": None,
                "generated_code_path": str((generated_code_root / f"iter_{iteration:03d}_attempt_{repair_rounds:02d}_model.py").resolve()),
                "model_config_path": None,
                "benchmark_config_path": None,
                "run_dir": None,
                "manifest_path": None,
                "metrics_path": None,
                "requested_device": requested_device,
                "resolved_device": None,
                "model_uses_gpu": None,
                "seed_model_name": seed_model_name,
                "agent_mode": agent_mode,
                "repair_rounds": repair_rounds,
                "protocol": None,
                "error_tail": previous_error,
                "evaluation_wall_clock_seconds": None,
            }
            history.append(failed_record)
            dump_json(history_root / f"iter_{iteration:03d}.json", failed_record)
            log(f"completed iteration={iteration} objective=None execution_status=failed")
            iteration += 1
            continue

        history.append(completed_record)
        dump_json(history_root / f"iter_{iteration:03d}.json", completed_record)
        log(f"completed iteration={iteration} objective={completed_record.get('objective_value')}")
        iteration += 1

    completed = [item for item in history if item.get("execution_status") == "completed" and item.get("objective_value") is not None]
    if not completed:
        raise RuntimeError(f"No completed direct-code iterations for {dataset_key}/{seed_model_name}/{agent_mode}")
    best_record = min(completed, key=lambda item: float(item["objective_value"]))
    if stopping_reason is None:
        stopping_reason = default_stopping_reason(history, budget_limit)
    llm_usage_summary = aggregate_llm_usage(llm_usage_rows)
    completed_count = completed_evaluation_count(history)
    failed_candidate_count = int(len([row for row in attempt_rows if str(row.get("execution_status")) != "completed"]))
    candidate_attempt_count = int(completed_count + failed_candidate_count)
    session_wall_clock_seconds = prior_session_wall_clock_seconds + float(perf_counter() - session_started_at)
    for record in history:
        record["best_so_far_iteration"] = int(best_record["iteration"])
        record["best_so_far_objective"] = float(best_record["objective_value"])
        record["accepted_as_best"] = bool(record["iteration"] == best_record["iteration"])

    dump_json(session_root / "iterations.json", {"iterations": history})
    _write_csv(session_root / "iterations.csv", history)
    dump_json(session_root / "best_iteration.json", best_record)
    dump_json(trace_root / "direct_code_attempts.json", {"rows": attempt_rows})
    if attempt_rows:
        _write_csv(trace_root / "direct_code_attempts.csv", attempt_rows)
    dump_json(trace_root / "llm_usage.json", {"rows": llm_usage_rows, "summary": llm_usage_summary})
    if llm_usage_rows:
        _write_csv(trace_root / "llm_usage.csv", llm_usage_rows)

    session_manifest = {
        "dataset_key": dataset_key,
        "method_family": "agent_direct_code_model",
        "method_name": f"{agent_mode}::{seed_model_name}",
        "agent_line": "direct_code_llm",
        "agent_mode": agent_mode,
        "agent_variant": agent_mode,
        "seed_model_name": seed_model_name,
        "seed_model_config_path": str(seed_model_config_path.resolve()),
        "seed_model_source_path": str(seed_model_source_path.resolve()),
        "session_id": session_slug,
        "session_root": str(session_root.resolve()),
        "generated_code_root": str(generated_code_root.resolve()),
        "requested_device": requested_device,
        "cuda_visible_devices": cuda_visible_devices,
        "llm_config_path": str(llm_config_path.resolve()),
        "llm_strategy": "llm_direct_code",
        "repair_rounds": int(repair_rounds),
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
        "llm_request_count": int(llm_usage_summary["llm_request_count"]),
        "llm_prompt_tokens": int(llm_usage_summary["llm_prompt_tokens"]),
        "llm_completion_tokens": int(llm_usage_summary["llm_completion_tokens"]),
        "llm_total_tokens": int(llm_usage_summary["llm_total_tokens"]),
        "llm_repair_request_count": int(llm_usage_summary["llm_repair_request_count"]),
        "session_hash": stable_hash(
            {
                "dataset_key": dataset_key,
                "agent_mode": agent_mode,
                "seed_model_name": seed_model_name,
                "session_id": session_slug,
                "seed": int(seed),
            }
        ),
    }
    dump_json(session_root / "agent_session.json", session_manifest)
    write_text(session_root / "agent.log", "\n".join(log_lines) + "\n")
    log(
        "search_stopped "
        f"completed_evaluations={completed_count}/{budget_limit} "
        f"candidate_attempts={candidate_attempt_count}/{candidate_attempt_budget_limit(budget_limit)} "
        f"failed_candidates={failed_candidate_count} "
        f"stopping_reason={stopping_reason}"
    )

    baseline_record = history[0]
    return {
        "dataset_key": dataset_key,
        "method_name": f"{agent_mode}::{seed_model_name}",
        "method_family": "agent_direct_code_model",
        "agent_line": "direct_code_llm",
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
        "llm_strategy": "llm_direct_code",
        "llm_enabled": True,
        "repair_rounds": int(repair_rounds),
        "budget_limit_completed_evaluations": int(budget_limit),
        "candidate_attempt_budget_limit": int(candidate_attempt_budget_limit(budget_limit)),
        "completed_evaluation_count": int(completed_count),
        "failed_candidate_count": int(failed_candidate_count),
        "candidate_attempt_count": int(candidate_attempt_count),
        "stopping_rule": MATCHED_BUDGET_STOPPING_RULE,
        "stopping_reason": stopping_reason,
        "session_wall_clock_seconds": float(session_wall_clock_seconds),
        "llm_request_count": int(llm_usage_summary["llm_request_count"]),
        "llm_prompt_tokens": int(llm_usage_summary["llm_prompt_tokens"]),
        "llm_completion_tokens": int(llm_usage_summary["llm_completion_tokens"]),
        "llm_total_tokens": int(llm_usage_summary["llm_total_tokens"]),
        "llm_repair_request_count": int(llm_usage_summary["llm_repair_request_count"]),
        **{f"baseline.{metric}": baseline_record.get(metric) for metric in REPORT_METRIC_PATHS},
        **{f"best.{metric}": best_record.get(metric) for metric in REPORT_METRIC_PATHS},
    }
