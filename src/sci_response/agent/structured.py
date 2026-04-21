from __future__ import annotations

import json
import subprocess
import sys
from time import perf_counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from sci_response.agent.compiler import compile_model_ir, render_generated_model_source
from sci_response.agent.config import default_mechanism_template_path, load_harness_config
from sci_response.agent.edits import apply_edit_sequence, dedupe_edit_sequences, describe_edit, enumerate_candidate_edits
from sci_response.agent.harness import (
    build_harness_review,
    changed_ir_paths,
    infer_hypothesis_axes,
    prioritize_mechanism_templates,
    summarize_rejection_reasons,
)
from sci_response.agent.ir import extract_model_ir
from sci_response.agent.llm import StructuredLLMClient, llm_propose_edit_sequence, load_llm_settings
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
from sci_response.baselines.artifacts import beijing_timestamp_slug
from sci_response.data.io import dump_json, dump_yaml, ensure_dir, load_json, load_yaml, stable_hash, write_text


OBJECTIVE_PATH = "val.delta.mse"
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
        "notes": "structured hypothesis search iteration",
    }


def _heuristic_edit_sequence(
    *,
    current_ir: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
    exclude_ir_hashes: Sequence[str] = (),
) -> Dict[str, Any]:
    seen_ir_hashes = {str(item.get("ir_hash")) for item in history}
    seen_ir_hashes.update(str(item) for item in exclude_ir_hashes)
    single_edits = [[edit] for edit in enumerate_candidate_edits(current_ir)]
    paired_edits: List[List[Dict[str, Any]]] = []
    primitive_edits = enumerate_candidate_edits(current_ir)
    for first in primitive_edits:
        for second in primitive_edits:
            if first["path"] == second["path"]:
                continue
            paired_edits.append([first, second])
    for sequence in dedupe_edit_sequences(single_edits + paired_edits):
        candidate_ir = apply_edit_sequence(current_ir, sequence)
        ir_hash = stable_hash(candidate_ir)
        if ir_hash in seen_ir_hashes:
            continue
        return {
            "proposal_source": "heuristic",
            "proposal_note": ", ".join(describe_edit(edit) for edit in sequence),
            "selected_mechanism_templates": infer_hypothesis_axes(changed_ir_paths(current_ir, candidate_ir)),
            "scientific_hypothesis": "Heuristic structured perturbation-response architecture edit.",
            "mechanistic_rationale": "Probe one local structural hypothesis while keeping data, split, and evaluation fixed.",
            "risk_notes": [],
            "edits": sequence,
            "candidate_ir": candidate_ir,
            "ir_hash": ir_hash,
        }
    return {
        "proposal_source": "heuristic",
        "proposal_note": "no_unseen_structured_edit_available",
        "selected_mechanism_templates": [],
        "scientific_hypothesis": "No unseen local structured edit remains under the current heuristic budget.",
        "mechanistic_rationale": "Search frontier exhausted under the current primitive set.",
        "risk_notes": ["search_frontier_exhausted"],
        "edits": [],
        "candidate_ir": current_ir,
        "ir_hash": stable_hash(current_ir),
    }


def _should_fallback_after_llm_failure(exc: Exception) -> bool:
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        return True
    message = str(exc).lower()
    fallback_signals = [
        "does not contain a json object",
        "expected json object",
        "unexpected llm response schema",
        "request failed across prompt variants",
        "request failed",
        "timed out",
        "connection reset",
        "temporarily unavailable",
        "already-seen ir",
        "repair_failed",
        "llm http error",
        "http error 400",
        "api key expired",
        "renew the api key",
        "v_api_biz_error",
    ]
    if any(signal in message for signal in fallback_signals):
        return True
    if isinstance(exc, RuntimeError) and ("llm" in message or "api key" in message or "http error" in message):
        return True
    return False


def _propose_next_ir(
    *,
    dataset_key: str,
    current_ir: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
    attempt_index: int,
    recent_rejections: Sequence[Dict[str, Any]],
    prioritized_templates: Dict[str, Any],
    llm_client: StructuredLLMClient | None,
    llm_strategy: str,
    exclude_ir_hashes: Sequence[str],
    log_fn,
) -> Dict[str, Any]:
    if llm_client is not None and llm_strategy in {"llm", "hybrid"}:
        try:
            proposal = llm_propose_edit_sequence(
                llm_client=llm_client,
                dataset_key=dataset_key,
                model_ir=current_ir,
                history=list(history),
                attempt_index=int(attempt_index),
                recent_rejections=list(recent_rejections),
                prioritized_templates=prioritized_templates,
            )
            candidate_ir = apply_edit_sequence(current_ir, proposal["edits"])
            ir_hash = stable_hash(candidate_ir)
            seen_ir_hashes = {str(item.get("ir_hash")) for item in history}
            if ir_hash in seen_ir_hashes:
                raise ValueError("LLM proposed an already-seen IR")
            log_fn(
                f"llm proposal accepted provider={llm_client.settings.provider} "
                f"model={llm_client.settings.model} note={proposal['proposal_note']}"
            )
            return {
                "proposal_source": "llm",
                "proposal_note": proposal["proposal_note"],
                "selected_mechanism_templates": proposal.get("selected_mechanism_templates", []),
                "scientific_hypothesis": proposal.get("scientific_hypothesis", ""),
                "mechanistic_rationale": proposal.get("mechanistic_rationale", ""),
                "risk_notes": proposal.get("risk_notes", []),
                "edits": proposal["edits"],
                "candidate_ir": candidate_ir,
                "ir_hash": ir_hash,
            }
        except Exception as exc:
            log_fn(f"llm proposal failed fallback_to_heuristic reason={type(exc).__name__}: {exc}")
            if llm_strategy == "llm" and not _should_fallback_after_llm_failure(exc):
                raise
    fallback = _heuristic_edit_sequence(
        current_ir=current_ir,
        history=history,
        exclude_ir_hashes=exclude_ir_hashes,
    )
    fallback["proposal_source"] = "heuristic_after_llm_failure"
    fallback["risk_notes"] = list(fallback.get("risk_notes", [])) + ["llm_proposal_failed_fallback_used"]
    return fallback


def _write_trace_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    import csv

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_structured_model_session(
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
    llm_config_path: Path | None,
    llm_strategy: str,
    agent_mode: str = "structured_llm_search",
) -> Dict[str, Any]:
    seed_model_config = load_yaml(seed_model_config_path)
    seed_model_name = str(seed_model_config["name"])
    session_slug = session_id or beijing_timestamp_slug()
    dataset_root = ensure_dir(agent_root / "datasets" / dataset_key)
    session_root = ensure_dir(dataset_root / "models" / _safe_slug(seed_model_name) / session_slug)
    ir_root = ensure_dir(session_root / "ir")
    edit_root = ensure_dir(session_root / "edits")
    compiled_root = ensure_dir(session_root / "compiled_models")
    generated_code_root = ensure_dir(session_root / "generated_code")
    benchmark_configs_root = ensure_dir(session_root / "benchmark_configs")
    benchmark_runs_root = ensure_dir(session_root / "benchmark_runs")
    history_root = ensure_dir(session_root / "history")
    harness_root = ensure_dir(session_root / "harness")
    trace_root = ensure_dir(session_root / "trace")
    rejection_root = ensure_dir(session_root / "rejections")
    harness_config = load_harness_config()
    max_attempts = int(harness_config.get("proposal_attempts_per_iteration", 3))
    budget_limit = int(max_iteration)
    session_started_at = perf_counter()
    prior_session_wall_clock_seconds = load_previous_session_wall_clock_seconds(session_root)
    llm_client = (
        StructuredLLMClient(load_llm_settings(llm_config_path))
        if llm_config_path is not None and llm_strategy in {"llm", "hybrid"}
        else None
    )

    log_lines = [
        f"dataset_key={dataset_key}",
        f"seed_model_config={seed_model_config_path}",
        f"seed_model_name={seed_model_name}",
        f"session_id={session_slug}",
        f"requested_device={requested_device}",
        f"cuda_visible_devices={cuda_visible_devices}",
        f"llm_config_path={llm_config_path}",
        f"llm_strategy={llm_strategy}",
        "harness_mode=mechanism_fidelity_structured_search_v1",
    ]
    log_lines = load_previous_log_lines(session_root / "agent.log", log_lines)

    def log(message: str) -> None:
        log_lines.append(message)
        print(f"[agent-structured] {message}", flush=True)

    history: List[Dict[str, Any]] = load_history_rows(history_root)
    accepted_trace_rows: List[Dict[str, Any]] = load_rows_payload(trace_root / "edit_program_trace.json")
    rejection_rows: List[Dict[str, Any]] = load_rows_payload(rejection_root / "proposal_rejections.json")
    template_priority_rows: List[Dict[str, Any]] = load_rows_payload(trace_root / "template_priority_trace.json")
    llm_usage_rows: List[Dict[str, Any]] = load_rows_payload(trace_root / "llm_usage.json")
    seed_ir = extract_model_ir(seed_model_config)
    current_reference_ir = seed_ir
    baseline_objective: float | None = baseline_objective_from_history(history)
    template_library = load_yaml(default_mechanism_template_path())
    stopping_reason: str | None = None
    iteration = next_iteration_index(history)
    if history:
        log(
            f"resume session_id={session_slug} next_iteration={iteration} "
            f"completed_evaluations={completed_evaluation_count(history)} candidate_attempts={len(history)}"
        )

    while should_continue_search(history, budget_limit):
        reference_objective_value: float | None = None
        if iteration == 0:
            proposal = {
                "proposal_source": "seed_model_ir",
                "proposal_note": f"iteration_0_seed_ir::{seed_model_name}",
                "selected_mechanism_templates": [],
                "edits": [],
                "candidate_ir": seed_ir,
                "ir_hash": stable_hash(seed_ir),
            }
        else:
            best_completed = [item for item in history if item.get("execution_status") == "completed" and item.get("objective_value") is not None]
            if best_completed:
                best_reference_record = min(best_completed, key=lambda item: float(item["objective_value"]))
                current_reference_ir = dict(best_reference_record["model_ir"])
                reference_objective_value = float(best_reference_record["objective_value"])
            proposal = None
            accepted_payload = None
            iteration_rejections: List[Dict[str, Any]] = []
            attempted_ir_hashes: set[str] = set()
            for attempt_index in range(1, max_attempts + 1):
                template_priority_payload = prioritize_mechanism_templates(
                    dataset_key=dataset_key,
                    model_ir=current_reference_ir,
                    history=history,
                    template_library=template_library,
                    harness_config=harness_config,
                    recent_rejections=iteration_rejections,
                )
                try:
                    candidate_proposal = _propose_next_ir(
                        dataset_key=dataset_key,
                        current_ir=current_reference_ir,
                        history=history,
                        attempt_index=attempt_index,
                        recent_rejections=iteration_rejections,
                        prioritized_templates=template_priority_payload,
                        llm_client=llm_client,
                        llm_strategy=llm_strategy,
                        exclude_ir_hashes=sorted(attempted_ir_hashes),
                        log_fn=log,
                    )
                finally:
                    if llm_client is not None:
                        for usage_event in llm_client.drain_usage_events():
                            llm_usage_rows.append(
                                {
                                    "iteration": int(iteration),
                                    "attempt_index": int(attempt_index),
                                    **usage_event,
                                }
                            )
                if str(candidate_proposal["ir_hash"]) in attempted_ir_hashes:
                    rejection_reasons = ["retry::duplicate_candidate_ir_within_iteration"]
                    rejection_record = {
                        "iteration": int(iteration),
                        "attempt_index": int(attempt_index),
                        "proposal_source": candidate_proposal.get("proposal_source"),
                        "proposal_note": candidate_proposal.get("proposal_note"),
                        "selected_mechanism_templates": list(candidate_proposal.get("selected_mechanism_templates", [])),
                        "hypothesis_axes": list(candidate_proposal.get("selected_mechanism_templates", [])),
                        "rejection_reasons": rejection_reasons,
                    }
                    iteration_rejections.append(rejection_record)
                    rejection_rows.append(
                        {
                            "iteration": int(iteration),
                            "attempt_index": int(attempt_index),
                            "proposal_source": candidate_proposal.get("proposal_source"),
                            "proposal_note": candidate_proposal.get("proposal_note"),
                            "selected_mechanism_templates_json": json.dumps(rejection_record["selected_mechanism_templates"], ensure_ascii=True),
                            "hypothesis_axes_json": json.dumps(rejection_record["hypothesis_axes"], ensure_ascii=True),
                            "rejection_reasons_json": json.dumps(rejection_reasons, ensure_ascii=True),
                            "rejection_reasons_count": int(len(rejection_reasons)),
                        }
                    )
                    log(
                        f"proposal_rejected iteration={iteration} attempt={attempt_index} "
                        f"reasons={rejection_reasons}"
                    )
                    continue
                attempted_ir_hashes.add(str(candidate_proposal["ir_hash"]))
                candidate_ir = candidate_proposal["candidate_ir"]
                generated_code_path = generated_code_root / f"iter_{iteration:03d}_attempt_{attempt_index:02d}_model.py"
                generated_class_name = "GeneratedStructuredHypothesisRegressor"
                compiled_model_config = compile_model_ir(
                    candidate_ir,
                    generated_module_path=generated_code_path,
                    generated_class_name=generated_class_name,
                )
                write_text(
                    generated_code_path,
                    render_generated_model_source(candidate_ir, class_name=generated_class_name),
                )
                harness_review = build_harness_review(
                    previous_ir=current_reference_ir,
                    candidate_ir=candidate_ir,
                    proposal=candidate_proposal,
                    compiled_model_config=compiled_model_config,
                    generated_code_path=generated_code_path,
                )
                if bool(harness_review["passed"]):
                    proposal = candidate_proposal
                    accepted_payload = {
                        "candidate_ir": candidate_ir,
                        "compiled_model_config": compiled_model_config,
                        "generated_code_path": generated_code_path,
                        "generated_class_name": generated_class_name,
                        "harness_review": harness_review,
                        "attempt_index": attempt_index,
                        "template_priority_payload": template_priority_payload,
                    }
                    break
                rejection_reasons = summarize_rejection_reasons(harness_review)
                rejection_record = {
                    "iteration": int(iteration),
                    "attempt_index": int(attempt_index),
                    "proposal_source": candidate_proposal.get("proposal_source"),
                    "proposal_note": candidate_proposal.get("proposal_note"),
                    "selected_mechanism_templates": list(candidate_proposal.get("selected_mechanism_templates", [])),
                    "hypothesis_axes": list(harness_review["mechanism_report"].get("hypothesis_axes", [])),
                    "rejection_reasons": rejection_reasons,
                }
                iteration_rejections.append(rejection_record)
                rejection_rows.append(
                    {
                        "iteration": int(iteration),
                        "attempt_index": int(attempt_index),
                        "proposal_source": candidate_proposal.get("proposal_source"),
                        "proposal_note": candidate_proposal.get("proposal_note"),
                        "selected_mechanism_templates_json": json.dumps(
                            list(candidate_proposal.get("selected_mechanism_templates", [])),
                            ensure_ascii=True,
                        ),
                        "hypothesis_axes_json": json.dumps(
                            list(harness_review["mechanism_report"].get("hypothesis_axes", [])),
                            ensure_ascii=True,
                        ),
                        "rejection_reasons_json": json.dumps(rejection_reasons, ensure_ascii=True),
                        "rejection_reasons_count": int(len(rejection_reasons)),
                    }
                )
                dump_json(rejection_root / f"iter_{iteration:03d}_attempt_{attempt_index:02d}.json", {
                    "proposal": candidate_proposal,
                    "template_priority": template_priority_payload,
                    "harness_review": harness_review,
                    "rejection_reasons": rejection_reasons,
                })
                log(
                    f"proposal_rejected iteration={iteration} attempt={attempt_index} "
                    f"reasons={rejection_reasons}"
                )
            if proposal is None or accepted_payload is None:
                stopping_reason = "no_acceptable_proposal_after_attempt_budget"
                log(f"stopping iteration={iteration} reason={stopping_reason}")
                break

        if iteration == 0:
            candidate_ir = proposal["candidate_ir"]
            ir_path = ir_root / f"iter_{iteration:03d}.json"
            edit_path = edit_root / f"iter_{iteration:03d}.json"
            compiled_model_path = compiled_root / f"iter_{iteration:03d}.yaml"
            generated_code_path = generated_code_root / f"iter_{iteration:03d}_model.py"
            benchmark_config_path = benchmark_configs_root / f"iter_{iteration:03d}.yaml"
            harness_review_path = harness_root / f"iter_{iteration:03d}.json"
            template_priority_path = trace_root / f"iter_{iteration:03d}_template_priority.json"
            generated_class_name = "GeneratedStructuredHypothesisRegressor"
            compiled_model_config = compile_model_ir(
                candidate_ir,
                generated_module_path=generated_code_path,
                generated_class_name=generated_class_name,
            )
            template_priority_payload = {
                "dataset_key": dataset_key,
                "ranked_templates": [],
                "dataset_tags": [],
                "top_k": 0,
            }
            attempt_index_value = 0
        else:
            candidate_ir = accepted_payload["candidate_ir"]
            ir_path = ir_root / f"iter_{iteration:03d}.json"
            edit_path = edit_root / f"iter_{iteration:03d}.json"
            compiled_model_path = compiled_root / f"iter_{iteration:03d}.yaml"
            generated_code_path = generated_code_root / f"iter_{iteration:03d}_model.py"
            benchmark_config_path = benchmark_configs_root / f"iter_{iteration:03d}.yaml"
            harness_review_path = harness_root / f"iter_{iteration:03d}.json"
            template_priority_path = trace_root / f"iter_{iteration:03d}_template_priority.json"
            generated_class_name = accepted_payload["generated_class_name"]
            template_priority_payload = accepted_payload["template_priority_payload"]
            attempt_index_value = int(accepted_payload["attempt_index"])
            compiled_model_config = compile_model_ir(
                candidate_ir,
                generated_module_path=generated_code_path,
                generated_class_name=generated_class_name,
            )

        dump_json(ir_path, candidate_ir)
        dump_json(
            edit_path,
            {
                "iteration": int(iteration),
                "proposal_source": proposal["proposal_source"],
                "proposal_note": proposal["proposal_note"],
                "attempt_index": int(attempt_index_value),
                "selected_mechanism_templates": list(proposal.get("selected_mechanism_templates", [])),
                "scientific_hypothesis": proposal.get("scientific_hypothesis", ""),
                "mechanistic_rationale": proposal.get("mechanistic_rationale", ""),
                "risk_notes": proposal.get("risk_notes", []),
                "edits": proposal["edits"],
                "ir_hash": proposal["ir_hash"],
            },
        )
        dump_json(template_priority_path, template_priority_payload)
        write_text(
            generated_code_path,
            render_generated_model_source(candidate_ir, class_name=generated_class_name),
        )
        harness_review = build_harness_review(
            previous_ir=current_reference_ir if iteration > 0 else seed_ir,
            candidate_ir=candidate_ir,
            proposal=proposal,
            compiled_model_config=compiled_model_config,
            generated_code_path=generated_code_path,
        )
        if iteration == 0:
            harness_review["passed"] = True
            harness_review["mechanism_report"]["mechanism_guard_passed"] = True
            harness_review["mechanism_report"]["risk_flags"] = [
                flag for flag in harness_review["mechanism_report"]["risk_flags"] if flag != "no_structural_change"
            ]
        dump_json(harness_review_path, harness_review)
        if not bool(harness_review["passed"]):
            raise RuntimeError(
                f"Harness review failed at iteration {iteration}: "
                f"{harness_review['edit_validation']['errors']} "
                f"{harness_review['mechanism_report']['risk_flags']} "
                f"{harness_review['semantic_code_consistency']['errors']}"
            )
        dump_yaml(compiled_model_path, compiled_model_config)
        dump_yaml(
            benchmark_config_path,
            _build_benchmark_payload(
                dataset_config=dataset_config,
                split_path=split_path,
                run_name=f"{dataset_key}_{seed_model_name}_agent_structured_iter_{iteration:03d}",
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
            objective_value_raw = _get_nested(metrics, OBJECTIVE_PATH)
            objective_value = float(objective_value_raw) if isinstance(objective_value_raw, (int, float)) else None
            if iteration == 0:
                baseline_objective = objective_value
            record = {
                "iteration": int(iteration),
                "phase": "baseline" if iteration == 0 else "agent",
                "proposal_source": proposal["proposal_source"],
                "proposal_note": proposal["proposal_note"],
                "attempt_index": int(attempt_index_value),
                "selected_mechanism_templates": list(proposal.get("selected_mechanism_templates", [])),
                "scientific_hypothesis": proposal.get("scientific_hypothesis", ""),
                "mechanistic_rationale": proposal.get("mechanistic_rationale", ""),
                "risk_notes": proposal.get("risk_notes", []),
                "edits": proposal["edits"],
                "edit_preview": [describe_edit(edit) for edit in proposal["edits"]],
                "ir_hash": proposal["ir_hash"],
                "harness_review_path": str(harness_review_path.resolve()),
                "template_priority_path": str(template_priority_path.resolve()),
                "hypothesis_axes": list(harness_review["mechanism_report"]["hypothesis_axes"]),
                "mechanism_guard_passed": bool(harness_review["mechanism_report"]["mechanism_guard_passed"]),
                "semantic_code_consistency_passed": bool(harness_review["semantic_code_consistency"]["passed"]),
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
                "reference_objective_value": reference_objective_value,
                "delta_vs_reference": (
                    float(reference_objective_value) - float(objective_value)
                    if reference_objective_value is not None and objective_value is not None
                    else None
                ),
                "delta_vs_iteration0": (
                    float(baseline_objective) - float(objective_value)
                    if baseline_objective is not None and objective_value is not None
                    else None
                ),
                "seed_model_name": seed_model_name,
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
                "attempt_index": int(attempt_index_value),
                "selected_mechanism_templates": list(proposal.get("selected_mechanism_templates", [])),
                "scientific_hypothesis": proposal.get("scientific_hypothesis", ""),
                "mechanistic_rationale": proposal.get("mechanistic_rationale", ""),
                "risk_notes": list(proposal.get("risk_notes", [])) + ["benchmark_execution_failed"],
                "edits": proposal["edits"],
                "edit_preview": [describe_edit(edit) for edit in proposal["edits"]],
                "ir_hash": proposal["ir_hash"],
                "harness_review_path": str(harness_review_path.resolve()),
                "template_priority_path": str(template_priority_path.resolve()),
                "hypothesis_axes": list(harness_review["mechanism_report"]["hypothesis_axes"]),
                "mechanism_guard_passed": bool(harness_review["mechanism_report"]["mechanism_guard_passed"]),
                "semantic_code_consistency_passed": bool(harness_review["semantic_code_consistency"]["passed"]),
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
                "reference_objective_value": reference_objective_value,
                "delta_vs_reference": None,
                "delta_vs_iteration0": None,
                "seed_model_name": seed_model_name,
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
        if str(record.get("execution_status")) == "completed":
            accepted_trace_rows.append(
                {
                    "iteration": int(iteration),
                    "attempt_index": int(attempt_index_value),
                    "proposal_source": proposal["proposal_source"],
                    "proposal_note": proposal["proposal_note"],
                    "selected_mechanism_templates_json": json.dumps(
                        list(proposal.get("selected_mechanism_templates", [])),
                        ensure_ascii=True,
                    ),
                    "scientific_hypothesis": proposal.get("scientific_hypothesis", ""),
                    "mechanistic_rationale": proposal.get("mechanistic_rationale", ""),
                    "changed_paths_json": json.dumps(harness_review["mechanism_report"]["changed_paths"], ensure_ascii=True),
                    "hypothesis_axes_json": json.dumps(harness_review["mechanism_report"]["hypothesis_axes"], ensure_ascii=True),
                    "template_priority_top_json": json.dumps(
                        [item.get("name") for item in template_priority_payload.get("ranked_templates", [])],
                        ensure_ascii=True,
                    ),
                    "generated_code_path": str(generated_code_path.resolve()),
                    "compiled_model_config_path": str(compiled_model_path.resolve()),
                    "benchmark_run_dir": str(record.get("run_dir")),
                    "objective_value": record.get("objective_value"),
                }
            )
            for row in template_priority_payload.get("ranked_templates", []):
                template_priority_rows.append(
                    {
                        "iteration": int(iteration),
                        "attempt_index": int(attempt_index_value),
                        "template_name": row.get("name"),
                        "rank": row.get("rank"),
                        "score": row.get("score"),
                        "matched_dataset_tags_json": json.dumps(row.get("matched_dataset_tags", []), ensure_ascii=True),
                        "matched_editable_paths_json": json.dumps(row.get("matched_editable_paths", []), ensure_ascii=True),
                        "rationale_json": json.dumps(row.get("rationale", []), ensure_ascii=True),
                        "selected_by_proposal": bool(row.get("name") in set(proposal.get("selected_mechanism_templates", []))),
                    }
                )
        dump_json(history_root / f"iter_{iteration:03d}.json", record)
        log(f"completed iteration={iteration} objective={record.get('objective_value')}")
        iteration += 1

    completed = [item for item in history if item.get("execution_status") == "completed" and item.get("objective_value") is not None]
    if not completed:
        raise RuntimeError(f"No completed structured iterations for {dataset_key}/{seed_model_name}")
    best_record = min(completed, key=lambda item: float(item["objective_value"]))
    if stopping_reason is None:
        stopping_reason = default_stopping_reason(history, budget_limit)
    llm_usage_summary = aggregate_llm_usage(llm_usage_rows)
    completed_count = completed_evaluation_count(history)
    failed_candidate_count = int(failed_history_count(history) + len(rejection_rows))
    candidate_attempt_count = int(completed_count + failed_candidate_count)
    session_wall_clock_seconds = prior_session_wall_clock_seconds + float(perf_counter() - session_started_at)

    for record in history:
        record["best_so_far_iteration"] = int(best_record["iteration"])
        record["best_so_far_objective"] = float(best_record["objective_value"])
        record["accepted_as_best"] = bool(record["iteration"] == best_record["iteration"])

    dump_json(session_root / "iterations.json", {"iterations": history})
    import csv
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
    dump_json(trace_root / "edit_program_trace.json", {"rows": accepted_trace_rows})
    _write_trace_csv(trace_root / "edit_program_trace.csv", accepted_trace_rows)
    dump_json(trace_root / "template_priority_trace.json", {"rows": template_priority_rows})
    if template_priority_rows:
        _write_trace_csv(trace_root / "template_priority_trace.csv", template_priority_rows)
    dump_json(trace_root / "llm_usage.json", {"rows": llm_usage_rows, "summary": llm_usage_summary})
    if llm_usage_rows:
        _write_trace_csv(trace_root / "llm_usage.csv", llm_usage_rows)
    reason_counts: Dict[str, int] = {}
    for row in rejection_rows:
        for reason in json.loads(str(row["rejection_reasons_json"])):
            reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
    dump_json(
        rejection_root / "proposal_rejections.json",
        {
            "rows": rejection_rows,
            "reason_counts": reason_counts,
        },
    )
    if rejection_rows:
        _write_trace_csv(rejection_root / "proposal_rejections.csv", rejection_rows)
    dump_json(rejection_root / "proposal_rejection_summary.json", {"reason_counts": reason_counts})
    dump_json(
        session_root / "agent_session.json",
        {
            "dataset_key": dataset_key,
            "method_family": "agent_structured_model",
            "method_name": f"agent_structured::{seed_model_name}",
            "agent_line": "main_agent",
            "agent_mode": agent_mode,
            "agent_variant": agent_mode,
            "seed_model_name": seed_model_name,
            "seed_model_config_path": str(seed_model_config_path.resolve()),
            "session_id": session_slug,
            "session_root": str(session_root.resolve()),
            "generated_code_root": str(generated_code_root.resolve()),
            "requested_device": requested_device,
            "cuda_visible_devices": cuda_visible_devices,
            "llm_config_path": str(llm_config_path.resolve()) if llm_config_path is not None else None,
            "llm_strategy": llm_strategy,
            "harness_mode": "mechanism_fidelity_structured_search_v1",
            "harness_config": harness_config,
            "mechanism_template_library_path": str(default_mechanism_template_path().resolve()),
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
        },
    )
    write_text(session_root / "agent.log", "\n".join(log_lines) + "\n")

    baseline_record = history[0]
    return {
        "dataset_key": dataset_key,
        "method_name": f"agent_structured::{seed_model_name}",
        "method_family": "agent_structured_model",
        "agent_line": "main_agent",
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
        "baseline_run_dir": baseline_record["run_dir"],
        "best_run_dir": best_record["run_dir"],
        "requested_device": requested_device,
        "best_resolved_device": best_record.get("resolved_device"),
        "best_model_uses_gpu": best_record.get("model_uses_gpu"),
        "llm_strategy": llm_strategy,
        "llm_enabled": llm_config_path is not None,
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
