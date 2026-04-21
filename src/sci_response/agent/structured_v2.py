from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Iterable, List, Sequence

from sci_response.agent.compiler import compile_model_ir, render_generated_model_source
from sci_response.agent.config import (
    default_harness_config_v2_path,
    default_mechanism_template_v2_path,
)
from sci_response.agent.edits import (
    apply_edit_sequence,
    dedupe_edit_sequences,
    describe_edit,
    enumerate_candidate_edits,
)
from sci_response.agent.harness import (
    build_harness_review,
    changed_ir_paths,
    dataset_mechanism_tags,
    infer_hypothesis_axes,
    prioritize_mechanism_templates,
    summarize_rejection_reasons,
)
from sci_response.agent.ir import enumerate_editable_sites, extract_model_ir
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
from sci_response.agent.memory import build_cross_dataset_memory
from sci_response.agent.structured import (
    OBJECTIVE_PATH,
    REPORT_METRIC_PATHS,
    _build_benchmark_payload,
    _flatten_selected,
    _get_nested,
    _run_benchmark_with_model,
    _safe_slug,
    _should_fallback_after_llm_failure,
    _write_trace_csv,
)
from sci_response.data.io import dump_json, dump_yaml, ensure_dir, load_json, load_yaml, stable_hash, write_text


def _load_harness_config_v2() -> Dict[str, Any]:
    return load_yaml(default_harness_config_v2_path())


def _task_group_for_dataset(dataset_key: str) -> str:
    key = str(dataset_key).lower()
    if key in {"norman", "adamson", "sciplex3"}:
        return "single_cell"
    if key in {"papalexi_arrayed_rna", "papalexi_arrayed_protein"}:
        return "multimodal"
    if key == "l1000_public":
        return "dose_time"
    if key == "cdsdb":
        return "paired_clinical"
    return "general"


def _action_family_for_path(path: str) -> str:
    path = str(path)
    if path.startswith("representation.optimizer.") or path.startswith("representation.loss."):
        return "optimization"
    if path.startswith("representation.trunk."):
        return "capacity"
    return "mechanism"


def _allowed_action_families(iteration: int, budget_limit: int, harness_config: Dict[str, Any]) -> List[str]:
    stage1_fraction = float(harness_config.get("stage1_fraction", 0.6))
    stage_cutoff = max(1, int(round(max(1, int(budget_limit)) * stage1_fraction)))
    if int(iteration) < stage_cutoff:
        return [str(item) for item in harness_config.get("stage1_allowed_action_families", ["mechanism", "capacity"])]
    return [str(item) for item in harness_config.get("stage2_allowed_action_families", ["mechanism", "capacity", "optimization"])]


def _stage_name(iteration: int, budget_limit: int, harness_config: Dict[str, Any]) -> str:
    allowed = set(_allowed_action_families(iteration, budget_limit, harness_config))
    if allowed == {"mechanism", "capacity"}:
        return "stage1_structure_first"
    return "stage2_refine_and_regularize"


def _filter_editable_sites_for_stage(
    model_ir: Dict[str, Any],
    *,
    allowed_action_families: Sequence[str],
) -> List[Dict[str, Any]]:
    allowed = set(str(item) for item in allowed_action_families)
    filtered: List[Dict[str, Any]] = []
    for site in enumerate_editable_sites(model_ir):
        action_family = _action_family_for_path(str(site["path"]))
        if action_family in allowed:
            payload = dict(site)
            payload["action_family"] = action_family
            filtered.append(payload)
    return filtered


def _filter_candidate_edits_for_stage(
    model_ir: Dict[str, Any],
    *,
    allowed_action_families: Sequence[str],
) -> List[Dict[str, Any]]:
    allowed_paths = {
        str(site["path"])
        for site in _filter_editable_sites_for_stage(model_ir, allowed_action_families=allowed_action_families)
    }
    return [edit for edit in enumerate_candidate_edits(model_ir) if str(edit["path"]) in allowed_paths]


def _memory_axis_lookup(memory_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(row["axis"]): dict(row) for row in memory_payload.get("ranked_axes", [])}


def _rerank_templates_v2(
    *,
    dataset_key: str,
    model_ir: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
    template_library: Dict[str, Any],
    harness_config: Dict[str, Any],
    recent_rejections: Sequence[Dict[str, Any]],
    memory_payload: Dict[str, Any],
    allowed_action_families: Sequence[str],
) -> Dict[str, Any]:
    base_payload = prioritize_mechanism_templates(
        dataset_key=dataset_key,
        model_ir=model_ir,
        history=history,
        template_library=template_library,
        harness_config=harness_config,
        recent_rejections=recent_rejections,
    )
    template_lookup = {str(row["name"]): dict(row) for row in template_library.get("templates", [])}
    memory_lookup = _memory_axis_lookup(memory_payload)
    task_group = _task_group_for_dataset(dataset_key)
    allowed = set(str(item) for item in allowed_action_families)
    boost_memory = float(harness_config.get("priority_boost_cross_dataset_memory", 3.5))

    reranked: List[Dict[str, Any]] = []
    for row in base_payload.get("ranked_templates", []):
        name = str(row.get("name"))
        template_meta = template_lookup.get(name, {})
        score = float(row.get("score", 0.0))
        rationale = list(row.get("rationale", []))
        preferred_task_groups = [str(item) for item in template_meta.get("preferred_task_groups", [])]
        action_families = [str(item) for item in template_meta.get("action_families", [])]
        if preferred_task_groups and task_group in set(preferred_task_groups):
            score += 1.5
            rationale.append(f"task_group+=1.50:{task_group}")
        if action_families and not (set(action_families) & allowed):
            score -= 3.0
            rationale.append("stage_mismatch-=3.00")
        elif action_families and (set(action_families) & allowed):
            score += 0.75
            rationale.append("stage_fit+=0.75")
        memory_row = memory_lookup.get(name)
        if memory_row is not None:
            delta = boost_memory * float(memory_row.get("score", 0.0))
            score += delta
            rationale.append(f"cross_dataset_memory+={delta:.2f}")
        updated = dict(row)
        updated["score"] = float(score)
        updated["rationale"] = rationale
        updated["task_group"] = task_group
        updated["action_families"] = action_families
        reranked.append(updated)
    reranked.sort(key=lambda item: (-float(item["score"]), str(item["name"])))
    for index, row in enumerate(reranked[: int(harness_config.get("template_priority_top_k", 6))], start=1):
        row["rank"] = int(index)
    return {
        "dataset_key": dataset_key,
        "dataset_tags": list(dataset_mechanism_tags(dataset_key)),
        "task_group": task_group,
        "allowed_action_families": list(allowed_action_families),
        "cross_dataset_memory": {
            "ranked_axes": list(memory_payload.get("ranked_axes", [])),
            "branch_roots": list(memory_payload.get("branch_roots", [])),
        },
        "history_outcome_summary": base_payload.get("history_outcome_summary", {}),
        "ranked_templates": reranked[: int(harness_config.get("template_priority_top_k", 6))],
    }


def _counterfactual_review(
    *,
    dataset_key: str,
    candidate_ir: Dict[str, Any],
    harness_config: Dict[str, Any],
) -> Dict[str, Any]:
    tags = set(dataset_mechanism_tags(dataset_key))
    rep = dict(candidate_ir.get("representation", {}))
    prediction = dict(rep.get("prediction", {}))
    conditioning = dict(rep.get("conditioning", {}))
    risk_flags: List[str] = []
    score = float(harness_config.get("counterfactual_pass_bonus", 1.5))
    major_penalty = float(harness_config.get("counterfactual_major_penalty", 3.0))
    minor_penalty = float(harness_config.get("counterfactual_minor_penalty", 1.0))

    if {"matched_control", "patient_paired"} & tags:
        if not bool(prediction.get("baseline_skip", False)):
            risk_flags.append("counterfactual_missing_baseline_anchor")
            score -= major_penalty
        if str(prediction.get("target", "delta")) != "delta":
            risk_flags.append("counterfactual_non_delta_target")
            score -= minor_penalty
    if {"dose_time", "context_rich", "drug_profile"} & tags:
        if int(conditioning.get("conditioning_dim", 0)) <= 0:
            risk_flags.append("counterfactual_missing_context_conditioning")
            score -= major_penalty
    if {"sparse_effect", "matched_control"} & tags:
        if not bool(prediction.get("zero_init_head", False)):
            risk_flags.append("counterfactual_weak_zero_init_prior")
            score -= minor_penalty
    if "patient_paired" in tags and str(conditioning.get("mode", "")) == "concat":
        risk_flags.append("counterfactual_weak_paired_conditioning")
        score -= minor_penalty
    return {
        "passed": not any(flag.startswith("counterfactual_missing_") for flag in risk_flags),
        "risk_flags": risk_flags,
        "score_adjustment": float(score),
    }


def _stage_edit_review(
    *,
    proposal: Dict[str, Any],
    allowed_action_families: Sequence[str],
) -> Dict[str, Any]:
    allowed = set(str(item) for item in allowed_action_families)
    action_families = sorted({_action_family_for_path(str(edit.get("path", ""))) for edit in proposal.get("edits", [])})
    errors: List[str] = []
    for edit in proposal.get("edits", []):
        path = str(edit.get("path", ""))
        family = _action_family_for_path(path)
        if family not in allowed:
            errors.append(f"stage_disallowed_action_family::{family}::{path}")
    return {
        "passed": not errors,
        "action_families": action_families,
        "errors": errors,
    }


def _heuristic_edit_sequence_v2(
    *,
    current_ir: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
    allowed_action_families: Sequence[str],
    exclude_ir_hashes: Sequence[str] = (),
) -> Dict[str, Any]:
    seen_ir_hashes = {str(item.get("ir_hash")) for item in history}
    seen_ir_hashes.update(str(item) for item in exclude_ir_hashes)
    primitive_edits = _filter_candidate_edits_for_stage(
        current_ir,
        allowed_action_families=allowed_action_families,
    )
    single_edits = [[edit] for edit in primitive_edits]
    paired_edits: List[List[Dict[str, Any]]] = []
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
            "proposal_source": "heuristic_v2",
            "proposal_note": ", ".join(describe_edit(edit) for edit in sequence),
            "selected_mechanism_templates": infer_hypothesis_axes(changed_ir_paths(current_ir, candidate_ir)),
            "scientific_hypothesis": "V2 heuristic structured perturbation-response architecture edit.",
            "mechanistic_rationale": "Probe a task-aware structural hypothesis while staying inside the current search stage action families.",
            "risk_notes": [],
            "edits": sequence,
            "candidate_ir": candidate_ir,
            "ir_hash": ir_hash,
        }
    return {
        "proposal_source": "heuristic_v2",
        "proposal_note": "no_unseen_structured_edit_available_v2",
        "selected_mechanism_templates": [],
        "scientific_hypothesis": "No unseen local structured edit remains under the current V2 heuristic frontier.",
        "mechanistic_rationale": "Search frontier exhausted under the current stage-gated primitive set.",
        "risk_notes": ["search_frontier_exhausted"],
        "edits": [],
        "candidate_ir": current_ir,
        "ir_hash": stable_hash(current_ir),
    }


def _propose_next_ir_v2(
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
    allowed_action_families: Sequence[str],
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
                f"llm proposal accepted note={proposal['proposal_note']} "
                f"stage={','.join(allowed_action_families)}"
            )
            return {
                "proposal_source": "llm_v2",
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
    fallback = _heuristic_edit_sequence_v2(
        current_ir=current_ir,
        history=history,
        allowed_action_families=allowed_action_families,
        exclude_ir_hashes=exclude_ir_hashes,
    )
    fallback["proposal_source"] = "heuristic_after_llm_failure_v2"
    fallback["risk_notes"] = list(fallback.get("risk_notes", [])) + ["llm_proposal_failed_fallback_used"]
    return fallback


def _candidate_beam_score(
    *,
    proposal: Dict[str, Any],
    template_priority_payload: Dict[str, Any],
    memory_payload: Dict[str, Any],
    stage_review: Dict[str, Any],
    counterfactual_review: Dict[str, Any],
    harness_review: Dict[str, Any],
    harness_config: Dict[str, Any],
) -> float:
    score = 0.0
    template_ranks = {
        str(row.get("name")): int(row.get("rank", 999))
        for row in template_priority_payload.get("ranked_templates", [])
    }
    rank_bonus = float(harness_config.get("beam_template_rank_bonus", 0.75))
    for name in proposal.get("selected_mechanism_templates", []):
        if str(name) in template_ranks:
            score += rank_bonus * max(1, 8 - int(template_ranks[str(name)]))
    memory_lookup = _memory_axis_lookup(memory_payload)
    memory_scale = float(harness_config.get("beam_memory_bonus_scale", 1.25))
    for axis in harness_review.get("mechanism_report", {}).get("hypothesis_axes", []):
        if str(axis) in memory_lookup:
            score += memory_scale * float(memory_lookup[str(axis)].get("score", 0.0))
    score += float(counterfactual_review.get("score_adjustment", 0.0))
    score -= float(harness_config.get("beam_edit_size_penalty", 0.15)) * len(
        harness_review.get("mechanism_report", {}).get("changed_paths", [])
    )
    if proposal.get("proposal_source") == "llm_v2":
        score += float(harness_config.get("beam_llm_proposal_bonus", 0.1))
    if not stage_review.get("passed", False):
        score -= 100.0
    if not counterfactual_review.get("passed", False):
        score -= 10.0
    if not harness_review.get("passed", False):
        score -= 100.0
    return float(score)


def run_structured_model_session_v2(
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
    agent_mode: str = "structured_llm_search_v2",
) -> Dict[str, Any]:
    seed_model_config = load_yaml(seed_model_config_path)
    seed_model_name = str(seed_model_config["name"])
    session_slug = session_id or f"{_safe_slug(agent_mode)}_{seed}"
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

    harness_config = _load_harness_config_v2()
    template_library = load_yaml(default_mechanism_template_v2_path())
    max_attempts = int(harness_config.get("proposal_attempts_per_iteration", 5))
    beam_width = int(harness_config.get("beam_width", 3))
    budget_limit = int(max_iteration)
    session_started_at = perf_counter()
    prior_session_wall_clock_seconds = load_previous_session_wall_clock_seconds(session_root)
    llm_client = (
        StructuredLLMClient(load_llm_settings(llm_config_path))
        if llm_config_path is not None and llm_strategy in {"llm", "hybrid"}
        else None
    )
    seed_ir = extract_model_ir(seed_model_config)
    baseline_objective: float | None = None
    current_reference_ir = seed_ir
    stopping_reason: str | None = None
    iteration = 0

    memory_roots = []
    for item in harness_config.get("cross_dataset_memory_roots", []):
        root = (repo_root / str(item)).resolve()
        memory_roots.append(root)
    if agent_root.resolve() not in memory_roots:
        memory_roots.append(agent_root.resolve())

    log_lines = [
        f"dataset_key={dataset_key}",
        f"seed_model_config={seed_model_config_path}",
        f"seed_model_name={seed_model_name}",
        f"session_id={session_slug}",
        f"requested_device={requested_device}",
        f"cuda_visible_devices={cuda_visible_devices}",
        f"llm_config_path={llm_config_path}",
        f"llm_strategy={llm_strategy}",
        "harness_mode=mechanism_fidelity_structured_search_v2",
    ]
    log_lines = load_previous_log_lines(session_root / "agent.log", log_lines)

    def log(message: str) -> None:
        log_lines.append(message)
        print(f"[agent-structured-v2] {message}", flush=True)

    history: List[Dict[str, Any]] = load_history_rows(history_root)
    accepted_trace_rows: List[Dict[str, Any]] = load_rows_payload(trace_root / "edit_program_trace.json")
    rejection_rows: List[Dict[str, Any]] = load_rows_payload(rejection_root / "proposal_rejections.json")
    template_priority_rows: List[Dict[str, Any]] = load_rows_payload(trace_root / "template_priority_trace.json")
    llm_usage_rows: List[Dict[str, Any]] = load_rows_payload(trace_root / "llm_usage.json")
    beam_rows: List[Dict[str, Any]] = load_rows_payload(trace_root / "beam_candidates.json")
    baseline_objective = baseline_objective_from_history(history)
    iteration = next_iteration_index(history)
    if history:
        log(
            f"resume session_id={session_slug} next_iteration={iteration} "
            f"completed_evaluations={completed_evaluation_count(history)} candidate_attempts={len(history)}"
        )

    while should_continue_search(history, budget_limit):
        reference_objective_value: float | None = None
        stage_allowed_families = _allowed_action_families(iteration, budget_limit, harness_config)
        stage_name = _stage_name(iteration, budget_limit, harness_config)
        memory_payload = build_cross_dataset_memory(
            target_dataset_key=dataset_key,
            branch_roots=memory_roots,
            min_similarity=float(harness_config.get("cross_dataset_memory_min_similarity", 0.3)),
            top_k=int(harness_config.get("cross_dataset_memory_top_k", 6)),
        )

        if iteration == 0:
            proposal = {
                "proposal_source": "seed_model_ir",
                "proposal_note": f"iteration_0_seed_ir::{seed_model_name}",
                "selected_mechanism_templates": [],
                "edits": [],
                "candidate_ir": seed_ir,
                "ir_hash": stable_hash(seed_ir),
                "scientific_hypothesis": "",
                "mechanistic_rationale": "",
                "risk_notes": [],
            }
            selected_payload = None
            template_priority_payload = {
                "dataset_key": dataset_key,
                "ranked_templates": [],
                "dataset_tags": list(dataset_mechanism_tags(dataset_key)),
                "task_group": _task_group_for_dataset(dataset_key),
                "allowed_action_families": stage_allowed_families,
                "cross_dataset_memory": {"ranked_axes": []},
            }
            attempt_index_value = 0
        else:
            best_completed = [item for item in history if item.get("execution_status") == "completed" and item.get("objective_value") is not None]
            if best_completed:
                best_reference_record = min(best_completed, key=lambda item: float(item["objective_value"]))
                current_reference_ir = dict(best_reference_record["model_ir"])
                reference_objective_value = float(best_reference_record["objective_value"])

            proposal = None
            selected_payload = None
            attempt_index_value = 0
            iteration_rejections: List[Dict[str, Any]] = []
            attempted_ir_hashes: set[str] = set()
            accepted_candidates: List[Dict[str, Any]] = []

            template_priority_payload = _rerank_templates_v2(
                dataset_key=dataset_key,
                model_ir=current_reference_ir,
                history=history,
                template_library=template_library,
                harness_config=harness_config,
                recent_rejections=iteration_rejections,
                memory_payload=memory_payload,
                allowed_action_families=stage_allowed_families,
            )

            for attempt_index in range(1, max_attempts + 1):
                candidate_proposal = _propose_next_ir_v2(
                    dataset_key=dataset_key,
                    current_ir=current_reference_ir,
                    history=history,
                    attempt_index=attempt_index,
                    recent_rejections=iteration_rejections,
                    prioritized_templates=template_priority_payload,
                    llm_client=llm_client,
                    llm_strategy=llm_strategy,
                    exclude_ir_hashes=sorted(attempted_ir_hashes),
                    allowed_action_families=stage_allowed_families,
                    log_fn=log,
                )
                if llm_client is not None:
                    for usage_event in llm_client.drain_usage_events():
                        llm_usage_rows.append({"iteration": int(iteration), "attempt_index": int(attempt_index), **usage_event})
                if str(candidate_proposal["ir_hash"]) in attempted_ir_hashes:
                    rejection_reasons = ["retry::duplicate_candidate_ir_within_iteration"]
                    iteration_rejections.append(
                        {
                            "iteration": int(iteration),
                            "attempt_index": int(attempt_index),
                            "proposal_source": candidate_proposal.get("proposal_source"),
                            "proposal_note": candidate_proposal.get("proposal_note"),
                            "selected_mechanism_templates": list(candidate_proposal.get("selected_mechanism_templates", [])),
                            "hypothesis_axes": list(candidate_proposal.get("selected_mechanism_templates", [])),
                            "rejection_reasons": rejection_reasons,
                        }
                    )
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
                                list(candidate_proposal.get("selected_mechanism_templates", [])),
                                ensure_ascii=True,
                            ),
                            "rejection_reasons_json": json.dumps(rejection_reasons, ensure_ascii=True),
                            "rejection_reasons_count": int(len(rejection_reasons)),
                        }
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
                write_text(generated_code_path, render_generated_model_source(candidate_ir, class_name=generated_class_name))
                harness_review = build_harness_review(
                    previous_ir=current_reference_ir,
                    candidate_ir=candidate_ir,
                    proposal=candidate_proposal,
                    compiled_model_config=compiled_model_config,
                    generated_code_path=generated_code_path,
                )
                stage_review = _stage_edit_review(
                    proposal=candidate_proposal,
                    allowed_action_families=stage_allowed_families,
                )
                counterfactual_review = _counterfactual_review(
                    dataset_key=dataset_key,
                    candidate_ir=candidate_ir,
                    harness_config=harness_config,
                )
                beam_score = _candidate_beam_score(
                    proposal=candidate_proposal,
                    template_priority_payload=template_priority_payload,
                    memory_payload=memory_payload,
                    stage_review=stage_review,
                    counterfactual_review=counterfactual_review,
                    harness_review=harness_review,
                    harness_config=harness_config,
                )
                beam_rows.append(
                    {
                        "iteration": int(iteration),
                        "attempt_index": int(attempt_index),
                        "proposal_source": candidate_proposal.get("proposal_source"),
                        "proposal_note": candidate_proposal.get("proposal_note"),
                        "beam_score": float(beam_score),
                        "selected_mechanism_templates_json": json.dumps(
                            list(candidate_proposal.get("selected_mechanism_templates", [])),
                            ensure_ascii=True,
                        ),
                        "stage_name": stage_name,
                        "allowed_action_families_json": json.dumps(stage_allowed_families, ensure_ascii=True),
                        "stage_review_passed": bool(stage_review["passed"]),
                        "stage_review_errors_json": json.dumps(stage_review["errors"], ensure_ascii=True),
                        "counterfactual_passed": bool(counterfactual_review["passed"]),
                        "counterfactual_flags_json": json.dumps(counterfactual_review["risk_flags"], ensure_ascii=True),
                        "harness_passed": bool(harness_review["passed"]),
                    }
                )
                rejection_reasons = []
                if not bool(harness_review["passed"]):
                    rejection_reasons.extend(summarize_rejection_reasons(harness_review))
                if not bool(stage_review["passed"]):
                    rejection_reasons.extend(stage_review["errors"])
                if not bool(counterfactual_review["passed"]):
                    rejection_reasons.extend([f"counterfactual::{flag}" for flag in counterfactual_review["risk_flags"]])
                if rejection_reasons:
                    iteration_rejections.append(
                        {
                            "iteration": int(iteration),
                            "attempt_index": int(attempt_index),
                            "proposal_source": candidate_proposal.get("proposal_source"),
                            "proposal_note": candidate_proposal.get("proposal_note"),
                            "selected_mechanism_templates": list(candidate_proposal.get("selected_mechanism_templates", [])),
                            "hypothesis_axes": list(harness_review["mechanism_report"].get("hypothesis_axes", [])),
                            "rejection_reasons": rejection_reasons,
                        }
                    )
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
                    dump_json(
                        rejection_root / f"iter_{iteration:03d}_attempt_{attempt_index:02d}.json",
                        {
                            "proposal": candidate_proposal,
                            "template_priority": template_priority_payload,
                            "harness_review": harness_review,
                            "stage_review": stage_review,
                            "counterfactual_review": counterfactual_review,
                            "rejection_reasons": rejection_reasons,
                        },
                    )
                    log(
                        f"proposal_rejected iteration={iteration} attempt={attempt_index} "
                        f"reasons={rejection_reasons}"
                    )
                    continue
                accepted_candidates.append(
                    {
                        "proposal": candidate_proposal,
                        "candidate_ir": candidate_ir,
                        "compiled_model_config": compiled_model_config,
                        "generated_code_path": generated_code_path,
                        "generated_class_name": generated_class_name,
                        "harness_review": harness_review,
                        "stage_review": stage_review,
                        "counterfactual_review": counterfactual_review,
                        "attempt_index": int(attempt_index),
                        "template_priority_payload": template_priority_payload,
                        "memory_payload": memory_payload,
                        "beam_score": float(beam_score),
                    }
                )
                if len(accepted_candidates) >= beam_width:
                    break

            if not accepted_candidates:
                stopping_reason = "no_acceptable_proposal_after_attempt_budget"
                log(f"stopping iteration={iteration} reason={stopping_reason}")
                break
            selected_payload = max(accepted_candidates, key=lambda item: float(item["beam_score"]))
            proposal = dict(selected_payload["proposal"])
            attempt_index_value = int(selected_payload["attempt_index"])
            log(
                f"beam_select iteration={iteration} attempt={attempt_index_value} "
                f"beam_score={selected_payload['beam_score']:.3f}"
            )

        if iteration == 0:
            candidate_ir = proposal["candidate_ir"]
            generated_class_name = "GeneratedStructuredHypothesisRegressor"
            generated_code_path = generated_code_root / f"iter_{iteration:03d}_model.py"
            compiled_model_config = compile_model_ir(
                candidate_ir,
                generated_module_path=generated_code_path,
                generated_class_name=generated_class_name,
            )
            harness_review = build_harness_review(
                previous_ir=seed_ir,
                candidate_ir=candidate_ir,
                proposal=proposal,
                compiled_model_config=compiled_model_config,
                generated_code_path=generated_code_path,
            )
            harness_review["passed"] = True
            harness_review["mechanism_report"]["mechanism_guard_passed"] = True
            harness_review["mechanism_report"]["risk_flags"] = [
                flag for flag in harness_review["mechanism_report"]["risk_flags"] if flag != "no_structural_change"
            ]
            stage_review = {
                "passed": True,
                "action_families": [],
                "errors": [],
            }
            counterfactual_review = {
                "passed": True,
                "risk_flags": [],
                "score_adjustment": 0.0,
            }
        else:
            candidate_ir = selected_payload["candidate_ir"]
            generated_class_name = str(selected_payload["generated_class_name"])
            generated_code_path = Path(selected_payload["generated_code_path"])
            compiled_model_config = dict(selected_payload["compiled_model_config"])
            harness_review = dict(selected_payload["harness_review"])
            stage_review = dict(selected_payload["stage_review"])
            counterfactual_review = dict(selected_payload["counterfactual_review"])

        ir_path = ir_root / f"iter_{iteration:03d}.json"
        edit_path = edit_root / f"iter_{iteration:03d}.json"
        compiled_model_path = compiled_root / f"iter_{iteration:03d}.yaml"
        benchmark_config_path = benchmark_configs_root / f"iter_{iteration:03d}.yaml"
        harness_review_path = harness_root / f"iter_{iteration:03d}.json"
        template_priority_path = trace_root / f"iter_{iteration:03d}_template_priority.json"
        memory_path = trace_root / f"iter_{iteration:03d}_memory.json"

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
                "stage_name": stage_name,
                "allowed_action_families": stage_allowed_families,
                "edits": proposal["edits"],
                "ir_hash": proposal["ir_hash"],
            },
        )
        dump_json(template_priority_path, template_priority_payload)
        dump_json(memory_path, memory_payload)
        write_text(generated_code_path, render_generated_model_source(candidate_ir, class_name=generated_class_name))
        dump_json(
            harness_review_path,
            {
                "harness_review": harness_review,
                "stage_review": stage_review,
                "counterfactual_review": counterfactual_review,
            },
        )
        dump_yaml(compiled_model_path, compiled_model_config)
        dump_yaml(
            benchmark_config_path,
            _build_benchmark_payload(
                dataset_config=dataset_config,
                split_path=split_path,
                run_name=f"{dataset_key}_{seed_model_name}_agent_structured_v2_iter_{iteration:03d}",
                artifacts_root=benchmark_runs_root,
                seed=int(seed),
                top_k=int(top_k),
                model_config=compiled_model_path,
            ),
        )
        log(
            f"dispatch iteration={iteration} stage={stage_name} proposal_source={proposal['proposal_source']} "
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
                "search_stage": stage_name,
                "allowed_action_families": list(stage_allowed_families),
                "beam_candidate_score": (
                    None if selected_payload is None else float(selected_payload["beam_score"])
                ),
                "memory_top_axes": [row.get("axis") for row in memory_payload.get("ranked_axes", [])],
                "harness_review_path": str(harness_review_path.resolve()),
                "template_priority_path": str(template_priority_path.resolve()),
                "counterfactual_memory_path": str(memory_path.resolve()),
                "hypothesis_axes": list(harness_review["mechanism_report"]["hypothesis_axes"]),
                "mechanism_guard_passed": bool(harness_review["mechanism_report"]["mechanism_guard_passed"]),
                "semantic_code_consistency_passed": bool(harness_review["semantic_code_consistency"]["passed"]),
                "stage_review_passed": bool(stage_review["passed"]),
                "counterfactual_passed": bool(counterfactual_review["passed"]),
                "counterfactual_flags": list(counterfactual_review.get("risk_flags", [])),
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
                "search_stage": stage_name,
                "allowed_action_families": list(stage_allowed_families),
                "beam_candidate_score": None if selected_payload is None else float(selected_payload["beam_score"]),
                "memory_top_axes": [row.get("axis") for row in memory_payload.get("ranked_axes", [])],
                "harness_review_path": str(harness_review_path.resolve()),
                "template_priority_path": str(template_priority_path.resolve()),
                "counterfactual_memory_path": str(memory_path.resolve()),
                "hypothesis_axes": list(harness_review["mechanism_report"]["hypothesis_axes"]),
                "mechanism_guard_passed": bool(harness_review["mechanism_report"]["mechanism_guard_passed"]),
                "semantic_code_consistency_passed": bool(harness_review["semantic_code_consistency"]["passed"]),
                "stage_review_passed": bool(stage_review["passed"]),
                "counterfactual_passed": bool(counterfactual_review["passed"]),
                "counterfactual_flags": list(counterfactual_review.get("risk_flags", [])),
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
                    "memory_top_axes_json": json.dumps(
                        [item.get("axis") for item in memory_payload.get("ranked_axes", [])],
                        ensure_ascii=True,
                    ),
                    "search_stage": stage_name,
                    "beam_candidate_score": record.get("beam_candidate_score"),
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
        raise RuntimeError(f"No completed structured-v2 iterations for {dataset_key}/{seed_model_name}")
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
    dump_json(trace_root / "beam_candidates.json", {"rows": beam_rows})
    if beam_rows:
        _write_trace_csv(trace_root / "beam_candidates.csv", beam_rows)
    dump_json(trace_root / "llm_usage.json", {"rows": llm_usage_rows, "summary": llm_usage_summary})
    if llm_usage_rows:
        _write_trace_csv(trace_root / "llm_usage.csv", llm_usage_rows)
    reason_counts: Dict[str, int] = {}
    for row in rejection_rows:
        for reason in json.loads(str(row["rejection_reasons_json"])):
            reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1
    dump_json(rejection_root / "proposal_rejections.json", {"rows": rejection_rows, "reason_counts": reason_counts})
    if rejection_rows:
        _write_trace_csv(rejection_root / "proposal_rejections.csv", rejection_rows)
    dump_json(rejection_root / "proposal_rejection_summary.json", {"reason_counts": reason_counts})
    dump_json(
        session_root / "agent_session.json",
        {
            "dataset_key": dataset_key,
            "method_family": "agent_structured_model",
            "method_name": f"agent_structured_v2::{seed_model_name}",
            "agent_line": "main_agent_v2",
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
            "harness_mode": "mechanism_fidelity_structured_search_v2",
            "harness_config": harness_config,
            "mechanism_template_library_path": str(default_mechanism_template_v2_path().resolve()),
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
        "method_name": f"agent_structured_v2::{seed_model_name}",
        "method_family": "agent_structured_model",
        "agent_line": "main_agent_v2",
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
