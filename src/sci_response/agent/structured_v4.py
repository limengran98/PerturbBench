from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence

from sci_response.agent.compiler import compile_model_ir, render_generated_model_source
from sci_response.agent.direct_code import _seed_model_source_path
from sci_response.agent.edits import describe_edit
from sci_response.agent.harness import build_harness_review
from sci_response.agent.ir import extract_model_ir
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
from sci_response.agent.structured import (
    OBJECTIVE_PATH,
    REPORT_METRIC_PATHS,
    _build_benchmark_payload,
    _flatten_selected,
    _get_nested,
    _run_benchmark_with_model,
    _safe_slug,
    _write_trace_csv,
)
from sci_response.agent.structured_v3 import (
    _local_refinement_candidates,
    _portfolio_candidates,
    _seed_family_baseline_proposal,
    _task_group,
)
from sci_response.baselines.artifacts import beijing_timestamp_slug
from sci_response.data.io import dump_json, dump_yaml, ensure_dir, load_json, load_yaml, stable_hash, write_text


def _trim_text(text: str, *, max_chars: int = 3200) -> str:
    stripped = str(text).strip()
    if len(stripped) <= int(max_chars):
        return stripped
    return stripped[: int(max_chars)].rstrip() + "\n# ... trimmed ..."


def _family_summary_payload(
    *,
    repo_root: Path,
    family_spec: Mapping[str, Any],
    dataset_key: str,
) -> Dict[str, Any]:
    seed_model_name = str(family_spec["seed_model_name"])
    source_path = _seed_model_source_path(repo_root, seed_model_name)
    source_text = source_path.read_text(encoding="utf-8")
    candidate_pool = _portfolio_candidates(dict(family_spec["seed_ir"]), dataset_key, seed_model_name)
    return {
        "seed_model_name": seed_model_name,
        "config_path": str(Path(family_spec["config_path"]).resolve()),
        "config_json": json.dumps(family_spec["config"], indent=2, ensure_ascii=True),
        "source_path": str(source_path.resolve()),
        "source_preview": _trim_text(source_text),
        "warmup_candidates": [
            {
                "proposal_note": proposal["proposal_note"],
                "scientific_hypothesis": proposal.get("scientific_hypothesis", ""),
                "mechanistic_rationale": proposal.get("mechanistic_rationale", ""),
                "edit_preview": [describe_edit(edit) for edit in proposal["edits"]],
            }
            for proposal in candidate_pool
        ],
    }


def _baseline_history_summary(history: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in history:
        if str(item.get("search_stage")) != "seed_family_warmup":
            continue
        rows.append(
            {
                "iteration": item.get("iteration"),
                "seed_family_name": item.get("seed_family_name"),
                "objective_value": item.get("objective_value"),
                "execution_status": item.get("execution_status"),
                "proposal_note": item.get("proposal_note"),
            }
        )
    return rows


def _build_code_first_warmup_prompts(
    *,
    repo_root: Path,
    dataset_key: str,
    task_group: str,
    family_specs: Sequence[Mapping[str, Any]],
    history: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    family_payloads = [
        _family_summary_payload(repo_root=repo_root, family_spec=family_spec, dataset_key=dataset_key)
        for family_spec in family_specs
    ]
    system_prompt = (
        "You are planning Stage A of a two-stage biological perturbation modeling agent. "
        "Think like a direct-code breadth explorer: prioritize broad, code-first hypotheses that can quickly reveal "
        "which seed family and structure direction are most promising. "
        "However, you must return exactly one JSON object, not code. "
        "Your job is to choose the best seed family for warmup and rank a small set of legal structured warmup directions."
    )
    user_prompt = (
        f"DATASET KEY: {dataset_key}\n"
        f"TASK GROUP: {task_group}\n\n"
        "STAGE A GOAL:\n"
        "- Do breadth-first warmup, not final mechanistic refinement.\n"
        "- Pick the seed family with the strongest headroom.\n"
        "- Rank the warmup directions that should be tried before the structured refinement stage begins.\n\n"
        "BASELINE RESULTS SO FAR:\n"
        f"{json.dumps(_baseline_history_summary(history), indent=2, ensure_ascii=True)}\n\n"
        "AVAILABLE SEED FAMILIES AND WARMUP DIRECTIONS:\n"
        f"{json.dumps(family_payloads, indent=2, ensure_ascii=True)}\n\n"
        "Return exactly one JSON object with this schema:\n"
        "{\n"
        '  "proposal_note": "code_first_warmup_plan",\n'
        '  "preferred_seed_family": "<one available seed_model_name>",\n'
        '  "warmup_order": ["<proposal_note>", "<proposal_note>", "..."],\n'
        '  "scientific_hypothesis": "<one concise sentence>",\n'
        '  "mechanistic_rationale": "<why this family/direction ordering should work>",\n'
        '  "risk_notes": ["<optional short note>", "..."]\n'
        "}\n\n"
        "Rules:\n"
        "- preferred_seed_family must match one listed seed_model_name exactly.\n"
        "- warmup_order must only use proposal_note values from that chosen family.\n"
        "- Rank at most 4 warmup directions.\n"
        "- Favor bolder breadth exploration now; later refinement will be mechanistically constrained.\n"
    )
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }


def _best_completed_record(history: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    completed = [item for item in history if item.get("execution_status") == "completed" and item.get("objective_value") is not None]
    if not completed:
        return None
    return min(completed, key=lambda item: float(item["objective_value"]))


def _best_record_for_seed_family(history: Sequence[Dict[str, Any]], seed_family_name: str) -> Dict[str, Any] | None:
    completed = [
        item
        for item in history
        if item.get("execution_status") == "completed"
        and item.get("objective_value") is not None
        and str(item.get("seed_family_name") or item.get("candidate_seed_model_name") or "") == str(seed_family_name)
    ]
    if not completed:
        return None
    return min(completed, key=lambda item: float(item["objective_value"]))


def _fallback_warmup_plan(
    *,
    history: Sequence[Dict[str, Any]],
    family_specs: Sequence[Mapping[str, Any]],
    dataset_key: str,
) -> Dict[str, Any]:
    by_family: List[tuple[float, str]] = []
    for family_spec in family_specs:
        family_name = str(family_spec["seed_model_name"])
        best_family_record = _best_record_for_seed_family(history, family_name)
        if best_family_record is None or best_family_record.get("objective_value") is None:
            continue
        by_family.append((float(best_family_record["objective_value"]), family_name))
    by_family.sort()
    preferred_seed_family = by_family[0][1] if by_family else str(family_specs[0]["seed_model_name"])
    family_spec = next(
        item for item in family_specs if str(item["seed_model_name"]) == str(preferred_seed_family)
    )
    candidate_pool = _portfolio_candidates(dict(family_spec["seed_ir"]), dataset_key, preferred_seed_family)
    return {
        "proposal_note": "code_first_warmup_plan::fallback",
        "preferred_seed_family": preferred_seed_family,
        "warmup_order": [proposal["proposal_note"] for proposal in candidate_pool[:4]],
        "scientific_hypothesis": "Use the strongest seed family baseline, then probe broad structural moves before refinement.",
        "mechanistic_rationale": "Fallback warmup chooses the family with the best early empirical signal and explores its highest-value broad moves first.",
        "risk_notes": ["fallback_without_llm_plan"],
    }


def _request_code_first_warmup_plan(
    *,
    repo_root: Path,
    dataset_key: str,
    task_group: str,
    family_specs: Sequence[Mapping[str, Any]],
    history: Sequence[Dict[str, Any]],
    llm_client: StructuredLLMClient,
) -> Dict[str, Any]:
    prompts = _build_code_first_warmup_prompts(
        repo_root=repo_root,
        dataset_key=dataset_key,
        task_group=task_group,
        family_specs=family_specs,
        history=history,
    )
    family_names = {str(item["seed_model_name"]) for item in family_specs}
    family_to_notes = {
        str(item["seed_model_name"]): {
            proposal["proposal_note"]
            for proposal in _portfolio_candidates(dict(item["seed_ir"]), dataset_key, str(item["seed_model_name"]))
        }
        for item in family_specs
    }
    try:
        plan = llm_client.chat_json(
            system_prompt=prompts["system_prompt"],
            user_prompt=prompts["user_prompt"],
            max_tokens_override=min(int(llm_client.settings.max_tokens), 1600),
        )
    except Exception:
        return _fallback_warmup_plan(history=history, family_specs=family_specs, dataset_key=dataset_key)

    preferred_seed_family = str(plan.get("preferred_seed_family", "")).strip()
    if preferred_seed_family not in family_names:
        return _fallback_warmup_plan(history=history, family_specs=family_specs, dataset_key=dataset_key)

    raw_order = plan.get("warmup_order", [])
    if not isinstance(raw_order, list):
        raw_order = []
    valid_notes = family_to_notes.get(preferred_seed_family, set())
    warmup_order = []
    seen = set()
    for item in raw_order:
        note = str(item).strip()
        if not note or note not in valid_notes or note in seen:
            continue
        seen.add(note)
        warmup_order.append(note)
    if not warmup_order:
        fallback = _fallback_warmup_plan(history=history, family_specs=family_specs, dataset_key=dataset_key)
        fallback["risk_notes"] = list(fallback.get("risk_notes", [])) + ["empty_llm_warmup_order"]
        return fallback

    return {
        "proposal_note": str(plan.get("proposal_note", "code_first_warmup_plan")),
        "preferred_seed_family": preferred_seed_family,
        "warmup_order": warmup_order[:4],
        "scientific_hypothesis": str(plan.get("scientific_hypothesis", "")),
        "mechanistic_rationale": str(plan.get("mechanistic_rationale", "")),
        "risk_notes": list(plan.get("risk_notes", [])) if isinstance(plan.get("risk_notes", []), list) else [],
    }


def _reorder_candidates_by_notes(
    candidates: Sequence[Dict[str, Any]],
    ordered_notes: Sequence[str],
) -> List[Dict[str, Any]]:
    order_map = {str(note): index for index, note in enumerate(ordered_notes)}
    ranked = sorted(
        (dict(candidate) for candidate in candidates),
        key=lambda candidate: (order_map.get(str(candidate.get("proposal_note")), 10_000), str(candidate.get("proposal_note", ""))),
    )
    return ranked


def _candidate_trace_payload(
    *,
    stage: str,
    task_group: str,
    reference: str,
    candidates: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "stage": stage,
        "task_group": task_group,
        "reference": reference,
        "candidates": [
            {
                "proposal_note": proposal.get("proposal_note"),
                "proposal_source": proposal.get("proposal_source"),
                "candidate_seed_model_name": proposal.get("candidate_seed_model_name"),
                "edit_preview": [describe_edit(edit) for edit in proposal.get("edits", [])],
                "ir_hash": proposal.get("ir_hash"),
            }
            for proposal in candidates
        ],
    }


def _pick_first_valid_candidate(
    *,
    candidate_pool: Sequence[Dict[str, Any]],
    previous_ir: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
    generated_code_root: Path,
    rejection_root: Path,
    iteration: int,
) -> tuple[Dict[str, Any] | None, int, List[Dict[str, Any]]]:
    seen_ir_hashes = {str(item.get("ir_hash")) for item in history}
    rejection_rows: List[Dict[str, Any]] = []
    for attempt_index, candidate in enumerate(candidate_pool, start=1):
        if str(candidate["ir_hash"]) in seen_ir_hashes:
            rejection_rows.append(
                {
                    "iteration": int(iteration),
                    "attempt_index": int(attempt_index),
                    "proposal_source": candidate.get("proposal_source"),
                    "proposal_note": candidate.get("proposal_note"),
                    "rejection_reasons_json": json.dumps(["duplicate_candidate_ir"], ensure_ascii=True),
                }
            )
            continue
        candidate_ir = dict(candidate["candidate_ir"])
        generated_code_path = generated_code_root / f"iter_{iteration:03d}_attempt_{attempt_index:02d}_model.py"
        compiled_model_config = compile_model_ir(
            candidate_ir,
            generated_module_path=generated_code_path,
            generated_class_name="GeneratedStructuredHypothesisRegressor",
        )
        write_text(
            generated_code_path,
            render_generated_model_source(candidate_ir, class_name="GeneratedStructuredHypothesisRegressor"),
        )
        harness_review = build_harness_review(
            previous_ir=previous_ir,
            candidate_ir=candidate_ir,
            proposal=candidate,
            compiled_model_config=compiled_model_config,
            generated_code_path=generated_code_path,
        )
        if bool(harness_review["passed"]):
            return dict(candidate), int(attempt_index), rejection_rows
        rejection_rows.append(
            {
                "iteration": int(iteration),
                "attempt_index": int(attempt_index),
                "proposal_source": candidate.get("proposal_source"),
                "proposal_note": candidate.get("proposal_note"),
                "rejection_reasons_json": json.dumps(["harness_failed"], ensure_ascii=True),
            }
        )
        dump_json(
            rejection_root / f"iter_{iteration:03d}_attempt_{attempt_index:02d}.json",
            {
                "proposal": candidate,
                "harness_review": harness_review,
            },
        )
    return None, 0, rejection_rows


def run_structured_model_session_v4(
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
    agent_mode: str = "structured_hybrid_search_v4",
    additional_seed_model_config_paths: Sequence[Path] = (),
) -> Dict[str, Any]:
    seed_model_config = load_yaml(seed_model_config_path)
    seed_model_name = str(seed_model_config["name"])
    additional_seed_model_config_paths = [Path(path).resolve() for path in additional_seed_model_config_paths]
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

    llm_client = StructuredLLMClient(load_llm_settings(llm_config_path))
    budget_limit = int(max_iteration)
    session_started_at = perf_counter()
    prior_session_wall_clock_seconds = load_previous_session_wall_clock_seconds(session_root)
    task_group = _task_group(dataset_key)
    seed_family_specs: List[Dict[str, Any]] = []
    seen_seed_paths = set()
    for config_path in [seed_model_config_path.resolve(), *additional_seed_model_config_paths]:
        if config_path in seen_seed_paths:
            continue
        seen_seed_paths.add(config_path)
        config_payload = load_yaml(config_path)
        seed_family_specs.append(
            {
                "config_path": config_path,
                "config": config_payload,
                "seed_model_name": str(config_payload["name"]),
                "seed_ir": extract_model_ir(config_payload),
            }
        )

    family_bootstrap_budget = min(len(seed_family_specs), max(0, budget_limit))
    warmup_stage_budget = min(3, max(0, budget_limit - family_bootstrap_budget - 4))
    warmup_stage_end_iteration = family_bootstrap_budget + warmup_stage_budget - 1

    log_lines = [
        f"dataset_key={dataset_key}",
        f"seed_model_config={seed_model_config_path}",
        f"seed_model_name={seed_model_name}",
        f"seed_families={json.dumps([item['seed_model_name'] for item in seed_family_specs], ensure_ascii=True)}",
        f"session_id={session_slug}",
        f"requested_device={requested_device}",
        f"cuda_visible_devices={cuda_visible_devices}",
        f"task_group={task_group}",
        "harness_mode=code_first_breadth_warmup_then_mechanistic_refinement_v4",
    ]
    log_lines = load_previous_log_lines(session_root / "agent.log", log_lines)

    def log(message: str) -> None:
        log_lines.append(message)
        print(f"[agent-structured-v4] {message}", flush=True)

    history: List[Dict[str, Any]] = load_history_rows(history_root)
    accepted_trace_rows: List[Dict[str, Any]] = load_rows_payload(trace_root / "edit_program_trace.json")
    rejection_rows: List[Dict[str, Any]] = load_rows_payload(rejection_root / "proposal_rejections.json")
    llm_usage_rows: List[Dict[str, Any]] = load_rows_payload(trace_root / "llm_usage.json")
    baseline_objective: float | None = baseline_objective_from_history(history)
    stopping_reason: str | None = None
    warmup_plan: Dict[str, Any] | None = None
    warmup_family_name: str | None = None
    iteration = next_iteration_index(history)
    if history:
        log(
            f"resume session_id={session_slug} next_iteration={iteration} "
            f"completed_evaluations={completed_evaluation_count(history)} candidate_attempts={len(history)}"
        )

    while should_continue_search(history, budget_limit):
        search_stage = "seed_family_warmup"
        proposal_note = ""
        attempt_index_value = 0
        trace_payload = {"stage": search_stage, "task_group": task_group, "candidates": []}
        if iteration < family_bootstrap_budget:
            family_spec = seed_family_specs[int(iteration)]
            previous_ir = dict(family_spec["seed_ir"])
            reference_objective_value = None
            proposal = _seed_family_baseline_proposal(
                reference_ir=previous_ir,
                seed_ir=dict(family_spec["seed_ir"]),
                seed_model_name=str(family_spec["seed_model_name"]),
            )
            proposal_note = str(proposal["proposal_note"])
            attempt_index_value = 1 if iteration > 0 else 0
            trace_payload = {
                "stage": search_stage,
                "task_group": task_group,
                "reference": "seed_family_baseline",
                "candidates": [
                    {
                        "proposal_note": proposal_note,
                        "proposal_source": proposal["proposal_source"],
                        "candidate_seed_model_name": proposal["candidate_seed_model_name"],
                        "edit_preview": [],
                        "ir_hash": proposal["ir_hash"],
                    }
                ],
            }
        else:
            best_completed = _best_completed_record(history)
            if best_completed is None:
                raise RuntimeError(f"No completed warmup records found for {dataset_key}/{seed_model_name}")

            if warmup_plan is None:
                warmup_plan = _request_code_first_warmup_plan(
                    repo_root=repo_root,
                    dataset_key=dataset_key,
                    task_group=task_group,
                    family_specs=seed_family_specs,
                    history=history,
                    llm_client=llm_client,
                )
                warmup_family_name = str(warmup_plan["preferred_seed_family"])
                dump_json(trace_root / "code_first_warmup_plan.json", warmup_plan)
                for usage_event in llm_client.drain_usage_events():
                    llm_usage_rows.append({"iteration": int(iteration), "attempt_index": 0, **usage_event})
                log(
                    f"warmup_plan preferred_seed_family={warmup_family_name} "
                    f"warmup_order={warmup_plan.get('warmup_order', [])}"
                )

            if iteration <= warmup_stage_end_iteration and warmup_stage_budget > 0:
                search_stage = "code_first_warmup"
                warmup_family_name = warmup_family_name or str(seed_family_specs[0]["seed_model_name"])
                family_spec = next(
                    item for item in seed_family_specs if str(item["seed_model_name"]) == str(warmup_family_name)
                )
                family_reference_record = _best_record_for_seed_family(history, warmup_family_name)
                reference_objective_value = (
                    float(family_reference_record["objective_value"])
                    if family_reference_record is not None and family_reference_record.get("objective_value") is not None
                    else None
                )
                previous_ir = dict(family_spec["seed_ir"])
                candidate_pool = _portfolio_candidates(previous_ir, dataset_key, warmup_family_name)
                candidate_pool = _reorder_candidates_by_notes(candidate_pool, list(warmup_plan.get("warmup_order", [])))
                trace_payload = _candidate_trace_payload(
                    stage=search_stage,
                    task_group=task_group,
                    reference=f"seed_family::{warmup_family_name}",
                    candidates=candidate_pool,
                )
                proposal, attempt_index_value, new_rejections = _pick_first_valid_candidate(
                    candidate_pool=candidate_pool,
                    previous_ir=previous_ir,
                    history=history,
                    generated_code_root=generated_code_root,
                    rejection_root=rejection_root,
                    iteration=int(iteration),
                )
                rejection_rows.extend(new_rejections)
                if proposal is None:
                    stopping_reason = "no_valid_v4_warmup_candidate"
                    log(f"stopping iteration={iteration} reason={stopping_reason}")
                    break
                proposal_note = str(proposal["proposal_note"])
            else:
                search_stage = "mechanistic_refinement"
                previous_ir = dict(best_completed["model_ir"])
                reference_objective_value = float(best_completed["objective_value"])
                current_reference_seed_family_name = str(
                    best_completed.get("candidate_seed_model_name")
                    or best_completed.get("seed_family_name")
                    or warmup_family_name
                    or seed_model_name
                )
                candidate_pool = _local_refinement_candidates(
                    previous_ir,
                    dataset_key,
                    current_reference_seed_family_name,
                )
                trace_payload = _candidate_trace_payload(
                    stage=search_stage,
                    task_group=task_group,
                    reference=f"best_ir::{current_reference_seed_family_name}",
                    candidates=candidate_pool,
                )
                proposal, attempt_index_value, new_rejections = _pick_first_valid_candidate(
                    candidate_pool=candidate_pool,
                    previous_ir=previous_ir,
                    history=history,
                    generated_code_root=generated_code_root,
                    rejection_root=rejection_root,
                    iteration=int(iteration),
                )
                rejection_rows.extend(new_rejections)
                if proposal is None:
                    stopping_reason = "no_valid_v4_refinement_candidate"
                    log(f"stopping iteration={iteration} reason={stopping_reason}")
                    break
                proposal_note = str(proposal["proposal_note"])

        candidate_ir = dict(proposal["candidate_ir"])
        ir_path = ir_root / f"iter_{iteration:03d}.json"
        edit_path = edit_root / f"iter_{iteration:03d}.json"
        compiled_model_path = compiled_root / f"iter_{iteration:03d}.yaml"
        generated_code_path = generated_code_root / f"iter_{iteration:03d}_model.py"
        benchmark_config_path = benchmark_configs_root / f"iter_{iteration:03d}.yaml"
        harness_review_path = harness_root / f"iter_{iteration:03d}.json"
        trace_stage_path = trace_root / f"iter_{iteration:03d}_search_stage.json"

        compiled_model_config = compile_model_ir(
            candidate_ir,
            generated_module_path=generated_code_path,
            generated_class_name="GeneratedStructuredHypothesisRegressor",
        )
        dump_json(ir_path, candidate_ir)
        dump_json(
            edit_path,
            {
                "iteration": int(iteration),
                "proposal_source": proposal["proposal_source"],
                "proposal_note": proposal_note,
                "attempt_index": int(attempt_index_value),
                "selected_mechanism_templates": list(proposal.get("selected_mechanism_templates", [])),
                "scientific_hypothesis": proposal.get("scientific_hypothesis", ""),
                "mechanistic_rationale": proposal.get("mechanistic_rationale", ""),
                "risk_notes": proposal.get("risk_notes", []),
                "candidate_seed_model_name": proposal.get("candidate_seed_model_name", seed_model_name),
                "search_stage": search_stage,
                "edits": proposal["edits"],
                "ir_hash": proposal["ir_hash"],
                "warmup_plan": warmup_plan if search_stage == "code_first_warmup" else None,
            },
        )
        dump_json(trace_stage_path, trace_payload)
        write_text(generated_code_path, render_generated_model_source(candidate_ir, class_name="GeneratedStructuredHypothesisRegressor"))
        harness_review = build_harness_review(
            previous_ir=previous_ir,
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
        dump_yaml(compiled_model_path, compiled_model_config)
        dump_yaml(
            benchmark_config_path,
            _build_benchmark_payload(
                dataset_config=dataset_config,
                split_path=split_path,
                run_name=f"{dataset_key}_{seed_model_name}_agent_structured_v4_iter_{iteration:03d}",
                artifacts_root=benchmark_runs_root,
                seed=int(seed),
                top_k=int(top_k),
                model_config=compiled_model_path,
            ),
        )

        log(
            f"dispatch iteration={iteration} stage={search_stage} "
            f"proposal_source={proposal['proposal_source']} proposal_note={proposal_note}"
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
                "phase": "baseline" if iteration < family_bootstrap_budget else "agent",
                "proposal_source": proposal["proposal_source"],
                "proposal_note": proposal_note,
                "attempt_index": int(attempt_index_value),
                "selected_mechanism_templates": list(proposal.get("selected_mechanism_templates", [])),
                "scientific_hypothesis": proposal.get("scientific_hypothesis", ""),
                "mechanistic_rationale": proposal.get("mechanistic_rationale", ""),
                "risk_notes": proposal.get("risk_notes", []),
                "edits": proposal["edits"],
                "edit_preview": [describe_edit(edit) for edit in proposal["edits"]],
                "ir_hash": proposal["ir_hash"],
                "candidate_seed_model_name": proposal.get("candidate_seed_model_name", seed_model_name),
                "seed_family_name": proposal.get("candidate_seed_model_name", seed_model_name),
                "search_stage": search_stage,
                "task_group": task_group,
                "warmup_seed_family_name": warmup_family_name,
                "warmup_plan_path": str((trace_root / "code_first_warmup_plan.json").resolve()) if warmup_plan else None,
                "harness_review_path": str(harness_review_path.resolve()),
                "search_stage_trace_path": str(trace_stage_path.resolve()),
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
                "proposal_note": proposal_note,
                "attempt_index": int(attempt_index_value),
                "selected_mechanism_templates": list(proposal.get("selected_mechanism_templates", [])),
                "scientific_hypothesis": proposal.get("scientific_hypothesis", ""),
                "mechanistic_rationale": proposal.get("mechanistic_rationale", ""),
                "risk_notes": list(proposal.get("risk_notes", [])) + ["benchmark_execution_failed"],
                "edits": proposal["edits"],
                "edit_preview": [describe_edit(edit) for edit in proposal["edits"]],
                "ir_hash": proposal["ir_hash"],
                "candidate_seed_model_name": proposal.get("candidate_seed_model_name", seed_model_name),
                "seed_family_name": proposal.get("candidate_seed_model_name", seed_model_name),
                "search_stage": search_stage,
                "task_group": task_group,
                "warmup_seed_family_name": warmup_family_name,
                "warmup_plan_path": str((trace_root / "code_first_warmup_plan.json").resolve()) if warmup_plan else None,
                "harness_review_path": str(harness_review_path.resolve()),
                "search_stage_trace_path": str(trace_stage_path.resolve()),
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
                    "search_stage": search_stage,
                    "proposal_source": proposal["proposal_source"],
                    "proposal_note": proposal_note,
                    "candidate_seed_model_name": proposal.get("candidate_seed_model_name", seed_model_name),
                    "scientific_hypothesis": proposal.get("scientific_hypothesis", ""),
                    "mechanistic_rationale": proposal.get("mechanistic_rationale", ""),
                    "edit_preview_json": json.dumps([describe_edit(edit) for edit in proposal["edits"]], ensure_ascii=True),
                    "generated_code_path": str(generated_code_path.resolve()),
                    "compiled_model_config_path": str(compiled_model_path.resolve()),
                    "benchmark_run_dir": str(record.get("run_dir")),
                    "objective_value": record.get("objective_value"),
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
    completed_count = completed_evaluation_count(history)
    failed_candidate_count = int(failed_history_count(history) + len(rejection_rows))
    candidate_attempt_count = int(completed_count + failed_candidate_count)
    session_wall_clock_seconds = prior_session_wall_clock_seconds + float(perf_counter() - session_started_at)
    llm_usage_summary = aggregate_llm_usage(llm_usage_rows)

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
    dump_json(rejection_root / "proposal_rejections.json", {"rows": rejection_rows})
    if rejection_rows:
        _write_trace_csv(rejection_root / "proposal_rejections.csv", rejection_rows)
    dump_json(trace_root / "llm_usage.json", {"rows": llm_usage_rows, "summary": llm_usage_summary})
    if llm_usage_rows:
        _write_trace_csv(trace_root / "llm_usage.csv", llm_usage_rows)
    dump_json(
        session_root / "agent_session.json",
        {
            "dataset_key": dataset_key,
            "method_family": "agent_structured_model",
            "method_name": f"agent_structured_v4::{seed_model_name}",
            "agent_line": "main_agent_v4",
            "agent_mode": agent_mode,
            "agent_variant": agent_mode,
            "seed_model_name": seed_model_name,
            "seed_model_config_path": str(seed_model_config_path.resolve()),
            "session_id": session_slug,
            "session_root": str(session_root.resolve()),
            "generated_code_root": str(generated_code_root.resolve()),
            "requested_device": requested_device,
            "cuda_visible_devices": cuda_visible_devices,
            "llm_config_path": str(llm_config_path.resolve()),
            "llm_strategy": "hybrid",
            "harness_mode": "code_first_breadth_warmup_then_mechanistic_refinement_v4",
            "task_group": task_group,
            "seed_family_candidates_json": json.dumps([item["seed_model_name"] for item in seed_family_specs], ensure_ascii=True),
            "warmup_plan": warmup_plan,
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
            "best_seed_family_name": best_record.get("candidate_seed_model_name", seed_model_name),
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
        "method_name": f"agent_structured_v4::{seed_model_name}",
        "method_family": "agent_structured_model",
        "agent_line": "main_agent_v4",
        "agent_mode": agent_mode,
        "agent_variant": agent_mode,
        "seed_model_name": seed_model_name,
        "session_id": session_slug,
        "session_root": str(session_root.resolve()),
        "baseline_iteration": 0,
        "baseline_objective": baseline_record.get("objective_value"),
        "best_iteration": int(best_record["iteration"]),
        "best_objective": best_record.get("objective_value"),
        "best_seed_family_name": best_record.get("candidate_seed_model_name", seed_model_name),
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
        "llm_strategy": "hybrid",
        "llm_enabled": True,
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
