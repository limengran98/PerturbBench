from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence

from sci_response.agent.compiler import compile_model_ir, render_generated_model_source
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
    _enum_to_edit,
    _local_refinement_candidates,
    _proposal_payload,
    _portfolio_candidates,
    _seed_family_baseline_proposal,
    _set_scalar_edit,
    _task_group,
    _toggle_to_edit,
)
from sci_response.agent.structured_v5 import (
    PROJECTABLE_PATHS,
    _best_completed_record,
    _best_record,
    _build_breadth_candidate_queue,
    _candidate_trace_payload,
    _deepcopy_ir,
    _dedupe_candidate_pool,
    _pick_first_valid_candidate,
    _projection_proposal,
    _projection_signature,
    _request_code_first_warmup_plan,
    _set_nested_value,
    _trust_region_candidates,
)
from sci_response.baselines.artifacts import beijing_timestamp_slug
from sci_response.data.io import dump_json, dump_yaml, ensure_dir, load_json, load_yaml, stable_hash, write_text


def _cross_family_baseline_candidates(
    *,
    seed_family_specs: Sequence[Mapping[str, Any]],
    reference_ir: Dict[str, Any],
    current_family_name: str | None,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for family_spec in seed_family_specs:
        family_name = str(family_spec["seed_model_name"])
        if current_family_name and family_name == str(current_family_name):
            continue
        candidates.append(
            _seed_family_baseline_proposal(
                reference_ir=reference_ir,
                seed_ir=dict(family_spec["seed_ir"]),
                seed_model_name=family_name,
            )
        )
    return candidates


def _breadth_macro_candidates(
    *,
    reference_ir: Dict[str, Any],
    dataset_key: str,
    current_family_name: str,
    seed_family_specs: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    group = _task_group(dataset_key)
    hidden_dim = int(_get_nested(reference_ir, "representation.trunk.hidden_dim") or 128)
    trunk_depth = int(_get_nested(reference_ir, "representation.trunk.trunk_depth") or 1)
    residual_depth = int(_get_nested(reference_ir, "representation.trunk.residual_depth") or 1)
    conditioning_dim = int(_get_nested(reference_ir, "representation.conditioning.conditioning_dim") or 64)
    learning_rate = float(_get_nested(reference_ir, "representation.optimizer.learning_rate") or 8e-4)
    batch_size = int(_get_nested(reference_ir, "representation.optimizer.batch_size") or 128)
    weight_decay = float(_get_nested(reference_ir, "representation.optimizer.weight_decay") or 1e-4)
    response_weight = float(_get_nested(reference_ir, "representation.loss.response_weight") or 0.35)

    macro_specs: List[tuple[str, str, str, List[Dict[str, Any] | None]]] = [
        (
            "breadth_capacity_jump",
            "Probe a larger capacity regime to escape the current local basin.",
            "When structured search stalls, a larger hidden width and a slightly deeper trunk are the fastest legal way to jump basins.",
            [
                _set_scalar_edit(reference_ir, "representation.trunk.hidden_dim", hidden_dim + 64),
                _set_scalar_edit(reference_ir, "representation.trunk.trunk_depth", trunk_depth + 1),
                _set_scalar_edit(reference_ir, "representation.trunk.residual_depth", residual_depth + 1),
            ],
        ),
        (
            "breadth_context_jump",
            "Probe a stronger context-routing regime before returning to local refinement.",
            "Direct-code style gains often come from changing how context is injected rather than from tiny optimizer nudges.",
            [
                _enum_to_edit(reference_ir, "representation.conditioning.mode", "film"),
                _set_scalar_edit(reference_ir, "representation.conditioning.conditioning_dim", conditioning_dim + 64),
                _set_scalar_edit(reference_ir, "representation.trunk.hidden_dim", hidden_dim + 32),
            ],
        ),
        (
            "breadth_optimizer_reset",
            "Probe a more conservative optimizer regime around the current structure.",
            "Some tasks reward a boring but stable optimizer reset before further architectural refinement.",
            [
                _set_scalar_edit(reference_ir, "representation.optimizer.learning_rate", min(learning_rate, 5e-4)),
                _set_scalar_edit(reference_ir, "representation.optimizer.batch_size", min(batch_size, 96)),
                _set_scalar_edit(reference_ir, "representation.optimizer.weight_decay", max(weight_decay, 2e-4)),
            ],
        ),
    ]
    if group in {"single_cell", "multimodal_rna", "multimodal_protein"}:
        macro_specs.append(
            (
                "breadth_stable_head",
                "Probe a more conservative response head before deeper local exploitation.",
                "Anchored sparse heads are a common direct-code win pattern that should be tested explicitly.",
                [
                    _toggle_to_edit(reference_ir, "representation.prediction.zero_init_head", True),
                    _set_scalar_edit(reference_ir, "representation.loss.response_weight", min(response_weight, 0.2)),
                ],
            )
        )
    if group == "paired_clinical":
        macro_specs.append(
            (
                "breadth_paired_anchor",
                "Re-anchor the model on delta-style paired response before local trust-region refinement.",
                "Clinical paired-response tasks often need an explicit anchor before capacity changes matter.",
                [
                    _toggle_to_edit(reference_ir, "representation.prediction.baseline_skip", True),
                    _enum_to_edit(reference_ir, "representation.prediction.target", "delta"),
                    _toggle_to_edit(reference_ir, "representation.prediction.zero_init_head", True),
                    _set_scalar_edit(reference_ir, "representation.loss.response_weight", min(response_weight, 0.25)),
                ],
            )
        )
    if group == "dose_time":
        macro_specs.append(
            (
                "breadth_dose_context",
                "Probe a larger conditioning bottleneck and conservative optimizer for dose/time response.",
                "Large public perturbation tasks tend to need explicit context width and stable optimization together.",
                [
                    _enum_to_edit(reference_ir, "representation.conditioning.mode", "film"),
                    _set_scalar_edit(reference_ir, "representation.conditioning.conditioning_dim", conditioning_dim + 64),
                    _set_scalar_edit(reference_ir, "representation.optimizer.learning_rate", min(learning_rate, 4e-4)),
                    _set_scalar_edit(reference_ir, "representation.optimizer.batch_size", min(batch_size, 96)),
                ],
            )
        )

    candidates = _cross_family_baseline_candidates(
        seed_family_specs=seed_family_specs,
        reference_ir=reference_ir,
        current_family_name=current_family_name,
    )
    candidates.extend(
        _proposal_payload(
            proposal_source="adaptive_breadth_v6",
            proposal_note=proposal_note,
            scientific_hypothesis=hypothesis,
            mechanistic_rationale=rationale,
            edits=[edit for edit in raw_edits if edit is not None],
            reference_ir=reference_ir,
            candidate_seed_model_name=current_family_name,
        )
        for proposal_note, hypothesis, rationale, raw_edits in macro_specs
        if any(edit is not None for edit in raw_edits)
    )
    candidates.extend(_portfolio_candidates(reference_ir, dataset_key, current_family_name))
    return _dedupe_candidate_pool(candidates)


def _constraint_optimizer_candidates(
    *,
    best_ir: Dict[str, Any],
    dataset_key: str,
    candidate_seed_model_name: str,
) -> List[Dict[str, Any]]:
    group = _task_group(dataset_key)
    learning_rate = float(_get_nested(best_ir, "representation.optimizer.learning_rate") or 8e-4)
    batch_size = int(_get_nested(best_ir, "representation.optimizer.batch_size") or 128)
    weight_decay = float(_get_nested(best_ir, "representation.optimizer.weight_decay") or 1e-4)
    response_weight = float(_get_nested(best_ir, "representation.loss.response_weight") or 0.35)
    hidden_dim = int(_get_nested(best_ir, "representation.trunk.hidden_dim") or 128)
    conditioning_dim = int(_get_nested(best_ir, "representation.conditioning.conditioning_dim") or 64)

    specs: List[tuple[str, str, str, List[Dict[str, Any] | None]]] = [
        (
            "constraint_optimizer_tighten",
            "Tighten optimizer settings while staying in the current mechanistic basin.",
            "Once a basin is found, smaller learning rate and modest regularization are the safest way to extract the remaining gain.",
            [
                _set_scalar_edit(best_ir, "representation.optimizer.learning_rate", min(learning_rate, 4e-4)),
                _set_scalar_edit(best_ir, "representation.optimizer.weight_decay", max(weight_decay, 2e-4)),
                _set_scalar_edit(best_ir, "representation.optimizer.batch_size", min(batch_size, 96)),
            ],
        ),
        (
            "constraint_stable_head",
            "Preserve a conservative response head during local refinement.",
            "If breadth exploration found a plausible basin, stabilizing the head usually improves transfer more than reopening the search space.",
            [
                _toggle_to_edit(best_ir, "representation.prediction.zero_init_head", True),
                _set_scalar_edit(best_ir, "representation.loss.response_weight", min(response_weight, 0.2)),
            ],
        ),
        (
            "constraint_capacity_nudge",
            "Make a small capacity nudge without leaving the current family or conditioning regime.",
            "This is the smallest safe step that still allows local exploitation to continue improving.",
            [
                _set_scalar_edit(best_ir, "representation.trunk.hidden_dim", hidden_dim + 16),
                _set_scalar_edit(best_ir, "representation.conditioning.conditioning_dim", conditioning_dim + 16),
            ],
        ),
    ]
    if group == "paired_clinical":
        specs.insert(
            0,
            (
                "constraint_paired_anchor_lock",
                "Lock in paired-response anchoring before final exploitation.",
                "Clinical response tasks usually punish unanchored late-stage drift more than they reward extra capacity.",
                [
                    _toggle_to_edit(best_ir, "representation.prediction.baseline_skip", True),
                    _enum_to_edit(best_ir, "representation.prediction.target", "delta"),
                    _toggle_to_edit(best_ir, "representation.prediction.zero_init_head", True),
                ],
            ),
        )

    candidates: List[Dict[str, Any]] = []
    for proposal_note, hypothesis, rationale, raw_edits in specs:
        edits = [edit for edit in raw_edits if edit is not None]
        if not edits:
            continue
        candidates.append(
            _proposal_payload(
                proposal_source="adaptive_constrain_v6",
                proposal_note=proposal_note,
                scientific_hypothesis=hypothesis,
                mechanistic_rationale=rationale,
                edits=edits,
                reference_ir=best_ir,
                candidate_seed_model_name=candidate_seed_model_name,
            )
        )
    return _dedupe_candidate_pool(candidates)


def _constraint_candidate_pool(
    *,
    projected_ir: Dict[str, Any],
    dataset_key: str,
    candidate_seed_model_name: str,
    projection_signature: Mapping[str, Any],
    lock_in: bool,
) -> List[Dict[str, Any]]:
    stage = "mechanistic_lock_in" if lock_in else "projected_trust_region"
    trust_region = _trust_region_candidates(
        projected_ir=projected_ir,
        dataset_key=dataset_key,
        candidate_seed_model_name=candidate_seed_model_name,
        projection_signature=projection_signature,
        stage=stage,
    )
    optimizer_pool = _constraint_optimizer_candidates(
        best_ir=projected_ir,
        dataset_key=dataset_key,
        candidate_seed_model_name=candidate_seed_model_name,
    )
    if lock_in:
        return _dedupe_candidate_pool(optimizer_pool + trust_region)
    return _dedupe_candidate_pool(trust_region + optimizer_pool)


def _bucket_int(value: Any, step: int) -> int:
    numeric = int(value or 0)
    return int((numeric // step) * step)


def _bucket_lr(value: Any) -> str:
    numeric = float(value or 0.0)
    if numeric <= 2.5e-4:
        return "low"
    if numeric <= 6.0e-4:
        return "mid"
    return "high"


def _basin_signature_from_ir(model_ir: Mapping[str, Any], seed_family_name: str) -> str:
    payload = {
        "seed_family_name": str(seed_family_name),
        "target": _get_nested(model_ir, "representation.prediction.target"),
        "baseline_skip": bool(_get_nested(model_ir, "representation.prediction.baseline_skip")),
        "zero_init_head": bool(_get_nested(model_ir, "representation.prediction.zero_init_head")),
        "conditioning_mode": _get_nested(model_ir, "representation.conditioning.mode"),
        "conditioning_dim_bucket": _bucket_int(_get_nested(model_ir, "representation.conditioning.conditioning_dim"), 64),
        "hidden_dim_bucket": _bucket_int(_get_nested(model_ir, "representation.trunk.hidden_dim"), 64),
        "trunk_depth": int(_get_nested(model_ir, "representation.trunk.trunk_depth") or 0),
        "residual_depth": int(_get_nested(model_ir, "representation.trunk.residual_depth") or 0),
        "lr_bucket": _bucket_lr(_get_nested(model_ir, "representation.optimizer.learning_rate")),
    }
    return stable_hash(payload)


def _diversify_breadth_candidate_pool(
    candidates: Sequence[Dict[str, Any]],
    *,
    per_signature_limit: int = 2,
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    signature_order: List[str] = []
    for candidate in candidates:
        candidate_seed_model_name = str(candidate.get("candidate_seed_model_name") or "")
        signature = _basin_signature_from_ir(dict(candidate.get("candidate_ir", {})), candidate_seed_model_name)
        if signature not in grouped:
            grouped[signature] = []
            signature_order.append(signature)
        grouped[signature].append(dict(candidate))

    ordered: List[Dict[str, Any]] = []
    for round_index in range(int(max(1, per_signature_limit))):
        for signature in signature_order:
            bucket = grouped[signature]
            if round_index < len(bucket):
                ordered.append(dict(bucket[round_index]))
    for signature in signature_order:
        bucket = grouped[signature]
        for candidate in bucket[int(max(1, per_signature_limit)) :]:
            ordered.append(dict(candidate))
    return _dedupe_candidate_pool(ordered)


def _actionable_projected_fields(projection_signature: Mapping[str, Any] | None) -> List[Dict[str, Any]]:
    if not projection_signature:
        return []
    return [
        dict(item)
        for item in projection_signature.get("projected_fields", [])
        if str(item.get("path")) != "__best_warmup_model__"
    ]


def _normalized_projection_signature_v6(
    *,
    history: Sequence[Dict[str, Any]],
    seed_family_specs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    normalized_history: List[Dict[str, Any]] = []
    for item in history:
        row = dict(item)
        if str(row.get("search_stage")) == "adaptive_breadth":
            row["search_stage"] = "breadth_warmup"
        normalized_history.append(row)

    signature = _projection_signature(history=normalized_history, seed_family_specs=seed_family_specs)
    if _actionable_projected_fields(signature):
        return signature

    breadth_records = [
        dict(item)
        for item in normalized_history
        if str(item.get("search_stage")) == "breadth_warmup"
        and str(item.get("execution_status")) == "completed"
        and item.get("objective_value") is not None
    ]
    best_breadth = _best_record(breadth_records)
    if best_breadth is None:
        return signature

    anchor_family_name = str(
        best_breadth.get("candidate_seed_model_name")
        or best_breadth.get("seed_family_name")
        or seed_family_specs[0]["seed_model_name"]
    )
    anchor_family_spec = next(
        (dict(item) for item in seed_family_specs if str(item["seed_model_name"]) == anchor_family_name),
        dict(seed_family_specs[0]),
    )
    anchor_seed_ir = _deepcopy_ir(anchor_family_spec["seed_ir"])
    anchor_objective_value = None
    for item in normalized_history:
        if (
            str(item.get("search_stage")) == "seed_family_warmup"
            and str(item.get("candidate_seed_model_name") or item.get("seed_family_name") or "") == anchor_family_name
            and item.get("objective_value") is not None
        ):
            anchor_objective_value = float(item["objective_value"])
            break

    best_breadth_ir = _deepcopy_ir(best_breadth["model_ir"])
    projected_ir = _deepcopy_ir(anchor_seed_ir)
    projected_fields: List[Dict[str, Any]] = []
    confidence_values: List[float] = []
    breadth_objective_value = float(best_breadth["objective_value"])
    relative_gain = 0.0
    if anchor_objective_value is not None and anchor_objective_value > 0:
        relative_gain = max(0.0, anchor_objective_value - breadth_objective_value) / float(anchor_objective_value)

    for path in PROJECTABLE_PATHS:
        anchor_value = _get_nested(anchor_seed_ir, path)
        breadth_value = _get_nested(best_breadth_ir, path)
        if anchor_value == breadth_value:
            continue
        if isinstance(anchor_value, (int, float)) and isinstance(breadth_value, (int, float)):
            if path.endswith(("hidden_dim", "conditioning_dim")) and abs(float(breadth_value) - float(anchor_value)) < 16.0:
                continue
            if path.endswith(("trunk_depth", "residual_depth")) and int(breadth_value) == int(anchor_value):
                continue
            if path.endswith("learning_rate"):
                low = max(min(abs(float(anchor_value)), abs(float(breadth_value))), 1.0e-12)
                high = max(abs(float(anchor_value)), abs(float(breadth_value)))
                if high / low < 1.25:
                    continue
            if path.endswith("weight_decay"):
                low = max(min(abs(float(anchor_value)), abs(float(breadth_value))), 1.0e-12)
                high = max(abs(float(anchor_value)), abs(float(breadth_value)))
                if high / low < 1.5:
                    continue
            if path.endswith("response_weight") and abs(float(breadth_value) - float(anchor_value)) < 0.05:
                continue
        _set_nested_value(projected_ir, path, breadth_value)
        confidence = min(0.92, 0.62 + relative_gain)
        projected_fields.append(
            {
                "path": path,
                "value": breadth_value,
                "confidence": float(confidence),
                "support_iterations": [int(best_breadth["iteration"])],
            }
        )
        confidence_values.append(float(confidence))

    if not projected_fields:
        return signature

    return {
        "anchor_seed_family_name": anchor_family_name,
        "anchor_source": "adaptive_breadth_macro_fallback",
        "anchor_objective_value": anchor_objective_value,
        "projected_ir": projected_ir,
        "projected_fields": projected_fields,
        "projection_confidence": float(sum(confidence_values) / len(confidence_values)) if confidence_values else 0.0,
        "evidence_iterations": [int(best_breadth["iteration"])],
        "evidence_count": 1,
    }


def _recent_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _session_memory_snapshot(
    *,
    history: Sequence[Dict[str, Any]],
    seed_family_specs: Sequence[Mapping[str, Any]],
    projection_signature: Mapping[str, Any] | None,
    warmup_plan_available: bool,
) -> Dict[str, Any]:
    completed = [
        dict(item)
        for item in history
        if str(item.get("execution_status")) == "completed" and item.get("objective_value") is not None
    ]
    best_record = _best_completed_record(history)
    last_record = dict(history[-1]) if history else {}
    best_iteration = int(best_record["iteration"]) if best_record is not None else -1
    current_iteration = int(last_record.get("iteration", -1))
    iterations_since_best = max(0, current_iteration - best_iteration) if best_iteration >= 0 and current_iteration >= 0 else 0

    action_history = [str(item.get("search_action")) for item in history if str(item.get("search_action")) in {"breadth", "constrain"}]
    last_action = action_history[-1] if action_history else None
    action_streak = 0
    for action in reversed(action_history):
        if action != last_action:
            break
        action_streak += 1

    family_history = [
        str(item.get("candidate_seed_model_name") or item.get("seed_family_name") or "")
        for item in completed
        if item.get("candidate_seed_model_name") or item.get("seed_family_name")
    ]
    current_best_family_name = (
        str(best_record.get("candidate_seed_model_name") or best_record.get("seed_family_name") or "")
        if best_record is not None
        else ""
    )
    same_family_streak = 0
    for family_name in reversed(family_history):
        if not family_name or family_name != current_best_family_name:
            break
        same_family_streak += 1

    breadth_completed = [item for item in completed if str(item.get("search_action")) == "breadth"]
    constrain_completed = [item for item in completed if str(item.get("search_action")) == "constrain"]

    recent_completed = completed[-3:]
    recent_gains = [
        float(item["delta_vs_reference"])
        for item in recent_completed
        if isinstance(item.get("delta_vs_reference"), (int, float))
    ]
    breadth_recent_gains = [
        float(item["delta_vs_reference"])
        for item in breadth_completed[-2:]
        if isinstance(item.get("delta_vs_reference"), (int, float))
    ]
    constrain_recent_gains = [
        float(item["delta_vs_reference"])
        for item in constrain_completed[-2:]
        if isinstance(item.get("delta_vs_reference"), (int, float))
    ]

    explored_seed_families = {
        str(item.get("candidate_seed_model_name") or item.get("seed_family_name") or "")
        for item in completed
        if item.get("candidate_seed_model_name") or item.get("seed_family_name")
    }
    unexplored_seed_family_names = [
        str(spec["seed_model_name"])
        for spec in seed_family_specs
        if str(spec["seed_model_name"]) not in explored_seed_families
    ]

    action_path = "".join("B" if action == "breadth" else "C" for action in action_history)
    return {
        "completed_count": int(len(completed)),
        "best_iteration": int(best_iteration),
        "best_stage": str(best_record.get("search_stage")) if best_record is not None else None,
        "best_family_name": current_best_family_name,
        "iterations_since_best": int(iterations_since_best),
        "last_action": last_action,
        "last_action_streak": int(action_streak),
        "same_family_streak": int(same_family_streak),
        "recent_gain_mean": _recent_mean(recent_gains),
        "breadth_recent_gain_mean": _recent_mean(breadth_recent_gains),
        "constrain_recent_gain_mean": _recent_mean(constrain_recent_gains),
        "last_gain": float(last_record.get("delta_vs_reference") or 0.0) if last_record else 0.0,
        "breadth_completed_count": int(len(breadth_completed)),
        "constrain_completed_count": int(len(constrain_completed)),
        "breadth_improve_count": int(sum(1 for item in breadth_completed if float(item.get("delta_vs_reference") or 0.0) > 0)),
        "constrain_improve_count": int(sum(1 for item in constrain_completed if float(item.get("delta_vs_reference") or 0.0) > 0)),
        "projection_available": bool(_actionable_projected_fields(projection_signature)),
        "projection_confidence": float(projection_signature.get("projection_confidence") or 0.0) if projection_signature else 0.0,
        "unexplored_seed_family_names": unexplored_seed_family_names,
        "warmup_plan_available": bool(warmup_plan_available),
        "action_path": action_path,
    }


def _score_actions(*, snapshot: Mapping[str, Any], task_group: str, pending_projection: bool) -> Dict[str, Any]:
    breadth_score = 0.0
    constrain_score = 0.0
    breadth_reasons: List[str] = []
    constrain_reasons: List[str] = []

    if snapshot["unexplored_seed_family_names"]:
        breadth_score += 6.0
        breadth_reasons.append("unexplored_seed_family")
    if int(snapshot["iterations_since_best"]) >= 2:
        breadth_score += 3.0
        breadth_reasons.append("stagnation_since_best")
    if int(snapshot["same_family_streak"]) >= 3:
        breadth_score += 2.0
        breadth_reasons.append("family_lock_in_detected")
    if float(snapshot["recent_gain_mean"]) <= 0.005:
        breadth_score += 1.5
        breadth_reasons.append("low_recent_gain")
    if snapshot["last_action"] == "constrain" and int(snapshot["last_action_streak"]) >= 2:
        breadth_score += 1.0
        breadth_reasons.append("constrain_streak")
    if task_group in {"dose_time", "paired_clinical"}:
        breadth_score += 0.5
        breadth_reasons.append("task_needs_escape_headroom")

    if bool(snapshot["projection_available"]):
        constrain_score += 2.0
        constrain_reasons.append("projection_available")
    if pending_projection:
        constrain_score += 2.0
        constrain_reasons.append("pending_projection_bridge")
    if snapshot["last_action"] == "breadth" and float(snapshot["last_gain"]) > 0:
        constrain_score += 3.0
        constrain_reasons.append("breadth_found_gain")
    if int(snapshot["iterations_since_best"]) == 0:
        constrain_score += 1.5
        constrain_reasons.append("recent_best_found")
    if float(snapshot["constrain_recent_gain_mean"]) > 0.005:
        constrain_score += 1.0
        constrain_reasons.append("constrain_is_working")
    if task_group in {"multimodal_protein", "paired_clinical"}:
        constrain_score += 0.5
        constrain_reasons.append("task_benefits_from_anchor")

    return {
        "breadth_score": float(breadth_score),
        "constrain_score": float(constrain_score),
        "breadth_reasons": breadth_reasons,
        "constrain_reasons": constrain_reasons,
    }


def _projection_fingerprint(projection_signature: Mapping[str, Any] | None) -> str | None:
    if not projection_signature:
        return None
    fields = _actionable_projected_fields(projection_signature)
    if not fields:
        return None
    normalized_fields = []
    for item in fields:
        normalized_fields.append(
            {
                "path": str(item.get("path")),
                "value": item.get("value"),
                "confidence": float(item.get("confidence") or 0.0),
            }
        )
    payload = {
        "anchor_seed_family_name": str(projection_signature.get("anchor_seed_family_name") or ""),
        "projected_fields": normalized_fields,
    }
    return stable_hash(payload)


def _should_refresh_warmup_plan(
    *,
    snapshot: Mapping[str, Any],
    warmup_plan: Mapping[str, Any] | None,
) -> bool:
    if warmup_plan is None:
        return True
    if int(snapshot["iterations_since_best"]) >= 3:
        return True
    if snapshot["last_action"] == "breadth" and int(snapshot["last_action_streak"]) >= 2 and float(snapshot["last_gain"]) <= 0:
        return True
    return False


def _choose_adaptive_action(
    *,
    snapshot: Mapping[str, Any],
    task_group: str,
    pending_projection: bool,
    warmup_plan: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    scores = _score_actions(snapshot=snapshot, task_group=task_group, pending_projection=pending_projection)
    if snapshot["unexplored_seed_family_names"]:
        return {
            "action": "breadth",
            "stage": "seed_family_warmup",
            "scores": scores,
            "decision_reason": "cover_each_seed_family_once",
            "refresh_warmup_plan": False,
            "lock_in": False,
        }
    breadth_score = float(scores["breadth_score"])
    constrain_score = float(scores["constrain_score"])
    if breadth_score > constrain_score:
        action = "breadth"
    elif constrain_score > breadth_score:
        action = "constrain"
    else:
        action = "constrain" if pending_projection or snapshot.get("last_action") == "breadth" else "breadth"
    lock_in = bool(
        action == "constrain"
        and bool(snapshot["projection_available"])
        and int(snapshot["iterations_since_best"]) <= 1
        and int(snapshot["constrain_improve_count"]) >= 1
    )
    return {
        "action": action,
        "stage": (
            "projection_bridge"
            if action == "constrain" and pending_projection
            else ("adaptive_lock_in" if lock_in else ("adaptive_breadth" if action == "breadth" else "adaptive_constrain"))
        ),
        "scores": scores,
        "decision_reason": (
            "pending_projection_bridge"
            if action == "constrain" and pending_projection
            else ("lock_in_after_projection_gain" if lock_in else ("breadth_escape" if action == "breadth" else "constrain_refine"))
        ),
        "refresh_warmup_plan": bool(action == "breadth" and _should_refresh_warmup_plan(snapshot=snapshot, warmup_plan=warmup_plan)),
        "lock_in": lock_in,
    }


def run_structured_model_session_v6(
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
    agent_mode: str = "structured_adaptive_tree_search_v6",
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

    log_lines = [
        f"dataset_key={dataset_key}",
        f"seed_model_config={seed_model_config_path}",
        f"seed_model_name={seed_model_name}",
        f"seed_families={json.dumps([item['seed_model_name'] for item in seed_family_specs], ensure_ascii=True)}",
        f"session_id={session_slug}",
        f"requested_device={requested_device}",
        f"cuda_visible_devices={cuda_visible_devices}",
        f"task_group={task_group}",
        "harness_mode=adaptive_breadth_constrain_tree_search_v6",
    ]
    log_lines = load_previous_log_lines(session_root / "agent.log", log_lines)

    def log(message: str) -> None:
        log_lines.append(message)
        print(f"[agent-structured-v6] {message}", flush=True)

    history: List[Dict[str, Any]] = load_history_rows(history_root)
    accepted_trace_rows: List[Dict[str, Any]] = load_rows_payload(trace_root / "edit_program_trace.json")
    rejection_rows: List[Dict[str, Any]] = load_rows_payload(rejection_root / "proposal_rejections.json")
    llm_usage_rows: List[Dict[str, Any]] = load_rows_payload(trace_root / "llm_usage.json")
    router_rows: List[Dict[str, Any]] = load_rows_payload(trace_root / "action_router.json")
    baseline_objective: float | None = baseline_objective_from_history(history)
    stopping_reason: str | None = None
    warmup_plan: Dict[str, Any] | None = None
    warmup_plan_refresh_count = 0
    projection_signature: Dict[str, Any] | None = None
    last_applied_projection_fingerprint: str | None = None
    iteration = next_iteration_index(history)
    if history:
        log(
            f"resume session_id={session_slug} next_iteration={iteration} "
            f"completed_evaluations={completed_evaluation_count(history)} candidate_attempts={len(history)}"
        )

    while should_continue_search(history, budget_limit):
        if iteration == 0:
            family_spec = dict(seed_family_specs[0])
            previous_ir = dict(family_spec["seed_ir"])
            reference_objective_value = None
            proposal = _seed_family_baseline_proposal(
                reference_ir=previous_ir,
                seed_ir=dict(family_spec["seed_ir"]),
                seed_model_name=str(family_spec["seed_model_name"]),
            )
            proposal_note = str(proposal["proposal_note"])
            search_stage = "baseline"
            search_action = "baseline"
            attempt_index_value = 0
            trace_payload = {
                "stage": search_stage,
                "task_group": task_group,
                "reference": "seed_baseline",
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
            action_scores_payload = {}
            action_decision_reason = "iteration_0_seed_baseline"
            action_path = ""
        else:
            projection_signature = _normalized_projection_signature_v6(history=history, seed_family_specs=seed_family_specs)
            pending_projection = False
            projection_fingerprint = _projection_fingerprint(projection_signature)
            if projection_fingerprint is not None and projection_fingerprint != last_applied_projection_fingerprint:
                pending_projection = True
            snapshot = _session_memory_snapshot(
                history=history,
                seed_family_specs=seed_family_specs,
                projection_signature=projection_signature,
                warmup_plan_available=warmup_plan is not None,
            )
            decision = _choose_adaptive_action(
                snapshot=snapshot,
                task_group=task_group,
                pending_projection=pending_projection,
                warmup_plan=warmup_plan,
            )
            search_action = str(decision["action"])
            search_stage = str(decision["stage"])
            action_scores_payload = dict(decision["scores"])
            action_decision_reason = str(decision["decision_reason"])
            action_path = str(snapshot["action_path"]) + ("B" if search_action == "breadth" else "C")
            router_rows.append(
                {
                    "iteration": int(iteration),
                    "search_action": search_action,
                    "search_stage": search_stage,
                    "decision_reason": action_decision_reason,
                    "action_path": action_path,
                    "snapshot_json": json.dumps(snapshot, ensure_ascii=True, sort_keys=True),
                    "scores_json": json.dumps(action_scores_payload, ensure_ascii=True, sort_keys=True),
                }
            )

            if search_action == "breadth" and search_stage == "seed_family_warmup":
                next_family_name = str(snapshot["unexplored_seed_family_names"][0])
                family_spec = next(
                    dict(item) for item in seed_family_specs if str(item["seed_model_name"]) == next_family_name
                )
                best_completed = _best_completed_record(history)
                previous_ir = (
                    _deepcopy_ir(best_completed["model_ir"])
                    if best_completed is not None
                    else dict(seed_family_specs[0]["seed_ir"])
                )
                reference_objective_value = (
                    float(best_completed["objective_value"])
                    if best_completed is not None and best_completed.get("objective_value") is not None
                    else None
                )
                proposal = _seed_family_baseline_proposal(
                    reference_ir=previous_ir,
                    seed_ir=dict(family_spec["seed_ir"]),
                    seed_model_name=next_family_name,
                )
                proposal_note = str(proposal["proposal_note"])
                attempt_index_value = 1
                trace_payload = {
                    "stage": search_stage,
                    "task_group": task_group,
                    "reference": "seed_family_coverage",
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
            elif search_action == "breadth":
                if bool(decision["refresh_warmup_plan"]):
                    warmup_plan = _request_code_first_warmup_plan(
                        repo_root=repo_root,
                        dataset_key=dataset_key,
                        task_group=task_group,
                        family_specs=seed_family_specs,
                        history=history,
                        llm_client=llm_client,
                    )
                    warmup_plan_refresh_count += 1
                    dump_json(trace_root / "code_first_warmup_plan.json", warmup_plan)
                    dump_json(trace_root / f"code_first_warmup_plan_iter_{iteration:03d}.json", warmup_plan)
                    for usage_event in llm_client.drain_usage_events():
                        llm_usage_rows.append({"iteration": int(iteration), "attempt_index": 0, **usage_event})
                    log(
                        f"warmup_plan preferred_seed_family={warmup_plan.get('preferred_seed_family')} "
                        f"warmup_order={warmup_plan.get('warmup_order', [])}"
                    )
                best_completed = _best_completed_record(history)
                current_reference_ir = (
                    _deepcopy_ir(best_completed["model_ir"])
                    if best_completed is not None
                    else _deepcopy_ir(seed_family_specs[0]["seed_ir"])
                )
                current_reference_family_name = (
                    str(best_completed.get("candidate_seed_model_name") or best_completed.get("seed_family_name") or seed_model_name)
                    if best_completed is not None
                    else seed_model_name
                )
                reference_objective_value = (
                    float(best_completed["objective_value"])
                    if best_completed is not None and best_completed.get("objective_value") is not None
                    else None
                )
                breadth_queue = _build_breadth_candidate_queue(
                    family_specs=seed_family_specs,
                    dataset_key=dataset_key,
                    history=history,
                    warmup_plan=warmup_plan or {},
                )
                macro_pool = _breadth_macro_candidates(
                    reference_ir=current_reference_ir,
                    dataset_key=dataset_key,
                    current_family_name=current_reference_family_name,
                    seed_family_specs=seed_family_specs,
                )
                candidate_pool = _diversify_breadth_candidate_pool(_dedupe_candidate_pool(breadth_queue + macro_pool))
                trace_payload = _candidate_trace_payload(
                    stage=search_stage,
                    task_group=task_group,
                    reference=f"best_ir::{current_reference_family_name}",
                    candidates=candidate_pool,
                )
                previous_ir = current_reference_ir
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
                    stopping_reason = "no_valid_v6_breadth_candidate"
                    log(f"stopping iteration={iteration} reason={stopping_reason}")
                    break
                proposal_note = str(proposal["proposal_note"])
            else:
                best_completed = _best_completed_record(history)
                if best_completed is None:
                    raise RuntimeError(f"No completed adaptive records found for {dataset_key}/{seed_model_name}")
                if pending_projection and projection_signature is not None and projection_fingerprint is not None:
                    search_stage = "projection_bridge"
                    anchor_family_name = str(projection_signature["anchor_seed_family_name"])
                    anchor_family_spec = next(
                        (dict(item) for item in seed_family_specs if str(item["seed_model_name"]) == anchor_family_name),
                        dict(seed_family_specs[0]),
                    )
                    previous_ir = _deepcopy_ir(anchor_family_spec["seed_ir"])
                    reference_objective_value = (
                        float(projection_signature["anchor_objective_value"])
                        if projection_signature.get("anchor_objective_value") is not None
                        else None
                    )
                    proposal = _projection_proposal(
                        anchor_seed_ir=previous_ir,
                        projected_ir=_deepcopy_ir(projection_signature["projected_ir"]),
                        anchor_seed_family_name=anchor_family_name,
                        projection_signature=projection_signature,
                    )
                    seen_ir_hashes = {str(item.get("ir_hash")) for item in history}
                    if str(proposal.get("ir_hash")) in seen_ir_hashes:
                        fallback_candidates = _constraint_candidate_pool(
                            projected_ir=_deepcopy_ir(projection_signature["projected_ir"]),
                            dataset_key=dataset_key,
                            candidate_seed_model_name=anchor_family_name,
                            projection_signature=projection_signature,
                            lock_in=bool(decision["lock_in"]),
                        )
                        replacement = next(
                            (dict(item) for item in fallback_candidates if str(item.get("ir_hash")) not in seen_ir_hashes),
                            None,
                        )
                        if replacement is not None:
                            proposal = replacement
                    proposal_note = str(proposal["proposal_note"])
                    attempt_index_value = 1
                    trace_payload = {
                        "stage": search_stage,
                        "task_group": task_group,
                        "reference": f"projection_anchor::{anchor_family_name}",
                        "projection_signature": projection_signature,
                        "candidates": [
                            {
                                "proposal_note": proposal_note,
                                "proposal_source": proposal["proposal_source"],
                                "candidate_seed_model_name": proposal.get("candidate_seed_model_name"),
                                "edit_preview": [describe_edit(edit) for edit in proposal.get("edits", [])],
                                "ir_hash": proposal.get("ir_hash"),
                            }
                        ],
                    }
                    last_applied_projection_fingerprint = projection_fingerprint
                else:
                    previous_ir = _deepcopy_ir(best_completed["model_ir"])
                    reference_objective_value = float(best_completed["objective_value"])
                    current_reference_family_name = str(
                        best_completed.get("candidate_seed_model_name")
                        or best_completed.get("seed_family_name")
                        or seed_model_name
                    )
                    search_stage = "adaptive_lock_in" if bool(decision["lock_in"]) else "adaptive_constrain"
                    candidate_pool = _constraint_candidate_pool(
                        projected_ir=previous_ir,
                        dataset_key=dataset_key,
                        candidate_seed_model_name=current_reference_family_name,
                        projection_signature=projection_signature or {},
                        lock_in=bool(decision["lock_in"]),
                    )
                    trace_payload = _candidate_trace_payload(
                        stage=search_stage,
                        task_group=task_group,
                        reference=f"best_ir::{current_reference_family_name}",
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
                        stopping_reason = "no_valid_v6_constraint_candidate"
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
        router_trace_path = trace_root / f"iter_{iteration:03d}_router.json"

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
                "search_action": search_action,
                "search_stage": search_stage,
                "action_path": action_path,
                "action_decision_reason": action_decision_reason,
                "action_scores": action_scores_payload,
                "proposal_source": proposal["proposal_source"],
                "proposal_note": proposal_note,
                "attempt_index": int(attempt_index_value),
                "selected_mechanism_templates": list(proposal.get("selected_mechanism_templates", [])),
                "scientific_hypothesis": proposal.get("scientific_hypothesis", ""),
                "mechanistic_rationale": proposal.get("mechanistic_rationale", ""),
                "risk_notes": proposal.get("risk_notes", []),
                "candidate_seed_model_name": proposal.get("candidate_seed_model_name", seed_model_name),
                "edits": proposal["edits"],
                "ir_hash": proposal["ir_hash"],
                "warmup_plan": warmup_plan if search_action == "breadth" else None,
                "projection_signature": projection_signature if search_action == "constrain" else None,
            },
        )
        dump_json(trace_stage_path, trace_payload)
        dump_json(
            router_trace_path,
            {
                "iteration": int(iteration),
                "search_action": search_action,
                "search_stage": search_stage,
                "action_path": action_path,
                "decision_reason": action_decision_reason,
                "scores": action_scores_payload,
            },
        )
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
                run_name=f"{dataset_key}_{seed_model_name}_agent_structured_v6_iter_{iteration:03d}",
                artifacts_root=benchmark_runs_root,
                seed=int(seed),
                top_k=int(top_k),
                model_config=compiled_model_path,
            ),
        )

        log(
            f"dispatch iteration={iteration} action={search_action} stage={search_stage} "
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
                "phase": "baseline" if iteration == 0 else "agent",
                "search_action": search_action,
                "search_stage": search_stage,
                "action_path": action_path,
                "action_decision_reason": action_decision_reason,
                "action_scores_json": json.dumps(action_scores_payload, ensure_ascii=True, sort_keys=True),
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
                "task_group": task_group,
                "preferred_warmup_seed_family_name": str(warmup_plan.get("preferred_seed_family")) if warmup_plan else None,
                "projection_anchor_seed_family_name": (
                    str(projection_signature.get("anchor_seed_family_name"))
                    if projection_signature is not None and projection_signature.get("anchor_seed_family_name") is not None
                    else None
                ),
                "projection_fingerprint": last_applied_projection_fingerprint,
                "warmup_plan_path": str((trace_root / "code_first_warmup_plan.json").resolve()) if warmup_plan else None,
                "projection_signature_path": str((trace_root / "projection_signature.json").resolve()) if projection_signature else None,
                "harness_review_path": str(harness_review_path.resolve()),
                "search_stage_trace_path": str(trace_stage_path.resolve()),
                "router_trace_path": str(router_trace_path.resolve()),
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
                "search_action": search_action,
                "search_stage": search_stage,
                "action_path": action_path,
                "action_decision_reason": action_decision_reason,
                "action_scores_json": json.dumps(action_scores_payload, ensure_ascii=True, sort_keys=True),
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
                "task_group": task_group,
                "preferred_warmup_seed_family_name": str(warmup_plan.get("preferred_seed_family")) if warmup_plan else None,
                "projection_anchor_seed_family_name": (
                    str(projection_signature.get("anchor_seed_family_name"))
                    if projection_signature is not None and projection_signature.get("anchor_seed_family_name") is not None
                    else None
                ),
                "projection_fingerprint": last_applied_projection_fingerprint,
                "warmup_plan_path": str((trace_root / "code_first_warmup_plan.json").resolve()) if warmup_plan else None,
                "projection_signature_path": str((trace_root / "projection_signature.json").resolve()) if projection_signature else None,
                "harness_review_path": str(harness_review_path.resolve()),
                "search_stage_trace_path": str(trace_stage_path.resolve()),
                "router_trace_path": str(router_trace_path.resolve()),
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
                    "search_action": search_action,
                    "search_stage": search_stage,
                    "action_path": action_path,
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
    dump_json(trace_root / "action_router.json", {"rows": router_rows})
    if router_rows:
        _write_trace_csv(trace_root / "action_router.csv", router_rows)
    dump_json(
        trace_root / "action_memory.json",
        {
            "warmup_plan_refresh_count": int(warmup_plan_refresh_count),
            "final_projection_signature": projection_signature,
            "last_applied_projection_fingerprint": last_applied_projection_fingerprint,
            "action_path": "".join(
                "B" if str(item.get("search_action")) == "breadth" else "C"
                for item in history
                if str(item.get("search_action")) in {"breadth", "constrain"}
            ),
        },
    )

    dump_json(
        session_root / "agent_session.json",
        {
            "dataset_key": dataset_key,
            "method_family": "agent_structured_model",
            "method_name": f"agent_structured_v6::{seed_model_name}",
            "agent_line": "main_agent_v6",
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
            "harness_mode": "adaptive_breadth_constrain_tree_search_v6",
            "task_group": task_group,
            "seed_family_candidates_json": json.dumps([item["seed_model_name"] for item in seed_family_specs], ensure_ascii=True),
            "warmup_plan": warmup_plan,
            "warmup_plan_refresh_count": int(warmup_plan_refresh_count),
            "projection_signature": projection_signature,
            "last_applied_projection_fingerprint": last_applied_projection_fingerprint,
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
        "method_name": f"agent_structured_v6::{seed_model_name}",
        "method_family": "agent_structured_model",
        "agent_line": "main_agent_v6",
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
