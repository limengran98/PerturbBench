from __future__ import annotations

import copy
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
    _proposal_payload,
    _portfolio_candidates,
    _set_scalar_edit,
    _seed_family_baseline_proposal,
    _task_group,
    _toggle_to_edit,
    _enum_to_edit,
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


PROJECTABLE_PATHS: tuple[str, ...] = (
    "representation.prediction.target",
    "representation.prediction.baseline_skip",
    "representation.prediction.zero_init_head",
    "representation.conditioning.mode",
    "representation.conditioning.conditioning_dim",
    "representation.trunk.hidden_dim",
    "representation.trunk.trunk_depth",
    "representation.trunk.residual_depth",
    "representation.optimizer.learning_rate",
    "representation.optimizer.weight_decay",
    "representation.optimizer.batch_size",
    "representation.loss.response_weight",
)


def _deepcopy_ir(model_ir: Mapping[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(dict(model_ir))


def _set_nested_value(payload: Dict[str, Any], path: str, value: Any) -> None:
    parts = [str(item) for item in str(path).split(".") if str(item)]
    cursor: Dict[str, Any] = payload
    for key in parts[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[parts[-1]] = value


def _candidate_objective(record: Mapping[str, Any]) -> float | None:
    value = record.get("objective_value")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _stage_completed_records(history: Sequence[Dict[str, Any]], stage: str) -> List[Dict[str, Any]]:
    return [
        dict(item)
        for item in history
        if str(item.get("search_stage")) == str(stage)
        and str(item.get("execution_status")) == "completed"
        and item.get("objective_value") is not None
    ]


def _best_record(records: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    valid = [dict(item) for item in records if _candidate_objective(item) is not None]
    if not valid:
        return None
    return min(valid, key=lambda item: float(item["objective_value"]))


def _projection_anchor(
    *,
    seed_family_records: Sequence[Dict[str, Any]],
    breadth_records: Sequence[Dict[str, Any]],
) -> Dict[str, Any] | None:
    best_seed_family = _best_record(seed_family_records)
    best_breadth = _best_record(breadth_records)
    if best_seed_family is None:
        return best_breadth
    if best_breadth is None:
        return best_seed_family
    best_seed_objective = float(best_seed_family["objective_value"])
    best_breadth_objective = float(best_breadth["objective_value"])
    if best_breadth_objective <= best_seed_objective * 0.97:
        return best_breadth
    return best_seed_family


def _projectable_edit(reference_ir: Dict[str, Any], path: str, target_value: Any) -> Dict[str, Any] | None:
    current_value = _get_nested(reference_ir, path)
    if current_value == target_value:
        return None
    if isinstance(target_value, bool):
        return _toggle_to_edit(reference_ir, path, bool(target_value))
    if isinstance(target_value, str):
        return _enum_to_edit(reference_ir, path, str(target_value))
    return _set_scalar_edit(reference_ir, path, target_value)


def _projection_signature(
    *,
    history: Sequence[Dict[str, Any]],
    seed_family_specs: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    seed_family_records = _stage_completed_records(history, "seed_family_warmup")
    breadth_records = _stage_completed_records(history, "breadth_warmup")
    anchor_record = _projection_anchor(
        seed_family_records=seed_family_records,
        breadth_records=breadth_records,
    )
    if anchor_record is None:
        family_spec = dict(seed_family_specs[0])
        anchor_seed_ir = _deepcopy_ir(family_spec["seed_ir"])
        return {
            "anchor_seed_family_name": str(family_spec["seed_model_name"]),
            "anchor_source": "fallback_seed_family",
            "anchor_objective_value": None,
            "projected_ir": anchor_seed_ir,
            "projected_fields": [],
            "projection_confidence": 0.0,
            "evidence_iterations": [],
            "evidence_count": 0,
        }

    anchor_family_name = str(anchor_record.get("candidate_seed_model_name") or anchor_record.get("seed_family_name") or seed_family_specs[0]["seed_model_name"])
    family_spec = next(
        (dict(item) for item in seed_family_specs if str(item["seed_model_name"]) == anchor_family_name),
        dict(seed_family_specs[0]),
    )
    anchor_seed_ir = _deepcopy_ir(family_spec["seed_ir"])
    anchor_objective = _candidate_objective(anchor_record)
    evidence_records = [
        dict(item)
        for item in breadth_records
        if _candidate_objective(item) is not None
        and anchor_objective is not None
        and float(item["objective_value"]) < float(anchor_objective)
    ]
    evidence_records.sort(key=lambda item: float(item["objective_value"]))

    projected_ir = _deepcopy_ir(anchor_seed_ir)
    projected_fields: List[Dict[str, Any]] = []
    confidence_values: List[float] = []
    for path in PROJECTABLE_PATHS:
        votes: Dict[str, Dict[str, Any]] = {}
        for record in evidence_records:
            candidate_ir = dict(record["model_ir"])
            candidate_value = _get_nested(candidate_ir, path)
            anchor_value = _get_nested(anchor_seed_ir, path)
            if candidate_value == anchor_value:
                continue
            objective_value = float(record["objective_value"])
            improvement = max(0.0, float(anchor_objective) - objective_value) if anchor_objective is not None else 0.0
            weight = max(1e-6, improvement)
            marker = json.dumps(candidate_value, sort_keys=True, ensure_ascii=True)
            bucket = votes.setdefault(marker, {"value": candidate_value, "weight": 0.0, "iterations": []})
            bucket["weight"] += float(weight)
            bucket["iterations"].append(int(record["iteration"]))
        if not votes:
            continue
        total_weight = sum(float(item["weight"]) for item in votes.values())
        winner = max(votes.values(), key=lambda item: float(item["weight"]))
        confidence = float(winner["weight"]) / float(total_weight) if total_weight > 0 else 0.0
        if confidence < 0.55:
            continue
        _set_nested_value(projected_ir, path, winner["value"])
        projected_fields.append(
            {
                "path": path,
                "value": winner["value"],
                "confidence": float(confidence),
                "support_iterations": list(winner["iterations"]),
            }
        )
        confidence_values.append(float(confidence))

    if not projected_fields and anchor_record in breadth_records:
        projected_ir = _deepcopy_ir(anchor_record["model_ir"])
        projected_fields.append(
            {
                "path": "__best_warmup_model__",
                "value": "record_model_ir",
                "confidence": 1.0,
                "support_iterations": [int(anchor_record["iteration"])],
            }
        )
        confidence_values.append(1.0)

    return {
        "anchor_seed_family_name": anchor_family_name,
        "anchor_source": "breadth_warmup" if anchor_record in breadth_records else "seed_family_warmup",
        "anchor_objective_value": anchor_objective,
        "projected_ir": projected_ir,
        "projected_fields": projected_fields,
        "projection_confidence": (
            float(sum(confidence_values) / len(confidence_values))
            if confidence_values
            else 0.0
        ),
        "evidence_iterations": [int(item["iteration"]) for item in evidence_records],
        "evidence_count": int(len(evidence_records)),
    }


def _projection_proposal(
    *,
    anchor_seed_ir: Dict[str, Any],
    projected_ir: Dict[str, Any],
    anchor_seed_family_name: str,
    projection_signature: Mapping[str, Any],
) -> Dict[str, Any]:
    edits: List[Dict[str, Any]] = []
    for item in projection_signature.get("projected_fields", []):
        path = str(item.get("path"))
        if path == "__best_warmup_model__":
            continue
        edit = _projectable_edit(anchor_seed_ir, path, item.get("value"))
        if edit is not None:
            edits.append(edit)
    if edits:
        return _proposal_payload(
            proposal_source="projection_v5",
            proposal_note="projected_signature_bridge",
            scientific_hypothesis="Bridge the strongest breadth-stage signals into a structured seed-space configuration before local refinement.",
            mechanistic_rationale="V5 turns broad warmup evidence into a small projected signature, then refines only around that projected basin.",
            edits=edits,
            reference_ir=anchor_seed_ir,
            candidate_seed_model_name=anchor_seed_family_name,
        )
    return {
        "proposal_source": "projection_v5",
        "proposal_note": "projected_signature_bridge::fallback",
        "selected_mechanism_templates": [],
        "scientific_hypothesis": "Keep the anchor family as-is because breadth warmup did not produce a stable projected signature.",
        "mechanistic_rationale": "When evidence is weak, V5 falls back to the strongest anchored structured configuration instead of hard-switching families.",
        "risk_notes": ["projection_fallback_anchor_ir"],
        "edits": [],
        "candidate_ir": _deepcopy_ir(projected_ir),
        "ir_hash": stable_hash(projected_ir),
        "candidate_seed_model_name": str(anchor_seed_family_name),
    }


def _dedupe_candidate_pool(candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for candidate in candidates:
        ir_hash = str(candidate.get("ir_hash"))
        if ir_hash in seen_hashes:
            continue
        seen_hashes.add(ir_hash)
        deduped.append(dict(candidate))
    return deduped


def _trust_region_candidates(
    *,
    projected_ir: Dict[str, Any],
    dataset_key: str,
    candidate_seed_model_name: str,
    projection_signature: Mapping[str, Any],
    stage: str,
) -> List[Dict[str, Any]]:
    projected_paths = {str(item.get("path")) for item in projection_signature.get("projected_fields", [])}
    custom_candidates: List[Dict[str, Any]] = []
    if projected_paths & {
        "representation.prediction.zero_init_head",
        "representation.loss.response_weight",
        "representation.optimizer.learning_rate",
    }:
        edits = [
            _toggle_to_edit(projected_ir, "representation.prediction.zero_init_head", True),
            _set_scalar_edit(
                projected_ir,
                "representation.loss.response_weight",
                min(float(_get_nested(projected_ir, "representation.loss.response_weight") or 0.35), 0.2),
            ),
            _set_scalar_edit(
                projected_ir,
                "representation.optimizer.learning_rate",
                min(float(_get_nested(projected_ir, "representation.optimizer.learning_rate") or 8e-4), 5e-4),
            ),
        ]
        edits = [edit for edit in edits if edit is not None]
        if edits:
            custom_candidates.append(
                _proposal_payload(
                    proposal_source="projection_v5_refine",
                    proposal_note="trust_projection_head_stability",
                    scientific_hypothesis="Preserve the projected conservative response head while tightening optimization.",
                    mechanistic_rationale="If breadth warmup identified stable anchored updates, the next step is to keep that bias and refine around it.",
                    edits=edits,
                    reference_ir=projected_ir,
                    candidate_seed_model_name=candidate_seed_model_name,
                )
            )
    if projected_paths & {
        "representation.conditioning.mode",
        "representation.conditioning.conditioning_dim",
    }:
        edits = [
            _set_scalar_edit(
                projected_ir,
                "representation.conditioning.conditioning_dim",
                int(_get_nested(projected_ir, "representation.conditioning.conditioning_dim") or 64) + 32,
            ),
            _set_scalar_edit(
                projected_ir,
                "representation.trunk.hidden_dim",
                int(_get_nested(projected_ir, "representation.trunk.hidden_dim") or 128) + 32,
            ),
        ]
        edits = [edit for edit in edits if edit is not None]
        if edits:
            custom_candidates.append(
                _proposal_payload(
                    proposal_source="projection_v5_refine",
                    proposal_note="trust_projection_conditioning",
                    scientific_hypothesis="Exploit the projected conditioning signal with a slightly wider conditioning bottleneck.",
                    mechanistic_rationale="When breadth warmup points to context routing, V5 refines that axis before trying unrelated edits.",
                    edits=edits,
                    reference_ir=projected_ir,
                    candidate_seed_model_name=candidate_seed_model_name,
                )
            )
    if projected_paths & {
        "representation.trunk.hidden_dim",
        "representation.trunk.trunk_depth",
        "representation.trunk.residual_depth",
    }:
        edits = [
            _set_scalar_edit(
                projected_ir,
                "representation.trunk.hidden_dim",
                int(_get_nested(projected_ir, "representation.trunk.hidden_dim") or 128) + 32,
            ),
            _set_scalar_edit(
                projected_ir,
                "representation.trunk.trunk_depth",
                int(_get_nested(projected_ir, "representation.trunk.trunk_depth") or 1) + 1,
            ),
            _set_scalar_edit(
                projected_ir,
                "representation.trunk.residual_depth",
                int(_get_nested(projected_ir, "representation.trunk.residual_depth") or 1) + 1,
            ),
        ]
        edits = [edit for edit in edits if edit is not None]
        if edits:
            custom_candidates.append(
                _proposal_payload(
                    proposal_source="projection_v5_refine",
                    proposal_note="trust_projection_capacity",
                    scientific_hypothesis="Refine around the projected capacity regime instead of re-opening the whole search space.",
                    mechanistic_rationale="This keeps V5 inside a structured trust region rather than letting warmup switch the model family again.",
                    edits=edits,
                    reference_ir=projected_ir,
                    candidate_seed_model_name=candidate_seed_model_name,
                )
            )
    generic_candidates = _local_refinement_candidates(projected_ir, dataset_key, candidate_seed_model_name)
    preferred_notes: List[str] = []
    if projected_paths & {"representation.conditioning.mode", "representation.conditioning.conditioning_dim"}:
        preferred_notes.append("refine_conditioning_dim")
    if projected_paths & {"representation.prediction.zero_init_head", "representation.loss.response_weight"}:
        preferred_notes.append("refine_response_weight")
    if projected_paths & {"representation.trunk.hidden_dim"}:
        preferred_notes.append("refine_hidden_dim")
    if projected_paths & {"representation.trunk.trunk_depth", "representation.trunk.residual_depth"}:
        preferred_notes.append("refine_depth")
    if projected_paths & {"representation.optimizer.learning_rate", "representation.optimizer.weight_decay", "representation.optimizer.batch_size"}:
        preferred_notes.extend(["refine_learning_rate", "refine_batch_size"])
    candidate_pool = custom_candidates + _reorder_candidates_by_notes(generic_candidates, preferred_notes)
    candidate_pool = _dedupe_candidate_pool(candidate_pool)
    if stage == "mechanistic_lock_in":
        return candidate_pool[:4]
    return candidate_pool


def _build_breadth_candidate_queue(
    *,
    family_specs: Sequence[Mapping[str, Any]],
    dataset_key: str,
    history: Sequence[Dict[str, Any]],
    warmup_plan: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    by_family: Dict[str, List[Dict[str, Any]]] = {}
    preferred_seed_family = str(warmup_plan.get("preferred_seed_family") or "")
    preferred_notes = [str(item) for item in warmup_plan.get("warmup_order", []) if str(item)]
    for family_spec in family_specs:
        family_name = str(family_spec["seed_model_name"])
        candidate_pool = _portfolio_candidates(dict(family_spec["seed_ir"]), dataset_key, family_name)
        if family_name == preferred_seed_family:
            candidate_pool = _reorder_candidates_by_notes(candidate_pool, preferred_notes)
        by_family[family_name] = _dedupe_candidate_pool(candidate_pool)

    best_seed_record = _best_record(_stage_completed_records(history, "seed_family_warmup"))
    empirical_family = (
        str(best_seed_record.get("candidate_seed_model_name") or best_seed_record.get("seed_family_name") or "")
        if best_seed_record is not None
        else ""
    )
    family_order: List[str] = []
    for family_name in [empirical_family, preferred_seed_family]:
        if family_name and family_name not in family_order and family_name in by_family:
            family_order.append(family_name)
    for family_spec in family_specs:
        family_name = str(family_spec["seed_model_name"])
        if family_name not in family_order:
            family_order.append(family_name)

    queue: List[Dict[str, Any]] = []
    for family_name in family_order:
        family_candidates = list(by_family.get(family_name, []))
        if family_candidates:
            queue.append(dict(family_candidates[0]))
    for family_name in family_order:
        family_candidates = list(by_family.get(family_name, []))
        for candidate in family_candidates[1:]:
            queue.append(dict(candidate))
    return _dedupe_candidate_pool(queue)


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


def run_structured_model_session_v5(
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
    agent_mode: str = "structured_projected_search_v5",
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
    breadth_stage_budget = min(len(seed_family_specs), max(0, budget_limit - family_bootstrap_budget - 1))
    projection_iteration = family_bootstrap_budget + breadth_stage_budget
    remaining_after_projection = max(0, budget_limit - projection_iteration - 1)
    trust_region_budget = min(3, remaining_after_projection)
    trust_region_end_iteration = projection_iteration + trust_region_budget

    log_lines = [
        f"dataset_key={dataset_key}",
        f"seed_model_config={seed_model_config_path}",
        f"seed_model_name={seed_model_name}",
        f"seed_families={json.dumps([item['seed_model_name'] for item in seed_family_specs], ensure_ascii=True)}",
        f"session_id={session_slug}",
        f"requested_device={requested_device}",
        f"cuda_visible_devices={cuda_visible_devices}",
        f"task_group={task_group}",
        "harness_mode=projected_breadth_warmup_then_trust_region_refinement_v5",
    ]
    log_lines = load_previous_log_lines(session_root / "agent.log", log_lines)

    def log(message: str) -> None:
        log_lines.append(message)
        print(f"[agent-structured-v5] {message}", flush=True)

    history: List[Dict[str, Any]] = load_history_rows(history_root)
    accepted_trace_rows: List[Dict[str, Any]] = load_rows_payload(trace_root / "edit_program_trace.json")
    rejection_rows: List[Dict[str, Any]] = load_rows_payload(rejection_root / "proposal_rejections.json")
    llm_usage_rows: List[Dict[str, Any]] = load_rows_payload(trace_root / "llm_usage.json")
    baseline_objective: float | None = baseline_objective_from_history(history)
    stopping_reason: str | None = None
    warmup_plan: Dict[str, Any] | None = None
    breadth_candidate_queue: List[Dict[str, Any]] | None = None
    projection_signature: Dict[str, Any] | None = None
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
            if warmup_plan is None:
                warmup_plan = _request_code_first_warmup_plan(
                    repo_root=repo_root,
                    dataset_key=dataset_key,
                    task_group=task_group,
                    family_specs=seed_family_specs,
                    history=history,
                    llm_client=llm_client,
                )
                dump_json(trace_root / "code_first_warmup_plan.json", warmup_plan)
                for usage_event in llm_client.drain_usage_events():
                    llm_usage_rows.append({"iteration": int(iteration), "attempt_index": 0, **usage_event})
                log(
                    f"warmup_plan preferred_seed_family={warmup_plan.get('preferred_seed_family')} "
                    f"warmup_order={warmup_plan.get('warmup_order', [])}"
                )
            if breadth_candidate_queue is None:
                breadth_candidate_queue = _build_breadth_candidate_queue(
                    family_specs=seed_family_specs,
                    dataset_key=dataset_key,
                    history=history,
                    warmup_plan=warmup_plan,
                )

            if iteration < projection_iteration and breadth_stage_budget > 0:
                search_stage = "breadth_warmup"
                reference_seed_record = _best_record(_stage_completed_records(history, "seed_family_warmup"))
                reference_objective_value = (
                    float(reference_seed_record["objective_value"])
                    if reference_seed_record is not None and reference_seed_record.get("objective_value") is not None
                    else None
                )
                previous_ir = (
                    _deepcopy_ir(reference_seed_record["model_ir"])
                    if reference_seed_record is not None
                    else _deepcopy_ir(seed_family_specs[0]["seed_ir"])
                )
                candidate_pool = list(breadth_candidate_queue or [])
                trace_payload = _candidate_trace_payload(
                    stage=search_stage,
                    task_group=task_group,
                    reference="cross_family_breadth_queue",
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
                    stopping_reason = "no_valid_v5_breadth_candidate"
                    log(f"stopping iteration={iteration} reason={stopping_reason}")
                    break
                proposal_note = str(proposal["proposal_note"])
            elif iteration == projection_iteration:
                search_stage = "signature_projection"
                projection_signature = _projection_signature(
                    history=history,
                    seed_family_specs=seed_family_specs,
                )
                dump_json(trace_root / "projection_signature.json", projection_signature)
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
                    fallback_candidates = _trust_region_candidates(
                        projected_ir=_deepcopy_ir(projection_signature["projected_ir"]),
                        dataset_key=dataset_key,
                        candidate_seed_model_name=anchor_family_name,
                        projection_signature=projection_signature,
                        stage="projected_trust_region",
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
                    "reference": f"anchor_seed_family::{anchor_family_name}",
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
            else:
                best_completed = _best_completed_record(history)
                if best_completed is None:
                    raise RuntimeError(f"No completed projected records found for {dataset_key}/{seed_model_name}")
                search_stage = "projected_trust_region" if iteration <= trust_region_end_iteration else "mechanistic_lock_in"
                previous_ir = dict(best_completed["model_ir"])
                reference_objective_value = float(best_completed["objective_value"])
                current_reference_seed_family_name = str(
                    best_completed.get("candidate_seed_model_name")
                    or best_completed.get("seed_family_name")
                    or seed_model_name
                )
                candidate_pool = _trust_region_candidates(
                    projected_ir=previous_ir,
                    dataset_key=dataset_key,
                    candidate_seed_model_name=current_reference_seed_family_name,
                    projection_signature=projection_signature or {},
                    stage=search_stage,
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
                    stopping_reason = "no_valid_v5_refinement_candidate"
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
                "warmup_plan": warmup_plan if search_stage == "breadth_warmup" else None,
                "projection_signature": projection_signature if search_stage == "signature_projection" else None,
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
                run_name=f"{dataset_key}_{seed_model_name}_agent_structured_v5_iter_{iteration:03d}",
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
                "preferred_warmup_seed_family_name": str(warmup_plan.get("preferred_seed_family")) if warmup_plan else None,
                "projection_anchor_seed_family_name": (
                    str(projection_signature.get("anchor_seed_family_name"))
                    if projection_signature is not None and projection_signature.get("anchor_seed_family_name") is not None
                    else None
                ),
                "warmup_plan_path": str((trace_root / "code_first_warmup_plan.json").resolve()) if warmup_plan else None,
                "projection_signature_path": str((trace_root / "projection_signature.json").resolve()) if projection_signature else None,
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
                "preferred_warmup_seed_family_name": str(warmup_plan.get("preferred_seed_family")) if warmup_plan else None,
                "projection_anchor_seed_family_name": (
                    str(projection_signature.get("anchor_seed_family_name"))
                    if projection_signature is not None and projection_signature.get("anchor_seed_family_name") is not None
                    else None
                ),
                "warmup_plan_path": str((trace_root / "code_first_warmup_plan.json").resolve()) if warmup_plan else None,
                "projection_signature_path": str((trace_root / "projection_signature.json").resolve()) if projection_signature else None,
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
            "method_name": f"agent_structured_v5::{seed_model_name}",
            "agent_line": "main_agent_v5",
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
            "harness_mode": "projected_breadth_warmup_then_trust_region_refinement_v5",
            "task_group": task_group,
            "seed_family_candidates_json": json.dumps([item["seed_model_name"] for item in seed_family_specs], ensure_ascii=True),
            "warmup_plan": warmup_plan,
            "projection_signature": projection_signature,
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
        "method_name": f"agent_structured_v5::{seed_model_name}",
        "method_family": "agent_structured_model",
        "agent_line": "main_agent_v5",
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
