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
    _portfolio_candidates,
    _proposal_payload,
    _seed_family_baseline_proposal,
    _task_group,
)
from sci_response.agent.structured_v5 import (
    _best_completed_record,
    _best_record,
    _build_breadth_candidate_queue,
    _candidate_trace_payload,
    _deepcopy_ir,
    _dedupe_candidate_pool,
    _pick_first_valid_candidate,
    _projection_proposal,
    _reorder_candidates_by_notes,
)
from sci_response.agent.structured_v6 import (
    _breadth_macro_candidates,
    _constraint_candidate_pool,
    _diversify_breadth_candidate_pool,
    _normalized_projection_signature_v6,
    _projection_fingerprint,
)
from sci_response.baselines.artifacts import beijing_timestamp_slug
from sci_response.data.io import dump_json, dump_yaml, ensure_dir, load_json, load_yaml, stable_hash, write_text


ANCHOR_DESCRIPTIONS: dict[str, str] = {
    "conditioned_residual": "Residual perturbation model that has been the strongest stable seed in prior agent runs.",
    "structured_hypothesis_hpo": "Structured hypothesis seed tuned for optimizer stability and HPO-style robustness.",
    "gears_anchor": "Single-cell CRISPR specialist-inspired anchor emphasizing stronger perturbation-aware capacity.",
    "cpa_anchor": "Conditional perturbation autoencoder-inspired anchor emphasizing context-aware routing.",
    "cellot_anchor": "Transport-inspired anchor emphasizing conservative anchored response mapping.",
    "xpert_anchor": "Paired clinical response anchor emphasizing anchored delta prediction and conservative optimization.",
    "random_edit_anchor": "High-entropy structured anchor inspired by the strongest random-edit search regimes.",
}

TASK_GROUP_PRIORS: dict[str, dict[str, float]] = {
    "single_cell": {
        "gears_anchor": 5.3,
        "conditioned_residual": 4.8,
        "structured_hypothesis_hpo": 4.4,
        "random_edit_anchor": 4.0,
        "cpa_anchor": 3.6,
        "cellot_anchor": 3.4,
        "xpert_anchor": 2.8,
    },
    "multimodal_rna": {
        "cellot_anchor": 5.3,
        "cpa_anchor": 4.5,
        "conditioned_residual": 4.0,
        "structured_hypothesis_hpo": 3.9,
        "random_edit_anchor": 3.7,
        "gears_anchor": 3.3,
        "xpert_anchor": 2.5,
    },
    "multimodal_protein": {
        "conditioned_residual": 5.1,
        "random_edit_anchor": 4.5,
        "structured_hypothesis_hpo": 4.2,
        "cpa_anchor": 3.8,
        "cellot_anchor": 3.5,
        "gears_anchor": 3.3,
        "xpert_anchor": 2.7,
    },
    "dose_time": {
        "cpa_anchor": 5.2,
        "random_edit_anchor": 4.8,
        "structured_hypothesis_hpo": 4.5,
        "conditioned_residual": 4.0,
        "cellot_anchor": 3.5,
        "xpert_anchor": 2.8,
        "gears_anchor": 2.7,
    },
    "paired_clinical": {
        "xpert_anchor": 5.4,
        "structured_hypothesis_hpo": 4.8,
        "conditioned_residual": 4.0,
        "random_edit_anchor": 3.7,
        "cpa_anchor": 3.0,
        "cellot_anchor": 2.9,
        "gears_anchor": 2.4,
    },
}

TASK_GROUP_BREADTH_MODES: dict[str, list[str]] = {
    "single_cell": ["anchor_restart", "context_jump", "macro_jump", "stable_head", "optimizer_reset"],
    "multimodal_rna": ["anchor_restart", "context_jump", "stable_head", "macro_jump", "optimizer_reset"],
    "multimodal_protein": ["anchor_restart", "stable_head", "macro_jump", "context_jump", "optimizer_reset"],
    "dose_time": ["anchor_restart", "context_jump", "macro_jump", "optimizer_reset", "stable_head"],
    "paired_clinical": ["anchor_restart", "paired_anchor", "optimizer_reset", "stable_head", "macro_jump"],
}

MODE_TO_PROPOSAL_NOTES: dict[str, list[str]] = {
    "anchor_restart": [],
    "macro_jump": ["breadth_capacity_jump"],
    "context_jump": ["breadth_context_jump", "breadth_dose_context"],
    "optimizer_reset": ["breadth_optimizer_reset"],
    "stable_head": ["breadth_stable_head"],
    "paired_anchor": ["breadth_paired_anchor"],
}


def _family_label(config_path: Path) -> str:
    return str(config_path.stem)


def _family_prior(task_group: str, family_name: str) -> float:
    return float(TASK_GROUP_PRIORS.get(task_group, {}).get(family_name, 0.0))


def _family_summary_payload_v7(family_spec: Mapping[str, Any], task_group: str) -> Dict[str, Any]:
    seed_ir = dict(family_spec["seed_ir"])
    return {
        "seed_family_name": str(family_spec["seed_model_name"]),
        "base_model_name": str(family_spec["base_model_name"]),
        "anchor_description": ANCHOR_DESCRIPTIONS.get(str(family_spec["seed_model_name"]), ""),
        "task_group_prior": _family_prior(task_group, str(family_spec["seed_model_name"])),
        "hidden_dim": _get_nested(seed_ir, "representation.trunk.hidden_dim"),
        "trunk_depth": _get_nested(seed_ir, "representation.trunk.trunk_depth"),
        "residual_depth": _get_nested(seed_ir, "representation.trunk.residual_depth"),
        "conditioning_mode": _get_nested(seed_ir, "representation.conditioning.mode"),
        "conditioning_dim": _get_nested(seed_ir, "representation.conditioning.conditioning_dim"),
        "zero_init_head": _get_nested(seed_ir, "representation.prediction.zero_init_head"),
        "baseline_skip": _get_nested(seed_ir, "representation.prediction.baseline_skip"),
        "learning_rate": _get_nested(seed_ir, "representation.optimizer.learning_rate"),
        "weight_decay": _get_nested(seed_ir, "representation.optimizer.weight_decay"),
    }


def _completed_records(history: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        dict(item)
        for item in history
        if str(item.get("execution_status")) == "completed" and item.get("objective_value") is not None
    ]


def _best_objective_by_family(history: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    best_by_family: Dict[str, float] = {}
    for item in _completed_records(history):
        family_name = str(item.get("candidate_seed_model_name") or item.get("seed_family_name") or "").strip()
        if not family_name:
            continue
        objective = float(item["objective_value"])
        current = best_by_family.get(family_name)
        if current is None or objective < current:
            best_by_family[family_name] = objective
    return best_by_family


def _fallback_anchor_plan(
    *,
    dataset_key: str,
    task_group: str,
    family_specs: Sequence[Mapping[str, Any]],
    history: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    empirical = _best_objective_by_family(history)
    ordered = sorted(
        [str(spec["seed_model_name"]) for spec in family_specs],
        key=lambda family_name: (
            0 if family_name in empirical else 1,
            empirical.get(family_name, 1.0e9),
            -_family_prior(task_group, family_name),
            family_name,
        ),
    )
    return {
        "proposal_note": "anchor_plan_v7::fallback",
        "preferred_seed_family": ordered[0],
        "anchor_order": ordered,
        "breadth_modes": list(TASK_GROUP_BREADTH_MODES.get(task_group, ["anchor_restart", "macro_jump", "optimizer_reset"])),
        "scientific_hypothesis": f"Use strong regime-aligned anchor families for {dataset_key}, then alternate breadth and constraint around the strongest basin.",
        "mechanistic_rationale": "V7 ranks a small anchor library by task-regime prior and any observed empirical gain, then uses those anchors as restart points instead of forcing all search to begin from one weak seed.",
        "risk_notes": ["fallback_without_anchor_llm"],
    }


def _build_anchor_plan_prompts(
    *,
    dataset_key: str,
    task_group: str,
    family_specs: Sequence[Mapping[str, Any]],
    history: Sequence[Dict[str, Any]],
) -> Dict[str, str]:
    family_payloads = [_family_summary_payload_v7(spec, task_group) for spec in family_specs]
    empirical = []
    for family_name, best_value in sorted(_best_objective_by_family(history).items()):
        empirical.append({"seed_family_name": family_name, "best_objective": best_value})
    system_prompt = (
        "You are planning V7 of a biological perturbation modeling agent. "
        "Your job is to choose the best anchor-family order for a strong start and define the breadth-first restart modes. "
        "Return exactly one JSON object and do not output prose outside JSON."
    )
    user_prompt = (
        f"DATASET KEY: {dataset_key}\n"
        f"TASK GROUP: {task_group}\n\n"
        "AVAILABLE ANCHOR FAMILIES:\n"
        f"{json.dumps(family_payloads, indent=2, ensure_ascii=True)}\n\n"
        "EMPIRICAL HISTORY SO FAR:\n"
        f"{json.dumps(empirical, indent=2, ensure_ascii=True)}\n\n"
        "You must output exactly one JSON object with this schema:\n"
        "{\n"
        '  "proposal_note": "anchor_plan_v7",\n'
        '  "preferred_seed_family": "<one seed_family_name>",\n'
        '  "anchor_order": ["<seed_family_name>", "..."],\n'
        '  "breadth_modes": ["anchor_restart", "macro_jump", "context_jump", "optimizer_reset", "stable_head", "paired_anchor"],\n'
        '  "scientific_hypothesis": "<one sentence>",\n'
        '  "mechanistic_rationale": "<one or two sentences>",\n'
        '  "risk_notes": ["<short note>", "..."]\n'
        "}\n\n"
        "Rules:\n"
        "- preferred_seed_family must be one listed seed_family_name.\n"
        "- anchor_order must be a unique ordering over listed seed_family_name values.\n"
        "- breadth_modes must be chosen from the allowed set in the schema.\n"
        "- Favor anchors that fit the regime and preserve headroom for later constrained refinement.\n"
    )
    return {"system_prompt": system_prompt, "user_prompt": user_prompt}


def _request_anchor_plan(
    *,
    dataset_key: str,
    task_group: str,
    family_specs: Sequence[Mapping[str, Any]],
    history: Sequence[Dict[str, Any]],
    llm_client: StructuredLLMClient,
) -> Dict[str, Any]:
    prompts = _build_anchor_plan_prompts(
        dataset_key=dataset_key,
        task_group=task_group,
        family_specs=family_specs,
        history=history,
    )
    family_names = [str(spec["seed_model_name"]) for spec in family_specs]
    allowed_modes = {"anchor_restart", "macro_jump", "context_jump", "optimizer_reset", "stable_head", "paired_anchor"}
    try:
        payload = llm_client.chat_json(
            system_prompt=prompts["system_prompt"],
            user_prompt=prompts["user_prompt"],
            max_tokens_override=min(int(llm_client.settings.max_tokens), 1800),
        )
    except Exception:
        return _fallback_anchor_plan(dataset_key=dataset_key, task_group=task_group, family_specs=family_specs, history=history)

    preferred_seed_family = str(payload.get("preferred_seed_family", "")).strip()
    raw_order = payload.get("anchor_order", [])
    if preferred_seed_family not in family_names or not isinstance(raw_order, list):
        return _fallback_anchor_plan(dataset_key=dataset_key, task_group=task_group, family_specs=family_specs, history=history)

    ordered: List[str] = []
    seen: set[str] = set()
    for item in raw_order:
        family_name = str(item).strip()
        if family_name in family_names and family_name not in seen:
            ordered.append(family_name)
            seen.add(family_name)
    for family_name in family_names:
        if family_name not in seen:
            ordered.append(family_name)
            seen.add(family_name)
    if preferred_seed_family in ordered:
        ordered.remove(preferred_seed_family)
    ordered.insert(0, preferred_seed_family)

    breadth_modes: List[str] = []
    for item in payload.get("breadth_modes", []) if isinstance(payload.get("breadth_modes", []), list) else []:
        mode = str(item).strip()
        if mode and mode in allowed_modes and mode not in breadth_modes:
            breadth_modes.append(mode)
    if not breadth_modes:
        breadth_modes = list(TASK_GROUP_BREADTH_MODES.get(task_group, ["anchor_restart", "macro_jump", "optimizer_reset"]))

    return {
        "proposal_note": str(payload.get("proposal_note", "anchor_plan_v7")),
        "preferred_seed_family": preferred_seed_family,
        "anchor_order": ordered,
        "breadth_modes": breadth_modes,
        "scientific_hypothesis": str(payload.get("scientific_hypothesis", "")),
        "mechanistic_rationale": str(payload.get("mechanistic_rationale", "")),
        "risk_notes": list(payload.get("risk_notes", [])) if isinstance(payload.get("risk_notes", []), list) else [],
    }


def _anchor_tournament_budget(task_group: str, family_count: int) -> int:
    if task_group in {"dose_time", "paired_clinical"}:
        return min(int(family_count), 4)
    return min(int(family_count), 3)


def _ranked_untried_anchors(
    *,
    history: Sequence[Dict[str, Any]],
    anchor_plan: Mapping[str, Any],
) -> List[str]:
    tried = {
        str(item.get("candidate_seed_model_name") or item.get("seed_family_name") or "")
        for item in _completed_records(history)
        if item.get("candidate_seed_model_name") or item.get("seed_family_name")
    }
    return [family_name for family_name in anchor_plan.get("anchor_order", []) if str(family_name) not in tried]


def _recent_no_gain_streak(history: Sequence[Dict[str, Any]]) -> int:
    streak = 0
    for item in reversed(_completed_records(history)):
        gain = item.get("delta_vs_reference")
        if not isinstance(gain, (int, float)) or float(gain) <= 0.0:
            streak += 1
            continue
        break
    return int(streak)


def _v7_snapshot(
    *,
    history: Sequence[Dict[str, Any]],
    anchor_plan: Mapping[str, Any],
    projection_signature: Mapping[str, Any] | None,
    task_group: str,
) -> Dict[str, Any]:
    completed = _completed_records(history)
    best_record = _best_completed_record(history)
    best_iteration = int(best_record["iteration"]) if best_record is not None else -1
    current_iteration = int(history[-1]["iteration"]) if history else -1
    iterations_since_best = max(0, current_iteration - best_iteration) if best_iteration >= 0 and current_iteration >= 0 else 0
    action_history = [str(item.get("search_action")) for item in history if str(item.get("search_action")) in {"breadth", "constrain"}]
    last_action = action_history[-1] if action_history else None
    last_action_streak = 0
    for action in reversed(action_history):
        if action != last_action:
            break
        last_action_streak += 1
    family_history = [
        str(item.get("candidate_seed_model_name") or item.get("seed_family_name") or "")
        for item in completed
        if item.get("candidate_seed_model_name") or item.get("seed_family_name")
    ]
    best_family_name = (
        str(best_record.get("candidate_seed_model_name") or best_record.get("seed_family_name") or "")
        if best_record is not None
        else ""
    )
    same_family_streak = 0
    for family_name in reversed(family_history):
        if family_name != best_family_name:
            break
        same_family_streak += 1
    recent = completed[-3:]
    breadth_completed = [item for item in completed if str(item.get("search_action")) == "breadth"]
    constrain_completed = [item for item in completed if str(item.get("search_action")) == "constrain"]
    recent_gain_mean = float(sum(float(item.get("delta_vs_reference") or 0.0) for item in recent) / len(recent)) if recent else 0.0
    breadth_gain_mean = (
        float(sum(float(item.get("delta_vs_reference") or 0.0) for item in breadth_completed[-2:]) / len(breadth_completed[-2:]))
        if breadth_completed[-2:]
        else 0.0
    )
    constrain_gain_mean = (
        float(sum(float(item.get("delta_vs_reference") or 0.0) for item in constrain_completed[-2:]) / len(constrain_completed[-2:]))
        if constrain_completed[-2:]
        else 0.0
    )
    return {
        "task_group": task_group,
        "completed_count": int(len(completed)),
        "best_iteration": int(best_iteration),
        "best_family_name": best_family_name,
        "iterations_since_best": int(iterations_since_best),
        "last_action": last_action,
        "last_action_streak": int(last_action_streak),
        "same_family_streak": int(same_family_streak),
        "recent_gain_mean": float(recent_gain_mean),
        "breadth_recent_gain_mean": float(breadth_gain_mean),
        "constrain_recent_gain_mean": float(constrain_gain_mean),
        "last_gain": float(history[-1].get("delta_vs_reference") or 0.0) if history else 0.0,
        "projection_available": bool(projection_signature and projection_signature.get("projected_fields")),
        "projection_confidence": float(projection_signature.get("projection_confidence") or 0.0) if projection_signature else 0.0,
        "untried_anchor_names": _ranked_untried_anchors(history=history, anchor_plan=anchor_plan),
        "anchor_trials_completed": int(len({name for name in family_history if name in set(anchor_plan.get("anchor_order", []))})),
        "recent_no_gain_streak": int(_recent_no_gain_streak(history)),
        "breadth_improve_count": int(sum(1 for item in breadth_completed if float(item.get("delta_vs_reference") or 0.0) > 0.0)),
        "constrain_improve_count": int(sum(1 for item in constrain_completed if float(item.get("delta_vs_reference") or 0.0) > 0.0)),
        "action_path": "".join("B" if action == "breadth" else "C" for action in action_history),
    }


def _should_refresh_anchor_plan(snapshot: Mapping[str, Any], anchor_plan_refresh_count: int) -> bool:
    if anchor_plan_refresh_count == 0:
        return False
    if int(snapshot["iterations_since_best"]) >= 3:
        return True
    if int(snapshot["recent_no_gain_streak"]) >= 2:
        return True
    return False


def _choose_action_v7(
    *,
    snapshot: Mapping[str, Any],
    pending_projection: bool,
    tournament_budget: int,
) -> Dict[str, Any]:
    reasons: List[str] = []
    if int(snapshot["anchor_trials_completed"]) < int(tournament_budget) and snapshot["untried_anchor_names"]:
        reasons.append("anchor_tournament_not_complete")
        return {
            "action": "breadth",
            "stage": "anchor_tournament",
            "decision_reason": "|".join(reasons),
            "lock_in": False,
            "refresh_anchor_plan": False,
        }
    if pending_projection:
        return {
            "action": "constrain",
            "stage": "projection_bridge",
            "decision_reason": "pending_projection_bridge",
            "lock_in": False,
            "refresh_anchor_plan": False,
        }
    if int(snapshot["recent_no_gain_streak"]) >= 2:
        return {
            "action": "breadth",
            "stage": "anchor_restart" if snapshot["untried_anchor_names"] else "adaptive_breadth",
            "decision_reason": "recent_no_gain_streak",
            "lock_in": False,
            "refresh_anchor_plan": True,
        }
    if int(snapshot["same_family_streak"]) >= 3 and float(snapshot["breadth_recent_gain_mean"]) <= 0.0:
        return {
            "action": "breadth",
            "stage": "anchor_restart" if snapshot["untried_anchor_names"] else "adaptive_breadth",
            "decision_reason": "family_lock_in",
            "lock_in": False,
            "refresh_anchor_plan": False,
        }
    if bool(snapshot["projection_available"]) and (
        float(snapshot["last_gain"]) > 0.0 or float(snapshot["constrain_recent_gain_mean"]) > 0.002
    ):
        lock_in = bool(
            int(snapshot["iterations_since_best"]) <= 1 and int(snapshot["constrain_improve_count"]) >= 1
        )
        return {
            "action": "constrain",
            "stage": "adaptive_lock_in" if lock_in else "adaptive_constrain",
            "decision_reason": "projected_refinement",
            "lock_in": lock_in,
            "refresh_anchor_plan": False,
        }
    if float(snapshot["breadth_recent_gain_mean"]) > float(snapshot["constrain_recent_gain_mean"]) + 0.002:
        return {
            "action": "breadth",
            "stage": "adaptive_breadth",
            "decision_reason": "breadth_has_more_headroom",
            "lock_in": False,
            "refresh_anchor_plan": False,
        }
    if float(snapshot["recent_gain_mean"]) <= 0.0:
        return {
            "action": "breadth",
            "stage": "adaptive_breadth",
            "decision_reason": "recent_gain_flat",
            "lock_in": False,
            "refresh_anchor_plan": False,
        }
    return {
        "action": "constrain",
        "stage": "adaptive_constrain",
        "decision_reason": "steady_mechanistic_refinement",
        "lock_in": False,
        "refresh_anchor_plan": False,
    }


def _anchor_seed_candidates(
    *,
    seed_family_specs: Sequence[Mapping[str, Any]],
    family_names: Sequence[str],
    reference_ir: Dict[str, Any],
    exclude_family_name: str | None = None,
    limit: int = 2,
) -> List[Dict[str, Any]]:
    by_name = {str(spec["seed_model_name"]): dict(spec) for spec in seed_family_specs}
    candidates: List[Dict[str, Any]] = []
    for family_name in family_names:
        family_name = str(family_name)
        if not family_name or family_name == str(exclude_family_name or ""):
            continue
        family_spec = by_name.get(family_name)
        if family_spec is None:
            continue
        candidates.append(
            _seed_family_baseline_proposal(
                reference_ir=reference_ir,
                seed_ir=dict(family_spec["seed_ir"]),
                seed_model_name=family_name,
            )
        )
        if len(candidates) >= int(limit):
            break
    return candidates


def _breadth_preference_notes(task_group: str, breadth_modes: Sequence[str]) -> List[str]:
    notes: List[str] = []
    for mode in breadth_modes:
        notes.extend(MODE_TO_PROPOSAL_NOTES.get(str(mode), []))
    if task_group == "paired_clinical" and "breadth_paired_anchor" not in notes:
        notes.insert(0, "breadth_paired_anchor")
    return notes


def run_structured_model_session_v7(
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
    agent_mode: str = "structured_anchor_tree_search_v7",
    additional_seed_model_config_paths: Sequence[Path] = (),
) -> Dict[str, Any]:
    seed_model_config = load_yaml(seed_model_config_path)
    seed_model_name = _family_label(seed_model_config_path)
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
    seen_family_names: set[str] = set()
    for config_path in [Path(seed_model_config_path).resolve(), *[Path(path).resolve() for path in additional_seed_model_config_paths]]:
        family_name = _family_label(config_path)
        if family_name in seen_family_names:
            continue
        config_payload = load_yaml(config_path)
        seed_ir = extract_model_ir(config_payload)
        seed_ir["seed_model_name"] = family_name
        seed_family_specs.append(
            {
                "config_path": config_path,
                "config": config_payload,
                "base_model_name": str(config_payload["name"]),
                "seed_model_name": family_name,
                "seed_ir": seed_ir,
            }
        )
        seen_family_names.add(family_name)

    log_lines = [
        f"dataset_key={dataset_key}",
        f"seed_model_config={seed_model_config_path}",
        f"seed_model_name={seed_model_name}",
        f"seed_families={json.dumps([item['seed_model_name'] for item in seed_family_specs], ensure_ascii=True)}",
        f"session_id={session_slug}",
        f"requested_device={requested_device}",
        f"cuda_visible_devices={cuda_visible_devices}",
        f"task_group={task_group}",
        "harness_mode=baseline_anchored_adaptive_tree_search_v7",
    ]
    log_lines = load_previous_log_lines(session_root / "agent.log", log_lines)

    def log(message: str) -> None:
        log_lines.append(message)
        print(f"[agent-structured-v7] {message}", flush=True)

    history = load_history_rows(history_root)
    accepted_trace_rows = load_rows_payload(trace_root / "edit_program_trace.json")
    rejection_rows = load_rows_payload(rejection_root / "proposal_rejections.json")
    llm_usage_rows = load_rows_payload(trace_root / "llm_usage.json")
    router_rows = load_rows_payload(trace_root / "action_router.json")
    baseline_objective = baseline_objective_from_history(history)
    stopping_reason: str | None = None
    projection_signature: Dict[str, Any] | None = None
    last_applied_projection_fingerprint: str | None = None
    anchor_plan_path = trace_root / "anchor_plan.json"
    action_memory_path = trace_root / "action_memory.json"
    action_memory_payload = load_json(action_memory_path) if action_memory_path.exists() else {}
    if isinstance(action_memory_payload.get("final_projection_signature"), dict):
        projection_signature = dict(action_memory_payload["final_projection_signature"])
    if action_memory_payload.get("last_applied_projection_fingerprint"):
        last_applied_projection_fingerprint = str(action_memory_payload["last_applied_projection_fingerprint"])
    if history:
        if anchor_plan_path.exists():
            anchor_plan = load_json(anchor_plan_path)
        else:
            anchor_plan = _request_anchor_plan(
                dataset_key=dataset_key,
                task_group=task_group,
                family_specs=seed_family_specs,
                history=history,
                llm_client=llm_client,
            )
            dump_json(anchor_plan_path, anchor_plan)
            for usage_event in llm_client.drain_usage_events():
                llm_usage_rows.append({"iteration": int(next_iteration_index(history)), "attempt_index": 0, **usage_event})
        anchor_plan_refresh_count = int(action_memory_payload.get("anchor_plan_refresh_count") or 1)
    else:
        anchor_plan = _request_anchor_plan(
            dataset_key=dataset_key,
            task_group=task_group,
            family_specs=seed_family_specs,
            history=history,
            llm_client=llm_client,
        )
        anchor_plan_refresh_count = 1
        dump_json(anchor_plan_path, anchor_plan)
        for usage_event in llm_client.drain_usage_events():
            llm_usage_rows.append({"iteration": 0, "attempt_index": 0, **usage_event})
    tournament_budget = _anchor_tournament_budget(task_group, len(seed_family_specs))
    iteration = next_iteration_index(history)
    if history:
        log(
            f"resume session_id={session_slug} next_iteration={iteration} "
            f"completed_iterations={len(history)}"
        )

    while should_continue_search(history, budget_limit):
        if iteration == 0:
            first_anchor_name = str(anchor_plan["anchor_order"][0])
            family_spec = next(dict(item) for item in seed_family_specs if str(item["seed_model_name"]) == first_anchor_name)
            previous_ir = _deepcopy_ir(family_spec["seed_ir"])
            reference_objective_value = None
            proposal = _seed_family_baseline_proposal(
                reference_ir=previous_ir,
                seed_ir=_deepcopy_ir(family_spec["seed_ir"]),
                seed_model_name=first_anchor_name,
            )
            proposal_note = str(proposal["proposal_note"])
            search_stage = "baseline"
            search_action = "baseline"
            attempt_index_value = 0
            trace_payload = {
                "stage": search_stage,
                "task_group": task_group,
                "reference": "baseline_anchor",
                "anchor_plan": anchor_plan,
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
            action_decision_reason = "iteration_0_anchor_seed"
            action_scores_payload = {}
            action_path = ""
        else:
            projection_signature = _normalized_projection_signature_v6(history=history, seed_family_specs=seed_family_specs)
            projection_fingerprint = _projection_fingerprint(projection_signature)
            pending_projection = bool(projection_fingerprint and projection_fingerprint != last_applied_projection_fingerprint)
            snapshot = _v7_snapshot(
                history=history,
                anchor_plan=anchor_plan,
                projection_signature=projection_signature,
                task_group=task_group,
            )
            if _should_refresh_anchor_plan(snapshot, anchor_plan_refresh_count):
                anchor_plan = _request_anchor_plan(
                    dataset_key=dataset_key,
                    task_group=task_group,
                    family_specs=seed_family_specs,
                    history=history,
                    llm_client=llm_client,
                )
                anchor_plan_refresh_count += 1
                dump_json(trace_root / "anchor_plan.json", anchor_plan)
                dump_json(trace_root / f"anchor_plan_iter_{iteration:03d}.json", anchor_plan)
                for usage_event in llm_client.drain_usage_events():
                    llm_usage_rows.append({"iteration": int(iteration), "attempt_index": 0, **usage_event})
                log(
                    f"anchor_plan_refresh preferred_seed_family={anchor_plan.get('preferred_seed_family')} "
                    f"anchor_order={anchor_plan.get('anchor_order', [])}"
                )

            decision = _choose_action_v7(
                snapshot=snapshot,
                pending_projection=pending_projection,
                tournament_budget=tournament_budget,
            )
            search_action = str(decision["action"])
            search_stage = str(decision["stage"])
            action_decision_reason = str(decision["decision_reason"])
            action_scores_payload = {
                "tournament_budget": float(tournament_budget),
                "projection_confidence": float(snapshot["projection_confidence"]),
                "recent_gain_mean": float(snapshot["recent_gain_mean"]),
                "breadth_recent_gain_mean": float(snapshot["breadth_recent_gain_mean"]),
                "constrain_recent_gain_mean": float(snapshot["constrain_recent_gain_mean"]),
                "iterations_since_best": float(snapshot["iterations_since_best"]),
            }
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

            best_completed = _best_completed_record(history)
            current_reference_ir = (
                _deepcopy_ir(best_completed["model_ir"])
                if best_completed is not None
                else _deepcopy_ir(seed_family_specs[0]["seed_ir"])
            )
            current_reference_family_name = (
                str(best_completed.get("candidate_seed_model_name") or best_completed.get("seed_family_name") or seed_model_name)
                if best_completed is not None
                else str(anchor_plan["preferred_seed_family"])
            )
            reference_objective_value = (
                float(best_completed["objective_value"])
                if best_completed is not None and best_completed.get("objective_value") is not None
                else None
            )

            if search_action == "breadth" and search_stage == "anchor_tournament":
                candidate_pool = _anchor_seed_candidates(
                    seed_family_specs=seed_family_specs,
                    family_names=snapshot["untried_anchor_names"],
                    reference_ir=current_reference_ir,
                    exclude_family_name=current_reference_family_name,
                    limit=1,
                )
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
                    search_stage = "adaptive_breadth"
                    search_action = "breadth"
                    action_decision_reason += "::fallback_after_duplicate_anchor"
                else:
                    proposal_note = str(proposal["proposal_note"])

            if search_action == "breadth" and search_stage != "anchor_tournament":
                breadth_pref_notes = _breadth_preference_notes(task_group, anchor_plan.get("breadth_modes", []))
                anchor_seed_pool = _anchor_seed_candidates(
                    seed_family_specs=seed_family_specs,
                    family_names=snapshot["untried_anchor_names"] or anchor_plan.get("anchor_order", []),
                    reference_ir=current_reference_ir,
                    exclude_family_name=current_reference_family_name,
                    limit=2 if search_stage == "anchor_restart" else 1,
                )
                queue = _build_breadth_candidate_queue(
                    family_specs=seed_family_specs,
                    dataset_key=dataset_key,
                    history=history,
                    warmup_plan={
                        "preferred_seed_family": str(anchor_plan["preferred_seed_family"]),
                        "warmup_order": [],
                    },
                )
                macro_pool = _breadth_macro_candidates(
                    reference_ir=current_reference_ir,
                    dataset_key=dataset_key,
                    current_family_name=current_reference_family_name,
                    seed_family_specs=seed_family_specs,
                )
                macro_pool = _reorder_candidates_by_notes(macro_pool, breadth_pref_notes)
                candidate_pool = _diversify_breadth_candidate_pool(
                    _dedupe_candidate_pool(anchor_seed_pool + queue + macro_pool),
                    per_signature_limit=2,
                )
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
                    stopping_reason = "no_valid_v7_breadth_candidate"
                    log(f"stopping iteration={iteration} reason={stopping_reason}")
                    break
                proposal_note = str(proposal["proposal_note"])
            elif search_action == "constrain":
                if pending_projection and projection_signature is not None and projection_fingerprint is not None:
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
                    previous_ir = current_reference_ir
                    candidate_pool = _constraint_candidate_pool(
                        projected_ir=current_reference_ir,
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
                        rescue_candidates = _diversify_breadth_candidate_pool(
                            _dedupe_candidate_pool(
                                _anchor_seed_candidates(
                                    seed_family_specs=seed_family_specs,
                                    family_names=snapshot["untried_anchor_names"] or anchor_plan.get("anchor_order", []),
                                    reference_ir=current_reference_ir,
                                    exclude_family_name=current_reference_family_name,
                                    limit=2,
                                )
                                + _breadth_macro_candidates(
                                    reference_ir=current_reference_ir,
                                    dataset_key=dataset_key,
                                    current_family_name=current_reference_family_name,
                                    seed_family_specs=seed_family_specs,
                                )
                            ),
                            per_signature_limit=2,
                        )
                        proposal, attempt_index_value, rescue_rejections = _pick_first_valid_candidate(
                            candidate_pool=rescue_candidates,
                            previous_ir=previous_ir,
                            history=history,
                            generated_code_root=generated_code_root,
                            rejection_root=rejection_root,
                            iteration=int(iteration),
                        )
                        rejection_rows.extend(rescue_rejections)
                        if proposal is None:
                            stopping_reason = "no_valid_v7_constraint_or_rescue_candidate"
                            log(f"stopping iteration={iteration} reason={stopping_reason}")
                            break
                        search_action = "breadth"
                        search_stage = "rescue_breadth_after_constraint_exhaustion"
                        action_decision_reason += "::constraint_rescue_breadth"
                        proposal_note = str(proposal["proposal_note"])
                        trace_payload = _candidate_trace_payload(
                            stage=search_stage,
                            task_group=task_group,
                            reference=f"rescue::{current_reference_family_name}",
                            candidates=rescue_candidates,
                        )
                    else:
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
                "anchor_plan": anchor_plan,
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
                run_name=f"{dataset_key}_{seed_model_name}_agent_structured_v7_iter_{iteration:03d}",
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
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
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
                "anchor_plan_path": str((trace_root / "anchor_plan.json").resolve()),
                "projection_anchor_seed_family_name": (
                    str(projection_signature.get("anchor_seed_family_name"))
                    if projection_signature is not None and projection_signature.get("anchor_seed_family_name") is not None
                    else None
                ),
                "projection_fingerprint": last_applied_projection_fingerprint,
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
                "anchor_plan_path": str((trace_root / "anchor_plan.json").resolve()),
                "projection_anchor_seed_family_name": (
                    str(projection_signature.get("anchor_seed_family_name"))
                    if projection_signature is not None and projection_signature.get("anchor_seed_family_name") is not None
                    else None
                ),
                "projection_fingerprint": last_applied_projection_fingerprint,
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

    completed = _completed_records(history)
    if not completed:
        raise RuntimeError(f"No completed structured iterations for {dataset_key}/{seed_model_name}")
    best_record = min(completed, key=lambda item: float(item["objective_value"]))
    if stopping_reason is None:
        stopping_reason = default_stopping_reason(history, budget_limit)
    completed_count = completed_evaluation_count(history)
    failed_candidate_count = int(failed_history_count(history) + len(rejection_rows))
    candidate_attempt_count = int(completed_count + failed_candidate_count)
    session_wall_clock_seconds = float(prior_session_wall_clock_seconds + (perf_counter() - session_started_at))
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
            "anchor_plan_refresh_count": int(anchor_plan_refresh_count),
            "final_anchor_plan": anchor_plan,
            "final_projection_signature": projection_signature,
            "last_applied_projection_fingerprint": last_applied_projection_fingerprint,
            "action_path": "".join(
                "B" if str(item.get("search_action")) == "breadth" else "C"
                for item in history
                if str(item.get("search_action")) in {"breadth", "constrain"}
            ),
        },
    )
    if projection_signature is not None:
        dump_json(trace_root / "projection_signature.json", projection_signature)

    dump_json(
        session_root / "agent_session.json",
        {
            "dataset_key": dataset_key,
            "method_family": "agent_structured_model",
            "method_name": f"agent_structured_v7::{seed_model_name}",
            "agent_line": "main_agent_v7",
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
            "harness_mode": "baseline_anchored_adaptive_tree_search_v7",
            "task_group": task_group,
            "seed_family_candidates_json": json.dumps([item["seed_model_name"] for item in seed_family_specs], ensure_ascii=True),
            "anchor_plan": anchor_plan,
            "anchor_plan_refresh_count": int(anchor_plan_refresh_count),
            "anchor_tournament_budget": int(tournament_budget),
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
        "method_name": f"agent_structured_v7::{seed_model_name}",
        "method_family": "agent_structured_model",
        "agent_line": "main_agent_v7",
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
        "llm_total_tokens": int(llm_usage_summary["llm_total_tokens"]),
        "llm_repair_request_count": int(llm_usage_summary["llm_repair_request_count"]),
    }
