from __future__ import annotations

import json
from string import Template
from typing import Any, Dict, List

from sci_response.agent.config import (
    default_harness_config_path,
    default_mechanism_template_path,
    default_prompt_config_path,
    default_skill_card_path,
)
from sci_response.agent.harness import prioritize_mechanism_templates
from sci_response.agent.ir import enumerate_editable_sites
from sci_response.data.io import load_yaml


def _recent_history(history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in history[-5:]:
        rows.append(
            {
                "iteration": item.get("iteration"),
                "objective_value": item.get("objective_value"),
                "proposal_note": item.get("proposal_note"),
                "accepted_as_best": item.get("accepted_as_best"),
                "hypothesis_axes": item.get("hypothesis_axes"),
            }
        )
    return rows


def _recent_rejections(rows: List[Dict[str, Any]], *, window: int) -> List[Dict[str, Any]]:
    trimmed = rows[-window:] if window > 0 else rows
    payload: List[Dict[str, Any]] = []
    for row in trimmed:
        payload.append(
            {
                "iteration": row.get("iteration"),
                "attempt_index": row.get("attempt_index"),
                "proposal_note": row.get("proposal_note"),
                "selected_mechanism_templates": row.get("selected_mechanism_templates", []),
                "hypothesis_axes": row.get("hypothesis_axes", []),
                "rejection_reasons": row.get("rejection_reasons", []),
            }
        )
    return payload


def _compact_skill_card(skill_card_md: str, *, compact_level: int) -> str:
    if compact_level <= 0:
        return skill_card_md
    lines = [line.rstrip() for line in skill_card_md.splitlines()]
    if compact_level == 1:
        kept = [line for line in lines if line.startswith("#") or line.startswith("-")][:12]
        return "\n".join(kept)
    kept = [line for line in lines if line.startswith("-")][:6]
    return "\n".join(kept)


def _compact_priority_payload(payload: Dict[str, Any], *, compact_level: int) -> Dict[str, Any]:
    if compact_level <= 0:
        return payload
    ranked_templates = list(payload.get("ranked_templates", []))
    if compact_level == 1:
        top_k = min(4, len(ranked_templates))
    else:
        top_k = min(3, len(ranked_templates))
    compact_rows: List[Dict[str, Any]] = []
    for row in ranked_templates[:top_k]:
        compact_rows.append(
            {
                "name": row.get("name"),
                "rank": row.get("rank"),
                "score": row.get("score"),
                "matched_dataset_tags": row.get("matched_dataset_tags", []),
                "matched_editable_paths": row.get("matched_editable_paths", []),
                "times_seen_in_history": row.get("times_seen_in_history", 0),
                "times_seen_in_recent_rejections": row.get("times_seen_in_recent_rejections", 0),
            }
        )
    history_summary = dict(payload.get("history_outcome_summary", {}))
    if compact_level >= 2:
        history_axes = list(history_summary.get("axes", []))[:3]
    else:
        history_axes = list(history_summary.get("axes", []))[:5]
    return {
        "dataset_key": payload.get("dataset_key"),
        "dataset_tags": payload.get("dataset_tags", []),
        "top_k": top_k,
        "history_outcome_summary": {"axes": history_axes},
        "ranked_templates": compact_rows,
    }


def build_mechanism_fidelity_prompts(
    *,
    dataset_key: str,
    model_ir: Dict[str, Any],
    history: List[Dict[str, Any]],
    attempt_index: int = 1,
    recent_rejections: List[Dict[str, Any]] | None = None,
    prioritized_templates: Dict[str, Any] | None = None,
    compact_level: int = 0,
) -> Dict[str, str]:
    prompt_config = load_yaml(default_prompt_config_path())
    harness_config = load_yaml(default_harness_config_path())
    mechanism_templates = load_yaml(default_mechanism_template_path())
    skill_card_md = default_skill_card_path().read_text(encoding="utf-8")
    sites = enumerate_editable_sites(model_ir)
    rejection_window = int(harness_config.get("rejection_memory_window", 5))
    if compact_level == 1:
        history = history[-3:]
        rejection_window = min(rejection_window, 3)
        sites = sites[:10]
    elif compact_level >= 2:
        history = history[-2:]
        rejection_window = min(rejection_window, 2)
        sites = sites[:8]
    rendered_rejections = _recent_rejections(list(recent_rejections or []), window=rejection_window)
    template_priority_payload = prioritized_templates or prioritize_mechanism_templates(
        dataset_key=dataset_key,
        model_ir=model_ir,
        history=history,
        template_library=mechanism_templates,
        harness_config=harness_config,
        recent_rejections=rendered_rejections,
    )
    rendered_priority_payload = _compact_priority_payload(template_priority_payload, compact_level=compact_level)
    if rendered_rejections:
        retry_guidance_text = (
            "This is a retry. Do not repeat rejected template combinations, paths, or semantics unless the new proposal "
            "explicitly explains why the previous rejection no longer applies."
        )
    else:
        retry_guidance_text = "No retry history yet. Propose the highest-priority mechanism-faithful structural edit."
    system_prompt = str(prompt_config["system"]).strip()
    user_prompt = Template(str(prompt_config["user_template"])).safe_substitute(
        dataset_key=dataset_key,
        attempt_index=int(attempt_index),
        harness_constraints_json=json.dumps(harness_config, indent=2, ensure_ascii=True),
        prioritized_mechanism_templates_json=json.dumps(rendered_priority_payload, indent=2, ensure_ascii=True),
        history_outcome_summary_json=json.dumps(rendered_priority_payload.get("history_outcome_summary", {}), indent=2, ensure_ascii=True),
        recent_rejections_json=json.dumps(rendered_rejections, indent=2, ensure_ascii=True),
        retry_guidance_text=retry_guidance_text,
        skill_card_md=_compact_skill_card(skill_card_md, compact_level=compact_level),
        model_ir_json=json.dumps(model_ir, indent=2, ensure_ascii=True),
        editable_sites_json=json.dumps(sites, indent=2, ensure_ascii=True),
        recent_history_json=json.dumps(_recent_history(history), indent=2, ensure_ascii=True),
    )
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
