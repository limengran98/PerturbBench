from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from sci_response.agent.ir import enumerate_editable_sites


def _flatten_ir_paths(payload: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in payload.items():
        next_prefix = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten_ir_paths(value, next_prefix))
        else:
            flat[next_prefix] = value
    return flat


def changed_ir_paths(before_ir: Dict[str, Any], after_ir: Dict[str, Any]) -> List[str]:
    before = _flatten_ir_paths(before_ir)
    after = _flatten_ir_paths(after_ir)
    all_paths = sorted(set(before) | set(after))
    return [path for path in all_paths if before.get(path) != after.get(path)]


def dataset_mechanism_tags(dataset_key: str) -> List[str]:
    key = str(dataset_key).lower()
    tags = {"general"}
    if key in {"norman", "adamson"}:
        tags.update({"single_cell", "matched_control", "high_dimensional", "sparse_effect", "combination"})
    elif key == "sciplex3":
        tags.update({"single_cell", "dose_time", "context_rich", "matched_control", "high_dimensional"})
    elif key in {"papalexi_arrayed_rna", "papalexi_arrayed_protein"}:
        tags.update({"single_cell", "matched_control", "context_rich"})
    elif key in {"l1000_public", "l1000"}:
        tags.update({"drug_profile", "dose_time", "context_rich", "high_dimensional"})
    elif key == "cdsdb":
        tags.update({"patient_paired", "drug_profile", "context_rich"})
    return sorted(tags)


def infer_hypothesis_axes(changed_paths: Sequence[str]) -> List[str]:
    axes = set()
    for path in changed_paths:
        if "prediction.baseline_skip" in path:
            axes.add("baseline_identity_skip")
        if "prediction.target" in path:
            axes.add("delta_vs_direct_response")
        if "conditioning.mode" in path or "conditioning.conditioning_dim" in path:
            axes.add("conditioning_operator")
        if "trunk.residual_depth" in path:
            axes.add("residual_refinement")
        if "trunk.hidden_dim" in path or "trunk.trunk_depth" in path:
            axes.add("trunk_capacity")
        if "trunk.use_se_block" in path or "trunk.se_reduction" in path:
            axes.add("se_channel_reweighting")
        if "prediction.zero_init_head" in path:
            axes.add("zero_init_stabilization")
        if "loss." in path:
            axes.add("loss_mixing")
        if "optimizer." in path:
            axes.add("optimizer_regularization")
    return sorted(axes)


def summarize_history_outcomes(history: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    positive: Counter[str] = Counter()
    negative: Counter[str] = Counter()
    positive_gain: Counter[str] = Counter()
    negative_gain: Counter[str] = Counter()
    best_axes: Counter[str] = Counter()
    for row in history:
        axes = [str(item) for item in row.get("hypothesis_axes", []) or []]
        if not axes:
            continue
        improvement = row.get("delta_vs_reference")
        if not isinstance(improvement, (int, float)):
            improvement = row.get("delta_vs_iteration0")
        improvement_value = float(improvement) if isinstance(improvement, (int, float)) else 0.0
        for axis in axes:
            if improvement_value > 0.0:
                positive[axis] += 1
                positive_gain[axis] += improvement_value
            elif improvement_value < 0.0:
                negative[axis] += 1
                negative_gain[axis] += abs(improvement_value)
            if bool(row.get("accepted_as_best", False)):
                best_axes[axis] += 1
    axes = sorted(set(positive) | set(negative) | set(best_axes))
    summary_rows: List[Dict[str, Any]] = []
    for axis in axes:
        summary_rows.append(
            {
                "axis": axis,
                "positive_count": int(positive.get(axis, 0)),
                "negative_count": int(negative.get(axis, 0)),
                "positive_gain_total": float(positive_gain.get(axis, 0.0)),
                "negative_gain_total": float(negative_gain.get(axis, 0.0)),
                "best_count": int(best_axes.get(axis, 0)),
            }
        )
    summary_rows.sort(
        key=lambda row: (
            -int(row["best_count"]),
            -float(row["positive_gain_total"]),
            int(row["negative_count"]),
            str(row["axis"]),
        )
    )
    return {"axes": summary_rows}


def _template_current_ir_affinity(template_name: str, model_ir: Dict[str, Any]) -> float:
    rep = dict(model_ir.get("representation", {}))
    prediction = dict(rep.get("prediction", {}))
    conditioning = dict(rep.get("conditioning", {}))
    trunk = dict(rep.get("trunk", {}))
    affinity = 0.0
    if template_name == "baseline_identity_skip" and bool(prediction.get("baseline_skip", False)):
        affinity += 1.0
    if template_name == "delta_vs_direct_response" and str(prediction.get("target", "delta")) == "delta":
        affinity += 0.75
    if template_name == "conditioning_operator" and int(conditioning.get("conditioning_dim", 0)) > 0:
        affinity += 1.0
    if template_name == "residual_refinement" and int(trunk.get("residual_depth", 0)) > 0:
        affinity += 1.0
    if template_name == "trunk_capacity" and int(trunk.get("hidden_dim", 0)) >= 64:
        affinity += 0.5
    if template_name == "se_channel_reweighting" and bool(trunk.get("use_se_block", False)):
        affinity += 1.0
    if template_name == "zero_init_stabilization" and bool(prediction.get("baseline_skip", False)):
        affinity += 0.75
    if template_name == "loss_mixing" and float(rep.get("loss", {}).get("delta_weight", 1.0)) != float(rep.get("loss", {}).get("response_weight", 1.0)):
        affinity += 0.5
    return affinity


def prioritize_mechanism_templates(
    *,
    dataset_key: str,
    model_ir: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
    template_library: Dict[str, Any],
    harness_config: Dict[str, Any],
    recent_rejections: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    templates = list(template_library.get("templates", []))
    editable_paths = {str(item["path"]) for item in enumerate_editable_sites(model_ir)}
    dataset_tags = dataset_mechanism_tags(dataset_key)
    top_k = int(harness_config.get("template_priority_top_k", 5))
    boost_unexplored = float(harness_config.get("priority_boost_unexplored_axis", 3.0))
    penalty_repeated = float(harness_config.get("priority_penalty_repeated_axis", 1.25))
    boost_dataset_tag = float(harness_config.get("priority_boost_dataset_tag", 2.5))
    boost_editable_path = float(harness_config.get("priority_boost_editable_path", 0.5))
    penalty_recent_rejection = float(harness_config.get("priority_penalty_recent_rejection", 2.0))
    boost_ir_affinity = float(harness_config.get("priority_boost_current_ir_affinity", 0.75))
    boost_positive_outcome = float(harness_config.get("priority_boost_positive_outcome", 4.0))
    penalty_negative_outcome = float(harness_config.get("priority_penalty_negative_outcome", 2.5))

    explored_axes = Counter(
        axis
        for row in history
        for axis in row.get("hypothesis_axes", []) or []
    )
    outcome_summary = summarize_history_outcomes(history)
    outcome_lookup = {str(row["axis"]): row for row in outcome_summary["axes"]}
    recent_rejections = list(recent_rejections or [])
    recent_rejection_names = Counter()
    for row in recent_rejections:
        for name in row.get("selected_mechanism_templates", []) or []:
            recent_rejection_names[str(name)] += 1
        for axis in row.get("hypothesis_axes", []) or []:
            recent_rejection_names[str(axis)] += 1

    ranked: List[Dict[str, Any]] = []
    for template in templates:
        name = str(template.get("name"))
        score = float(template.get("priority_base", 0.0))
        rationale: List[str] = [f"base={score:.2f}"]
        preferred_dataset_tags = [str(item) for item in template.get("preferred_dataset_tags", [])]
        matched_tags = sorted(set(preferred_dataset_tags) & set(dataset_tags))
        if matched_tags:
            delta = boost_dataset_tag * len(matched_tags)
            score += delta
            rationale.append(f"dataset_tags+={delta:.2f}:{','.join(matched_tags)}")
        preferred_paths = [str(item) for item in template.get("preferred_paths", [])]
        matched_paths = [path for path in preferred_paths if path in editable_paths]
        if matched_paths:
            delta = boost_editable_path * len(matched_paths)
            score += delta
            rationale.append(f"editable_paths+={delta:.2f}:{','.join(matched_paths)}")
        seen_count = int(explored_axes.get(name, 0))
        if seen_count == 0:
            score += boost_unexplored
            rationale.append(f"unexplored+={boost_unexplored:.2f}")
        else:
            delta = penalty_repeated * seen_count
            score -= delta
            rationale.append(f"repeated-={delta:.2f}")
        rejection_count = int(recent_rejection_names.get(name, 0))
        if rejection_count:
            delta = penalty_recent_rejection * rejection_count
            score -= delta
            rationale.append(f"recent_rejection-={delta:.2f}")
        outcome_row = outcome_lookup.get(name)
        if outcome_row is not None:
            positive_count = int(outcome_row.get("positive_count", 0))
            negative_count = int(outcome_row.get("negative_count", 0))
            positive_gain = float(outcome_row.get("positive_gain_total", 0.0))
            negative_gain = float(outcome_row.get("negative_gain_total", 0.0))
            best_count = int(outcome_row.get("best_count", 0))
            if positive_count > 0:
                delta = boost_positive_outcome * positive_count + positive_gain
                score += delta
                rationale.append(f"positive_outcome+={delta:.2f}")
            if negative_count > 0:
                delta = penalty_negative_outcome * negative_count + negative_gain
                score -= delta
                rationale.append(f"negative_outcome-={delta:.2f}")
            if best_count > 0:
                delta = float(best_count)
                score += delta
                rationale.append(f"best_history+={delta:.2f}")
        affinity = _template_current_ir_affinity(name, model_ir)
        if affinity:
            delta = affinity * boost_ir_affinity
            score += delta
            rationale.append(f"ir_affinity+={delta:.2f}")
        ranked.append(
            {
                "name": name,
                "score": float(score),
                "preferred_paths": preferred_paths,
                "matched_dataset_tags": matched_tags,
                "matched_editable_paths": matched_paths,
                "times_seen_in_history": seen_count,
                "times_seen_in_recent_rejections": rejection_count,
                "rationale": rationale,
                "hypothesis": str(template.get("hypothesis", "")),
                "risks": list(template.get("risks", [])),
            }
        )

    ranked.sort(key=lambda item: (-float(item["score"]), str(item["name"])))
    top_ranked = ranked[:top_k]
    for index, row in enumerate(top_ranked, start=1):
        row["rank"] = int(index)
    return {
        "dataset_key": dataset_key,
        "dataset_tags": dataset_tags,
        "top_k": top_k,
        "history_outcome_summary": outcome_summary,
        "ranked_templates": top_ranked,
    }


def validate_edit_sequence_against_ir(model_ir: Dict[str, Any], edits: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    editable_sites = {str(item["path"]): dict(item) for item in enumerate_editable_sites(model_ir)}
    allowed_primitives = {"set_scalar", "toggle_boolean", "scale_numeric", "increment", "cycle_enum"}
    errors: List[str] = []
    for index, edit in enumerate(edits):
        primitive = str(edit.get("primitive"))
        path = str(edit.get("path"))
        if primitive not in allowed_primitives:
            errors.append(f"edit[{index}] unsupported primitive: {primitive}")
            continue
        if path not in editable_sites:
            errors.append(f"edit[{index}] path is not registered editable site: {path}")
            continue
        site = editable_sites[path]
        site_primitives = set(str(item) for item in site.get("primitive_family", []))
        if primitive not in site_primitives:
            errors.append(f"edit[{index}] primitive {primitive} not allowed for path {path}")
    return {
        "passed": not errors,
        "errors": errors,
        "editable_site_count": len(editable_sites),
    }


def build_mechanism_fidelity_report(
    *,
    previous_ir: Dict[str, Any],
    candidate_ir: Dict[str, Any],
    proposal: Dict[str, Any],
) -> Dict[str, Any]:
    changed_paths = changed_ir_paths(previous_ir, candidate_ir)
    axes = infer_hypothesis_axes(changed_paths)
    risk_flags: List[str] = []
    if not changed_paths:
        risk_flags.append("no_structural_change")
    if len(changed_paths) > 4:
        risk_flags.append("too_many_structural_changes")
    if "representation.prediction.target" in changed_paths and "representation.loss.response_weight" in changed_paths:
        risk_flags.append("prediction_target_and_loss_changed_together")
    if "representation.optimizer.learning_rate" in changed_paths and len(changed_paths) == 1:
        risk_flags.append("optimizer_only_change")
    proposal_note = str(proposal.get("proposal_note", ""))
    mechanistic_rationale = str(proposal.get("mechanistic_rationale", ""))
    scientific_hypothesis = str(proposal.get("scientific_hypothesis", ""))
    declared_templates = [str(item) for item in proposal.get("selected_mechanism_templates", []) if str(item).strip()]
    narrative_ok = any(token in (proposal_note + " " + mechanistic_rationale + " " + scientific_hypothesis).lower() for token in [
        "baseline",
        "delta",
        "response",
        "conditioning",
        "residual",
        "skip",
        "se",
        "loss",
        "mechan",
    ])
    if not narrative_ok:
        risk_flags.append("weak_mechanistic_narrative")
    if declared_templates and not (set(declared_templates) & set(axes)):
        risk_flags.append("declared_template_mismatch")
    return {
        "changed_paths": changed_paths,
        "hypothesis_axes": axes,
        "declared_templates": declared_templates,
        "risk_flags": risk_flags,
        "scientific_hypothesis": scientific_hypothesis,
        "mechanistic_rationale": mechanistic_rationale,
        "risk_notes": list(proposal.get("risk_notes", [])) if isinstance(proposal.get("risk_notes", []), list) else [],
        "mechanism_guard_passed": "no_structural_change" not in risk_flags,
    }


def semantic_code_consistency_check(
    *,
    candidate_ir: Dict[str, Any],
    compiled_model_config: Dict[str, Any],
    generated_code_path: Path,
) -> Dict[str, Any]:
    errors: List[str] = []
    if str(compiled_model_config.get("name")) != "generated_structured_hypothesis":
        errors.append("compiled model is not routed through generated_structured_hypothesis")
    factory = dict(compiled_model_config.get("factory", {}))
    if str(factory.get("module_path", "")) != str(generated_code_path.resolve()):
        errors.append("factory.module_path does not match generated code path")
    if not generated_code_path.exists():
        errors.append("generated code file does not exist")
    generated_text = generated_code_path.read_text(encoding="utf-8") if generated_code_path.exists() else ""
    if "GeneratedStructuredHypothesisRegressor" not in generated_text:
        errors.append("generated code missing GeneratedStructuredHypothesisRegressor class")
    expected_hash = hashlib.sha256(json.dumps(candidate_ir, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    found_hash = hashlib.sha256(json.dumps(compiled_model_config.get("model_ir", {}), sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    if expected_hash != found_hash:
        errors.append("compiled model_ir does not match candidate IR")
    return {
        "passed": not errors,
        "errors": errors,
        "candidate_ir_hash": expected_hash,
        "compiled_ir_hash": found_hash,
    }


def build_harness_review(
    *,
    previous_ir: Dict[str, Any],
    candidate_ir: Dict[str, Any],
    proposal: Dict[str, Any],
    compiled_model_config: Dict[str, Any],
    generated_code_path: Path,
) -> Dict[str, Any]:
    edit_validation = validate_edit_sequence_against_ir(previous_ir, proposal.get("edits", []))
    mechanism_report = build_mechanism_fidelity_report(
        previous_ir=previous_ir,
        candidate_ir=candidate_ir,
        proposal=proposal,
    )
    semantic_report = semantic_code_consistency_check(
        candidate_ir=candidate_ir,
        compiled_model_config=compiled_model_config,
        generated_code_path=generated_code_path,
    )
    passed = bool(edit_validation["passed"] and mechanism_report["mechanism_guard_passed"] and semantic_report["passed"])
    return {
        "passed": passed,
        "edit_validation": edit_validation,
        "mechanism_report": mechanism_report,
        "semantic_code_consistency": semantic_report,
    }


def summarize_rejection_reasons(review: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    for error in review.get("edit_validation", {}).get("errors", []):
        reasons.append(f"edit_validation::{error}")
    for flag in review.get("mechanism_report", {}).get("risk_flags", []):
        reasons.append(f"mechanism::{flag}")
    for error in review.get("semantic_code_consistency", {}).get("errors", []):
        reasons.append(f"semantic_code_consistency::{error}")
    return reasons
