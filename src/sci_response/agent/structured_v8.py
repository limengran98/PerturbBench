from __future__ import annotations

import csv
import difflib
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Mapping, Sequence

from sci_response.agent.direct_code import (
    _build_benchmark_payload,
    _extract_python_code,
    _flatten_selected,
    _get_nested,
    _load_generated_class,
    _run_benchmark_with_capture,
    _safe_slug,
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
from sci_response.agent.structured import OBJECTIVE_PATH, REPORT_METRIC_PATHS, _write_trace_csv
from sci_response.agent.structured_v3 import _task_group
from sci_response.agent.v8_behavior import (
    run_behavior_probes_v8,
    stability_probe_v8,
    summarize_behavior_probe_for_trace,
)
from sci_response.agent.v8_parent_selection import (
    choose_champion_baseline_v8,
    choose_practical_parent_v8,
    parent_candidates_json,
)
from sci_response.baselines.artifacts import beijing_timestamp_slug
from sci_response.data.io import dump_json, dump_yaml, ensure_dir, load_json, load_yaml, stable_hash, write_text
from sci_response.data.registry import load_dataset_spec, load_prepared_dataset
from sci_response.data.splits import indices_from_split, load_split
from sci_response.models.runtime import build_split_batch, fit_intervention_encoder


V8_CLASS_NAME = "GeneratedV8Regressor"

MECHANISM_PRIOR_LIBRARY: dict[str, dict[str, str]] = {
    "global_linear_skip": {
        "summary": "Add an explicit baseline-to-output skip so perturbation modeling anchors to the pre-state rather than relearning the full response from scratch.",
        "focus": "global_skip",
    },
    "deep_residual_stack": {
        "summary": "Use a deeper residual stack so perturbation effects are accumulated through repeated residual refinements instead of one shallow projection.",
        "focus": "residual_stack",
    },
    "se_block": {
        "summary": "Insert channel-wise reweighting so the model can suppress irrelevant response dimensions under each perturbation.",
        "focus": "se_block",
    },
    "context_gate": {
        "summary": "Use explicit context gating so intervention effects are modulated by cellular or study context instead of being globally shared.",
        "focus": "context_gate",
    },
    "film_conditioning": {
        "summary": "Use FiLM-style conditioning so context can scale and shift intermediate activations, not just concatenate to the input.",
        "focus": "film_conditioning",
    },
    "dual_head_delta_response": {
        "summary": "Split post-response prediction and delta prediction into separate but coupled heads to reduce interference between anchored and differential targets.",
        "focus": "dual_head",
    },
    "mse_plus_pcc_loss": {
        "summary": "Combine MSE with a correlation-aware term to align both absolute response magnitude and response direction.",
        "focus": "pcc_loss",
    },
    "ema_plugin": {
        "summary": "Track an EMA model during training to stabilize evaluation under noisy perturbation-response targets.",
        "focus": "ema_plugin",
    },
    "mixup_plugin": {
        "summary": "Use perturbation-aware mixup to regularize the model when sample variability is high and the target space is smooth.",
        "focus": "mixup_plugin",
    },
    "input_noise_plugin": {
        "summary": "Add calibrated baseline/input noise during training to encourage local robustness around observed states.",
        "focus": "input_noise",
    },
}

TASK_GROUP_PRIOR_SEQUENCE: dict[str, list[str]] = {
    "single_cell": ["global_linear_skip", "deep_residual_stack", "context_gate", "dual_head_delta_response", "mse_plus_pcc_loss"],
    "multimodal_rna": ["film_conditioning", "dual_head_delta_response", "mse_plus_pcc_loss", "context_gate", "global_linear_skip"],
    "multimodal_protein": ["dual_head_delta_response", "global_linear_skip", "context_gate", "deep_residual_stack", "mse_plus_pcc_loss"],
    "dose_time": ["film_conditioning", "context_gate", "mse_plus_pcc_loss", "deep_residual_stack", "mixup_plugin"],
    "paired_clinical": ["global_linear_skip", "dual_head_delta_response", "mse_plus_pcc_loss", "ema_plugin", "input_noise_plugin"],
}

ALLOWED_STRUCTURAL_WITNESSES = {
    "global_skip",
    "residual_stack",
    "se_block",
    "context_gate",
    "film_conditioning",
    "dual_head",
    "residual_output_head",
    "pcc_loss",
    "ema_plugin",
    "mixup_plugin",
    "input_noise",
    "layer_norm",
    "batch_norm",
    "dropout_gate",
    "attention_pool",
    "bottleneck_block",
    "mlp_block",
}

ALLOWED_BEHAVIORAL_WITNESSES = {
    "stability_improves",
    "conditioning_sensitivity",
    "delta_head_specialization",
    "skip_path_contribution",
    "regularization_effect",
    "loss_coupling_effect",
}

WITNESS_KEYWORDS: dict[str, list[str]] = {
    "global_skip": ["skip", "identity", "residual", "baseline_proj", "output_skip"],
    "residual_stack": ["residual", "skip", "modulelist", "res_block"],
    "se_block": ["adaptiveavgpool", "sigmoid", "squeeze", "excitation", "se_"],
    "context_gate": ["gate", "gating", "context", "sigmoid"],
    "film_conditioning": ["film", "gamma", "beta", "conditioning"],
    "dual_head": ["head_post", "head_delta", "dual_head", "post_head", "delta_head"],
    "residual_output_head": ["residual_output", "output_skip", "baseline_proj"],
    "pcc_loss": ["pearson", "corr", "correlation"],
    "ema_plugin": ["ema", "moving_average", "shadow_params"],
    "mixup_plugin": ["mixup"],
    "input_noise": ["noise", "jitter", "gaussian"],
    "layer_norm": ["layernorm"],
    "batch_norm": ["batchnorm", "batch_norm"],
    "dropout_gate": ["dropout"],
    "attention_pool": ["attention", "attn"],
    "bottleneck_block": ["bottleneck"],
    "mlp_block": ["nn.linear", "gelu", "relu", "mlp"],
}

ALLOWED_EDIT_SCOPES = {
    "structure",
    "structure+training",
    "consistency_repair",
    "training_stabilization",
}
ALLOWED_RISK_LEVELS = {"low", "medium", "high"}


def _normalize_string_list(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    values: list[str] = []
    seen: set[str] = set()
    for item in raw:
        value = str(item).strip()
        if not value or value in seen:
            continue
        values.append(value)
        seen.add(value)
    return values


def _safe_log_text_v8(message: Any) -> str:
    if isinstance(message, str):
        return message
    try:
        return json.dumps(message, ensure_ascii=True, sort_keys=True)
    except TypeError:
        return repr(message)


def _shape_tuple_v8(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return ()
    try:
        return tuple(int(part) for part in shape)
    except TypeError:
        return ()


def _build_harness_batches_v8(
    *,
    dataset_config: Path,
    split_path: Path,
    max_train_samples: int = 64,
    max_val_samples: int = 32,
) -> Dict[str, Any]:
    dataset_spec = load_dataset_spec(dataset_config)
    dataset = load_prepared_dataset(dataset_spec)
    split = load_split(split_path)
    split_indices = indices_from_split(dataset.sample_ids, split)
    train_indices = split_indices["train"][: min(int(max_train_samples), len(split_indices["train"]))]
    val_indices = split_indices["val"][: min(int(max_val_samples), len(split_indices["val"]))]
    vocabulary = fit_intervention_encoder(dataset.intervention_ids[split_indices["train"]])
    train_batch = build_split_batch(
        dataset.baseline,
        dataset.post,
        dataset.delta,
        dataset.context,
        dataset.intervention_ids,
        dataset.sample_ids,
        train_indices,
        vocabulary,
    )
    val_batch = build_split_batch(
        dataset.baseline,
        dataset.post,
        dataset.delta,
        dataset.context,
        dataset.intervention_ids,
        dataset.sample_ids,
        val_indices,
        vocabulary,
    )
    return {
        "train_batch": train_batch,
        "val_batch": val_batch,
        "dimension_summary": {
            "baseline_dim": int(train_batch["baseline"].shape[1]),
            "intervention_dim": int(train_batch["intervention_onehot"].shape[1]),
            "context_dim": int(train_batch["context"].shape[1]),
            "output_dim": int(train_batch["post"].shape[1]),
            "train_sample_count": int(train_batch["baseline"].shape[0]),
            "val_sample_count": int(val_batch["baseline"].shape[0]),
            "context_feature_names": dataset.context_feature_names.astype(str).tolist(),
        },
    }


def _build_baseline_payload_v8(
    *,
    method_name: str,
    family: str,
    params: Mapping[str, Any],
    dataset_config: Path,
    split_path: Path,
    run_name: str,
    artifacts_root: Path,
    seed: int,
    top_k: int,
    baseline_root: Path,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "run_name": run_name,
        "seed": int(seed),
        "artifacts_root": str(artifacts_root.resolve()),
        "dataset_config": str(dataset_config.resolve()),
        "split_path": str(split_path.resolve()),
        "metrics": {"top_k": int(top_k)},
        "notes": "v8 champion baseline start",
    }
    if family == "universal":
        payload["baseline"] = {
            "name": str(method_name),
            "feature_builder": {"intervention_hash_dim": 64},
            "tuning_budget": {"definition": "v8_champion_start", "num_trials": 1},
            "params": dict(params),
        }
        return payload
    if family == "specialist":
        payload["baseline_name"] = str(method_name)
        payload["baseline_root"] = str(baseline_root.resolve())
        payload["baseline"] = dict(params)
        return payload
    raise ValueError(f"Unsupported champion baseline family for V8: {family}")


def _precheck_generated_candidate_v8(
    *,
    generated_code_path: Path,
    class_name: str,
    mechanism_card: Mapping[str, Any],
    seed_model_config: Mapping[str, Any],
    harness_batches: Mapping[str, Any],
) -> Dict[str, Any]:
    train_batch = dict(harness_batches["train_batch"])
    val_batch = dict(harness_batches["val_batch"])
    dimension_summary = dict(harness_batches["dimension_summary"])
    smoke_config = dict(seed_model_config)
    smoke_config["epochs"] = 1
    smoke_config["patience"] = 1
    smoke_config["batch_size"] = min(16, int(dimension_summary["train_sample_count"]))
    smoke_config["eval_batch_size"] = min(32, int(dimension_summary["val_sample_count"]))
    smoke_config["learning_rate"] = float(smoke_config.get("learning_rate", 1e-3))
    smoke_logs: list[str] = []

    def _smoke_log(message: Any) -> None:
        smoke_logs.append(_safe_log_text_v8(message))

    try:
        generated_cls = _load_generated_class(generated_code_path, class_name)
        model = generated_cls(
            baseline_dim=int(dimension_summary["baseline_dim"]),
            intervention_dim=int(dimension_summary["intervention_dim"]),
            context_dim=int(dimension_summary["context_dim"]),
            output_dim=int(dimension_summary["output_dim"]),
            config=smoke_config,
            seed=0,
            requested_device="cpu",
        )
        training_summary = model.fit(train_batch, val_batch, smoke_config, _smoke_log)
        predictions = model.predict(val_batch)
        if not isinstance(predictions, dict):
            raise RuntimeError("predict must return a dict")
        if "post" not in predictions or "delta" not in predictions:
            raise RuntimeError("predict must return both 'post' and 'delta'")
        post_shape = _shape_tuple_v8(predictions["post"])
        delta_shape = _shape_tuple_v8(predictions["delta"])
        expected_shape = (
            int(dimension_summary["val_sample_count"]),
            int(dimension_summary["output_dim"]),
        )
        if post_shape != expected_shape:
            raise RuntimeError(f"post prediction shape mismatch: expected {expected_shape}, got {post_shape}")
        if delta_shape != expected_shape:
            raise RuntimeError(f"delta prediction shape mismatch: expected {expected_shape}, got {delta_shape}")
        return {
            "passed": True,
            "error": None,
            "training_summary": training_summary,
            "log_lines": smoke_logs[-8:],
            "dimension_summary": dimension_summary,
            "mechanism_title": str(mechanism_card.get("title") or ""),
        }
    except Exception as exc:
        return {
            "passed": False,
            "error": f"{type(exc).__name__}: {exc}",
            "training_summary": {},
            "log_lines": smoke_logs[-8:],
            "dimension_summary": dimension_summary,
            "mechanism_title": str(mechanism_card.get("title") or ""),
        }


def _history_prompt_rows(history: Sequence[Dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in history[-6:]:
        rows.append(
            {
                "iteration": item.get("iteration"),
                "execution_status": item.get("execution_status"),
                "search_action": item.get("search_action"),
                "search_stage": item.get("search_stage"),
                "objective_value": item.get("objective_value"),
                "delta_vs_reference": item.get("delta_vs_reference"),
                "delta_vs_iteration0": item.get("delta_vs_iteration0"),
                "mechanism_title": item.get("mechanism_title"),
                "claimed_mechanism": item.get("claimed_mechanism"),
                "mechanism_consistency_passed": item.get("mechanism_consistency_passed"),
                "behavior_probe_passed": item.get("behavior_probe_passed"),
                "stability_probe_passed": item.get("stability_probe_passed"),
            }
        )
    return rows


def _recent_no_gain_streak_v8(history: Sequence[Dict[str, Any]]) -> int:
    streak = 0
    for item in reversed(history):
        if str(item.get("execution_status")) != "completed":
            streak += 1
            continue
        gain = item.get("delta_vs_reference")
        if not isinstance(gain, (int, float)) or float(gain) <= 0.0:
            streak += 1
            continue
        break
    return int(streak)


def _best_completed_record(history: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    completed = [
        dict(item)
        for item in history
        if str(item.get("execution_status")) == "completed" and item.get("objective_value") is not None
    ]
    if not completed:
        return None
    return min(completed, key=lambda item: float(item["objective_value"]))


def _best_record_by(history: Sequence[Dict[str, Any]], key_name: str) -> Dict[str, Any] | None:
    rows = [
        dict(item)
        for item in history
        if str(item.get("execution_status")) == "completed" and item.get(key_name) is not None
    ]
    if not rows:
        return None
    return max(rows, key=lambda item: float(item.get(key_name) or 0.0))


def _build_archive_summary(history: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    best_score = _best_completed_record(history)
    stable_rows = [
        item
        for item in history
        if str(item.get("execution_status")) == "completed" and bool(item.get("stability_probe_passed"))
    ]
    best_stable = min(stable_rows, key=lambda item: float(item["objective_value"])) if stable_rows else None
    novel_rows = [
        item
        for item in history
        if str(item.get("execution_status")) == "completed" and item.get("novelty_score") is not None
    ]
    best_novel = max(novel_rows, key=lambda item: float(item.get("novelty_score") or 0.0)) if novel_rows else None
    consistent_rows = [
        item
        for item in history
        if str(item.get("execution_status")) == "completed"
        and bool(item.get("mechanism_consistency_passed"))
        and bool(item.get("behavior_probe_passed"))
    ]
    best_mechanism_consistent = (
        min(consistent_rows, key=lambda item: float(item["objective_value"])) if consistent_rows else None
    )

    def _row_summary(row: Dict[str, Any] | None) -> Dict[str, Any] | None:
        if row is None:
            return None
        return {
            "iteration": int(row["iteration"]),
            "objective_value": float(row["objective_value"]),
            "search_action": row.get("search_action"),
            "mechanism_title": row.get("mechanism_title"),
            "claimed_mechanism": row.get("claimed_mechanism"),
            "structural_signature": row.get("structural_signature"),
            "novelty_score": row.get("novelty_score"),
            "mechanism_consistency_score": row.get("mechanism_consistency_score"),
            "behavior_probe_score": row.get("behavior_probe_score"),
            "stability_probe_score": row.get("stability_probe_score"),
            "recommended_next_action": row.get("recommended_next_action"),
        }

    knowledge_rows: list[dict[str, Any]] = []
    for item in history:
        if item.get("mechanism_title"):
            knowledge_rows.append(
                {
                    "iteration": item.get("iteration"),
                    "mechanism_title": item.get("mechanism_title"),
                    "claimed_mechanism": item.get("claimed_mechanism"),
                    "search_action": item.get("search_action"),
                    "objective_value": item.get("objective_value"),
                    "delta_vs_reference": item.get("delta_vs_reference"),
                    "mechanism_consistency_passed": item.get("mechanism_consistency_passed"),
                    "behavior_probe_passed": item.get("behavior_probe_passed"),
                    "stability_probe_passed": item.get("stability_probe_passed"),
                    "structural_signature": item.get("structural_signature"),
                    "recommended_next_action": item.get("recommended_next_action"),
                }
            )
    return {
        "best_score": _row_summary(best_score),
        "best_stable": _row_summary(best_stable),
        "best_novel": _row_summary(best_novel),
        "best_mechanism_consistent": _row_summary(best_mechanism_consistent),
        "knowledge_rows": knowledge_rows[-24:],
    }


def _scheduler_snapshot_v8(history: Sequence[Dict[str, Any]], archive_summary: Mapping[str, Any]) -> Dict[str, Any]:
    completed = [
        item
        for item in history
        if str(item.get("execution_status")) == "completed" and item.get("objective_value") is not None
    ]
    breadth_rows = [item for item in completed if str(item.get("search_action")) == "breadth"]
    constrain_rows = [item for item in completed if str(item.get("search_action")) == "constrain"]
    best_record = _best_completed_record(history)
    best_iteration = int(best_record["iteration"]) if best_record is not None else -1
    current_iteration = int(history[-1]["iteration"]) if history else -1
    recent = completed[-3:]
    recent_consistency = history[-2:]
    novelty_values = [float(item.get("novelty_score") or 0.0) for item in breadth_rows[-2:]]
    route_path = "".join(
        "B" if str(item.get("search_action")) == "breadth" else "C"
        for item in history
        if str(item.get("search_action")) in {"breadth", "constrain"}
    )
    unique_signatures = {
        str(item.get("structural_signature"))
        for item in completed
        if item.get("structural_signature")
    }
    consistency_debt = sum(1 for item in recent_consistency if item.get("mechanism_consistency_passed") is False)
    stability_debt = sum(1 for item in recent_consistency if item.get("stability_probe_passed") is False)
    return {
        "completed_count": int(len(completed)),
        "iterations_since_best": max(0, current_iteration - best_iteration) if best_iteration >= 0 else 0,
        "recent_no_gain_streak": int(_recent_no_gain_streak_v8(history)),
        "recent_gain_mean": (
            float(sum(float(item.get("delta_vs_reference") or 0.0) for item in recent) / len(recent))
            if recent
            else 0.0
        ),
        "breadth_recent_gain_mean": (
            float(sum(float(item.get("delta_vs_reference") or 0.0) for item in breadth_rows[-2:]) / len(breadth_rows[-2:]))
            if breadth_rows[-2:]
            else 0.0
        ),
        "constrain_recent_gain_mean": (
            float(sum(float(item.get("delta_vs_reference") or 0.0) for item in constrain_rows[-2:]) / len(constrain_rows[-2:]))
            if constrain_rows[-2:]
            else 0.0
        ),
        "consistency_debt": int(consistency_debt),
        "stability_debt": int(stability_debt),
        "recent_novelty_mean": (float(sum(novelty_values) / len(novelty_values)) if novelty_values else 0.0),
        "archive_diversity": int(len(unique_signatures)),
        "last_action": str(history[-1].get("search_action")) if history else None,
        "route_path": route_path,
        "has_consistent_anchor": bool(archive_summary.get("best_mechanism_consistent")),
        "best_score_iteration": (
            int(archive_summary["best_score"]["iteration"]) if archive_summary.get("best_score") else -1
        ),
    }


def _choose_action_v8(snapshot: Mapping[str, Any]) -> Dict[str, Any]:
    completed_count = int(snapshot["completed_count"])
    breadth_count = int(str(snapshot["route_path"]).count("B"))
    if completed_count <= 1:
        return {
            "action": "breadth",
            "stage": "open_breadth_bootstrap",
            "decision_reason": "iteration_1_force_breadth",
        }
    if breadth_count < min(3, completed_count):
        return {
            "action": "breadth",
            "stage": "exploration_quota",
            "decision_reason": "ensure_open_structural_search",
        }
    if not bool(snapshot["has_consistent_anchor"]):
        return {
            "action": "breadth",
            "stage": "consistency_anchor_search",
            "decision_reason": "no_mechanism_consistent_candidate_yet",
        }
    if int(snapshot["consistency_debt"]) > 0 or int(snapshot["stability_debt"]) > 0:
        return {
            "action": "constrain",
            "stage": "alignment_repair",
            "decision_reason": "consistency_or_stability_debt",
        }
    if int(snapshot["recent_no_gain_streak"]) >= 2:
        return {
            "action": "breadth",
            "stage": "adaptive_breadth",
            "decision_reason": "recent_no_gain_streak",
        }
    if float(snapshot["breadth_recent_gain_mean"]) > float(snapshot["constrain_recent_gain_mean"]) + 0.002:
        return {
            "action": "breadth",
            "stage": "adaptive_breadth",
            "decision_reason": "breadth_has_more_headroom",
        }
    if str(snapshot["last_action"]) == "breadth" and float(snapshot["recent_gain_mean"]) > 0.0:
        return {
            "action": "constrain",
            "stage": "mechanism_alignment",
            "decision_reason": "promising_structure_needs_alignment",
        }
    if float(snapshot["recent_novelty_mean"]) < 0.35:
        return {
            "action": "breadth",
            "stage": "novelty_refresh",
            "decision_reason": "novelty_debt",
        }
    return {
        "action": "constrain",
        "stage": "adaptive_constrain",
        "decision_reason": "steady_alignment_refinement",
    }


def _archive_parent_record(history: Sequence[Dict[str, Any]], route: str) -> Dict[str, Any] | None:
    archive_summary = _build_archive_summary(history)
    if route == "constrain":
        if archive_summary.get("best_mechanism_consistent"):
            iteration = int(archive_summary["best_mechanism_consistent"]["iteration"])
            return next((dict(item) for item in history if int(item.get("iteration", -1)) == iteration), None)
        if archive_summary.get("best_score"):
            iteration = int(archive_summary["best_score"]["iteration"])
            return next((dict(item) for item in history if int(item.get("iteration", -1)) == iteration), None)
    if archive_summary.get("best_novel") and _recent_no_gain_streak_v8(history) >= 2:
        iteration = int(archive_summary["best_novel"]["iteration"])
        return next((dict(item) for item in history if int(item.get("iteration", -1)) == iteration), None)
    if archive_summary.get("best_score"):
        iteration = int(archive_summary["best_score"]["iteration"])
        return next((dict(item) for item in history if int(item.get("iteration", -1)) == iteration), None)
    return _best_completed_record(history)


def _reference_code_payload(
    *,
    seed_model_source_path: Path,
    practical_parent_selection: Mapping[str, Any],
    parent_record: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    if parent_record is not None and parent_record.get("generated_code_path"):
        parent_code_path = Path(str(parent_record["generated_code_path"]))
        if parent_code_path.exists():
            return {
                "parent_iteration": int(parent_record["iteration"]),
                "parent_source_kind": "generated_code",
                "parent_code_path": str(parent_code_path.resolve()),
                "parent_code": parent_code_path.read_text(encoding="utf-8"),
                "parent_selection": dict(practical_parent_selection),
            }
    selected_practical = practical_parent_selection.get("selected")
    if isinstance(selected_practical, dict):
        practical_code_path = Path(str(selected_practical.get("generated_code_path") or ""))
        if practical_code_path.exists():
            return {
                "parent_iteration": int(selected_practical.get("iteration") or 0),
                "parent_source_kind": "historical_codegen_parent",
                "parent_code_path": str(practical_code_path.resolve()),
                "parent_code": practical_code_path.read_text(encoding="utf-8"),
                "parent_selection": dict(practical_parent_selection),
            }
    return {
        "parent_iteration": 0,
        "parent_source_kind": "seed_model_source",
        "parent_code_path": str(seed_model_source_path.resolve()),
        "parent_code": seed_model_source_path.read_text(encoding="utf-8"),
        "parent_selection": dict(practical_parent_selection),
    }


def _task_group_priors(task_group: str) -> list[dict[str, str]]:
    names = TASK_GROUP_PRIOR_SEQUENCE.get(task_group, TASK_GROUP_PRIOR_SEQUENCE.get("single_cell", []))
    return [{"name": name, **MECHANISM_PRIOR_LIBRARY[name]} for name in names if name in MECHANISM_PRIOR_LIBRARY]


def _candidate_specs_for_route(task_group: str, route: str) -> list[dict[str, Any]]:
    priors = _task_group_priors(task_group)
    if route == "breadth":
        return [
            {"candidate_kind": "free_form_explore_a", "prior": None},
            {"candidate_kind": "free_form_explore_b", "prior": None},
            {"candidate_kind": "prior_guided_structural", "prior": priors[0] if priors else None},
        ]
    return [
        {"candidate_kind": "consistency_repair", "prior": None},
        {"candidate_kind": "stability_repair", "prior": priors[0] if priors else None},
        {"candidate_kind": "local_refine", "prior": priors[1] if len(priors) > 1 else None},
    ]


def _build_mechanism_card_prompts(
    *,
    dataset_key: str,
    task_group: str,
    iteration: int,
    route: str,
    stage: str,
    candidate_kind: str,
    seed_model_name: str,
    archive_summary: Mapping[str, Any],
    history: Sequence[Dict[str, Any]],
    parent_record: Mapping[str, Any] | None,
    prior: Mapping[str, Any] | None,
    champion_baseline: Mapping[str, Any] | None,
) -> Dict[str, str]:
    prior_text = json.dumps(prior, ensure_ascii=True, indent=2) if prior else "<none>"
    champion_text = json.dumps(champion_baseline, ensure_ascii=True, indent=2) if champion_baseline else "<none>"
    system_prompt = (
        "You are designing one mechanism card for an automated biological perturbation modeling agent. "
        "Return exactly one JSON object. The mechanism card is binding: it must be specific enough to constrain the later implementation. "
        "Do not output markdown or prose outside the JSON object."
    )
    user_prompt = (
        f"DATASET KEY: {dataset_key}\n"
        f"TASK GROUP: {task_group}\n"
        f"ITERATION: {iteration}\n"
        f"ROUTE: {route}\n"
        f"STAGE: {stage}\n"
        f"CANDIDATE KIND: {candidate_kind}\n"
        f"SEED MODEL NAME: {seed_model_name}\n\n"
        "ARCHIVE SUMMARY:\n"
        f"{json.dumps(archive_summary, ensure_ascii=True, indent=2)}\n\n"
        "RECENT HISTORY:\n"
        f"{json.dumps(_history_prompt_rows(history), ensure_ascii=True, indent=2)}\n\n"
        "CURRENT DATASET CHAMPION BASELINE (THIS DEFINES THE INCUMBENT TO MATCH OR BEAT):\n"
        f"{champion_text}\n\n"
        "PARENT RECORD:\n"
        f"{json.dumps(parent_record or {}, ensure_ascii=True, indent=2)}\n\n"
        "OPTIONAL RETRIEVAL PRIOR (INSPIRATION ONLY, NOT A HARD BOUNDARY):\n"
        f"{prior_text}\n\n"
        "Return exactly one JSON object with this schema:\n"
        "{\n"
        '  "proposal_note": "v8_mechanism_card",\n'
        '  "title": "<short title>",\n'
        '  "claimed_mechanism": "<one concrete mechanism claim>",\n'
        '  "biological_rationale": "<one or two sentences>",\n'
        '  "implementation_directive": "<what the later code must actually implement>",\n'
        '  "structural_witnesses": ["<choose from controlled vocabulary>"],\n'
        '  "behavioral_witnesses": ["<choose from controlled vocabulary>"],\n'
        '  "forbidden_shortcuts": ["optimizer_only_change", "batch_size_only_change", "hidden_dim_only_change", "cosmetic_refactor_only", "claim_without_witness"],\n'
        '  "edit_scope": "<one of structure, structure+training, consistency_repair, training_stabilization>",\n'
        '  "risk_level": "<one of low, medium, high>"\n'
        "}\n\n"
        "Controlled structural_witness vocabulary:\n"
        f"{json.dumps(sorted(ALLOWED_STRUCTURAL_WITNESSES), ensure_ascii=True)}\n\n"
        "Controlled behavioral_witness vocabulary:\n"
        f"{json.dumps(sorted(ALLOWED_BEHAVIORAL_WITNESSES), ensure_ascii=True)}\n\n"
        "Rules:\n"
        "- The card must come before implementation and must be specific enough to constrain later code.\n"
        "- Breadth routes must include at least one macro structural witness such as skip, block, gate, conditioning, head, normalization, loss coupling, or plugin.\n"
        "- Constrain routes must preserve the same broad mechanism family and focus on alignment, repair, or stabilization.\n"
        "- Do not propose a card that is only optimizer tuning.\n"
        "- The implementation_directive must describe what code must exist, not just the intended story.\n"
    )
    return {"system_prompt": system_prompt, "user_prompt": user_prompt}


def _fallback_mechanism_card(
    *,
    route: str,
    candidate_kind: str,
    prior: Mapping[str, Any] | None,
    parent_record: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    focus = str((prior or {}).get("focus") or ("global_skip" if route == "breadth" else "residual_stack"))
    if focus not in ALLOWED_STRUCTURAL_WITNESSES:
        focus = "global_skip"
    return {
        "proposal_note": f"v8_mechanism_card::{candidate_kind}::fallback",
        "title": f"Fallback {focus.replace('_', ' ').title()}",
        "claimed_mechanism": (
            "Introduce one explicit structural mechanism and make its implementation traceable."
            if route == "breadth"
            else "Preserve the current mechanism family but align the implementation and stabilize training."
        ),
        "biological_rationale": str((prior or {}).get("summary") or "Use a conservative, auditable mechanism proposal when LLM card generation fails."),
        "implementation_directive": (
            "The generated code must materially implement the named structural witness and keep the training interface unchanged."
        ),
        "structural_witnesses": [focus],
        "behavioral_witnesses": ["stability_improves"] if route == "constrain" else ["skip_path_contribution"],
        "forbidden_shortcuts": [
            "optimizer_only_change",
            "batch_size_only_change",
            "hidden_dim_only_change",
            "claim_without_witness",
        ],
        "edit_scope": "structure" if route == "breadth" else "consistency_repair",
        "risk_level": "medium",
        "parent_iteration": parent_record.get("iteration") if parent_record else None,
    }


def _normalize_mechanism_card(
    payload: Mapping[str, Any],
    *,
    route: str,
    candidate_kind: str,
    prior: Mapping[str, Any] | None,
    parent_record: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    card = dict(payload)
    title = str(card.get("title", "")).strip() or f"V8 {candidate_kind.replace('_', ' ').title()}"
    claimed_mechanism = str(card.get("claimed_mechanism", "")).strip() or "Mechanism claim missing"
    biological_rationale = str(card.get("biological_rationale", "")).strip() or "Biological rationale missing"
    implementation_directive = str(card.get("implementation_directive", "")).strip() or claimed_mechanism
    structural_witnesses = [
        item for item in _normalize_string_list(card.get("structural_witnesses")) if item in ALLOWED_STRUCTURAL_WITNESSES
    ]
    behavioral_witnesses = [
        item for item in _normalize_string_list(card.get("behavioral_witnesses")) if item in ALLOWED_BEHAVIORAL_WITNESSES
    ]
    forbidden_shortcuts = _normalize_string_list(card.get("forbidden_shortcuts"))
    edit_scope = str(card.get("edit_scope", "")).strip()
    risk_level = str(card.get("risk_level", "")).strip().lower()
    if route == "breadth" and not structural_witnesses:
        structural_witnesses = [str((prior or {}).get("focus") or "global_skip")]
    if route == "breadth" and edit_scope not in {"structure", "structure+training"}:
        edit_scope = "structure"
    if route == "constrain" and edit_scope not in {"consistency_repair", "training_stabilization", "structure+training"}:
        edit_scope = "consistency_repair"
    if edit_scope not in ALLOWED_EDIT_SCOPES:
        edit_scope = "structure" if route == "breadth" else "consistency_repair"
    if risk_level not in ALLOWED_RISK_LEVELS:
        risk_level = "medium"
    if not behavioral_witnesses:
        behavioral_witnesses = ["stability_improves"] if route == "constrain" else ["skip_path_contribution"]
    if not forbidden_shortcuts:
        forbidden_shortcuts = ["optimizer_only_change", "batch_size_only_change", "hidden_dim_only_change", "claim_without_witness"]
    normalized = {
        "proposal_note": str(card.get("proposal_note", f"v8_mechanism_card::{candidate_kind}")),
        "title": title,
        "claimed_mechanism": claimed_mechanism,
        "biological_rationale": biological_rationale,
        "implementation_directive": implementation_directive,
        "structural_witnesses": structural_witnesses,
        "behavioral_witnesses": behavioral_witnesses,
        "forbidden_shortcuts": forbidden_shortcuts,
        "edit_scope": edit_scope,
        "risk_level": risk_level,
        "route": route,
        "candidate_kind": candidate_kind,
        "prior_name": str((prior or {}).get("name") or ""),
        "parent_iteration": int(parent_record["iteration"]) if parent_record and parent_record.get("iteration") is not None else 0,
    }
    normalized["mechanism_card_id"] = stable_hash(normalized)
    return normalized


def _request_mechanism_card(
    *,
    llm_client: StructuredLLMClient,
    dataset_key: str,
    task_group: str,
    iteration: int,
    route: str,
    stage: str,
    candidate_kind: str,
    seed_model_name: str,
    archive_summary: Mapping[str, Any],
    history: Sequence[Dict[str, Any]],
    parent_record: Mapping[str, Any] | None,
    prior: Mapping[str, Any] | None,
    champion_baseline: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    prompts = _build_mechanism_card_prompts(
        dataset_key=dataset_key,
        task_group=task_group,
        iteration=iteration,
        route=route,
        stage=stage,
        candidate_kind=candidate_kind,
        seed_model_name=seed_model_name,
        archive_summary=archive_summary,
        history=history,
        parent_record=parent_record,
        prior=prior,
        champion_baseline=champion_baseline,
    )
    try:
        payload = llm_client.chat_json(
            system_prompt=prompts["system_prompt"],
            user_prompt=prompts["user_prompt"],
            max_tokens_override=min(int(llm_client.settings.max_tokens), 1600),
        )
        return _normalize_mechanism_card(
            payload,
            route=route,
            candidate_kind=candidate_kind,
            prior=prior,
            parent_record=parent_record,
        )
    except Exception:
        return _normalize_mechanism_card(
            _fallback_mechanism_card(
                route=route,
                candidate_kind=candidate_kind,
                prior=prior,
                parent_record=parent_record,
            ),
            route=route,
            candidate_kind=candidate_kind,
            prior=prior,
            parent_record=parent_record,
        )


def _build_v8_code_prompts(
    *,
    dataset_key: str,
    task_group: str,
    route: str,
    stage: str,
    iteration: int,
    candidate_kind: str,
    seed_model_name: str,
    mechanism_card: Mapping[str, Any],
    archive_summary: Mapping[str, Any],
    history: Sequence[Dict[str, Any]],
    parent_code: str,
    parent_source_kind: str,
    previous_code: str | None,
    previous_error: str | None,
    prior: Mapping[str, Any] | None,
    dimension_summary: Mapping[str, Any],
    champion_baseline: Mapping[str, Any] | None,
) -> Dict[str, str]:
    previous_code_text = previous_code.strip() if previous_code and previous_code.strip() else "<none>"
    previous_error_text = previous_error.strip() if previous_error and previous_error.strip() else "<none>"
    prior_text = json.dumps(prior, ensure_ascii=True, indent=2) if prior else "<none>"
    champion_text = json.dumps(champion_baseline, ensure_ascii=True, indent=2) if champion_baseline else "<none>"
    system_prompt = (
        "You are editing a scientific response prediction model by directly rewriting Python model code. "
        f"Return only executable Python source code for one module. The module must define a class named {V8_CLASS_NAME}. "
        "Preserve the required training interface exactly:\n"
        "- __init__(self, *, baseline_dim, intervention_dim, context_dim, output_dim, config, seed, requested_device='cpu')\n"
        "- fit(self, train_batch, val_batch, config, log_fn) -> dict\n"
        "- predict(self, batch) -> dict with keys 'post' and 'delta'\n"
        "Use only the Python standard library, numpy, and torch. "
        "The Mechanism Card is binding: the code must materially implement the named structural witnesses. "
        "Do not hardcode larger replacement input widths or impose arbitrary minimum dimensions. "
        "Every projection consuming baseline/intervention/context must accept exactly the constructor-provided dimensions."
    )
    user_prompt = (
        f"DATASET KEY: {dataset_key}\n"
        f"TASK GROUP: {task_group}\n"
        f"ROUTE: {route}\n"
        f"STAGE: {stage}\n"
        f"ITERATION: {iteration}\n"
        f"CANDIDATE KIND: {candidate_kind}\n"
        f"SEED MODEL NAME: {seed_model_name}\n"
        f"PARENT SOURCE KIND: {parent_source_kind}\n\n"
        "MECHANISM CARD (BINDING CONTRACT):\n"
        f"{json.dumps(mechanism_card, ensure_ascii=True, indent=2)}\n\n"
        "ARCHIVE SUMMARY:\n"
        f"{json.dumps(archive_summary, ensure_ascii=True, indent=2)}\n\n"
        "RECENT HISTORY:\n"
        f"{json.dumps(_history_prompt_rows(history), ensure_ascii=True, indent=2)}\n\n"
        "OPTIONAL RETRIEVAL PRIOR (INSPIRATION ONLY, NOT A HARD BOUNDARY):\n"
        f"{prior_text}\n\n"
        "CURRENT DATASET CHAMPION BASELINE (THIS DEFINES THE INCUMBENT FLOOR TO BEAT):\n"
        f"{champion_text}\n\n"
        "ACTUAL BATCH DIMENSIONS FOR THIS DATASET (MUST BE RESPECTED EXACTLY):\n"
        f"{json.dumps(dimension_summary, ensure_ascii=True, indent=2)}\n\n"
        "PREVIOUS FAILED CODE:\n"
        f"{previous_code_text}\n\n"
        "PREVIOUS ERROR / TRACEBACK:\n"
        f"{previous_error_text}\n\n"
        "PARENT CODE TO EDIT:\n"
        f"{parent_code}\n\n"
        "REQUIRED INTERFACE:\n"
        f"- Define class {V8_CLASS_NAME}\n"
        "- Constructor signature:\n"
        "  __init__(self, *, baseline_dim, intervention_dim, context_dim, output_dim, config, seed, requested_device='cpu')\n"
        "- fit(train_batch, val_batch, config, log_fn) returns a dict containing at least best_val_loss and best_epoch\n"
        "- predict(batch) returns {'post': ..., 'delta': ...}\n"
        "- Use batch['baseline'], batch['intervention_onehot'], batch['context'], batch['post'], batch['delta']\n"
        "- Respect requested_device when torch is available\n\n"
        "FAIRNESS CONSTRAINTS:\n"
        "- Do not read any external files beyond this module.\n"
        "- Do not alter data loading, split logic, evaluation, or artifact writing.\n"
        "- Do not call the network or subprocesses.\n"
        "- Keep all changes inside the model and its local training loop only.\n\n"
        "ROUTE-SPECIFIC RULES:\n"
        + (
            "- This is a Breadth candidate. Make at least one real structural change. "
            "Do not return a candidate that is only optimizer tuning, hidden-dim retuning, or cosmetic refactoring.\n"
            if route == "breadth"
            else "- This is a Constrain candidate. Preserve the broad mechanism family but tighten implementation fidelity, runtime robustness, and training stability. "
            "Do not invent a completely unrelated new backbone family.\n"
        )
        + "\nProduce only the full Python module source."
    )
    return {"system_prompt": system_prompt, "user_prompt": user_prompt}


def _code_feature_flags(code_text: str) -> Dict[str, bool]:
    lowered = str(code_text).lower()
    flags = {
        witness: any(keyword in lowered for keyword in keywords)
        for witness, keywords in WITNESS_KEYWORDS.items()
    }
    flags["context_gate"] = ("gate" in lowered or "gating" in lowered) and "context" in lowered
    flags["film_conditioning"] = "film" in lowered or ("gamma" in lowered and "beta" in lowered)
    flags["residual_stack"] = "residual" in lowered or "modulelist" in lowered or "res_block" in lowered
    flags["global_skip"] = (
        "skip" in lowered
        or "identity" in lowered
        or "baseline_proj" in lowered
        or "output_skip" in lowered
    )
    return flags


def _structural_signature_from_features(flags: Mapping[str, bool]) -> str:
    active = sorted(name for name, enabled in flags.items() if enabled)
    if not active:
        return "plain_codegen"
    return "|".join(active[:6])


def _line_diff_count(before_code: str, after_code: str) -> int:
    diff = list(
        difflib.unified_diff(
            before_code.splitlines(),
            after_code.splitlines(),
            lineterm="",
        )
    )
    return int(sum(1 for line in diff if line.startswith("+") or line.startswith("-")))


def _review_candidate_code(
    *,
    parent_code: str,
    candidate_code: str,
    mechanism_card: Mapping[str, Any],
    route: str,
) -> Dict[str, Any]:
    flags = _code_feature_flags(candidate_code)
    witness_hits = [name for name in mechanism_card.get("structural_witnesses", []) if flags.get(str(name), False)]
    missing_witnesses = [
        name for name in mechanism_card.get("structural_witnesses", []) if name not in witness_hits
    ]
    structural_edit_detected = bool(witness_hits)
    line_diff_count = _line_diff_count(parent_code, candidate_code)
    novelty_score = min(1.5, float(line_diff_count) / 48.0 + 0.2 * len(witness_hits))
    shortcut_flags: list[str] = []
    if route == "breadth" and not structural_edit_detected:
        shortcut_flags.append("breadth_without_structural_witness")
    if line_diff_count < 10:
        shortcut_flags.append("low_delta_code_change")
    if not candidate_code.strip():
        shortcut_flags.append("empty_code")
    forbidden_shortcuts = {str(item) for item in mechanism_card.get("forbidden_shortcuts", [])}
    if "claim_without_witness" in forbidden_shortcuts and not structural_edit_detected:
        shortcut_flags.append("claim_without_witness")
    consistency_score = max(
        0.0,
        min(
            1.0,
            0.55 * (len(witness_hits) / max(1, len(mechanism_card.get("structural_witnesses", []) or [])))
            + 0.25 * min(1.0, line_diff_count / 24.0)
            + 0.20 * (1.0 if route == "constrain" or structural_edit_detected else 0.0),
        ),
    )
    passed = bool(structural_edit_detected or route == "constrain")
    if route == "breadth" and len(witness_hits) == 0:
        passed = False
    if line_diff_count < 4:
        passed = False
    return {
        "passed": bool(passed and "empty_code" not in shortcut_flags),
        "mechanism_consistency_score": float(consistency_score),
        "feature_flags": flags,
        "witness_hits": witness_hits,
        "missing_witnesses": missing_witnesses,
        "shortcut_flags": shortcut_flags,
        "structural_edit_detected": bool(structural_edit_detected),
        "line_diff_count": int(line_diff_count),
        "novelty_score": float(novelty_score),
        "structural_signature": _structural_signature_from_features(flags),
    }


def _cheap_candidate_score(
    *,
    route: str,
    candidate_kind: str,
    review: Mapping[str, Any],
    mechanism_card: Mapping[str, Any],
) -> float:
    score = 4.0 * float(review["mechanism_consistency_score"]) + 1.5 * float(review["novelty_score"])
    if review.get("structural_edit_detected"):
        score += 0.6
    if route == "breadth" and candidate_kind.startswith("free_form"):
        score += 0.3
    if route == "constrain" and mechanism_card.get("edit_scope") in {"consistency_repair", "training_stabilization"}:
        score += 0.4
    if str(mechanism_card.get("risk_level")) == "high":
        score -= 0.25
    return float(score)


def _run_behavior_probes(
    *,
    mechanism_card: Mapping[str, Any],
    review: Mapping[str, Any],
    metrics_payload: Mapping[str, Any],
    train_log_text: str,
    reference_record: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    return run_behavior_probes_v8(
        mechanism_card=mechanism_card,
        review=review,
        metrics_payload=metrics_payload,
        train_log_text=train_log_text,
        reference_record=reference_record,
    )


def _stability_probe(
    *,
    train_log_text: str,
    metrics_payload: Mapping[str, Any],
    reference_record: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    return stability_probe_v8(
        train_log_text=train_log_text,
        metrics_payload=metrics_payload,
        reference_record=reference_record,
    )


def _recommended_next_action(
    *,
    execution_status: str,
    delta_vs_reference: float | None,
    mechanism_consistency_passed: bool,
    behavior_probe_passed: bool,
    stability_probe_passed: bool,
    novelty_score: float | None,
) -> str:
    if execution_status != "completed":
        return "constrain"
    if not mechanism_consistency_passed or not stability_probe_passed:
        return "constrain"
    if delta_vs_reference is not None and delta_vs_reference > 0.0 and behavior_probe_passed:
        return "constrain"
    if novelty_score is not None and novelty_score >= 0.5:
        return "breadth"
    return "constrain"


def run_structured_model_session_v8(
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
    baseline_root: Path,
    seed: int,
    top_k: int,
    max_iteration: int,
    session_id: str | None,
    llm_config_path: Path,
    agent_mode: str = "structured_open_mechanism_search_v8",
) -> Dict[str, Any]:
    llm_client = StructuredLLMClient(load_llm_settings(llm_config_path))
    seed_model_config = load_yaml(seed_model_config_path)
    seed_model_name = str(seed_model_config["name"])
    seed_model_source_path = repo_root / "src" / "sci_response" / "models" / f"{seed_model_name}.py"
    if not seed_model_source_path.exists():
        raise FileNotFoundError(f"Could not resolve seed model source for {seed_model_name}: {seed_model_source_path}")

    session_slug = session_id or beijing_timestamp_slug()
    practical_warm_start = str(agent_mode).strip() in {
        "structured_open_mechanism_search_v8",
        "structured_open_mechanism_search_v8_practical",
    }
    agent_line = "main_agent_v8"
    method_name = f"agent_structured_v8::{seed_model_name}"
    harness_mode = "open_mechanism_search_v8"
    budget_limit = int(max_iteration)
    seed_value = int(seed)
    top_k_value = int(top_k)
    task_group = _task_group(dataset_key)
    session_started_at = perf_counter()
    prior_session_wall_clock_seconds = load_previous_session_wall_clock_seconds(session_root)
    harness_batches = _build_harness_batches_v8(
        dataset_config=dataset_config.resolve(),
        split_path=split_path.resolve(),
    )

    dataset_root = ensure_dir(agent_root / "datasets" / dataset_key)
    session_root = ensure_dir(dataset_root / "models" / _safe_slug(Path(seed_model_config_path).stem) / session_slug)
    generated_code_root = ensure_dir(session_root / "generated_code")
    raw_response_root = ensure_dir(session_root / "raw_responses")
    benchmark_configs_root = ensure_dir(session_root / "benchmark_configs")
    benchmark_runs_root = ensure_dir(session_root / "benchmark_runs")
    model_configs_root = ensure_dir(session_root / "model_configs")
    history_root = ensure_dir(session_root / "history")
    trace_root = ensure_dir(session_root / "trace")
    prompts_root = ensure_dir(session_root / "prompts")
    mechanism_root = ensure_dir(session_root / "mechanism_cards")
    review_root = ensure_dir(session_root / "consistency_reviews")
    candidate_root = ensure_dir(session_root / "candidate_pools")
    attempt_logs_root = ensure_dir(session_root / "attempt_logs")

    log_lines = [
        f"dataset_key={dataset_key}",
        f"seed_model_name={seed_model_name}",
        f"seed_model_config={seed_model_config_path}",
        f"seed_model_source={seed_model_source_path}",
        f"session_id={session_slug}",
        f"requested_device={requested_device}",
        f"cuda_visible_devices={cuda_visible_devices}",
        f"llm_config_path={llm_config_path}",
        f"task_group={task_group}",
        f"harness_mode={harness_mode}",
    ]
    log_lines = load_previous_log_lines(session_root / "agent.log", log_lines)

    def log(message: str) -> None:
        log_lines.append(message)
        print(f"[agent-structured-v8] {message}", flush=True)

    history = load_history_rows(history_root)
    llm_usage_rows = load_rows_payload(trace_root / "llm_usage.json")
    mechanism_rows = load_rows_payload(trace_root / "mechanism_cards.json")
    review_rows = load_rows_payload(trace_root / "consistency_reviews.json")
    router_rows = load_rows_payload(trace_root / "action_router.json")
    parent_selection_payload = load_json(trace_root / "parent_selection.json") if (trace_root / "parent_selection.json").exists() else {}
    parent_rows = list(parent_selection_payload.get("rows", [])) if isinstance(parent_selection_payload.get("rows"), list) else []
    behavior_rows = load_rows_payload(trace_root / "behavior_probes.json")
    candidate_pool_rows = load_rows_payload(trace_root / "candidate_pools.json")
    rejection_rows = load_rows_payload(trace_root / "proposal_rejections.json")
    baseline_objective = baseline_objective_from_history(history)
    stopping_reason: str | None = None
    iteration = next_iteration_index(history)
    champion_baseline_selection = (
        parent_selection_payload.get("champion_baseline_selection")
        if isinstance(parent_selection_payload.get("champion_baseline_selection"), dict)
        else choose_champion_baseline_v8(
            repo_root=repo_root,
            dataset_key=dataset_key,
        )
    )
    practical_parent_selection = (
        parent_selection_payload.get("session_selection")
        if isinstance(parent_selection_payload.get("session_selection"), dict)
        else choose_practical_parent_v8(
            repo_root=repo_root,
            dataset_key=dataset_key,
            seed_model_name=seed_model_name,
            current_session_id=session_slug,
        )
    )
    if history:
        log(
            f"resume session_id={session_slug} next_iteration={iteration} "
            f"completed_iterations={len(history)}"
        )

    while should_continue_search(history, budget_limit):
        if iteration == 0:
            selected_champion_baseline = champion_baseline_selection.get("selected")
            selected_practical = practical_parent_selection.get("selected")
            practical_code_path = (
                Path(str(selected_practical.get("generated_code_path") or ""))
                if isinstance(selected_practical, dict)
                else None
            )
            use_practical_warm_start = bool(
                practical_warm_start and isinstance(selected_practical, dict) and practical_code_path is not None and practical_code_path.exists()
            )
            if isinstance(selected_champion_baseline, dict):
                benchmark_config_path = benchmark_configs_root / f"iter_{iteration:03d}.yaml"
                dump_yaml(
                    benchmark_config_path,
                    _build_baseline_payload_v8(
                        method_name=str(selected_champion_baseline["method_name"]),
                        family=str(selected_champion_baseline["family"]),
                        params=dict(selected_champion_baseline.get("params", {})),
                        dataset_config=dataset_config,
                        split_path=split_path,
                        run_name=f"{dataset_key}_{seed_model_name}_{agent_mode}_iter_{iteration:03d}",
                        artifacts_root=benchmark_runs_root,
                        seed=seed_value,
                        top_k=top_k_value,
                        baseline_root=baseline_root,
                    ),
                )
                model_config_path = None
                baseline_proposal_source = "champion_baseline"
                baseline_proposal_note = (
                    "champion_baseline::"
                    f"{selected_champion_baseline.get('method_name')}::"
                    f"{selected_champion_baseline.get('mse_mean')}"
                )
                baseline_candidate_kind = f"baseline::{selected_champion_baseline.get('method_name')}"
                baseline_generated_code_path = None
            else:
                if use_practical_warm_start:
                    model_config_path = model_configs_root / f"iter_{iteration:03d}.yaml"
                    model_payload = dict(seed_model_config)
                    model_payload["name"] = f"generated_direct_code::{_safe_slug(agent_mode)}"
                    model_payload["factory"] = {
                        "module_path": str(practical_code_path.resolve()),
                        "class_name": str(selected_practical.get("generated_class_name") or V8_CLASS_NAME),
                    }
                    model_payload["seed_model_name"] = seed_model_name
                    model_payload["agent_mode"] = agent_mode
                    model_payload["warm_start_source"] = {
                        "strategy": "historical_codegen_parent",
                        "selection": selected_practical,
                    }
                    dump_yaml(model_config_path, model_payload)
                    baseline_proposal_source = "practical_warm_start"
                    baseline_proposal_note = (
                        "practical_warm_start::"
                        f"{selected_practical.get('branch_root', 'unknown')}::"
                        f"{selected_practical.get('session_id', 'unknown')}"
                    )
                    baseline_candidate_kind = "practical_warm_start"
                    baseline_generated_code_path = str(practical_code_path.resolve())
                else:
                    model_config_path = seed_model_config_path.resolve()
                    baseline_proposal_source = "seed_model"
                    baseline_proposal_note = f"seed_model_baseline::{seed_model_name}"
                    baseline_candidate_kind = "baseline"
                    baseline_generated_code_path = None
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
            log(
                f"dispatch iteration=0 proposal_source={baseline_proposal_source} "
                f"proposal_note={baseline_proposal_note}"
            )
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
                raise RuntimeError(f"Seed model baseline failed for V8 session: {stderr_text or stdout_text}")
            manifest = load_json(run_dir / "manifest.json")
            metrics = load_json(run_dir / "metrics.json")
            objective_raw = _get_nested(metrics, OBJECTIVE_PATH)
            objective_value = float(objective_raw) if isinstance(objective_raw, (int, float)) else None
            baseline_objective = objective_value
            record = {
                "iteration": 0,
                "phase": "baseline",
                "search_action": "baseline",
                "search_stage": "baseline",
                "proposal_source": baseline_proposal_source,
                "proposal_note": baseline_proposal_note,
                "candidate_kind": baseline_candidate_kind,
                "execution_status": str(manifest.get("execution_status")),
                "objective_path": OBJECTIVE_PATH,
                "objective_value": objective_value,
                "reference_objective_value": None,
                "delta_vs_reference": None,
                "delta_vs_iteration0": 0.0 if objective_value is not None else None,
                "generated_code_path": baseline_generated_code_path,
                "model_config_path": str(model_config_path.resolve()) if isinstance(model_config_path, Path) else None,
                "benchmark_config_path": str(benchmark_config_path.resolve()),
                "run_dir": str(run_dir.resolve()),
                "manifest_path": str((run_dir / "manifest.json").resolve()),
                "metrics_path": str((run_dir / "metrics.json").resolve()),
                "requested_device": manifest.get("requested_device"),
                "resolved_device": manifest.get("resolved_device"),
                "model_uses_gpu": manifest.get("model_uses_gpu"),
                "seed_model_name": seed_model_name,
                "agent_mode": agent_mode,
                "protocol": manifest.get("protocol"),
                "evaluation_wall_clock_seconds": float(evaluation_wall_clock_seconds),
                "parent_iteration": 0,
                "parent_source_kind": (
                    "champion_baseline"
                    if isinstance(selected_champion_baseline, dict)
                    else "historical_codegen_parent"
                    if use_practical_warm_start
                    else "seed_model_source"
                ),
                "parent_selection_json": parent_candidates_json(practical_parent_selection),
                "champion_baseline_selection_json": parent_candidates_json(champion_baseline_selection),
                "mechanism_title": (
                    str(selected_champion_baseline.get("display_name"))
                    if isinstance(selected_champion_baseline, dict)
                    else
                    str(selected_practical.get("mechanism_title"))
                    if use_practical_warm_start and selected_practical.get("mechanism_title") is not None
                    else None
                ),
                "claimed_mechanism": (
                    f"champion baseline {selected_champion_baseline.get('method_name')}"
                    if isinstance(selected_champion_baseline, dict)
                    else
                    str(selected_practical.get("payload", {}).get("claimed_mechanism"))
                    if use_practical_warm_start and isinstance(selected_practical.get("payload"), dict)
                    else None
                ),
                "mechanism_card_path": None,
                "mechanism_consistency_passed": True,
                "mechanism_consistency_score": 1.0,
                "behavior_probe_passed": True,
                "behavior_probe_score": 1.0,
                "stability_probe_passed": True,
                "stability_probe_score": 1.0,
                "novelty_score": 0.0,
                "structural_signature": (
                    f"champion_baseline::{selected_champion_baseline.get('method_name')}"
                    if isinstance(selected_champion_baseline, dict)
                    else
                    str(selected_practical.get("structural_signature"))
                    if use_practical_warm_start and selected_practical.get("structural_signature") is not None
                    else "seed_model_baseline"
                ),
                "recommended_next_action": "breadth",
                "warm_start_strategy": (
                    "champion_baseline"
                    if isinstance(selected_champion_baseline, dict)
                    else "historical_codegen_parent"
                    if use_practical_warm_start
                    else "seed_model_baseline"
                ),
                "warm_start_used": bool(isinstance(selected_champion_baseline, dict) or use_practical_warm_start),
                "champion_baseline_method": (
                    str(selected_champion_baseline.get("method_name"))
                    if isinstance(selected_champion_baseline, dict)
                    else None
                ),
                "champion_baseline_mse_mean": (
                    float(selected_champion_baseline["mse_mean"])
                    if isinstance(selected_champion_baseline, dict) and isinstance(selected_champion_baseline.get("mse_mean"), (int, float))
                    else None
                ),
            }
            record.update(_flatten_selected(metrics, REPORT_METRIC_PATHS))
            history.append(record)
            dump_json(history_root / f"iter_{iteration:03d}.json", record)
            log(f"completed iteration=0 objective={objective_value}")
            iteration += 1
            continue

        archive_summary = _build_archive_summary(history)
        snapshot = _scheduler_snapshot_v8(history, archive_summary)
        decision = _choose_action_v8(snapshot)
        search_action = str(decision["action"])
        search_stage = str(decision["stage"])
        decision_reason = str(decision["decision_reason"])
        router_rows.append(
            {
                "iteration": int(iteration),
                "search_action": search_action,
                "search_stage": search_stage,
                "decision_reason": decision_reason,
                "snapshot_json": json.dumps(snapshot, ensure_ascii=True, sort_keys=True),
                "archive_summary_json": json.dumps(archive_summary, ensure_ascii=True, sort_keys=True),
                "parent_selection_json": parent_candidates_json(practical_parent_selection),
            }
        )
        parent_record = _archive_parent_record(history, search_action)
        reference_payload = _reference_code_payload(
            practical_parent_selection=practical_parent_selection,
            seed_model_source_path=seed_model_source_path,
            parent_record=parent_record,
        )
        selected_parent = reference_payload.get("parent_selection", {}).get("selected")
        parent_rows.append(
            {
                "iteration": int(iteration),
                "search_action": search_action,
                "search_stage": search_stage,
                "parent_source_kind": str(reference_payload["parent_source_kind"]),
                "parent_iteration": int(reference_payload["parent_iteration"]),
                "parent_code_path": str(reference_payload["parent_code_path"]),
                "archive_parent_iteration": (
                    int(parent_record["iteration"])
                    if parent_record is not None and isinstance(parent_record.get("iteration"), int)
                    else None
                ),
                "historical_parent_session_id": (
                    str(selected_parent.get("session_id")) if isinstance(selected_parent, dict) else None
                ),
                "historical_parent_agent_line": (
                    str(selected_parent.get("agent_line")) if isinstance(selected_parent, dict) else None
                ),
                "historical_parent_agent_mode": (
                    str(selected_parent.get("agent_mode")) if isinstance(selected_parent, dict) else None
                ),
                "historical_parent_objective": (
                    float(selected_parent["objective_value"])
                    if isinstance(selected_parent, dict) and isinstance(selected_parent.get("objective_value"), (int, float))
                    else None
                ),
                "parent_selection_json": parent_candidates_json(reference_payload["parent_selection"]),
            }
        )
        candidate_specs = _candidate_specs_for_route(task_group, search_action)
        candidate_pool: list[dict[str, Any]] = []
        candidate_rejections: list[dict[str, Any]] = []

        for candidate_spec in candidate_specs:
            candidate_kind = str(candidate_spec["candidate_kind"])
            prior = candidate_spec.get("prior")
            mechanism_card = _request_mechanism_card(
                llm_client=llm_client,
                dataset_key=dataset_key,
                task_group=task_group,
                iteration=int(iteration),
                route=search_action,
                stage=search_stage,
                candidate_kind=candidate_kind,
                seed_model_name=seed_model_name,
                archive_summary=archive_summary,
                history=history,
                parent_record=parent_record,
                prior=prior,
                champion_baseline=champion_baseline_selection.get("selected"),
            )
            for usage_event in llm_client.drain_usage_events():
                llm_usage_rows.append(
                    {
                        "iteration": int(iteration),
                        "candidate_kind": candidate_kind,
                        "request_phase": "mechanism_card",
                        **usage_event,
                    }
                )
            mechanism_card_path = mechanism_root / f"iter_{iteration:03d}_{candidate_kind}.json"
            dump_json(mechanism_card_path, mechanism_card)
            mechanism_rows.append(
                {
                    "iteration": int(iteration),
                    "search_action": search_action,
                    "search_stage": search_stage,
                    "candidate_kind": candidate_kind,
                    "mechanism_card_id": mechanism_card["mechanism_card_id"],
                    "title": mechanism_card["title"],
                    "claimed_mechanism": mechanism_card["claimed_mechanism"],
                    "edit_scope": mechanism_card["edit_scope"],
                    "risk_level": mechanism_card["risk_level"],
                    "structural_witnesses_json": json.dumps(mechanism_card["structural_witnesses"], ensure_ascii=True),
                    "behavioral_witnesses_json": json.dumps(mechanism_card["behavioral_witnesses"], ensure_ascii=True),
                    "prior_name": mechanism_card["prior_name"],
                }
            )

            prompts = _build_v8_code_prompts(
                dataset_key=dataset_key,
                task_group=task_group,
                route=search_action,
                stage=search_stage,
                iteration=int(iteration),
                candidate_kind=candidate_kind,
                seed_model_name=seed_model_name,
                mechanism_card=mechanism_card,
                archive_summary=archive_summary,
                history=history,
                parent_code=str(reference_payload["parent_code"]),
                parent_source_kind=str(reference_payload["parent_source_kind"]),
                previous_code=None,
                previous_error=None,
                prior=prior,
                dimension_summary=harness_batches["dimension_summary"],
                champion_baseline=champion_baseline_selection.get("selected"),
            )
            write_text(prompts_root / f"iter_{iteration:03d}_{candidate_kind}_system.txt", prompts["system_prompt"] + "\n")
            write_text(prompts_root / f"iter_{iteration:03d}_{candidate_kind}_user.txt", prompts["user_prompt"] + "\n")

            generated_code_path = generated_code_root / f"iter_{iteration:03d}_{candidate_kind}.py"
            raw_response_path = raw_response_root / f"iter_{iteration:03d}_{candidate_kind}.txt"
            review_path = review_root / f"iter_{iteration:03d}_{candidate_kind}.json"
            try:
                raw_response_text = llm_client.chat_text(
                    system_prompt=prompts["system_prompt"],
                    user_prompt=prompts["user_prompt"],
                )
                for usage_event in llm_client.drain_usage_events():
                    llm_usage_rows.append(
                        {
                            "iteration": int(iteration),
                            "candidate_kind": candidate_kind,
                            "request_phase": "codegen",
                            **usage_event,
                        }
                    )
                code_text = _extract_python_code(raw_response_text)
                write_text(raw_response_path, raw_response_text + "\n")
                write_text(generated_code_path, code_text.rstrip() + "\n")
                _load_generated_class(generated_code_path, V8_CLASS_NAME)
                review = _review_candidate_code(
                    parent_code=str(reference_payload["parent_code"]),
                    candidate_code=code_text,
                    mechanism_card=mechanism_card,
                    route=search_action,
                )
                precheck = _precheck_generated_candidate_v8(
                    generated_code_path=generated_code_path,
                    class_name=V8_CLASS_NAME,
                    mechanism_card=mechanism_card,
                    seed_model_config=seed_model_config,
                    harness_batches=harness_batches,
                )
                dump_json(review_path, review)
                review_rows.append(
                    {
                        "iteration": int(iteration),
                        "search_action": search_action,
                        "search_stage": search_stage,
                        "candidate_kind": candidate_kind,
                        "mechanism_card_id": mechanism_card["mechanism_card_id"],
                        "passed": review["passed"],
                        "mechanism_consistency_score": review["mechanism_consistency_score"],
                        "novelty_score": review["novelty_score"],
                        "structural_signature": review["structural_signature"],
                        "witness_hits_json": json.dumps(review["witness_hits"], ensure_ascii=True),
                        "missing_witnesses_json": json.dumps(review["missing_witnesses"], ensure_ascii=True),
                        "shortcut_flags_json": json.dumps(review["shortcut_flags"], ensure_ascii=True),
                        "precheck_passed": bool(precheck["passed"]),
                        "precheck_error": precheck["error"],
                        "precheck_logs_json": json.dumps(precheck["log_lines"], ensure_ascii=True),
                    }
                )
                candidate_pool_rows.append(
                    {
                        "iteration": int(iteration),
                        "search_action": search_action,
                        "search_stage": search_stage,
                        "candidate_kind": candidate_kind,
                        "mechanism_card_id": mechanism_card["mechanism_card_id"],
                        "title": mechanism_card["title"],
                        "claimed_mechanism": mechanism_card["claimed_mechanism"],
                        "mechanism_consistency_score": review["mechanism_consistency_score"],
                        "novelty_score": review["novelty_score"],
                        "cheap_score": _cheap_candidate_score(
                            route=search_action,
                            candidate_kind=candidate_kind,
                            review=review,
                            mechanism_card=mechanism_card,
                        ),
                        "passed_static_review": review["passed"],
                        "generated_code_path": str(generated_code_path.resolve()),
                        "mechanism_card_path": str(mechanism_card_path.resolve()),
                        "review_path": str(review_path.resolve()),
                        "prior_name": mechanism_card["prior_name"],
                        "precheck_passed": bool(precheck["passed"]),
                        "precheck_error": precheck["error"],
                        "precheck_logs_json": json.dumps(precheck["log_lines"], ensure_ascii=True),
                    }
                )
                if review["passed"] and bool(precheck["passed"]):
                    candidate_pool.append(
                        {
                            "candidate_kind": candidate_kind,
                            "mechanism_card": mechanism_card,
                            "mechanism_card_path": mechanism_card_path,
                            "generated_code_path": generated_code_path,
                            "review": review,
                            "precheck": precheck,
                            "cheap_score": _cheap_candidate_score(
                                route=search_action,
                                candidate_kind=candidate_kind,
                                review=review,
                                mechanism_card=mechanism_card,
                            ),
                        }
                    )
                else:
                    candidate_rejections.append(
                        {
                            "iteration": int(iteration),
                            "search_action": search_action,
                            "search_stage": search_stage,
                            "candidate_kind": candidate_kind,
                            "reason": "mechanism_code_mismatch" if not review["passed"] else "precheck_failed",
                            "review_path": str(review_path.resolve()),
                            "mechanism_card_id": mechanism_card["mechanism_card_id"],
                            "precheck_error": precheck["error"],
                        }
                    )
            except Exception as exc:
                for usage_event in llm_client.drain_usage_events():
                    llm_usage_rows.append(
                        {
                            "iteration": int(iteration),
                            "candidate_kind": candidate_kind,
                            "request_phase": "codegen",
                            **usage_event,
                        }
                    )
                review = {
                    "passed": False,
                    "mechanism_consistency_score": 0.0,
                    "novelty_score": 0.0,
                    "structural_signature": "codegen_failed",
                    "witness_hits": [],
                    "missing_witnesses": mechanism_card.get("structural_witnesses", []),
                    "shortcut_flags": ["code_generation_failed", type(exc).__name__],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                dump_json(review_path, review)
                candidate_rejections.append(
                    {
                        "iteration": int(iteration),
                        "search_action": search_action,
                        "search_stage": search_stage,
                        "candidate_kind": candidate_kind,
                        "reason": f"{type(exc).__name__}: {exc}",
                        "review_path": str(review_path.resolve()),
                        "mechanism_card_id": mechanism_card["mechanism_card_id"],
                    }
                )

        rejection_rows.extend(candidate_rejections)
        dump_json(candidate_root / f"iter_{iteration:03d}_candidate_pool.json", {"rows": candidate_pool_rows[-len(candidate_specs):]})
        if not candidate_pool:
            stopping_reason = f"no_valid_v8_{search_action}_candidate"
            log(f"stopping iteration={iteration} reason={stopping_reason}")
            break

        candidate_pool.sort(key=lambda item: (-float(item["cheap_score"]), str(item["candidate_kind"])))
        selected = candidate_pool[0]
        generated_code_path = Path(selected["generated_code_path"])
        mechanism_card = dict(selected["mechanism_card"])
        review = dict(selected["review"])
        mechanism_card_path = Path(selected["mechanism_card_path"])
        model_config_path = model_configs_root / f"iter_{iteration:03d}.yaml"
        benchmark_config_path = benchmark_configs_root / f"iter_{iteration:03d}.yaml"

        model_payload = dict(seed_model_config)
        model_payload["name"] = f"generated_direct_code::{_safe_slug(agent_mode)}"
        model_payload["factory"] = {
            "module_path": str(generated_code_path.resolve()),
            "class_name": V8_CLASS_NAME,
        }
        model_payload["seed_model_name"] = seed_model_name
        model_payload["agent_mode"] = agent_mode
        model_payload["proposal_note"] = mechanism_card["proposal_note"]
        model_payload["mechanism_card"] = mechanism_card
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
            f"dispatch iteration={iteration} action={search_action} stage={search_stage} "
            f"candidate_kind={selected['candidate_kind']} title={mechanism_card['title']}"
        )
        try:
            incumbent_record = _best_completed_record(history)
            incumbent_objective = (
                float(incumbent_record["objective_value"])
                if incumbent_record is not None and isinstance(incumbent_record.get("objective_value"), (int, float))
                else baseline_objective
            )
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
                raise RuntimeError(stderr_text or stdout_text or "benchmark_failed_without_error_output")
            manifest = load_json(run_dir / "manifest.json")
            metrics = load_json(run_dir / "metrics.json")
            train_log_path = run_dir / "train.log"
            train_log_text = train_log_path.read_text(encoding="utf-8") if train_log_path.exists() else ""
            objective_raw = _get_nested(metrics, OBJECTIVE_PATH)
            objective_value = float(objective_raw) if isinstance(objective_raw, (int, float)) else None
            reference_objective_value = incumbent_objective
            behavior_probe = _run_behavior_probes(
                mechanism_card=mechanism_card,
                review=review,
                metrics_payload=metrics,
                train_log_text=train_log_text,
                reference_record=parent_record,
            )
            stability_probe = _stability_probe(
                train_log_text=train_log_text,
                metrics_payload=metrics,
                reference_record=parent_record,
            )
            delta_vs_reference = (
                float(reference_objective_value) - float(objective_value)
                if reference_objective_value is not None and objective_value is not None
                else None
            )
            behavior_rows.append(
                {
                    "iteration": int(iteration),
                    "search_action": search_action,
                    "search_stage": search_stage,
                    "candidate_kind": str(selected["candidate_kind"]),
                    "mechanism_card_id": mechanism_card["mechanism_card_id"],
                    "mechanism_title": mechanism_card["title"],
                    "passed": bool(behavior_probe["passed"]),
                    "score": float(behavior_probe["score"]),
                    "results_json": summarize_behavior_probe_for_trace(behavior_probe),
                    "details_json": json.dumps(behavior_probe.get("details", {}), ensure_ascii=True, sort_keys=True),
                    "stable_training": bool(behavior_probe.get("stable_training")),
                    "stability_probe_passed": bool(stability_probe["passed"]),
                    "stability_probe_score": float(stability_probe["score"]),
                    "stability_probe_details_json": json.dumps(stability_probe, ensure_ascii=True, sort_keys=True),
                }
            )
            record = {
                "iteration": int(iteration),
                "phase": "agent",
                "search_action": search_action,
                "search_stage": search_stage,
                "proposal_source": "v8_open_mechanism_codegen",
                "proposal_note": mechanism_card["proposal_note"],
                "candidate_kind": str(selected["candidate_kind"]),
                "execution_status": str(manifest.get("execution_status")),
                "objective_path": OBJECTIVE_PATH,
                "objective_value": objective_value,
                "reference_objective_value": reference_objective_value,
                "delta_vs_reference": delta_vs_reference,
                "delta_vs_iteration0": (
                    float(baseline_objective) - float(objective_value)
                    if baseline_objective is not None and objective_value is not None
                    else None
                ),
                "generated_code_path": str(generated_code_path.resolve()),
                "model_config_path": str(model_config_path.resolve()),
                "benchmark_config_path": str(benchmark_config_path.resolve()),
                "run_dir": str(run_dir.resolve()),
                "manifest_path": str((run_dir / "manifest.json").resolve()),
                "metrics_path": str((run_dir / "metrics.json").resolve()),
                "requested_device": manifest.get("requested_device"),
                "resolved_device": manifest.get("resolved_device"),
                "model_uses_gpu": manifest.get("model_uses_gpu"),
                "seed_model_name": seed_model_name,
                "agent_mode": agent_mode,
                "protocol": manifest.get("protocol"),
                "evaluation_wall_clock_seconds": float(evaluation_wall_clock_seconds),
                "parent_iteration": int(reference_payload["parent_iteration"]),
                "parent_source_kind": str(reference_payload["parent_source_kind"]),
                "parent_selection_json": parent_candidates_json(reference_payload["parent_selection"]),
                "champion_baseline_selection_json": parent_candidates_json(champion_baseline_selection),
                "mechanism_title": mechanism_card["title"],
                "claimed_mechanism": mechanism_card["claimed_mechanism"],
                "mechanism_card_path": str(mechanism_card_path.resolve()),
                "mechanism_consistency_passed": bool(review["passed"]),
                "mechanism_consistency_score": float(review["mechanism_consistency_score"]),
                "behavior_probe_passed": bool(behavior_probe["passed"]),
                "behavior_probe_score": float(behavior_probe["score"]),
                "stability_probe_passed": bool(stability_probe["passed"]),
                "stability_probe_score": float(stability_probe["score"]),
                "novelty_score": float(review["novelty_score"]),
                "structural_signature": str(review["structural_signature"]),
                "selected_structural_witnesses_json": json.dumps(mechanism_card["structural_witnesses"], ensure_ascii=True),
                "behavior_probe_results_json": json.dumps(behavior_probe["results"], ensure_ascii=True, sort_keys=True),
                "behavior_probe_summary_json": summarize_behavior_probe_for_trace(behavior_probe),
                "behavior_probe_details_json": json.dumps(behavior_probe.get("details", {}), ensure_ascii=True, sort_keys=True),
                "stability_probe_json": json.dumps(stability_probe, ensure_ascii=True, sort_keys=True),
                "consistency_review_path": str((review_root / f"iter_{iteration:03d}_{selected['candidate_kind']}.json").resolve()),
                "recommended_next_action": _recommended_next_action(
                    execution_status=str(manifest.get("execution_status")),
                    delta_vs_reference=delta_vs_reference,
                    mechanism_consistency_passed=bool(review["passed"]),
                    behavior_probe_passed=bool(behavior_probe["passed"]),
                    stability_probe_passed=bool(stability_probe["passed"]),
                    novelty_score=float(review["novelty_score"]),
                ),
            }
            record.update(_flatten_selected(metrics, REPORT_METRIC_PATHS))
        except Exception as exc:
            incumbent_record = _best_completed_record(history)
            incumbent_objective = (
                float(incumbent_record["objective_value"])
                if incumbent_record is not None and isinstance(incumbent_record.get("objective_value"), (int, float))
                else baseline_objective
            )
            behavior_rows.append(
                {
                    "iteration": int(iteration),
                    "search_action": search_action,
                    "search_stage": search_stage,
                    "candidate_kind": str(selected["candidate_kind"]),
                    "mechanism_card_id": mechanism_card["mechanism_card_id"],
                    "mechanism_title": mechanism_card["title"],
                    "passed": False,
                    "score": 0.0,
                    "results_json": summarize_behavior_probe_for_trace({"passed": False, "score": 0.0, "results": {}}),
                    "details_json": json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=True, sort_keys=True),
                    "stable_training": False,
                    "stability_probe_passed": False,
                    "stability_probe_score": 0.0,
                    "stability_probe_details_json": json.dumps({}, ensure_ascii=True, sort_keys=True),
                }
            )
            record = {
                "iteration": int(iteration),
                "phase": "agent",
                "search_action": search_action,
                "search_stage": search_stage,
                "proposal_source": "v8_open_mechanism_codegen",
                "proposal_note": mechanism_card["proposal_note"],
                "candidate_kind": str(selected["candidate_kind"]),
                "execution_status": "failed",
                "objective_path": OBJECTIVE_PATH,
                "objective_value": None,
                "reference_objective_value": (
                    incumbent_objective
                ),
                "delta_vs_reference": None,
                "delta_vs_iteration0": None,
                "generated_code_path": str(generated_code_path.resolve()),
                "model_config_path": str(model_config_path.resolve()),
                "benchmark_config_path": str(benchmark_config_path.resolve()),
                "run_dir": None,
                "manifest_path": None,
                "metrics_path": None,
                "requested_device": requested_device,
                "resolved_device": None,
                "model_uses_gpu": None,
                "seed_model_name": seed_model_name,
                "agent_mode": agent_mode,
                "protocol": None,
                "evaluation_wall_clock_seconds": None,
                "parent_iteration": int(reference_payload["parent_iteration"]),
                "parent_source_kind": str(reference_payload["parent_source_kind"]),
                "parent_selection_json": parent_candidates_json(reference_payload["parent_selection"]),
                "champion_baseline_selection_json": parent_candidates_json(champion_baseline_selection),
                "mechanism_title": mechanism_card["title"],
                "claimed_mechanism": mechanism_card["claimed_mechanism"],
                "mechanism_card_path": str(mechanism_card_path.resolve()),
                "mechanism_consistency_passed": bool(review["passed"]),
                "mechanism_consistency_score": float(review["mechanism_consistency_score"]),
                "behavior_probe_passed": False,
                "behavior_probe_score": 0.0,
                "stability_probe_passed": False,
                "stability_probe_score": 0.0,
                "novelty_score": float(review["novelty_score"]),
                "structural_signature": str(review["structural_signature"]),
                "selected_structural_witnesses_json": json.dumps(mechanism_card["structural_witnesses"], ensure_ascii=True),
                "behavior_probe_results_json": json.dumps({}, ensure_ascii=True),
                "behavior_probe_summary_json": summarize_behavior_probe_for_trace({"passed": False, "score": 0.0, "results": {}}),
                "behavior_probe_details_json": json.dumps({}, ensure_ascii=True, sort_keys=True),
                "stability_probe_json": json.dumps({}, ensure_ascii=True, sort_keys=True),
                "consistency_review_path": str((review_root / f"iter_{iteration:03d}_{selected['candidate_kind']}.json").resolve()),
                "recommended_next_action": "constrain",
                "error_tail": f"{type(exc).__name__}: {exc}",
            }
            for metric_path in REPORT_METRIC_PATHS:
                record[metric_path] = None
            log(f"candidate_failed iteration={iteration} reason={type(exc).__name__}: {exc}")

        history.append(record)
        dump_json(history_root / f"iter_{iteration:03d}.json", record)
        log(f"completed iteration={iteration} objective={record.get('objective_value')}")
        iteration += 1

    completed = [
        item
        for item in history
        if str(item.get("execution_status")) == "completed" and item.get("objective_value") is not None
    ]
    if not completed:
        raise RuntimeError(f"No completed V8 iterations for {dataset_key}/{seed_model_name}")
    best_record = min(completed, key=lambda item: float(item["objective_value"]))
    if stopping_reason is None:
        stopping_reason = default_stopping_reason(history, budget_limit)

    completed_count = completed_evaluation_count(history)
    failed_candidate_count = int(failed_history_count(history) + len(rejection_rows))
    candidate_attempt_count = int(completed_count + failed_candidate_count)
    session_wall_clock_seconds = float(prior_session_wall_clock_seconds + (perf_counter() - session_started_at))
    llm_usage_summary = aggregate_llm_usage(llm_usage_rows)
    archive_summary = _build_archive_summary(history)

    for record in history:
        record["best_so_far_iteration"] = int(best_record["iteration"])
        record["best_so_far_objective"] = float(best_record["objective_value"])
        record["accepted_as_best"] = bool(record["iteration"] == best_record["iteration"])

    dump_json(session_root / "iterations.json", {"iterations": history})
    fieldnames = sorted({key for row in history for key in row.keys()})
    with (session_root / "iterations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    dump_json(session_root / "best_iteration.json", best_record)
    dump_json(trace_root / "mechanism_cards.json", {"rows": mechanism_rows})
    if mechanism_rows:
        _write_trace_csv(trace_root / "mechanism_cards.csv", mechanism_rows)
    dump_json(trace_root / "consistency_reviews.json", {"rows": review_rows})
    if review_rows:
        _write_trace_csv(trace_root / "consistency_reviews.csv", review_rows)
    dump_json(trace_root / "action_router.json", {"rows": router_rows})
    if router_rows:
        _write_trace_csv(trace_root / "action_router.csv", router_rows)
    dump_json(
        trace_root / "parent_selection.json",
        {
            "rows": parent_rows,
            "session_selection": practical_parent_selection,
            "champion_baseline_selection": champion_baseline_selection,
        },
    )
    if parent_rows:
        _write_trace_csv(trace_root / "parent_selection.csv", parent_rows)
    dump_json(trace_root / "behavior_probes.json", {"rows": behavior_rows})
    if behavior_rows:
        _write_trace_csv(trace_root / "behavior_probes.csv", behavior_rows)
    dump_json(trace_root / "candidate_pools.json", {"rows": candidate_pool_rows})
    if candidate_pool_rows:
        _write_trace_csv(trace_root / "candidate_pools.csv", candidate_pool_rows)
    dump_json(trace_root / "archive.json", archive_summary)
    dump_json(trace_root / "proposal_rejections.json", {"rows": rejection_rows})
    if rejection_rows:
        _write_trace_csv(trace_root / "proposal_rejections.csv", rejection_rows)
    dump_json(trace_root / "llm_usage.json", {"rows": llm_usage_rows, "summary": llm_usage_summary})
    if llm_usage_rows:
        _write_trace_csv(trace_root / "llm_usage.csv", llm_usage_rows)

    dump_json(
        session_root / "agent_session.json",
        {
            "dataset_key": dataset_key,
            "method_family": "agent_structured_model",
            "method_name": method_name,
            "agent_line": agent_line,
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
            "llm_strategy": "open_mechanism_codegen",
            "harness_mode": harness_mode,
            "warm_start_strategy": history[0].get("warm_start_strategy"),
            "warm_start_used": bool(history and history[0].get("warm_start_used")),
            "task_group": task_group,
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
            "best_structural_signature": best_record.get("structural_signature"),
            "best_mechanism_title": best_record.get("mechanism_title"),
            "objective_path": OBJECTIVE_PATH,
            "llm_request_count": int(llm_usage_summary["llm_request_count"]),
            "llm_prompt_tokens": int(llm_usage_summary["llm_prompt_tokens"]),
            "llm_completion_tokens": int(llm_usage_summary["llm_completion_tokens"]),
            "llm_total_tokens": int(llm_usage_summary["llm_total_tokens"]),
            "llm_repair_request_count": int(llm_usage_summary["llm_repair_request_count"]),
            "archive_summary": archive_summary,
            "practical_parent_selection": practical_parent_selection,
            "champion_baseline_selection": champion_baseline_selection,
        },
    )
    write_text(session_root / "agent.log", "\n".join(log_lines) + "\n")

    baseline_record = history[0]
    return {
        "dataset_key": dataset_key,
        "method_name": method_name,
        "method_family": "agent_structured_model",
        "agent_line": agent_line,
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
        "llm_strategy": "open_mechanism_codegen",
        "llm_enabled": True,
        "harness_mode": harness_mode,
        "warm_start_strategy": baseline_record.get("warm_start_strategy"),
        "warm_start_used": bool(history and history[0].get("warm_start_used")),
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
        "best_mechanism_title": best_record.get("mechanism_title"),
        "best_structural_signature": best_record.get("structural_signature"),
        **{f"baseline.{metric}": baseline_record.get(metric) for metric in REPORT_METRIC_PATHS},
        **{f"best.{metric}": best_record.get(metric) for metric in REPORT_METRIC_PATHS},
    }
