from __future__ import annotations

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence

from sci_response.agent.compiler import compile_model_ir, render_generated_model_source
from sci_response.agent.edits import apply_edit_sequence, describe_edit
from sci_response.agent.harness import build_harness_review, changed_ir_paths, infer_hypothesis_axes
from sci_response.agent.ir import enumerate_editable_sites, extract_model_ir
from sci_response.agent.matched_budget import (
    MATCHED_BUDGET_STOPPING_RULE,
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
from sci_response.baselines.artifacts import beijing_timestamp_slug
from sci_response.data.io import dump_json, dump_yaml, ensure_dir, load_json, load_yaml, stable_hash, write_text


def _task_group(dataset_key: str) -> str:
    key = str(dataset_key).lower()
    if key in {"norman", "adamson", "sciplex3"}:
        return "single_cell"
    if key == "papalexi_arrayed_protein":
        return "multimodal_protein"
    if key == "papalexi_arrayed_rna":
        return "multimodal_rna"
    if key == "l1000_public":
        return "dose_time"
    if key == "cdsdb":
        return "paired_clinical"
    return "general"


def _site_map(model_ir: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {str(site["path"]): dict(site) for site in enumerate_editable_sites(model_ir)}


def _dedupe_edits(edits: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for edit in edits:
        marker = (
            str(edit.get("primitive")),
            str(edit.get("path")),
            repr(edit.get("value")),
            repr(edit.get("delta")),
            repr(edit.get("factor")),
            repr(edit.get("choices")),
            repr(edit.get("step")),
        )
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(dict(edit))
    return deduped


def _set_scalar_edit(model_ir: Dict[str, Any], path: str, value: int | float) -> Dict[str, Any] | None:
    site = _site_map(model_ir).get(path)
    if site is None:
        return None
    if "set_scalar" not in set(str(item) for item in site.get("primitive_family", [])):
        return None
    current_value = site.get("current_value")
    if current_value == value:
        return None
    if str(site.get("type")) == "int":
        value = int(max(1, round(float(value))))
    elif str(site.get("type")) == "float":
        value = float(value)
    return {"primitive": "set_scalar", "path": path, "value": value}


def _toggle_to_edit(model_ir: Dict[str, Any], path: str, target: bool) -> Dict[str, Any] | None:
    site = _site_map(model_ir).get(path)
    if site is None:
        return None
    if "toggle_boolean" not in set(str(item) for item in site.get("primitive_family", [])):
        return None
    current_value = bool(site.get("current_value"))
    if current_value == bool(target):
        return None
    return {"primitive": "toggle_boolean", "path": path}


def _enum_to_edit(model_ir: Dict[str, Any], path: str, target: str) -> Dict[str, Any] | None:
    site = _site_map(model_ir).get(path)
    if site is None:
        return None
    if "cycle_enum" not in set(str(item) for item in site.get("primitive_family", [])):
        return None
    choices = [str(item) for item in site.get("choices", [])]
    current_value = str(site.get("current_value"))
    if current_value == str(target) or current_value not in choices or str(target) not in choices:
        return None
    current_index = choices.index(current_value)
    target_index = choices.index(str(target))
    step = (target_index - current_index) % len(choices)
    if step == 0:
        return None
    return {
        "primitive": "cycle_enum",
        "path": path,
        "choices": choices,
        "step": int(step),
    }


def _proposal_payload(
    *,
    proposal_source: str,
    proposal_note: str,
    scientific_hypothesis: str,
    mechanistic_rationale: str,
    edits: Sequence[Dict[str, Any]],
    reference_ir: Dict[str, Any],
    candidate_seed_model_name: str,
) -> Dict[str, Any]:
    deduped = _dedupe_edits(edits)
    candidate_ir = apply_edit_sequence(reference_ir, deduped) if deduped else reference_ir
    hypothesis_axes = infer_hypothesis_axes(changed_ir_paths(reference_ir, candidate_ir))
    return {
        "proposal_source": proposal_source,
        "proposal_note": proposal_note,
        "scientific_hypothesis": scientific_hypothesis,
        "mechanistic_rationale": mechanistic_rationale,
        "selected_mechanism_templates": hypothesis_axes,
        "risk_notes": [],
        "edits": deduped,
        "candidate_ir": candidate_ir,
        "ir_hash": stable_hash(candidate_ir),
        "candidate_seed_model_name": str(candidate_seed_model_name),
    }


def _seed_family_baseline_proposal(
    *,
    reference_ir: Dict[str, Any],
    seed_ir: Dict[str, Any],
    seed_model_name: str,
) -> Dict[str, Any]:
    return {
        "proposal_source": "seed_family_baseline_v3",
        "proposal_note": f"seed_family_baseline::{seed_model_name}",
        "selected_mechanism_templates": [],
        "scientific_hypothesis": f"Evaluate {seed_model_name} as a direct seed-family starting point.",
        "mechanistic_rationale": (
            "V3 first compares a small set of strong seed families, then spends the remaining budget only around the best one."
        ),
        "risk_notes": [],
        "edits": [],
        "candidate_ir": dict(seed_ir),
        "ir_hash": stable_hash(seed_ir),
        "candidate_seed_model_name": str(seed_model_name),
    }


def _portfolio_candidates(reference_ir: Dict[str, Any], dataset_key: str, candidate_seed_model_name: str) -> List[Dict[str, Any]]:
    rep = dict(reference_ir["representation"])
    trunk = dict(rep["trunk"])
    conditioning = dict(rep["conditioning"])
    loss = dict(rep["loss"])
    optimizer = dict(rep["optimizer"])
    group = _task_group(dataset_key)

    lr_tight = min(float(optimizer.get("learning_rate", 8e-4)), 6e-4)
    lr_tighter = min(float(optimizer.get("learning_rate", 8e-4)), 5e-4)
    wd_up = max(float(optimizer.get("weight_decay", 1e-4)), 2e-4)
    batch_down = min(int(optimizer.get("batch_size", 128)), 96)
    hidden_mid = max(int(trunk.get("hidden_dim", 128)), 160)
    hidden_big = max(int(trunk.get("hidden_dim", 128)), 192)
    cond_mid = max(int(conditioning.get("conditioning_dim", 64)), 96)
    cond_big = max(int(conditioning.get("conditioning_dim", 64)), 128)
    response_low = min(float(loss.get("response_weight", 0.35)), 0.2)
    response_mid = min(float(loss.get("response_weight", 0.35)), 0.25)

    common_capacity = [
        _set_scalar_edit(reference_ir, "representation.trunk.hidden_dim", hidden_big),
        _set_scalar_edit(reference_ir, "representation.trunk.trunk_depth", max(int(trunk.get("trunk_depth", 1)), 2)),
        _set_scalar_edit(reference_ir, "representation.trunk.residual_depth", max(int(trunk.get("residual_depth", 1)), 2)),
    ]
    common_regularize = [
        _set_scalar_edit(reference_ir, "representation.optimizer.learning_rate", lr_tight),
        _set_scalar_edit(reference_ir, "representation.optimizer.weight_decay", wd_up),
        _set_scalar_edit(reference_ir, "representation.optimizer.batch_size", batch_down),
    ]
    common_stabilize = [
        _toggle_to_edit(reference_ir, "representation.prediction.zero_init_head", True),
        _set_scalar_edit(reference_ir, "representation.loss.response_weight", response_low),
    ]
    context_film = [
        _enum_to_edit(reference_ir, "representation.conditioning.mode", "film"),
        _set_scalar_edit(reference_ir, "representation.conditioning.conditioning_dim", cond_mid),
        _set_scalar_edit(reference_ir, "representation.trunk.hidden_dim", hidden_mid),
    ]

    group_programs: Dict[str, List[tuple[str, str, str, List[Dict[str, Any] | None]]]] = {
        "single_cell": [
            (
                "context_film_capacity",
                "Use stronger context conditioning and a wider trunk for single-cell perturbation response.",
                "Single-cell tasks benefit from sharper perturbation-context modulation before heavy optimizer tuning.",
                context_film,
            ),
            (
                "deeper_capacity",
                "Increase trunk and residual capacity to fit richer perturbation-response structure.",
                "A small but explicit capacity jump is often the easiest way to beat the default seed on CRISPR tasks.",
                common_capacity,
            ),
            (
                "zero_init_delta_focus",
                "Encourage sparse, baseline-anchored response updates.",
                "Single-cell perturbations often benefit from a conservative response head with lower response loss weight.",
                common_stabilize,
            ),
            (
                "optimizer_tighten",
                "Regularize optimization after the structural probes.",
                "If structure is already reasonable, slightly smaller learning rate and smaller batch size often help.",
                common_regularize,
            ),
        ],
        "multimodal_protein": [
            (
                "deeper_capacity",
                "Increase shared capacity before changing supervision aggressively.",
                "Protein readouts appear to reward stronger latent capacity more than extra prompt logic.",
                common_capacity,
            ),
            (
                "zero_init_delta_focus",
                "Stabilize the response head and lower response loss weight.",
                "Protein perturbation tasks frequently improve with conservative response updates.",
                common_stabilize,
            ),
            (
                "context_film_capacity",
                "Upgrade conditioning strength without abandoning the current structured family.",
                "A stronger context operator can matter even when the default seed is already competitive.",
                context_film,
            ),
            (
                "optimizer_tighten",
                "Regularize learning once the structure is expanded.",
                "Smaller learning rate and batch size can preserve gains from added capacity.",
                common_regularize,
            ),
        ],
        "multimodal_rna": [
            (
                "zero_init_response_down",
                "Reduce response overfitting and stabilize the prediction head.",
                "RNA paired-response tasks looked under-regularized in V1/V2, so this portfolio starts with a conservative head.",
                [
                    _toggle_to_edit(reference_ir, "representation.prediction.zero_init_head", True),
                    _set_scalar_edit(reference_ir, "representation.loss.response_weight", response_mid),
                    _set_scalar_edit(reference_ir, "representation.optimizer.learning_rate", lr_tighter),
                ],
            ),
            (
                "context_film_capacity",
                "Strengthen context routing and widen the trunk together.",
                "If the failure mode is under-conditioning, moving to FiLM with a wider hidden state is the cleanest legal structural jump.",
                context_film,
            ),
            (
                "deeper_capacity",
                "Try a deeper residual trunk for multimodal RNA perturbation response.",
                "This is the main capacity-only probe before optimizer refinements.",
                common_capacity,
            ),
            (
                "optimizer_tighten",
                "Tighten optimizer settings after the structural probes.",
                "Lower learning rate and modest weight decay are a cheap hedge against unstable later iterations.",
                common_regularize,
            ),
        ],
        "dose_time": [
            (
                "film_big_context",
                "Use FiLM-style conditioning with a larger context bottleneck.",
                "Dose/time tasks are the clearest case where stronger explicit conditioning should help.",
                [
                    _enum_to_edit(reference_ir, "representation.conditioning.mode", "film"),
                    _set_scalar_edit(reference_ir, "representation.conditioning.conditioning_dim", cond_big),
                    _set_scalar_edit(reference_ir, "representation.trunk.hidden_dim", hidden_big),
                ],
            ),
            (
                "optimizer_for_long_runs",
                "Shrink learning rate and batch size for the long-horizon L1000 training regime.",
                "Large public drug-response tasks reward boring but stable optimization settings.",
                common_regularize,
            ),
            (
                "deeper_capacity",
                "Try a moderately deeper trunk after the context upgrade.",
                "If conditioning alone is not enough, extra capacity is the next simple move.",
                common_capacity,
            ),
            (
                "zero_init_delta_focus",
                "Make the response head more conservative.",
                "This final bootstrap move is useful when the model over-shoots on broad-response tasks.",
                common_stabilize,
            ),
        ],
        "paired_clinical": [
            (
                "paired_anchor",
                "Explicitly enforce paired-response anchoring and a conservative response head.",
                "Clinical pre/post response needs anchored delta-style behavior before capacity tweaks.",
                [
                    _toggle_to_edit(reference_ir, "representation.prediction.baseline_skip", True),
                    _enum_to_edit(reference_ir, "representation.prediction.target", "delta"),
                    _toggle_to_edit(reference_ir, "representation.prediction.zero_init_head", True),
                    _set_scalar_edit(reference_ir, "representation.loss.response_weight", response_mid),
                ],
            ),
            (
                "optimizer_tighten",
                "Use more conservative optimizer settings for paired clinical data.",
                "This branch borrows the one thing HPO consistently did better than V1: tighter optimization.",
                common_regularize,
            ),
            (
                "context_film_capacity",
                "Add stronger context routing on top of the paired anchor.",
                "Clinical state shift should benefit from more expressive conditioning after anchoring is in place.",
                context_film,
            ),
            (
                "deeper_capacity",
                "Increase model capacity only after anchoring and optimizer regularization.",
                "This is the final bootstrap direction before local exploitation.",
                common_capacity,
            ),
        ],
    }
    programs = group_programs.get(group, group_programs["single_cell"])
    proposals: List[Dict[str, Any]] = []
    for macro_name, hypothesis, rationale, raw_edits in programs:
        edits = [edit for edit in raw_edits if edit is not None]
        if not edits:
            continue
        proposals.append(
            _proposal_payload(
                proposal_source="portfolio_v3",
                proposal_note=macro_name,
                scientific_hypothesis=hypothesis,
                mechanistic_rationale=rationale,
                edits=edits,
                reference_ir=reference_ir,
                candidate_seed_model_name=candidate_seed_model_name,
            )
        )
    return proposals


def _local_refinement_candidates(best_ir: Dict[str, Any], dataset_key: str, candidate_seed_model_name: str) -> List[Dict[str, Any]]:
    rep = dict(best_ir["representation"])
    trunk = dict(rep["trunk"])
    conditioning = dict(rep["conditioning"])
    loss = dict(rep["loss"])
    optimizer = dict(rep["optimizer"])
    group = _task_group(dataset_key)

    candidates: List[tuple[str, str, str, List[Dict[str, Any] | None]]] = [
        (
            "refine_hidden_dim",
            "Adjust hidden width around the current best configuration.",
            "A small capacity step often captures the remaining easy gain after the initial portfolio stage.",
            [
                _set_scalar_edit(best_ir, "representation.trunk.hidden_dim", int(trunk.get("hidden_dim", 128)) + 32),
            ],
        ),
        (
            "refine_learning_rate",
            "Lower the learning rate around the current best configuration.",
            "This is the simplest way to stabilize later-iteration improvements without changing model semantics.",
            [
                _set_scalar_edit(
                    best_ir,
                    "representation.optimizer.learning_rate",
                    min(float(optimizer.get("learning_rate", 8e-4)), 5e-4),
                ),
            ],
        ),
        (
            "refine_conditioning_dim",
            "Increase conditioning width around the current best.",
            "Context-heavy tasks often still improve from a slightly larger conditioning bottleneck.",
            [
                _set_scalar_edit(best_ir, "representation.conditioning.conditioning_dim", int(conditioning.get("conditioning_dim", 64)) + 32),
            ],
        ),
        (
            "refine_response_weight",
            "Reduce response loss weight to sharpen delta prediction.",
            "When structure is already good, this is the cleanest last-stage supervision adjustment.",
            [
                _set_scalar_edit(best_ir, "representation.loss.response_weight", min(float(loss.get("response_weight", 0.35)), 0.2)),
            ],
        ),
        (
            "refine_depth",
            "Increase depth one more step around the current best.",
            "If the best model is still capacity-limited, an extra depth increment is a high-yield final probe.",
            [
                _set_scalar_edit(best_ir, "representation.trunk.trunk_depth", int(trunk.get("trunk_depth", 1)) + 1),
                _set_scalar_edit(best_ir, "representation.trunk.residual_depth", int(trunk.get("residual_depth", 1)) + 1),
            ],
        ),
    ]
    if group in {"dose_time", "paired_clinical"}:
        candidates.insert(
            0,
            (
                "refine_batch_size",
                "Lower batch size for the larger-context regime.",
                "The long-run structured models on L1000/CDSDB benefited from more conservative optimization in prior branches.",
                [
                    _set_scalar_edit(best_ir, "representation.optimizer.batch_size", min(int(optimizer.get("batch_size", 128)), 96)),
                    _set_scalar_edit(best_ir, "representation.optimizer.weight_decay", max(float(optimizer.get("weight_decay", 1e-4)), 2e-4)),
                ],
            ),
        )
    if group == "multimodal_protein":
        candidates.insert(
            0,
            (
                "refine_zero_init",
                "Keep the protein response head conservative while widening capacity.",
                "Protein readouts looked strongest when the response head stayed close to zero at initialization.",
                [
                    _toggle_to_edit(best_ir, "representation.prediction.zero_init_head", True),
                    _set_scalar_edit(best_ir, "representation.trunk.hidden_dim", int(trunk.get("hidden_dim", 128)) + 48),
                ],
            ),
        )
    proposals: List[Dict[str, Any]] = []
    for macro_name, hypothesis, rationale, raw_edits in candidates:
        edits = [edit for edit in raw_edits if edit is not None]
        if not edits:
            continue
        proposals.append(
            _proposal_payload(
                proposal_source="portfolio_v3_refine",
                proposal_note=macro_name,
                scientific_hypothesis=hypothesis,
                mechanistic_rationale=rationale,
                edits=edits,
                reference_ir=best_ir,
                candidate_seed_model_name=candidate_seed_model_name,
            )
        )
    return proposals


def run_structured_model_session_v3(
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
    agent_mode: str = "structured_portfolio_search_v3",
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
        ir_payload = extract_model_ir(config_payload)
        seed_family_specs.append(
            {
                "config_path": config_path,
                "config": config_payload,
                "seed_model_name": str(config_payload["name"]),
                "seed_ir": ir_payload,
                "ir_hash": stable_hash(ir_payload),
            }
        )
    seed_family_bootstrap_budget = min(max(0, len(seed_family_specs) - 1), max(0, budget_limit - 1))
    portfolio_stage_budget = min(4, max(0, budget_limit - 1 - seed_family_bootstrap_budget))
    portfolio_stage_end_iteration = seed_family_bootstrap_budget + portfolio_stage_budget

    log_lines = [
        f"dataset_key={dataset_key}",
        f"seed_model_config={seed_model_config_path}",
        f"seed_model_name={seed_model_name}",
        f"seed_families={json.dumps([item['seed_model_name'] for item in seed_family_specs], ensure_ascii=True)}",
        f"session_id={session_slug}",
        f"requested_device={requested_device}",
        f"cuda_visible_devices={cuda_visible_devices}",
        f"task_group={task_group}",
        "harness_mode=performance_first_multi_seed_portfolio_search_v3",
    ]
    log_lines = load_previous_log_lines(session_root / "agent.log", log_lines)

    def log(message: str) -> None:
        log_lines.append(message)
        print(f"[agent-structured-v3] {message}", flush=True)

    history: List[Dict[str, Any]] = load_history_rows(history_root)
    accepted_trace_rows: List[Dict[str, Any]] = load_rows_payload(trace_root / "edit_program_trace.json")
    rejection_rows: List[Dict[str, Any]] = load_rows_payload(rejection_root / "proposal_rejections.json")
    seed_ir = dict(seed_family_specs[0]["seed_ir"])
    baseline_objective: float | None = baseline_objective_from_history(history)
    stopping_reason: str | None = None
    iteration = next_iteration_index(history)
    if history:
        log(
            f"resume session_id={session_slug} next_iteration={iteration} "
            f"completed_evaluations={completed_evaluation_count(history)} candidate_attempts={len(history)}"
        )

    while should_continue_search(history, budget_limit):
        reference_objective_value: float | None = None
        search_stage = "baseline"
        current_reference_seed_family_name = seed_model_name
        if iteration == 0:
            proposal = {
                "proposal_source": "seed_model_ir",
                "proposal_note": f"iteration_0_seed_ir::{seed_model_name}",
                "selected_mechanism_templates": [],
                "scientific_hypothesis": "Evaluate the stronger seed model before search begins.",
                "mechanistic_rationale": "V3 starts from a strong default seed family, then compares one alternate family before spending the rest of the budget on structured search.",
                "risk_notes": [],
                "edits": [],
                "candidate_ir": seed_ir,
                "ir_hash": stable_hash(seed_ir),
                "candidate_seed_model_name": seed_model_name,
            }
            attempt_index_value = 0
            search_candidates_payload = {"stage": "baseline", "task_group": task_group, "candidates": []}
        else:
            best_completed = [item for item in history if item.get("execution_status") == "completed" and item.get("objective_value") is not None]
            if best_completed:
                best_reference_record = min(best_completed, key=lambda item: float(item["objective_value"]))
                current_reference_ir = dict(best_reference_record["model_ir"])
                reference_objective_value = float(best_reference_record["objective_value"])
                current_reference_seed_family_name = str(best_reference_record.get("candidate_seed_model_name") or best_reference_record.get("seed_family_name") or seed_model_name)
            else:
                current_reference_ir = seed_ir
            if iteration <= seed_family_bootstrap_budget:
                search_stage = "seed_family_bootstrap"
                family_index = int(iteration)
                family_spec = seed_family_specs[family_index]
                proposal = _seed_family_baseline_proposal(
                    reference_ir=current_reference_ir,
                    seed_ir=dict(family_spec["seed_ir"]),
                    seed_model_name=str(family_spec["seed_model_name"]),
                )
                attempt_index_value = 1
                search_candidates_payload = {
                    "stage": search_stage,
                    "task_group": task_group,
                    "reference": "best_seed_family_so_far",
                    "candidates": [
                        {
                            "proposal_note": proposal["proposal_note"],
                            "proposal_source": proposal["proposal_source"],
                            "candidate_seed_model_name": proposal["candidate_seed_model_name"],
                            "edit_preview": [],
                            "ir_hash": proposal["ir_hash"],
                        }
                    ],
                }
            else:
                search_stage = "portfolio_bootstrap" if iteration <= portfolio_stage_end_iteration else "local_exploit"
                candidate_pool = (
                    _portfolio_candidates(current_reference_ir, dataset_key, current_reference_seed_family_name)
                    if search_stage == "portfolio_bootstrap"
                    else _local_refinement_candidates(current_reference_ir, dataset_key, current_reference_seed_family_name)
                )
                search_candidates_payload = {
                    "stage": search_stage,
                    "task_group": task_group,
                    "reference": "best_seed_family_so_far" if search_stage == "portfolio_bootstrap" else "best_ir",
                    "candidates": [
                        {
                            "proposal_note": item["proposal_note"],
                            "proposal_source": item["proposal_source"],
                            "candidate_seed_model_name": item.get("candidate_seed_model_name"),
                            "edit_preview": [describe_edit(edit) for edit in item["edits"]],
                            "ir_hash": item["ir_hash"],
                        }
                        for item in candidate_pool
                    ],
                }
                seen_ir_hashes = {str(item.get("ir_hash")) for item in history}
                proposal = None
                attempt_index_value = 0
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
                        proposal=candidate,
                        compiled_model_config=compiled_model_config,
                        generated_code_path=generated_code_path,
                    )
                    if bool(harness_review["passed"]):
                        proposal = dict(candidate)
                        attempt_index_value = int(attempt_index)
                        break
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
                if proposal is None:
                    stopping_reason = "no_valid_v3_candidate"
                    log(f"stopping iteration={iteration} reason={stopping_reason}")
                    break

        candidate_ir = dict(proposal["candidate_ir"])
        ir_path = ir_root / f"iter_{iteration:03d}.json"
        edit_path = edit_root / f"iter_{iteration:03d}.json"
        compiled_model_path = compiled_root / f"iter_{iteration:03d}.yaml"
        generated_code_path = generated_code_root / f"iter_{iteration:03d}_model.py"
        benchmark_config_path = benchmark_configs_root / f"iter_{iteration:03d}.yaml"
        harness_review_path = harness_root / f"iter_{iteration:03d}.json"
        trace_stage_path = trace_root / f"iter_{iteration:03d}_search_stage.json"
        generated_class_name = "GeneratedStructuredHypothesisRegressor"
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
                "candidate_seed_model_name": proposal.get("candidate_seed_model_name", seed_model_name),
                "search_stage": search_stage,
                "edits": proposal["edits"],
                "ir_hash": proposal["ir_hash"],
            },
        )
        dump_json(trace_stage_path, search_candidates_payload)
        write_text(generated_code_path, render_generated_model_source(candidate_ir, class_name=generated_class_name))
        harness_review = build_harness_review(
            previous_ir=seed_ir if iteration == 0 else current_reference_ir,
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
                run_name=f"{dataset_key}_{seed_model_name}_agent_structured_v3_iter_{iteration:03d}",
                artifacts_root=benchmark_runs_root,
                seed=int(seed),
                top_k=int(top_k),
                model_config=compiled_model_path,
            ),
        )

        log(
            f"dispatch iteration={iteration} stage={search_stage} "
            f"proposal_source={proposal['proposal_source']} proposal_note={proposal['proposal_note']}"
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
                "candidate_seed_model_name": proposal.get("candidate_seed_model_name", seed_model_name),
                "seed_family_name": proposal.get("candidate_seed_model_name", seed_model_name),
                "search_stage": search_stage,
                "task_group": task_group,
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
                "proposal_note": proposal["proposal_note"],
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
                    "proposal_note": proposal["proposal_note"],
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
    dump_json(
        session_root / "agent_session.json",
        {
            "dataset_key": dataset_key,
            "method_family": "agent_structured_model",
            "method_name": f"agent_structured_v3::{seed_model_name}",
            "agent_line": "main_agent_v3",
            "agent_mode": agent_mode,
            "agent_variant": agent_mode,
            "seed_model_name": seed_model_name,
            "seed_model_config_path": str(seed_model_config_path.resolve()),
            "session_id": session_slug,
            "session_root": str(session_root.resolve()),
            "generated_code_root": str(generated_code_root.resolve()),
            "requested_device": requested_device,
            "cuda_visible_devices": cuda_visible_devices,
            "llm_config_path": None,
            "llm_strategy": "portfolio",
            "harness_mode": "performance_first_multi_seed_portfolio_search_v3",
            "task_group": task_group,
            "seed_family_candidates_json": json.dumps([item["seed_model_name"] for item in seed_family_specs], ensure_ascii=True),
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
            "llm_request_count": 0,
            "llm_prompt_tokens": 0,
            "llm_completion_tokens": 0,
            "llm_total_tokens": 0,
            "llm_repair_request_count": 0,
        },
    )
    write_text(session_root / "agent.log", "\n".join(log_lines) + "\n")

    baseline_record = history[0]
    return {
        "dataset_key": dataset_key,
        "method_name": f"agent_structured_v3::{seed_model_name}",
        "method_family": "agent_structured_model",
        "agent_line": "main_agent_v3",
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
        "llm_strategy": "portfolio",
        "llm_enabled": False,
        "budget_limit_completed_evaluations": int(budget_limit),
        "candidate_attempt_budget_limit": int(candidate_attempt_budget_limit(budget_limit)),
        "completed_evaluation_count": int(completed_count),
        "failed_candidate_count": int(failed_candidate_count),
        "candidate_attempt_count": int(candidate_attempt_count),
        "stopping_rule": MATCHED_BUDGET_STOPPING_RULE,
        "stopping_reason": stopping_reason,
        "session_wall_clock_seconds": float(session_wall_clock_seconds),
        "llm_request_count": 0,
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
        "llm_total_tokens": 0,
        "llm_repair_request_count": 0,
        **{f"baseline.{metric}": baseline_record.get(metric) for metric in REPORT_METRIC_PATHS},
        **{f"best.{metric}": best_record.get(metric) for metric in REPORT_METRIC_PATHS},
    }
