from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List, Sequence

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
from sci_response.agent.structured import OBJECTIVE_PATH, REPORT_METRIC_PATHS
from sci_response.data.io import dump_json, dump_yaml, ensure_dir, load_json, load_yaml, stable_hash, write_text


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


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
        "notes": "matched-budget hpo control iteration",
    }


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


def _base_search_space(model_name: str) -> List[Dict[str, Any]]:
    if model_name == "structured_hypothesis":
        return [
            {"name": "hidden_dim", "kind": "choice", "values": [64, 96, 128, 192, 256]},
            {"name": "trunk_depth", "kind": "choice", "values": [1, 2, 3]},
            {"name": "residual_depth", "kind": "choice", "values": [0, 1, 2]},
            {"name": "conditioning_dim", "kind": "choice", "values": [32, 64, 96]},
            {"name": "dropout", "kind": "choice", "values": [0.0, 0.1, 0.2, 0.3]},
            {"name": "learning_rate", "kind": "logfloat", "low": 3.0e-4, "high": 3.0e-3},
            {"name": "weight_decay", "kind": "logfloat", "low": 1.0e-6, "high": 1.0e-3},
            {"name": "batch_size", "kind": "choice", "values": [64, 128, 256]},
            {"name": "use_lr_scheduler", "kind": "choice", "values": [False, True]},
            {"name": "scheduler_factor", "kind": "choice", "values": [0.3, 0.5, 0.7]},
            {"name": "scheduler_patience", "kind": "choice", "values": [4, 8, 12]},
        ]
    if model_name == "conditioned_residual":
        return [
            {"name": "hidden_dim", "kind": "choice", "values": [64, 96, 128, 192, 256]},
            {"name": "conditioning_dim", "kind": "choice", "values": [32, 64, 96]},
            {"name": "learning_rate", "kind": "logfloat", "low": 3.0e-4, "high": 3.0e-3},
            {"name": "weight_decay", "kind": "logfloat", "low": 1.0e-6, "high": 1.0e-3},
            {"name": "batch_size", "kind": "choice", "values": [64, 128, 256]},
        ]
    raise ValueError(f"Unsupported HPO seed model family: {model_name}")


def _search_space_manifest(model_name: str, base_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    manifest: List[Dict[str, Any]] = []
    for spec in _base_search_space(model_name):
        row = dict(spec)
        if row["kind"] == "choice":
            row["default"] = base_config.get(row["name"], row["values"][0])
        else:
            row["default"] = base_config.get(row["name"])
        manifest.append(row)
    return manifest


def _normalize_params(params: Dict[str, Any], search_space: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    spec_by_name = {str(item["name"]): dict(item) for item in search_space}
    for key, value in params.items():
        spec = spec_by_name[str(key)]
        if spec["kind"] == "choice":
            exemplar = spec["values"][0]
            if isinstance(exemplar, bool):
                normalized[str(key)] = bool(value)
            elif isinstance(exemplar, int) and not isinstance(exemplar, bool):
                normalized[str(key)] = int(value)
            elif isinstance(exemplar, float):
                normalized[str(key)] = float(value)
            else:
                normalized[str(key)] = value
        elif spec["kind"] == "logfloat":
            normalized[str(key)] = float(value)
        else:
            normalized[str(key)] = value
    return normalized


def _apply_hpo_params(base_config: Dict[str, Any], sampled_params: Dict[str, Any]) -> Dict[str, Any]:
    config = dict(base_config)
    config.update(sampled_params)
    if str(config["name"]) == "structured_hypothesis" and not bool(config.get("use_lr_scheduler", False)):
        config["scheduler_factor"] = float(base_config.get("scheduler_factor", 0.5))
        config["scheduler_patience"] = int(base_config.get("scheduler_patience", 8))
        config["scheduler_min_lr"] = float(base_config.get("scheduler_min_lr", 1.0e-5))
    return config


def _complete_search_params(
    *,
    base_config: Dict[str, Any],
    sampled_params: Dict[str, Any],
    search_space: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    complete: Dict[str, Any] = {}
    for spec in search_space:
        name = str(spec["name"])
        if name in sampled_params:
            complete[name] = sampled_params[name]
        elif "default" in spec and spec.get("default") is not None:
            complete[name] = spec["default"]
        elif name in base_config:
            complete[name] = base_config[name]
    return _normalize_params(complete, search_space)


def _suggest_optuna_trial(trial: Any, search_space: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for spec in search_space:
        name = str(spec["name"])
        if spec["kind"] == "choice":
            params[name] = trial.suggest_categorical(name, list(spec["values"]))
        elif spec["kind"] == "logfloat":
            params[name] = trial.suggest_float(name, float(spec["low"]), float(spec["high"]), log=True)
        else:
            raise ValueError(f"Unsupported search-space kind for optuna: {spec['kind']}")
    return _normalize_params(params, search_space)


def _build_flaml_space(search_space: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    from flaml import tune

    config: Dict[str, Any] = {}
    for spec in search_space:
        name = str(spec["name"])
        if spec["kind"] == "choice":
            config[name] = tune.choice(list(spec["values"]))
        elif spec["kind"] == "logfloat":
            config[name] = tune.loguniform(float(spec["low"]), float(spec["high"]))
        else:
            raise ValueError(f"Unsupported search-space kind for flaml: {spec['kind']}")
    return config


def _optuna_distributions(search_space: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    import optuna

    distributions: Dict[str, Any] = {}
    for spec in search_space:
        name = str(spec["name"])
        if spec["kind"] == "choice":
            distributions[name] = optuna.distributions.CategoricalDistribution(list(spec["values"]))
        elif spec["kind"] == "logfloat":
            distributions[name] = optuna.distributions.FloatDistribution(
                float(spec["low"]),
                float(spec["high"]),
                log=True,
            )
        else:
            raise ValueError(f"Unsupported search-space kind for optuna distributions: {spec['kind']}")
    return distributions


def run_hpo_model_session(
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
    agent_mode: str,
) -> Dict[str, Any]:
    if agent_mode not in {"hpo_optuna_tpe", "hpo_flaml_cfo"}:
        raise ValueError(f"Unsupported hpo mode: {agent_mode}")

    seed_model_config = load_yaml(seed_model_config_path)
    seed_model_name = str(seed_model_config["name"])
    if seed_model_name not in {"structured_hypothesis", "conditioned_residual"}:
        raise ValueError(
            f"HPO currently supports generic seed families structured_hypothesis/conditioned_residual, got {seed_model_name!r}"
        )
    search_space = _search_space_manifest(seed_model_name, seed_model_config)
    search_space_hash = stable_hash(search_space)
    budget_limit = int(max_iteration)
    session_started_at = perf_counter()

    session_slug = session_id or f"{_safe_slug(agent_mode)}_{seed}"
    dataset_root = ensure_dir(agent_root / "datasets" / dataset_key)
    session_root = ensure_dir(dataset_root / "hpo" / _safe_slug(agent_mode) / _safe_slug(seed_model_name) / session_slug)
    configs_root = ensure_dir(session_root / "configs")
    benchmark_configs_root = ensure_dir(session_root / "benchmark_configs")
    benchmark_runs_root = ensure_dir(session_root / "benchmark_runs")
    history_root = ensure_dir(session_root / "history")
    trace_root = ensure_dir(session_root / "trace")
    prior_session_wall_clock_seconds = load_previous_session_wall_clock_seconds(session_root)

    dump_json(trace_root / "search_space.json", {"rows": search_space})

    log_lines = [
        f"dataset_key={dataset_key}",
        f"agent_mode={agent_mode}",
        f"seed_model_name={seed_model_name}",
        f"seed_model_config={seed_model_config_path}",
        f"requested_device={requested_device}",
        f"cuda_visible_devices={cuda_visible_devices}",
        f"search_space_hash={search_space_hash}",
        f"matched_budget_trials={int(max_iteration)}",
        "structure_edits=disabled",
        "semantic_checker=disabled",
        "biological_constraints=disabled",
    ]
    log_lines = load_previous_log_lines(session_root / "agent.log", log_lines)

    def log(message: str) -> None:
        log_lines.append(message)
        print(f"[agent-hpo] {message}", flush=True)

    history: List[Dict[str, Any]] = load_history_rows(history_root)
    trace_rows: List[Dict[str, Any]] = load_rows_payload(trace_root / "hpo_trace.json")
    baseline_objective: float | None = baseline_objective_from_history(history)
    stopping_reason: str | None = None

    optuna_study = None
    flaml_searcher = None
    if agent_mode == "hpo_optuna_tpe":
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        optuna_study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=int(seed)),
        )
    else:
        from flaml.tune.searcher import CFO

        flaml_searcher = CFO(
            metric="loss",
            mode="min",
            space=_build_flaml_space(search_space),
            seed=int(seed),
        )

    if history:
        if agent_mode == "hpo_optuna_tpe":
            assert optuna_study is not None
            import optuna

            distributions = _optuna_distributions(search_space)
            for row in history:
                if int(row.get("iteration", 0) or 0) == 0:
                    continue
                params = _normalize_params(
                    dict(row.get("completed_search_params") or row.get("sampled_params") or {}),
                    search_space,
                )
                if not params:
                    continue
                objective_value = row.get("objective_value")
                state = (
                    optuna.trial.TrialState.COMPLETE
                    if str(row.get("execution_status")) == "completed" and objective_value is not None
                    else optuna.trial.TrialState.FAIL
                )
                trial_kwargs: Dict[str, Any] = {
                    "params": params,
                    "distributions": distributions,
                    "state": state,
                }
                if state == optuna.trial.TrialState.COMPLETE:
                    trial_kwargs["value"] = float(objective_value)
                optuna_study.add_trial(optuna.trial.create_trial(**trial_kwargs))
        else:
            assert flaml_searcher is not None
            for row in history:
                if int(row.get("iteration", 0) or 0) == 0:
                    continue
                params = _normalize_params(
                    dict(row.get("completed_search_params") or row.get("sampled_params") or {}),
                    search_space,
                )
                if not params:
                    continue
                backend_trial_id = str(row.get("backend_trial_id") or f"trial_{int(row.get('iteration', 0)):03d}")
                objective_value = row.get("objective_value")
                loss = (
                    float(objective_value)
                    if str(row.get("execution_status")) == "completed" and objective_value is not None
                    else float("inf")
                )
                flaml_searcher.on_trial_complete(
                    backend_trial_id,
                    result={"loss": loss, "config": params},
                )
    iteration = next_iteration_index(history)
    if history:
        log(
            f"resume session_id={session_slug} next_iteration={iteration} "
            f"completed_evaluations={completed_evaluation_count(history)} candidate_attempts={len(history)}"
        )
    while should_continue_search(history, budget_limit):
        if iteration == 0:
            sampled_params: Dict[str, Any] = {}
            proposal_source = "seed_model_config"
            proposal_note = f"iteration_0_seed_config::{seed_model_name}"
            backend_trial_id = None
            backend_trial = None
        else:
            if agent_mode == "hpo_optuna_tpe":
                assert optuna_study is not None
                backend_trial = optuna_study.ask()
                backend_trial_id = str(getattr(backend_trial, "number", iteration))
                sampled_params = _suggest_optuna_trial(backend_trial, search_space)
            else:
                assert flaml_searcher is not None
                backend_trial_id = f"trial_{iteration:03d}"
                sampled_params = _normalize_params(
                    dict(flaml_searcher.suggest(str(backend_trial_id)) or {}),
                    search_space,
                )
                backend_trial = None
            proposal_source = agent_mode
            proposal_note = ", ".join(f"{key}={sampled_params[key]}" for key in sorted(sampled_params)) or "no-op"

        candidate_config = _apply_hpo_params(seed_model_config, sampled_params)
        completed_search_params = _complete_search_params(
            base_config=seed_model_config,
            sampled_params=sampled_params,
            search_space=search_space,
        )
        model_config_path = configs_root / f"iter_{iteration:03d}.yaml"
        benchmark_config_path = benchmark_configs_root / f"iter_{iteration:03d}.yaml"

        dump_yaml(model_config_path, candidate_config)
        dump_yaml(
            benchmark_config_path,
            _build_benchmark_payload(
                dataset_config=dataset_config,
                split_path=split_path,
                run_name=f"{dataset_key}_{seed_model_name}_{agent_mode}_iter_{iteration:03d}",
                artifacts_root=benchmark_runs_root,
                seed=int(seed),
                top_k=int(top_k),
                model_config=model_config_path,
            ),
        )
        log(
            f"dispatch iteration={iteration} proposal_source={proposal_source} "
            f"proposal_note={proposal_note}"
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
            objective_raw = _get_nested(metrics, OBJECTIVE_PATH)
            objective_value = float(objective_raw) if isinstance(objective_raw, (int, float)) else None
            if iteration == 0:
                baseline_objective = objective_value
            elif objective_value is not None:
                if agent_mode == "hpo_optuna_tpe":
                    assert optuna_study is not None and backend_trial is not None
                    optuna_study.tell(backend_trial, float(objective_value))
                else:
                    assert flaml_searcher is not None and backend_trial_id is not None
                    flaml_searcher.on_trial_complete(
                        str(backend_trial_id),
                        result={"loss": float(objective_value), "config": completed_search_params},
                    )
            record = {
                "iteration": int(iteration),
                "phase": "baseline" if iteration == 0 else "agent",
                "proposal_source": proposal_source,
                "proposal_note": proposal_note,
                "backend_trial_id": backend_trial_id,
                "sampled_params": sampled_params,
                "completed_search_params": completed_search_params,
                "model_config": candidate_config,
                "config_hash": stable_hash(candidate_config),
                "model_config_path": str(model_config_path.resolve()),
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
                "delta_vs_iteration0": (
                    float(baseline_objective) - float(objective_value)
                    if baseline_objective is not None and objective_value is not None
                    else None
                ),
                "seed_model_name": seed_model_name,
                "agent_mode": agent_mode,
                "search_backend": "optuna_tpe" if agent_mode == "hpo_optuna_tpe" else "flaml_cfo",
                "budget_matched_to_iterations": int(max_iteration),
                "protocol": manifest.get("protocol"),
                "start_time_beijing": manifest.get("start_time_beijing"),
                "end_time_beijing": manifest.get("end_time_beijing"),
                "evaluation_wall_clock_seconds": float(evaluation_wall_clock_seconds),
            }
            record.update(_flatten_selected(metrics, REPORT_METRIC_PATHS))
        except Exception as exc:
            if iteration == 0:
                raise
            if agent_mode == "hpo_optuna_tpe":
                assert optuna_study is not None and backend_trial is not None
                import optuna

                optuna_study.tell(backend_trial, state=optuna.trial.TrialState.FAIL)
            else:
                assert flaml_searcher is not None and backend_trial_id is not None
                flaml_searcher.on_trial_complete(
                    str(backend_trial_id),
                    result={"loss": float("inf"), "config": completed_search_params},
                )
            record = {
                "iteration": int(iteration),
                "phase": "agent",
                "proposal_source": proposal_source,
                "proposal_note": proposal_note,
                "backend_trial_id": backend_trial_id,
                "sampled_params": sampled_params,
                "completed_search_params": completed_search_params,
                "model_config": candidate_config,
                "config_hash": stable_hash(candidate_config),
                "model_config_path": str(model_config_path.resolve()),
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
                "delta_vs_iteration0": None,
                "seed_model_name": seed_model_name,
                "agent_mode": agent_mode,
                "search_backend": "optuna_tpe" if agent_mode == "hpo_optuna_tpe" else "flaml_cfo",
                "budget_matched_to_iterations": int(max_iteration),
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
        trace_rows.append(
            {
                "iteration": int(iteration),
                "proposal_source": proposal_source,
                "proposal_note": proposal_note,
                "backend_trial_id": backend_trial_id,
                "sampled_params_json": json.dumps(sampled_params, ensure_ascii=True, sort_keys=True),
                "objective_value": record.get("objective_value"),
                "benchmark_run_dir": str(record.get("run_dir")),
                "model_config_path": str(model_config_path.resolve()),
            }
        )
        dump_json(history_root / f"iter_{iteration:03d}.json", record)
        log(f"completed iteration={iteration} objective={record.get('objective_value')}")
        iteration += 1

    completed = [item for item in history if item.get("execution_status") == "completed" and item.get("objective_value") is not None]
    if not completed:
        raise RuntimeError(f"No completed hpo iterations for {dataset_key}/{seed_model_name}/{agent_mode}")
    best_record = min(completed, key=lambda item: float(item["objective_value"]))
    if stopping_reason is None:
        stopping_reason = default_stopping_reason(history, budget_limit)
    completed_count = completed_evaluation_count(history)
    failed_candidate_count = int(failed_history_count(history))
    candidate_attempt_count = int(len(history))
    session_wall_clock_seconds = prior_session_wall_clock_seconds + float(perf_counter() - session_started_at)
    for record in history:
        record["best_so_far_iteration"] = int(best_record["iteration"])
        record["best_so_far_objective"] = float(best_record["objective_value"])
        record["accepted_as_best"] = bool(record["iteration"] == best_record["iteration"])

    dump_json(session_root / "iterations.json", {"iterations": history})
    fieldnames = sorted(
        {key for row in history for key in row.keys() if key not in {"model_config", "sampled_params", "completed_search_params"}}
        | {"model_config_json", "sampled_params_json", "completed_search_params_json"}
    )
    with (session_root / "iterations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            flattened = {
                key: value
                for key, value in row.items()
                if key not in {"model_config", "sampled_params", "completed_search_params"}
            }
            flattened["model_config_json"] = json.dumps(row["model_config"], ensure_ascii=True, sort_keys=True)
            flattened["sampled_params_json"] = json.dumps(row["sampled_params"], ensure_ascii=True, sort_keys=True)
            flattened["completed_search_params_json"] = json.dumps(
                row["completed_search_params"],
                ensure_ascii=True,
                sort_keys=True,
            )
            writer.writerow(flattened)
    dump_json(session_root / "best_iteration.json", best_record)
    dump_json(trace_root / "hpo_trace.json", {"rows": trace_rows})
    if trace_rows:
        _write_csv(trace_root / "hpo_trace.csv", trace_rows)
    dump_json(
        session_root / "agent_session.json",
        {
            "dataset_key": dataset_key,
            "method_family": "agent_hpo_model",
            "method_name": f"{agent_mode}::{seed_model_name}",
            "agent_line": "hpo_search",
            "agent_mode": agent_mode,
            "agent_variant": agent_mode,
            "seed_model_name": seed_model_name,
            "seed_model_config_path": str(seed_model_config_path.resolve()),
            "session_id": session_slug,
            "session_root": str(session_root.resolve()),
            "requested_device": requested_device,
            "cuda_visible_devices": cuda_visible_devices,
            "search_backend": "optuna_tpe" if agent_mode == "hpo_optuna_tpe" else "flaml_cfo",
            "matched_budget_trials": int(max_iteration),
            "budget_limit_completed_evaluations": int(budget_limit),
            "candidate_attempt_budget_limit": int(candidate_attempt_budget_limit(budget_limit)),
            "completed_evaluation_count": int(completed_count),
            "failed_candidate_count": int(failed_candidate_count),
            "candidate_attempt_count": int(candidate_attempt_count),
            "stopping_rule": MATCHED_BUDGET_STOPPING_RULE,
            "stopping_reason": stopping_reason,
            "session_wall_clock_seconds": float(session_wall_clock_seconds),
            "search_space_hash": search_space_hash,
            "search_space_path": str((trace_root / "search_space.json").resolve()),
            "best_iteration": int(best_record["iteration"]),
            "best_objective": best_record["objective_value"],
            "best_run_dir": best_record["run_dir"],
            "objective_path": OBJECTIVE_PATH,
        },
    )
    write_text(session_root / "agent.log", "\n".join(log_lines) + "\n")

    baseline_record = history[0]
    return {
        "dataset_key": dataset_key,
        "method_name": f"{agent_mode}::{seed_model_name}",
        "method_family": "agent_hpo_model",
        "agent_line": "hpo_search",
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
        "search_backend": "optuna_tpe" if agent_mode == "hpo_optuna_tpe" else "flaml_cfo",
        "matched_budget_trials": int(max_iteration),
        "budget_limit_completed_evaluations": int(budget_limit),
        "candidate_attempt_budget_limit": int(candidate_attempt_budget_limit(budget_limit)),
        "completed_evaluation_count": int(completed_count),
        "failed_candidate_count": int(failed_candidate_count),
        "candidate_attempt_count": int(candidate_attempt_count),
        "stopping_rule": MATCHED_BUDGET_STOPPING_RULE,
        "stopping_reason": stopping_reason,
        "session_wall_clock_seconds": float(session_wall_clock_seconds),
        **{f"baseline.{metric}": baseline_record.get(metric) for metric in REPORT_METRIC_PATHS},
        **{f"best.{metric}": best_record.get(metric) for metric in REPORT_METRIC_PATHS},
    }
