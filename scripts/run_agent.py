#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sci_response.agent import (
    build_codeevo_command,
    default_agent_root,
    default_session_id,
    list_codeevo_configs,
    resolve_codeevo_root,
    resolve_dataset_keys,
    resolve_method_names,
    run_direct_code_model_session,
    run_hpo_model_session,
    run_method_session,
    run_random_edit_model_session,
    run_structured_model_session,
    run_structured_model_session_v2,
    run_structured_model_session_v3,
    run_structured_model_session_v4,
    run_structured_model_session_v5,
    run_structured_model_session_v6,
    run_structured_model_session_v7,
    run_structured_model_session_v8,
    update_global_agent_summary,
    write_dataset_pack_summary,
)
from sci_response.agent.formal import RAW_ITERATION_BUDGET_SEMANTICS
from sci_response.agent.config import default_llm_config_path, load_agent_defaults
from sci_response.agent.llm import StructuredLLMClient, load_llm_settings
from sci_response.data.io import ensure_dir, load_yaml


def _legacy_mode_requested(args: Any) -> bool:
    return bool(args.legacy_codeevo or args.list_configs or args.config or args.dry_run)


def _parse_csv_list(raw: str | None) -> List[str]:
    if raw is None:
        return []
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _requires_llm(agent_mode: str) -> bool:
    return agent_mode in {
        "structured_llm_search",
        "structured_llm_search_v2",
        "structured_hybrid_search_v4",
        "structured_projected_search_v5",
        "structured_adaptive_tree_search_v6",
        "structured_anchor_tree_search_v7",
        "structured_open_mechanism_search_v8",
        "structured_open_mechanism_search_v8_practical",
        "direct_code_llm_singleshot",
        "direct_code_llm_repairloop",
    }


def _preflight_llm_config(llm_config_path: Path) -> None:
    if str(os.environ.get("SCI_RESPONSE_SKIP_LLM_PREFLIGHT", "")).strip() == "1":
        return
    settings = load_llm_settings(llm_config_path)
    client = StructuredLLMClient(settings)
    try:
        text_payload = client.chat_text(
            system_prompt="Reply with exactly the word OK.",
            user_prompt="Reply with exactly the word OK.",
            max_tokens_override=16,
        )
    except Exception as text_exc:
        try:
            payload = client.chat_json(
                system_prompt="Return JSON only.",
                user_prompt=(
                    "Reply with exactly one JSON object: "
                    "{\"status\":\"ok\",\"provider\":\"%s\",\"model\":\"%s\"}"
                    % (settings.provider, settings.model)
                ),
                max_tokens_override=64,
            )
        except Exception as json_exc:
            raise SystemExit(
                "LLM preflight failed for "
                f"{llm_config_path}: text_probe={type(text_exc).__name__}: {text_exc}; "
                f"json_probe={type(json_exc).__name__}: {json_exc}. "
                "The run was stopped before search started so we do not silently fall back to a degraded mode."
            ) from json_exc
        if str(payload.get("status", "")).strip().lower() == "ok":
            return
        raise SystemExit(
            f"LLM preflight returned unexpected payload for {llm_config_path}: {payload!r}. "
            "The run was stopped before search started."
        )
    if "ok" not in str(text_payload).strip().lower():
        raise SystemExit(
            f"LLM preflight returned unexpected text payload for {llm_config_path}: {text_payload!r}. "
            "The run was stopped before search started."
        )


def _method_supports_cuda(method_name: str, method_defaults: Dict[str, Any]) -> bool:
    family = str(method_defaults.get("family", ""))
    normalized = str(method_name).lower()
    if family == "universal" and normalized == "catboost":
        return True
    if family == "specialist" and normalized in {"gperturb", "gears", "cpa", "xpert", "cellot", "transigen"}:
        return True
    return False


def _order_methods(method_names: List[str], matrix_config: Path) -> List[str]:
    matrix = load_yaml(matrix_config)
    defaults = dict(matrix.get("baseline_defaults", {}))
    decorated = []
    for index, method_name in enumerate(method_names):
        method_defaults = dict(defaults[method_name])
        gpu_priority = 0 if _method_supports_cuda(method_name, method_defaults) else 1
        decorated.append((gpu_priority, index, method_name))
    decorated.sort()
    return [method_name for _, _, method_name in decorated]


def _run_legacy_mode(args: Any, passthrough: List[str]) -> int:
    if args.dataset_keys or args.all_main_datasets:
        raise SystemExit("Legacy CodeEvo mode cannot be combined with new dataset/method agent arguments.")
    if args.list_configs:
        print(f"codeevo_root={resolve_codeevo_root(ROOT)}")
        for path in list_codeevo_configs(ROOT):
            print(path)
        return 0

    config_path = Path(args.config).resolve() if args.config else None
    command = build_codeevo_command(ROOT, config_path=config_path, extra_args=passthrough)
    if args.dry_run:
        print("resolved_command:")
        print(" ".join(command))
        return 0
    completed = subprocess.run(command, cwd=str(resolve_codeevo_root(ROOT)))
    return int(completed.returncode)


def main() -> None:
    import argparse

    agent_defaults = load_agent_defaults()

    parser = argparse.ArgumentParser(
        description="Formal MechCell agent runner on top of the unified benchmark interface, with legacy CodeEvo fallback."
    )
    parser.add_argument("--legacy-codeevo", action="store_true", help="Use the old legacy/CodeEvo launcher.")
    parser.add_argument("--config", default=None, help="Legacy CodeEvo mode: optional config JSON path.")
    parser.add_argument("--list-configs", action="store_true", help="Legacy CodeEvo mode: list CodeEvo config JSON files.")
    parser.add_argument("--dry-run", action="store_true", help="Legacy CodeEvo mode: print the resolved command and exit.")

    parser.add_argument("--dataset-keys", default=None, help="Comma-separated dataset keys from configs/benchmark_matrix.yaml.")
    parser.add_argument("--all-main-datasets", action="store_true", help="Run on all 7 public main datasets.")
    parser.add_argument(
        "--agent-mode",
        default=str(agent_defaults.get("agent_mode", "structured_llm_search")),
        choices=[
            "structured_llm_search",
            "structured_heuristic_search",
            "structured_llm_search_v2",
            "structured_heuristic_search_v2",
            "structured_portfolio_search_v3",
            "structured_hybrid_search_v4",
            "structured_projected_search_v5",
            "structured_adaptive_tree_search_v6",
            "structured_anchor_tree_search_v7",
            "structured_open_mechanism_search_v8",
            "structured_open_mechanism_search_v8_practical",
            "ablation_method_search",
            "direct_code_llm_singleshot",
            "direct_code_llm_repairloop",
            "random_edit_uniform",
            "random_edit_stratified",
            "hpo_optuna_tpe",
            "hpo_flaml_cfo",
        ],
        help="Default paper path is structured_llm_search. The old benchmark-method search remains as ablation_method_search only.",
    )
    parser.add_argument("--methods", default=None, help="Optional comma-separated subset of runnable benchmark methods for each dataset.")
    parser.add_argument("--all-methods", action="store_true", help="Run all runnable benchmark methods for each selected dataset.")
    parser.add_argument("--seed-model-config", default=None, help="Structured hypothesis mode seed model config YAML path. Defaults to configs/agent/defaults.yaml -> seed_model_config.")
    parser.add_argument(
        "--max-iteration",
        type=int,
        default=int(agent_defaults.get("max_iteration", 10)),
        help="Matched-budget upper bound on completed model evaluations per dataset/split. Iteration 0 counts as the no-agent baseline evaluation.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--top-k", type=int, default=int(agent_defaults.get("top_k", 20)))
    parser.add_argument("--device", default="cpu", help="Requested device, e.g. cpu or cuda:0")
    parser.add_argument("--cuda-visible-devices", default=None, help="Optional CUDA_VISIBLE_DEVICES mask, e.g. 0 or 2")
    parser.add_argument("--matrix-config", default=str(ROOT / "configs" / "benchmark_matrix.yaml"))
    parser.add_argument("--runtime-env-config", default=str(ROOT / "configs" / "runtime_envs.yaml"))
    parser.add_argument("--runtime-mode", default=str(agent_defaults.get("runtime_mode", "primary")), choices=["primary", "fallback", "upstream"])
    parser.add_argument("--runtime-env-group", default=None, help="Optional explicit env-group override.")
    parser.add_argument("--agent-root", default=str(default_agent_root(ROOT)))
    parser.add_argument("--baseline-root", default=str(ROOT / "baseline"))
    parser.add_argument("--session-id", default=None, help="Optional fixed session id. Default is a Beijing timestamp slug.")
    parser.add_argument("--llm-config", default=None, help="JSON/YAML config with llm/providers sections. Defaults to configs/agent/llm.yaml.")
    parser.add_argument("--llm-strategy", default="llm", choices=["heuristic", "llm", "hybrid", "portfolio"])
    args, passthrough = parser.parse_known_args()

    if _legacy_mode_requested(args):
        raise SystemExit(_run_legacy_mode(args, passthrough))

    if args.max_iteration < 0:
        raise SystemExit("--max-iteration must be >= 0")

    matrix_config = Path(args.matrix_config).resolve()
    runtime_env_config = Path(args.runtime_env_config).resolve()
    requested_agent_root = Path(args.agent_root).resolve()
    if args.agent_mode in {"structured_llm_search_v2", "structured_heuristic_search_v2"} and requested_agent_root == Path(default_agent_root(ROOT)).resolve():
        requested_agent_root = (ROOT / "agent_runs_v2").resolve()
    if args.agent_mode == "structured_portfolio_search_v3" and requested_agent_root == Path(default_agent_root(ROOT)).resolve():
        requested_agent_root = (ROOT / "agent_runs_v3").resolve()
    if args.agent_mode == "structured_hybrid_search_v4" and requested_agent_root == Path(default_agent_root(ROOT)).resolve():
        requested_agent_root = (ROOT / "agent_runs_v4").resolve()
    if args.agent_mode == "structured_projected_search_v5" and requested_agent_root == Path(default_agent_root(ROOT)).resolve():
        requested_agent_root = (ROOT / "agent_runs_v5").resolve()
    if args.agent_mode == "structured_adaptive_tree_search_v6" and requested_agent_root == Path(default_agent_root(ROOT)).resolve():
        requested_agent_root = (ROOT / "agent_runs_v6").resolve()
    if args.agent_mode == "structured_anchor_tree_search_v7" and requested_agent_root == Path(default_agent_root(ROOT)).resolve():
        requested_agent_root = (ROOT / "agent_runs_v7").resolve()
    if args.agent_mode == "structured_open_mechanism_search_v8" and requested_agent_root == Path(default_agent_root(ROOT)).resolve():
        requested_agent_root = (ROOT / "agent_runs_v8").resolve()
    if args.agent_mode == "structured_open_mechanism_search_v8_practical" and requested_agent_root == Path(default_agent_root(ROOT)).resolve():
        requested_agent_root = (ROOT / "agent_runs_v8").resolve()
    agent_root = ensure_dir(requested_agent_root)
    baseline_root = Path(args.baseline_root).resolve()
    session_id = args.session_id or default_session_id()
    default_seed_model_config = (ROOT / str(agent_defaults.get("seed_model_config", "configs/models/conditioned_residual.yaml"))).resolve()
    default_hpo_seed_model_config = (ROOT / str(agent_defaults.get("hpo_seed_model_config", "configs/models/structured_hypothesis_hpo.yaml"))).resolve()
    default_llm_config = (ROOT / str(agent_defaults.get("llm_config", str(default_llm_config_path())))).resolve()

    dataset_keys = resolve_dataset_keys(
        matrix_config,
        dataset_keys=args.dataset_keys,
        all_main_datasets=bool(args.all_main_datasets),
    )
    matrix = load_yaml(matrix_config)

    print(
        f"[agent] session_id={session_id} agent_root={agent_root} "
        f"datasets={dataset_keys} max_iteration={args.max_iteration} "
        f"device={args.device} cuda_visible_devices={args.cuda_visible_devices} "
        f"agent_mode={args.agent_mode}",
        flush=True,
    )

    global_rows: List[Dict[str, Any]] = []
    for dataset_key in dataset_keys:
        dataset_rows: List[Dict[str, Any]] = []
        if args.agent_mode in {
            "structured_llm_search",
            "structured_heuristic_search",
            "structured_llm_search_v2",
            "structured_heuristic_search_v2",
            "structured_portfolio_search_v3",
            "structured_hybrid_search_v4",
            "structured_projected_search_v5",
            "structured_adaptive_tree_search_v6",
            "structured_anchor_tree_search_v7",
            "structured_open_mechanism_search_v8",
            "structured_open_mechanism_search_v8_practical",
        }:
            if args.seed_model_config:
                seed_model_config_path = Path(args.seed_model_config).resolve()
            elif args.agent_mode == "structured_portfolio_search_v3":
                seed_model_config_path = default_seed_model_config
            else:
                seed_model_config_path = default_seed_model_config
            if not seed_model_config_path.exists():
                raise SystemExit(f"Structured seed model config does not exist: {seed_model_config_path}")
            llm_config_path = Path(args.llm_config).resolve() if args.llm_config else default_llm_config
            llm_strategy = str(args.llm_strategy)
            if args.agent_mode in {"structured_llm_search", "structured_llm_search_v2"}:
                llm_strategy = "llm"
                if not llm_config_path.exists():
                    raise SystemExit(
                        "structured_llm_search requires a reachable llm config. "
                        f"Expected {llm_config_path}. Pass --llm-config explicitly if you keep it elsewhere."
                    )
            elif args.agent_mode in {"structured_heuristic_search", "structured_heuristic_search_v2"}:
                llm_config_path = Path(args.llm_config).resolve() if args.llm_config else None
                llm_strategy = "heuristic"
            elif args.agent_mode == "structured_portfolio_search_v3":
                llm_config_path = None
                llm_strategy = "portfolio"
            elif args.agent_mode == "structured_hybrid_search_v4":
                llm_strategy = "hybrid"
                if not llm_config_path.exists():
                    raise SystemExit(
                        "structured_hybrid_search_v4 requires a reachable llm config. "
                        f"Expected {llm_config_path}. Pass --llm-config explicitly if you keep it elsewhere."
                    )
            elif args.agent_mode == "structured_projected_search_v5":
                llm_strategy = "hybrid"
                if not llm_config_path.exists():
                    raise SystemExit(
                        "structured_projected_search_v5 requires a reachable llm config. "
                        f"Expected {llm_config_path}. Pass --llm-config explicitly if you keep it elsewhere."
                    )
            elif args.agent_mode == "structured_adaptive_tree_search_v6":
                llm_strategy = "hybrid"
                if not llm_config_path.exists():
                    raise SystemExit(
                        "structured_adaptive_tree_search_v6 requires a reachable llm config. "
                        f"Expected {llm_config_path}. Pass --llm-config explicitly if you keep it elsewhere."
                    )
            elif args.agent_mode == "structured_anchor_tree_search_v7":
                llm_strategy = "hybrid"
                if not llm_config_path.exists():
                    raise SystemExit(
                        "structured_anchor_tree_search_v7 requires a reachable llm config. "
                        f"Expected {llm_config_path}. Pass --llm-config explicitly if you keep it elsewhere."
                    )
            elif args.agent_mode == "structured_open_mechanism_search_v8":
                llm_strategy = "hybrid"
                if not llm_config_path.exists():
                    raise SystemExit(
                        "structured_open_mechanism_search_v8 requires a reachable llm config. "
                        f"Expected {llm_config_path}. Pass --llm-config explicitly if you keep it elsewhere."
                    )
            elif args.agent_mode == "structured_open_mechanism_search_v8_practical":
                llm_strategy = "hybrid"
                if not llm_config_path.exists():
                    raise SystemExit(
                        "structured_open_mechanism_search_v8_practical requires a reachable llm config. "
                        f"Expected {llm_config_path}. Pass --llm-config explicitly if you keep it elsewhere."
                    )
            dataset_entry = dict(dict(matrix["datasets"])[dataset_key])
            dataset_config = (ROOT / str(dataset_entry["dataset_config"])).resolve()
            split_path = (ROOT / str(dataset_entry["primary_split"])).resolve()
            if _requires_llm(str(args.agent_mode)):
                _preflight_llm_config(llm_config_path)
            print(
                f"[agent] dataset={dataset_key} structured_seed_model={seed_model_config_path.name} "
                f"llm_strategy={llm_strategy} start",
                flush=True,
            )
            if args.agent_mode in {"structured_llm_search_v2", "structured_heuristic_search_v2"}:
                row = run_structured_model_session_v2(
                    repo_root=ROOT,
                    dataset_key=dataset_key,
                    dataset_config=dataset_config,
                    split_path=split_path,
                    seed_model_config_path=seed_model_config_path,
                    runtime_env_config=runtime_env_config,
                    requested_device=str(args.device),
                    cuda_visible_devices=args.cuda_visible_devices,
                    runtime_mode=str(args.runtime_mode),
                    runtime_env_group=args.runtime_env_group,
                    agent_root=agent_root,
                    seed=int(args.seed),
                    top_k=int(args.top_k),
                    max_iteration=int(args.max_iteration),
                    session_id=session_id,
                    llm_config_path=llm_config_path,
                    llm_strategy=llm_strategy,
                    agent_mode=str(args.agent_mode),
                )
            elif args.agent_mode == "structured_portfolio_search_v3":
                alternate_seed_model_config_paths = []
                for candidate_seed_config_path in (default_seed_model_config, default_hpo_seed_model_config):
                    resolved_candidate_seed_config_path = Path(candidate_seed_config_path).resolve()
                    if resolved_candidate_seed_config_path == seed_model_config_path:
                        continue
                    alternate_seed_model_config_paths.append(resolved_candidate_seed_config_path)
                row = run_structured_model_session_v3(
                    repo_root=ROOT,
                    dataset_key=dataset_key,
                    dataset_config=dataset_config,
                    split_path=split_path,
                    seed_model_config_path=seed_model_config_path,
                    runtime_env_config=runtime_env_config,
                    requested_device=str(args.device),
                    cuda_visible_devices=args.cuda_visible_devices,
                    runtime_mode=str(args.runtime_mode),
                    runtime_env_group=args.runtime_env_group,
                    agent_root=agent_root,
                    seed=int(args.seed),
                    top_k=int(args.top_k),
                    max_iteration=int(args.max_iteration),
                    session_id=session_id,
                    agent_mode=str(args.agent_mode),
                    additional_seed_model_config_paths=alternate_seed_model_config_paths,
                )
            elif args.agent_mode == "structured_hybrid_search_v4":
                alternate_seed_model_config_paths = []
                for candidate_seed_config_path in (default_seed_model_config, default_hpo_seed_model_config):
                    resolved_candidate_seed_config_path = Path(candidate_seed_config_path).resolve()
                    if resolved_candidate_seed_config_path == seed_model_config_path:
                        continue
                    alternate_seed_model_config_paths.append(resolved_candidate_seed_config_path)
                row = run_structured_model_session_v4(
                    repo_root=ROOT,
                    dataset_key=dataset_key,
                    dataset_config=dataset_config,
                    split_path=split_path,
                    seed_model_config_path=seed_model_config_path,
                    runtime_env_config=runtime_env_config,
                    requested_device=str(args.device),
                    cuda_visible_devices=args.cuda_visible_devices,
                    runtime_mode=str(args.runtime_mode),
                    runtime_env_group=args.runtime_env_group,
                    agent_root=agent_root,
                    seed=int(args.seed),
                    top_k=int(args.top_k),
                    max_iteration=int(args.max_iteration),
                    session_id=session_id,
                    llm_config_path=llm_config_path,
                    agent_mode=str(args.agent_mode),
                    additional_seed_model_config_paths=alternate_seed_model_config_paths,
                )
            elif args.agent_mode == "structured_projected_search_v5":
                alternate_seed_model_config_paths = []
                for candidate_seed_config_path in (default_seed_model_config, default_hpo_seed_model_config):
                    resolved_candidate_seed_config_path = Path(candidate_seed_config_path).resolve()
                    if resolved_candidate_seed_config_path == seed_model_config_path:
                        continue
                    alternate_seed_model_config_paths.append(resolved_candidate_seed_config_path)
                row = run_structured_model_session_v5(
                    repo_root=ROOT,
                    dataset_key=dataset_key,
                    dataset_config=dataset_config,
                    split_path=split_path,
                    seed_model_config_path=seed_model_config_path,
                    runtime_env_config=runtime_env_config,
                    requested_device=str(args.device),
                    cuda_visible_devices=args.cuda_visible_devices,
                    runtime_mode=str(args.runtime_mode),
                    runtime_env_group=args.runtime_env_group,
                    agent_root=agent_root,
                    seed=int(args.seed),
                    top_k=int(args.top_k),
                    max_iteration=int(args.max_iteration),
                    session_id=session_id,
                    llm_config_path=llm_config_path,
                    agent_mode=str(args.agent_mode),
                    additional_seed_model_config_paths=alternate_seed_model_config_paths,
                )
            elif args.agent_mode == "structured_adaptive_tree_search_v6":
                alternate_seed_model_config_paths = []
                for candidate_seed_config_path in (default_seed_model_config, default_hpo_seed_model_config):
                    resolved_candidate_seed_config_path = Path(candidate_seed_config_path).resolve()
                    if resolved_candidate_seed_config_path == seed_model_config_path:
                        continue
                    alternate_seed_model_config_paths.append(resolved_candidate_seed_config_path)
                row = run_structured_model_session_v6(
                    repo_root=ROOT,
                    dataset_key=dataset_key,
                    dataset_config=dataset_config,
                    split_path=split_path,
                    seed_model_config_path=seed_model_config_path,
                    runtime_env_config=runtime_env_config,
                    requested_device=str(args.device),
                    cuda_visible_devices=args.cuda_visible_devices,
                    runtime_mode=str(args.runtime_mode),
                    runtime_env_group=args.runtime_env_group,
                    agent_root=agent_root,
                    seed=int(args.seed),
                    top_k=int(args.top_k),
                    max_iteration=int(args.max_iteration),
                    session_id=session_id,
                    llm_config_path=llm_config_path,
                    agent_mode=str(args.agent_mode),
                    additional_seed_model_config_paths=alternate_seed_model_config_paths,
                )
            elif args.agent_mode == "structured_anchor_tree_search_v7":
                alternate_seed_model_config_paths = []
                v7_anchor_config_paths = [
                    ROOT / "configs" / "models" / "gears_anchor.yaml",
                    ROOT / "configs" / "models" / "cpa_anchor.yaml",
                    ROOT / "configs" / "models" / "cellot_anchor.yaml",
                    ROOT / "configs" / "models" / "xpert_anchor.yaml",
                    ROOT / "configs" / "models" / "random_edit_anchor.yaml",
                ]
                for candidate_seed_config_path in (default_seed_model_config, default_hpo_seed_model_config, *v7_anchor_config_paths):
                    resolved_candidate_seed_config_path = Path(candidate_seed_config_path).resolve()
                    if resolved_candidate_seed_config_path == seed_model_config_path:
                        continue
                    alternate_seed_model_config_paths.append(resolved_candidate_seed_config_path)
                row = run_structured_model_session_v7(
                    repo_root=ROOT,
                    dataset_key=dataset_key,
                    dataset_config=dataset_config,
                    split_path=split_path,
                    seed_model_config_path=seed_model_config_path,
                    runtime_env_config=runtime_env_config,
                    requested_device=str(args.device),
                    cuda_visible_devices=args.cuda_visible_devices,
                    runtime_mode=str(args.runtime_mode),
                    runtime_env_group=args.runtime_env_group,
                    agent_root=agent_root,
                    seed=int(args.seed),
                    top_k=int(args.top_k),
                    max_iteration=int(args.max_iteration),
                    session_id=session_id,
                    llm_config_path=llm_config_path,
                    agent_mode=str(args.agent_mode),
                    additional_seed_model_config_paths=alternate_seed_model_config_paths,
                )
            elif args.agent_mode == "structured_open_mechanism_search_v8":
                row = run_structured_model_session_v8(
                    repo_root=ROOT,
                    dataset_key=dataset_key,
                    dataset_config=dataset_config,
                    split_path=split_path,
                    seed_model_config_path=seed_model_config_path,
                    runtime_env_config=runtime_env_config,
                    requested_device=str(args.device),
                    cuda_visible_devices=args.cuda_visible_devices,
                    runtime_mode=str(args.runtime_mode),
                    runtime_env_group=args.runtime_env_group,
                    agent_root=agent_root,
                    baseline_root=baseline_root,
                    seed=int(args.seed),
                    top_k=int(args.top_k),
                    max_iteration=int(args.max_iteration),
                    session_id=session_id,
                    llm_config_path=llm_config_path,
                    agent_mode=str(args.agent_mode),
                )
            elif args.agent_mode == "structured_open_mechanism_search_v8_practical":
                row = run_structured_model_session_v8(
                    repo_root=ROOT,
                    dataset_key=dataset_key,
                    dataset_config=dataset_config,
                    split_path=split_path,
                    seed_model_config_path=seed_model_config_path,
                    runtime_env_config=runtime_env_config,
                    requested_device=str(args.device),
                    cuda_visible_devices=args.cuda_visible_devices,
                    runtime_mode=str(args.runtime_mode),
                    runtime_env_group=args.runtime_env_group,
                    agent_root=agent_root,
                    baseline_root=baseline_root,
                    seed=int(args.seed),
                    top_k=int(args.top_k),
                    max_iteration=int(args.max_iteration),
                    session_id=session_id,
                    llm_config_path=llm_config_path,
                    agent_mode=str(args.agent_mode),
                )
            else:
                row = run_structured_model_session(
                    repo_root=ROOT,
                    dataset_key=dataset_key,
                    dataset_config=dataset_config,
                    split_path=split_path,
                    seed_model_config_path=seed_model_config_path,
                    runtime_env_config=runtime_env_config,
                    requested_device=str(args.device),
                    cuda_visible_devices=args.cuda_visible_devices,
                    runtime_mode=str(args.runtime_mode),
                    runtime_env_group=args.runtime_env_group,
                    agent_root=agent_root,
                    seed=int(args.seed),
                    top_k=int(args.top_k),
                    max_iteration=int(args.max_iteration),
                    session_id=session_id,
                    llm_config_path=llm_config_path,
                    llm_strategy=llm_strategy,
                    agent_mode=str(args.agent_mode),
                )
            dataset_rows.append(row)
            global_rows.append(row)
            print(
                f"[agent] dataset={dataset_key} structured_seed_model={seed_model_config_path.name} completed "
                f"best_iteration={row['best_iteration']} objective_improvement={row['objective_improvement']}",
                flush=True,
            )
            methods_for_summary = [row["method_name"]]
        elif args.agent_mode in {"direct_code_llm_singleshot", "direct_code_llm_repairloop"}:
            seed_model_config_path = Path(args.seed_model_config).resolve() if args.seed_model_config else default_seed_model_config
            if not seed_model_config_path.exists():
                raise SystemExit(f"Direct-code seed model config does not exist: {seed_model_config_path}")
            llm_config_path = Path(args.llm_config).resolve() if args.llm_config else default_llm_config
            if not llm_config_path.exists():
                raise SystemExit(f"{args.agent_mode} requires a reachable llm config. Expected {llm_config_path}.")
            dataset_entry = dict(dict(matrix["datasets"])[dataset_key])
            dataset_config = (ROOT / str(dataset_entry["dataset_config"])).resolve()
            split_path = (ROOT / str(dataset_entry["primary_split"])).resolve()
            print(
                f"[agent] dataset={dataset_key} direct_code_seed_model={seed_model_config_path.name} "
                f"agent_mode={args.agent_mode} start",
                flush=True,
            )
            row = run_direct_code_model_session(
                repo_root=ROOT,
                dataset_key=dataset_key,
                dataset_config=dataset_config,
                split_path=split_path,
                seed_model_config_path=seed_model_config_path,
                runtime_env_config=runtime_env_config,
                requested_device=str(args.device),
                cuda_visible_devices=args.cuda_visible_devices,
                runtime_mode=str(args.runtime_mode),
                runtime_env_group=args.runtime_env_group,
                agent_root=agent_root,
                seed=int(args.seed),
                top_k=int(args.top_k),
                max_iteration=int(args.max_iteration),
                session_id=session_id,
                llm_config_path=llm_config_path,
                agent_mode=str(args.agent_mode),
            )
            dataset_rows.append(row)
            global_rows.append(row)
            print(
                f"[agent] dataset={dataset_key} direct_code_seed_model={seed_model_config_path.name} completed "
                f"best_iteration={row['best_iteration']} objective_improvement={row['objective_improvement']}",
                flush=True,
            )
            methods_for_summary = [row["method_name"]]
        elif args.agent_mode in {"random_edit_uniform", "random_edit_stratified"}:
            seed_model_config_path = Path(args.seed_model_config).resolve() if args.seed_model_config else default_seed_model_config
            if not seed_model_config_path.exists():
                raise SystemExit(f"Random-edit seed model config does not exist: {seed_model_config_path}")
            dataset_entry = dict(dict(matrix["datasets"])[dataset_key])
            dataset_config = (ROOT / str(dataset_entry["dataset_config"])).resolve()
            split_path = (ROOT / str(dataset_entry["primary_split"])).resolve()
            print(
                f"[agent] dataset={dataset_key} random_edit_seed_model={seed_model_config_path.name} "
                f"agent_mode={args.agent_mode} start",
                flush=True,
            )
            row = run_random_edit_model_session(
                repo_root=ROOT,
                dataset_key=dataset_key,
                dataset_config=dataset_config,
                split_path=split_path,
                seed_model_config_path=seed_model_config_path,
                runtime_env_config=runtime_env_config,
                requested_device=str(args.device),
                cuda_visible_devices=args.cuda_visible_devices,
                runtime_mode=str(args.runtime_mode),
                runtime_env_group=args.runtime_env_group,
                agent_root=agent_root,
                seed=int(args.seed),
                top_k=int(args.top_k),
                max_iteration=int(args.max_iteration),
                session_id=session_id,
                agent_mode=str(args.agent_mode),
            )
            dataset_rows.append(row)
            global_rows.append(row)
            print(
                f"[agent] dataset={dataset_key} random_edit_seed_model={seed_model_config_path.name} completed "
                f"best_iteration={row['best_iteration']} objective_improvement={row['objective_improvement']}",
                flush=True,
            )
            methods_for_summary = [row["method_name"]]
        elif args.agent_mode in {"hpo_optuna_tpe", "hpo_flaml_cfo"}:
            seed_model_config_path = Path(args.seed_model_config).resolve() if args.seed_model_config else default_hpo_seed_model_config
            if not seed_model_config_path.exists():
                raise SystemExit(f"HPO seed model config does not exist: {seed_model_config_path}")
            dataset_entry = dict(dict(matrix["datasets"])[dataset_key])
            dataset_config = (ROOT / str(dataset_entry["dataset_config"])).resolve()
            split_path = (ROOT / str(dataset_entry["primary_split"])).resolve()
            print(
                f"[agent] dataset={dataset_key} hpo_seed_model={seed_model_config_path.name} "
                f"agent_mode={args.agent_mode} start",
                flush=True,
            )
            row = run_hpo_model_session(
                repo_root=ROOT,
                dataset_key=dataset_key,
                dataset_config=dataset_config,
                split_path=split_path,
                seed_model_config_path=seed_model_config_path,
                runtime_env_config=runtime_env_config,
                requested_device=str(args.device),
                cuda_visible_devices=args.cuda_visible_devices,
                runtime_mode=str(args.runtime_mode),
                runtime_env_group=args.runtime_env_group,
                agent_root=agent_root,
                seed=int(args.seed),
                top_k=int(args.top_k),
                max_iteration=int(args.max_iteration),
                session_id=session_id,
                agent_mode=str(args.agent_mode),
            )
            dataset_rows.append(row)
            global_rows.append(row)
            print(
                f"[agent] dataset={dataset_key} hpo_seed_model={seed_model_config_path.name} completed "
                f"best_iteration={row['best_iteration']} objective_improvement={row['objective_improvement']}",
                flush=True,
            )
            methods_for_summary = [row["method_name"]]
        else:
            if args.seed_model_config:
                print("[agent] ignoring --seed-model-config because agent_mode=ablation_method_search", flush=True)
            method_names = resolve_method_names(
                matrix_config,
                dataset_key,
                methods=args.methods,
                all_methods=bool(args.all_methods),
            )
            method_names = _order_methods(method_names, matrix_config)
            print(f"[agent] dataset={dataset_key} methods={method_names}", flush=True)
            for method_name in method_names:
                print(f"[agent] dataset={dataset_key} method={method_name} start", flush=True)
                row = run_method_session(
                    repo_root=ROOT,
                    matrix_config=matrix_config,
                    runtime_env_config=runtime_env_config,
                    dataset_key=dataset_key,
                    method_name=method_name,
                    session_id=session_id,
                    max_iteration=int(args.max_iteration),
                    seed=int(args.seed),
                    top_k=int(args.top_k),
                    requested_device=str(args.device),
                    cuda_visible_devices=args.cuda_visible_devices,
                    runtime_mode=str(args.runtime_mode),
                    runtime_env_group=args.runtime_env_group,
                    agent_root=agent_root,
                    baseline_root=baseline_root,
                    agent_mode=str(args.agent_mode),
                )
                dataset_rows.append(row)
                global_rows.append(row)
                print(
                    f"[agent] dataset={dataset_key} method={method_name} completed "
                    f"best_iteration={row['best_iteration']} objective_improvement={row['objective_improvement']}",
                    flush=True,
                )
            methods_for_summary = method_names

        pack_root = write_dataset_pack_summary(
            agent_root=agent_root,
            dataset_key=dataset_key,
            session_id=session_id,
            rows=dataset_rows,
            resolved_config={
                "session_id": session_id,
                "dataset_key": dataset_key,
                "methods": methods_for_summary,
                "max_iteration": int(args.max_iteration),
                "seed": int(args.seed),
                "top_k": int(args.top_k),
                "device": str(args.device),
                "budget_semantics": RAW_ITERATION_BUDGET_SEMANTICS,
                "cuda_visible_devices": args.cuda_visible_devices,
                "matrix_config": str(matrix_config),
                "runtime_env_config": str(runtime_env_config),
                "runtime_mode": str(args.runtime_mode),
                "runtime_env_group": args.runtime_env_group,
                "agent_root": str(agent_root),
                "baseline_root": str(baseline_root),
                "agent_mode": str(args.agent_mode),
                "seed_model_config": (
                    str(seed_model_config_path)
                    if args.agent_mode in {
                        "structured_llm_search",
                        "structured_heuristic_search",
                        "structured_hybrid_search_v4",
                        "structured_projected_search_v5",
                        "structured_adaptive_tree_search_v6",
                        "structured_open_mechanism_search_v8",
                        "structured_open_mechanism_search_v8_practical",
                        "direct_code_llm_singleshot",
                        "direct_code_llm_repairloop",
                        "random_edit_uniform",
                        "random_edit_stratified",
                        "hpo_optuna_tpe",
                        "hpo_flaml_cfo",
                    }
                    else None
                ),
                "llm_config": (
                    str((Path(args.llm_config).resolve() if args.llm_config else default_llm_config))
                    if _requires_llm(str(args.agent_mode))
                    else (str(Path(args.llm_config).resolve()) if args.llm_config else None)
                ),
                "llm_strategy": (
                    "llm"
                    if args.agent_mode in {"structured_llm_search", "direct_code_llm_singleshot", "direct_code_llm_repairloop"}
                    else ("heuristic" if args.agent_mode == "structured_heuristic_search" else ("hybrid" if args.agent_mode in {"structured_hybrid_search_v4", "structured_projected_search_v5", "structured_adaptive_tree_search_v6", "structured_anchor_tree_search_v7", "structured_open_mechanism_search_v8", "structured_open_mechanism_search_v8_practical"} else str(args.llm_strategy)))
                ),
                "dataset_config": str((ROOT / str(dict(matrix["datasets"])[dataset_key]["dataset_config"])).resolve()),
                "split_path": str((ROOT / str(dict(matrix["datasets"])[dataset_key]["primary_split"])).resolve()),
            },
        )
        print(f"[agent] dataset={dataset_key} pack_summary={pack_root / 'agent_pack_summary.csv'}", flush=True)

    update_global_agent_summary(agent_root)
    print(f"[agent] global_summary={agent_root / 'global' / 'agent_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
