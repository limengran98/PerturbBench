#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sci_response.agent import (
    default_agent_root,
    default_session_id,
    resolve_dataset_keys,
    run_direct_code_model_session,
    run_hpo_model_session,
    run_random_edit_model_session,
    update_global_agent_summary,
    write_dataset_pack_summary,
)
from sci_response.agent.config import default_llm_config_path, load_agent_defaults
from sci_response.agent.formal import RAW_ITERATION_BUDGET_SEMANTICS
from sci_response.data.io import ensure_dir, load_yaml


def main() -> None:
    import argparse

    agent_defaults = load_agent_defaults()
    parser = argparse.ArgumentParser(
        description=(
            "Public PerturbBench control-line runner. "
            "This release keeps direct-code, random-edit, and HPO controls, "
            "and intentionally excludes the main structured-agent v1-v8 line."
        )
    )
    parser.add_argument("--dataset-keys", default=None, help="Comma-separated dataset keys from configs/benchmark_matrix.yaml.")
    parser.add_argument("--all-main-datasets", action="store_true", help="Run on all 7 public main datasets.")
    parser.add_argument(
        "--agent-mode",
        default=str(agent_defaults.get("agent_mode", "direct_code_llm_singleshot")),
        choices=[
            "direct_code_llm_singleshot",
            "direct_code_llm_repairloop",
            "random_edit_uniform",
            "random_edit_stratified",
            "hpo_optuna_tpe",
            "hpo_flaml_cfo",
        ],
    )
    parser.add_argument("--seed-model-config", default=None)
    parser.add_argument("--max-iteration", type=int, default=int(agent_defaults.get("max_iteration", 10)))
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--top-k", type=int, default=int(agent_defaults.get("top_k", 20)))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--matrix-config", default=str(ROOT / "configs" / "benchmark_matrix.yaml"))
    parser.add_argument("--runtime-env-config", default=str(ROOT / "configs" / "runtime_envs.yaml"))
    parser.add_argument("--runtime-mode", default=str(agent_defaults.get("runtime_mode", "primary")), choices=["primary", "fallback", "upstream"])
    parser.add_argument("--runtime-env-group", default=None)
    parser.add_argument("--agent-root", default=str(default_agent_root(ROOT)))
    parser.add_argument("--baseline-root", default=str(ROOT / "baseline"))
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--llm-config", default=None, help="Required for direct-code modes.")
    args = parser.parse_args()

    if args.max_iteration < 0:
        raise SystemExit("--max-iteration must be >= 0")

    matrix_config = Path(args.matrix_config).resolve()
    runtime_env_config = Path(args.runtime_env_config).resolve()
    agent_root = ensure_dir(Path(args.agent_root).resolve())
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

    for dataset_key in dataset_keys:
        dataset_rows: List[Dict[str, Any]] = []
        dataset_entry = dict(dict(matrix["datasets"])[dataset_key])
        dataset_config = (ROOT / str(dataset_entry["dataset_config"])).resolve()
        split_path = (ROOT / str(dataset_entry["primary_split"])).resolve()

        if args.agent_mode in {"direct_code_llm_singleshot", "direct_code_llm_repairloop"}:
            seed_model_config_path = Path(args.seed_model_config).resolve() if args.seed_model_config else default_seed_model_config
            llm_config_path = Path(args.llm_config).resolve() if args.llm_config else default_llm_config
            if not seed_model_config_path.exists():
                raise SystemExit(f"Direct-code seed model config does not exist: {seed_model_config_path}")
            if not llm_config_path.exists():
                raise SystemExit(
                    f"{args.agent_mode} requires an llm config. Expected {llm_config_path}. "
                    "Copy configs/agent/llm.example.yaml to a local file and pass --llm-config."
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
            resolved_seed_model_config = str(seed_model_config_path)
            resolved_llm_config = str(llm_config_path)
        elif args.agent_mode in {"random_edit_uniform", "random_edit_stratified"}:
            seed_model_config_path = Path(args.seed_model_config).resolve() if args.seed_model_config else default_seed_model_config
            if not seed_model_config_path.exists():
                raise SystemExit(f"Random-edit seed model config does not exist: {seed_model_config_path}")
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
            resolved_seed_model_config = str(seed_model_config_path)
            resolved_llm_config = None
        else:
            seed_model_config_path = Path(args.seed_model_config).resolve() if args.seed_model_config else default_hpo_seed_model_config
            if not seed_model_config_path.exists():
                raise SystemExit(f"HPO seed model config does not exist: {seed_model_config_path}")
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
            resolved_seed_model_config = str(seed_model_config_path)
            resolved_llm_config = None

        dataset_rows.append(row)
        pack_root = write_dataset_pack_summary(
            agent_root=agent_root,
            dataset_key=dataset_key,
            session_id=session_id,
            rows=dataset_rows,
            resolved_config={
                "session_id": session_id,
                "dataset_key": dataset_key,
                "methods": [row["method_name"]],
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
                "seed_model_config": resolved_seed_model_config,
                "llm_config": resolved_llm_config,
                "llm_strategy": ("llm" if args.agent_mode in {"direct_code_llm_singleshot", "direct_code_llm_repairloop"} else None),
                "dataset_config": str(dataset_config),
                "split_path": str(split_path),
            },
        )
        print(
            f"[agent] dataset={dataset_key} completed best_iteration={row['best_iteration']} "
            f"objective_improvement={row['objective_improvement']}",
            flush=True,
        )
        print(f"[agent] dataset={dataset_key} pack_summary={pack_root / 'agent_pack_summary.csv'}", flush=True)

    update_global_agent_summary(agent_root)
    print(f"[agent] global_summary={agent_root / 'global' / 'agent_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
