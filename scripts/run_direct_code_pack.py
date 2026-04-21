#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sci_response.agent.config import default_llm_config_path, load_agent_defaults
from sci_response.agent.formal import RAW_ITERATION_BUDGET_SEMANTICS, normalize_agent_result_row
from sci_response.baselines.artifacts import beijing_timestamp_slug
from sci_response.data.io import dump_json, dump_yaml, ensure_dir, load_json, write_text
from sci_response.final_results import (
    iterative_seed_run_is_complete,
    latest_iterative_session_info,
    refresh_all_final_results,
)
from sci_response.pathing import repo_relative_str


DIRECT_CODE_MODES = ["direct_code_llm_singleshot", "direct_code_llm_repairloop"]


def _parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _flatten_numeric(payload: Any, prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_numeric(value, next_prefix))
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        flat[prefix] = float(payload)
    return flat


def _aggregate_metric_tables(metric_tables: Sequence[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = sorted(set().union(*(table.keys() for table in metric_tables)))
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = [table[key] for table in metric_tables if key in table]
        if not values:
            continue
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        summary[key] = {
            "mean": float(mean_value),
            "std": float(variance ** 0.5),
            "count": int(len(values)),
        }
    return summary


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_latest(path: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)


def _progress(message: str) -> None:
    print(f"[direct-code-pack] {message}", flush=True)


def _update_direct_code_global_summary(agent_root: Path, *, snapshot_id: str, modes: Sequence[str]) -> None:
    global_summary_path = agent_root / "global" / "agent_summary.json"
    if not global_summary_path.exists():
        return
    payload = load_json(global_summary_path)
    rows = [
        dict(row)
        for row in list(payload.get("rows", []))
        if str(row.get("agent_line")) == "direct_code_llm" or str(row.get("agent_mode")) in set(modes)
    ]
    if not rows:
        return
    line_root = ensure_dir(agent_root / "direct_code_llm" / "global")
    history_root = ensure_dir(line_root / "history" / snapshot_id)
    dump_json(line_root / "direct_code_summary.json", {"rows": rows})
    _write_csv(line_root / "direct_code_summary.csv", rows)
    dump_json(history_root / "direct_code_summary.json", {"rows": rows})
    _write_csv(history_root / "direct_code_summary.csv", rows)
    dump_json(
        history_root / "summary_manifest.json",
        {
            "row_count": int(len(rows)),
            "modes": list(modes),
            "latest_csv": repo_relative_str(line_root / "direct_code_summary.csv"),
            "latest_json": repo_relative_str(line_root / "direct_code_summary.json"),
        },
    )
    write_text(line_root / "LATEST_DIRECT_CODE_SUMMARY.txt", repo_relative_str(history_root) + "\n")


def main() -> None:
    import argparse

    agent_defaults = load_agent_defaults()
    parser = argparse.ArgumentParser(description="Run the direct_code_llm control line for one dataset across both variants and multiple seeds.")
    parser.add_argument("--dataset-key", required=True)
    parser.add_argument("--agent-modes", default=",".join(DIRECT_CODE_MODES))
    parser.add_argument("--seeds", default="11,12,13")
    parser.add_argument("--max-iteration", type=int, default=int(agent_defaults.get("max_iteration", 10)))
    parser.add_argument("--seed-model-config", default=str((ROOT / str(agent_defaults.get("seed_model_config", "configs/models/conditioned_residual.yaml"))).resolve()))
    parser.add_argument("--llm-config", default=str((ROOT / str(agent_defaults.get("llm_config", str(default_llm_config_path())))).resolve()))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default=None)
    parser.add_argument("--runtime-env-config", default=str(ROOT / "configs" / "runtime_envs.yaml"))
    parser.add_argument("--runtime-mode", default=str(agent_defaults.get("runtime_mode", "primary")), choices=["primary", "fallback", "upstream"])
    parser.add_argument("--runtime-env-group", default=None)
    parser.add_argument("--agent-root", default=str(ROOT / "direct_code_runs"))
    parser.add_argument("--top-k", type=int, default=int(agent_defaults.get("top_k", 20)))
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    modes = [mode for mode in _parse_csv_list(args.agent_modes) if mode in set(DIRECT_CODE_MODES)]
    if not modes:
        raise SystemExit(f"No valid direct-code modes requested from {args.agent_modes!r}")

    agent_root = ensure_dir(Path(args.agent_root).resolve())
    line_dataset_root = ensure_dir(agent_root / "direct_code_llm" / "datasets" / str(args.dataset_key))
    pack_id = beijing_timestamp_slug()
    pack_root = ensure_dir(line_dataset_root / "pack_runs" / pack_id)
    latest_root = ensure_dir(line_dataset_root / "latest")

    per_seed_rows: List[Dict[str, Any]] = []
    seed_values = [int(item) for item in _parse_csv_list(args.seeds)]
    seed_model_name = Path(args.seed_model_config).stem
    for mode in modes:
        for index, seed in enumerate(seed_values, start=1):
            session_id = f"{pack_id}__{mode}__seed{seed}"
            session_pack_path = agent_root / "datasets" / str(args.dataset_key) / "pack_runs" / session_id / "agent_pack_summary.json"
            _progress(
                f"dataset={args.dataset_key} mode={mode} seed[{index}/{len(seed_values)}]={seed} "
                f"max_iteration={args.max_iteration}"
            )
            existing_completed = None
            if args.skip_existing:
                existing_completed = iterative_seed_run_is_complete(
                    repo_root=ROOT,
                    branch_root=agent_root,
                    comparison_group="direct_code_llm",
                    comparison_method=str(mode),
                    dataset_key=str(args.dataset_key),
                    seed_model_name=seed_model_name,
                    seed=int(seed),
                    budget_limit=int(args.max_iteration),
                )
                if existing_completed is not None:
                    session_id = str(existing_completed["session_id"])
                    session_pack_path = agent_root / "datasets" / str(args.dataset_key) / "pack_runs" / session_id / "agent_pack_summary.json"
                else:
                    existing_partial = latest_iterative_session_info(
                        repo_root=ROOT,
                        branch_root=agent_root,
                        comparison_group="direct_code_llm",
                        comparison_method=str(mode),
                        dataset_key=str(args.dataset_key),
                        seed_model_name=seed_model_name,
                        seed=int(seed),
                    )
                    if existing_partial is not None:
                        session_id = str(existing_partial["session_id"])
                        session_pack_path = agent_root / "datasets" / str(args.dataset_key) / "pack_runs" / session_id / "agent_pack_summary.json"
            if existing_completed is None:
                cmd = [
                    sys.executable,
                    str(ROOT / "scripts" / "run_agent.py"),
                    "--dataset-keys",
                    str(args.dataset_key),
                    "--agent-mode",
                    str(mode),
                    "--seed",
                    str(seed),
                    "--top-k",
                    str(int(args.top_k)),
                    "--max-iteration",
                    str(int(args.max_iteration)),
                    "--device",
                    str(args.device),
                    "--runtime-env-config",
                    str(Path(args.runtime_env_config).resolve()),
                    "--runtime-mode",
                    str(args.runtime_mode),
                    "--agent-root",
                    str(agent_root),
                    "--session-id",
                    session_id,
                    "--seed-model-config",
                    str(Path(args.seed_model_config).resolve()),
                    "--llm-config",
                    str(Path(args.llm_config).resolve()),
                ]
                if args.cuda_visible_devices is not None:
                    cmd.extend(["--cuda-visible-devices", str(args.cuda_visible_devices)])
                if args.runtime_env_group is not None:
                    cmd.extend(["--runtime-env-group", str(args.runtime_env_group)])
                subprocess.run(cmd, check=True, cwd=str(ROOT))
            else:
                _progress(
                    f"reuse_existing dataset={args.dataset_key} mode={mode} seed={seed} session_id={session_id}"
                )
            if existing_completed is None and session_id != f"{pack_id}__{mode}__seed{seed}":
                _progress(
                    f"resume_existing dataset={args.dataset_key} mode={mode} seed={seed} session_id={session_id}"
                )

            payload = load_json(session_pack_path)
            rows = list(payload.get("rows", []))
            if len(rows) != 1:
                raise RuntimeError(f"Expected exactly one row in {session_pack_path}, found {len(rows)}")
            row = normalize_agent_result_row(dict(rows[0]), max_iteration=int(args.max_iteration))
            row["seed"] = int(seed)
            row["session_id"] = session_id
            row["agent_mode"] = str(mode)
            per_seed_rows.append(row)

    summary_rows: List[Dict[str, Any]] = []
    for mode in modes:
        grouped = [row for row in per_seed_rows if str(row.get("agent_mode")) == mode]
        aggregate_metrics = _aggregate_metric_tables([_flatten_numeric(row) for row in grouped])
        summary_row: Dict[str, Any] = {
            "dataset_key": str(args.dataset_key),
            "agent_line": "direct_code_llm",
            "agent_mode": str(mode),
            "budget_semantics": RAW_ITERATION_BUDGET_SEMANTICS,
            "seed_count": int(len(grouped)),
            "seeds_json": json.dumps([int(row["seed"]) for row in grouped], ensure_ascii=True),
            "pack_id": pack_id,
            "max_iteration": int(args.max_iteration),
            "seed_model_config": repo_relative_str(Path(args.seed_model_config)),
            "llm_config": repo_relative_str(Path(args.llm_config)),
            "device": str(args.device),
            "cuda_visible_devices": args.cuda_visible_devices,
            "session_ids_json": json.dumps([row["session_id"] for row in grouped], ensure_ascii=True),
            "method_name": grouped[0].get("method_name") if grouped else None,
            "seed_model_name": grouped[0].get("seed_model_name") if grouped else None,
        }
        for metric_key, metric_value in aggregate_metrics.items():
            summary_row[f"{metric_key}_mean"] = metric_value["mean"]
            summary_row[f"{metric_key}_std"] = metric_value["std"]
            summary_row[f"{metric_key}_count"] = metric_value["count"]
        summary_rows.append(summary_row)

    dump_json(pack_root / "direct_code_pack_summary.json", {"rows": summary_rows})
    _write_csv(pack_root / "direct_code_pack_summary.csv", summary_rows)
    dump_json(pack_root / "per_seed_rows.json", {"rows": per_seed_rows})
    _write_csv(pack_root / "per_seed_rows.csv", per_seed_rows)
    dump_yaml(
        pack_root / "resolved_direct_code_pack_config.yaml",
        {
            "dataset_key": str(args.dataset_key),
            "agent_modes": list(modes),
            "seeds": seed_values,
            "max_iteration": int(args.max_iteration),
            "seed_model_config": repo_relative_str(Path(args.seed_model_config)),
            "llm_config": repo_relative_str(Path(args.llm_config)),
            "device": str(args.device),
            "cuda_visible_devices": args.cuda_visible_devices,
            "runtime_env_config": repo_relative_str(Path(args.runtime_env_config)),
            "runtime_mode": str(args.runtime_mode),
            "runtime_env_group": args.runtime_env_group,
            "agent_root": repo_relative_str(agent_root),
            "budget_semantics": RAW_ITERATION_BUDGET_SEMANTICS,
        },
    )

    _copy_latest(pack_root / "direct_code_pack_summary.json", latest_root / "direct_code_pack_summary.json")
    _copy_latest(pack_root / "direct_code_pack_summary.csv", latest_root / "direct_code_pack_summary.csv")
    _copy_latest(pack_root / "per_seed_rows.json", latest_root / "per_seed_rows.json")
    _copy_latest(pack_root / "per_seed_rows.csv", latest_root / "per_seed_rows.csv")
    _copy_latest(pack_root / "resolved_direct_code_pack_config.yaml", latest_root / "resolved_direct_code_pack_config.yaml")
    write_text(line_dataset_root / "LATEST_DIRECT_CODE_PACK_RUN.txt", repo_relative_str(pack_root) + "\n")

    _update_direct_code_global_summary(agent_root, snapshot_id=pack_id, modes=modes)
    refresh_all_final_results(repo_root=ROOT, updated_branch_root=agent_root)

    print(f"direct_code_pack_dir={repo_relative_str(pack_root)}")
    print(f"direct_code_pack_summary_csv={repo_relative_str(pack_root / 'direct_code_pack_summary.csv')}")
    print(f"direct_code_pack_summary_json={repo_relative_str(pack_root / 'direct_code_pack_summary.json')}")


if __name__ == "__main__":
    main()
