#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sci_response.baselines.artifacts import beijing_timestamp_slug
from sci_response.data.io import dump_json, ensure_dir, load_json, write_text
from sci_response.final_results import refresh_all_final_results
from sci_response.pathing import repo_relative_str


def flatten_metrics(run_id: str, manifest: Dict[str, object], metrics: Dict[str, object]) -> Dict[str, object]:
    training_metrics = metrics.get("training", {})
    training_method_family = training_metrics.get("method_family") if isinstance(training_metrics, dict) else None
    training_method_name = training_metrics.get("method_name") if isinstance(training_metrics, dict) else None
    execution_status = manifest.get("execution_status")
    if execution_status is None and isinstance(training_metrics, dict):
        execution_status = training_metrics.get("status", "completed")
    row: Dict[str, object] = {
        "run_id": run_id,
        "dataset_name": manifest.get("dataset_name"),
        "method_family": manifest.get("method_family", training_method_family),
        "method_name": manifest.get("method_name", training_method_name),
        "execution_status": execution_status,
        "baseline_mode": manifest.get("baseline_mode"),
        "seed": manifest.get("seed"),
        "requested_device": manifest.get("requested_device"),
        "resolved_device": manifest.get("resolved_device"),
        "enhanced_information": manifest.get("enhanced_information"),
    }
    prediction_bundle = manifest.get("prediction_bundle", {})
    if isinstance(prediction_bundle, dict):
        row["prediction_available"] = prediction_bundle.get("available")
    for split_name in ["train", "val", "test"]:
        split_metrics = metrics.get(split_name, {})
        if not isinstance(split_metrics, dict):
            continue
        for target_name, target_metrics in split_metrics.items():
            if not isinstance(target_metrics, dict):
                continue
            for metric_name, value in target_metrics.items():
                row[f"{split_name}.{target_name}.{metric_name}"] = value
    if isinstance(training_metrics, dict):
        for key, value in training_metrics.items():
            row[f"training.{key}"] = value
    return row


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate artifact metrics into a single CSV and JSON summary.")
    parser.add_argument("--artifacts-root", default="artifacts", help="Root directory containing run subdirectories.")
    parser.add_argument("--output-csv", default="artifacts/global/benchmark_summary.csv")
    parser.add_argument("--output-json", default="artifacts/global/benchmark_summary.json")
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts_root).resolve()
    rows: List[Dict[str, object]] = []
    for manifest_path in sorted(artifacts_root.rglob("manifest.json")):
        run_dir = manifest_path.parent
        if run_dir.name == "predictions":
            continue
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        manifest = load_json(manifest_path)
        metrics = load_json(metrics_path)
        run_id = str(run_dir.relative_to(artifacts_root))
        rows.append(flatten_metrics(run_id, manifest, metrics))

    if not rows:
        raise RuntimeError(f"No completed runs found under {artifacts_root}")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    output_csv = Path(args.output_csv).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    output_json = Path(args.output_json).resolve()
    dump_json(output_json, {"runs": rows})

    global_root = ensure_dir(output_csv.parent)
    history_root = ensure_dir(global_root / "history" / beijing_timestamp_slug())
    history_csv = history_root / output_csv.name
    history_json = history_root / output_json.name
    with history_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    dump_json(history_json, {"runs": rows})
    dump_json(
        history_root / "summary_manifest.json",
        {
            "row_count": int(len(rows)),
            "latest_csv": repo_relative_str(output_csv),
            "latest_json": repo_relative_str(output_json),
            "history_csv": repo_relative_str(history_csv),
            "history_json": repo_relative_str(history_json),
        },
    )
    write_text(global_root / "LATEST_BENCHMARK_SUMMARY.txt", repo_relative_str(history_root) + "\n")
    refresh_all_final_results(repo_root=ROOT, updated_branch_root=artifacts_root)

    print(f"aggregate_csv={repo_relative_str(output_csv)}")
    print(f"aggregate_json={repo_relative_str(output_json)}")
    print(f"aggregate_history_dir={repo_relative_str(history_root)}")


if __name__ == "__main__":
    main()
