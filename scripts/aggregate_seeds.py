#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _flatten_numeric(payload: Any, prefix: str = "") -> Dict[str, float]:
    flat: Dict[str, float] = {}
    if isinstance(payload, dict):
        for key, value in payload.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_flatten_numeric(value, next_prefix))
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        flat[prefix] = float(payload)
    return flat


def aggregate_artifacts(artifact_paths: Iterable[Path]) -> Dict[str, Any]:
    artifact_paths = [path.resolve() for path in artifact_paths]
    metric_tables: List[Dict[str, float]] = []
    for path in artifact_paths:
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        metric_tables.append(_flatten_numeric(metrics))

    keys = sorted(set().union(*(table.keys() for table in metric_tables)))
    summary_metrics: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = [table[key] for table in metric_tables if key in table]
        if not values:
            continue
        mean_value = sum(values) / len(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        summary_metrics[key] = {
            "mean": float(mean_value),
            "std": float(variance ** 0.5),
            "count": int(len(values)),
        }

    return {
        "artifact_paths": [str(path) for path in artifact_paths],
        "metrics": summary_metrics,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate multiple seed runs into mean/std summaries.")
    parser.add_argument("--artifact-path", action="append", required=True, help="Artifact directory. Repeat this flag.")
    parser.add_argument(
        "--output",
        default="seed_summary.json",
        help="Output JSON path for the aggregated seed summary.",
    )
    args = parser.parse_args()

    artifact_paths = [Path(item) for item in args.artifact_path]
    summary = aggregate_artifacts(artifact_paths)

    output_path = Path(args.output).resolve()
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"output: {output_path}")
    sample_keys = [
        key
        for key in sorted(summary["metrics"].keys())
        if key.startswith("test.delta.") and key.split(".")[-1] in {"mse", "mae", "pearson", "spearman", "r2"}
    ]
    for key in sample_keys:
        item = summary["metrics"][key]
        print(f"{key}: mean={item['mean']:.6f} std={item['std']:.6f} n={item['count']}")


if __name__ == "__main__":
    main()
