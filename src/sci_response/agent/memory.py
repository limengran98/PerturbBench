from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from sci_response.agent.harness import dataset_mechanism_tags


def _parse_json_list(raw: Any) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except Exception:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload if str(item).strip()]


def _safe_float(raw: Any) -> float | None:
    try:
        if raw is None or str(raw).strip() == "":
            return None
        return float(raw)
    except Exception:
        return None


def _dataset_similarity(left_dataset_key: str, right_dataset_key: str) -> float:
    left = set(dataset_mechanism_tags(left_dataset_key))
    right = set(dataset_mechanism_tags(right_dataset_key))
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    union = len(left | right)
    return float(overlap / union) if union else 0.0


def _candidate_iteration_files(branch_root: Path) -> Iterable[Path]:
    for path in branch_root.glob("datasets/*/models/*/*/iterations.csv"):
        yield path


def build_cross_dataset_memory(
    *,
    target_dataset_key: str,
    branch_roots: Sequence[Path],
    min_similarity: float = 0.3,
    top_k: int = 6,
) -> Dict[str, Any]:
    axis_counter: Dict[str, Dict[str, float]] = defaultdict(lambda: {
        "weighted_positive_gain": 0.0,
        "weighted_negative_gain": 0.0,
        "positive_count": 0.0,
        "negative_count": 0.0,
        "best_count": 0.0,
        "support_count": 0.0,
    })
    source_rows: List[Dict[str, Any]] = []
    seen_files: set[str] = set()

    for root in branch_roots:
        resolved_root = root.resolve()
        if not resolved_root.exists():
            continue
        for iteration_csv in _candidate_iteration_files(resolved_root):
            csv_key = str(iteration_csv.resolve())
            if csv_key in seen_files:
                continue
            seen_files.add(csv_key)
            dataset_key = iteration_csv.parts[-5]
            similarity = _dataset_similarity(target_dataset_key, dataset_key)
            if similarity < float(min_similarity):
                continue
            with iteration_csv.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    if str(row.get("execution_status")) != "completed":
                        continue
                    axes = _parse_json_list(row.get("hypothesis_axes"))
                    if not axes:
                        continue
                    improvement = _safe_float(row.get("delta_vs_reference"))
                    if improvement is None:
                        improvement = _safe_float(row.get("delta_vs_iteration0"))
                    if improvement is None:
                        continue
                    weight = float(similarity)
                    if str(row.get("accepted_as_best", "")).lower() == "true":
                        weight += 0.25
                    for axis in axes:
                        payload = axis_counter[axis]
                        payload["support_count"] += 1.0
                        if improvement > 0:
                            payload["weighted_positive_gain"] += float(improvement) * weight
                            payload["positive_count"] += weight
                        elif improvement < 0:
                            payload["weighted_negative_gain"] += abs(float(improvement)) * weight
                            payload["negative_count"] += weight
                        if str(row.get("accepted_as_best", "")).lower() == "true":
                            payload["best_count"] += weight
                    source_rows.append(
                        {
                            "source_dataset_key": dataset_key,
                            "target_dataset_key": target_dataset_key,
                            "dataset_similarity": float(similarity),
                            "iteration": int(row.get("iteration", 0)),
                            "objective_value": _safe_float(row.get("objective_value")),
                            "delta_vs_reference": _safe_float(row.get("delta_vs_reference")),
                            "delta_vs_iteration0": _safe_float(row.get("delta_vs_iteration0")),
                            "accepted_as_best": str(row.get("accepted_as_best", "")).lower() == "true",
                            "hypothesis_axes": axes,
                            "iterations_csv": str(iteration_csv.resolve()),
                        }
                    )

    ranked_axes: List[Dict[str, Any]] = []
    for axis, payload in axis_counter.items():
        score = (
            float(payload["weighted_positive_gain"])
            + 0.5 * float(payload["best_count"])
            - 0.75 * float(payload["weighted_negative_gain"])
        )
        ranked_axes.append(
            {
                "axis": axis,
                "score": float(score),
                "weighted_positive_gain": float(payload["weighted_positive_gain"]),
                "weighted_negative_gain": float(payload["weighted_negative_gain"]),
                "positive_count": float(payload["positive_count"]),
                "negative_count": float(payload["negative_count"]),
                "best_count": float(payload["best_count"]),
                "support_count": int(payload["support_count"]),
            }
        )
    ranked_axes.sort(
        key=lambda row: (
            -float(row["score"]),
            -float(row["best_count"]),
            -float(row["weighted_positive_gain"]),
            str(row["axis"]),
        )
    )
    return {
        "target_dataset_key": target_dataset_key,
        "branch_roots": [str(Path(root).resolve()) for root in branch_roots],
        "min_similarity": float(min_similarity),
        "ranked_axes": ranked_axes[: int(top_k)],
        "source_rows": source_rows,
    }
