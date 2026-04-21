#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw


def _resolve_prediction_file(run_dir: Path, filename: str) -> Path:
    direct_path = run_dir / filename
    if direct_path.exists():
        return direct_path
    prediction_path = run_dir / "predictions" / filename
    if prediction_path.exists():
        return prediction_path
    return direct_path


def _rank_1d(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(values.shape[0], dtype=np.float32)
    return ranks


def _pearson_1d(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = float(np.sqrt(np.sum(np.square(x_centered)) * np.sum(np.square(y_centered))))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denom)


def _spearman_1d(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson_1d(_rank_1d(x), _rank_1d(y))


def _per_sample_pearson(target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    if target.ndim != 2 or target.shape[1] < 2:
        return np.asarray([], dtype=np.float32)
    values = np.zeros(target.shape[0], dtype=np.float32)
    for row_idx in range(target.shape[0]):
        values[row_idx] = float(_pearson_1d(target[row_idx], pred[row_idx]))
    return values


def _draw_scatter_png(path: Path, x: np.ndarray, y: np.ndarray, title: str) -> None:
    width, height = 900, 700
    margin = 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    x_min = float(min(np.min(x), np.min(y)))
    x_max = float(max(np.max(x), np.max(y)))
    if abs(x_max - x_min) < 1e-6:
        x_min -= 1.0
        x_max += 1.0

    def map_x(value: float) -> int:
        return int(margin + (value - x_min) / (x_max - x_min) * (width - 2 * margin))

    def map_y(value: float) -> int:
        return int(height - margin - (value - x_min) / (x_max - x_min) * (height - 2 * margin))

    draw.rectangle((margin, margin, width - margin, height - margin), outline="black", width=2)
    draw.line((margin, height - margin, width - margin, margin), fill=(160, 160, 160), width=1)

    max_points = min(12000, x.shape[0])
    if x.shape[0] > max_points:
        rng = np.random.default_rng(0)
        indices = np.sort(rng.choice(np.arange(x.shape[0]), size=max_points, replace=False))
        x = x[indices]
        y = y[indices]

    for x_value, y_value in zip(x.tolist(), y.tolist()):
        px = map_x(float(x_value))
        py = map_y(float(y_value))
        draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=(31, 119, 180))

    draw.text((margin, 20), title, fill="black")
    draw.text((margin, height - margin + 20), "target", fill="black")
    draw.text((10, margin - 10), "pred", fill="black")
    image.save(path)


def _draw_hist_png(path: Path, values: np.ndarray, title: str) -> None:
    width, height = 900, 700
    margin = 60
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((margin, margin, width - margin, height - margin), outline="black", width=2)

    if values.size == 0:
        draw.text((margin, 20), title + " (empty)", fill="black")
        image.save(path)
        return

    bins = 40
    hist, edges = np.histogram(values, bins=bins)
    max_count = max(int(hist.max()), 1)

    for idx in range(bins):
        left = margin + idx * (width - 2 * margin) / bins
        right = margin + (idx + 1) * (width - 2 * margin) / bins
        bar_height = (hist[idx] / max_count) * (height - 2 * margin)
        top = height - margin - bar_height
        draw.rectangle((left, top, right - 1, height - margin), fill=(214, 39, 40))

    draw.text((margin, 20), title, fill="black")
    draw.text((margin, height - margin + 20), f"range [{edges[0]:.3g}, {edges[-1]:.3g}]", fill="black")
    image.save(path)


def analyze_artifact(run_dir: Path) -> Dict[str, object]:
    pred = np.load(_resolve_prediction_file(run_dir, "pred_test.npy"), allow_pickle=False).astype(np.float32)
    target = np.load(_resolve_prediction_file(run_dir, "target_test.npy"), allow_pickle=False).astype(np.float32)
    residual = pred - target

    flat_pred = pred.reshape(-1).astype(np.float64)
    flat_target = target.reshape(-1).astype(np.float64)
    per_sample_pearson = _per_sample_pearson(target, pred)

    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    _draw_scatter_png(
        analysis_dir / "scatter_pred_vs_target.png",
        x=flat_target,
        y=flat_pred,
        title="Prediction vs Target (test)",
    )
    _draw_hist_png(
        analysis_dir / "residual_hist.png",
        values=residual.reshape(-1).astype(np.float64),
        title="Residual Histogram (pred - target)",
    )

    summary: Dict[str, object] = {
        "artifact_path": str(run_dir),
        "prediction_shape": list(pred.shape),
        "target_shape": list(target.shape),
        "pred_mean": float(np.mean(pred)),
        "pred_std": float(np.std(pred)),
        "target_mean": float(np.mean(target)),
        "target_std": float(np.std(target)),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "global_pearson": float(_pearson_1d(flat_target, flat_pred)),
        "global_spearman": float(_spearman_1d(flat_target, flat_pred)),
        "per_sample_pearson": {
            "available": bool(per_sample_pearson.size > 0),
            "mean": float(np.mean(per_sample_pearson)) if per_sample_pearson.size else None,
            "std": float(np.std(per_sample_pearson)) if per_sample_pearson.size else None,
            "p05": float(np.quantile(per_sample_pearson, 0.05)) if per_sample_pearson.size else None,
            "p50": float(np.quantile(per_sample_pearson, 0.50)) if per_sample_pearson.size else None,
            "p95": float(np.quantile(per_sample_pearson, 0.95)) if per_sample_pearson.size else None,
        },
        "analysis_files": {
            "scatter_pred_vs_target": str(analysis_dir / "scatter_pred_vs_target.png"),
            "residual_hist": str(analysis_dir / "residual_hist.png"),
        },
    }
    (analysis_dir / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze saved test predictions from a run artifact.")
    parser.add_argument("--artifact-path", required=True, help="Path to artifacts/<run_id>/")
    args = parser.parse_args()

    artifact_path = Path(args.artifact_path).resolve()
    summary = analyze_artifact(artifact_path)

    print(f"artifact_path: {summary['artifact_path']}")
    print(f"prediction_shape: {summary['prediction_shape']}")
    print(
        f"global_pearson: {summary['global_pearson']:.6f} "
        f"global_spearman: {summary['global_spearman']:.6f}"
    )
    print(
        f"residual_mean: {summary['residual_mean']:.6f} "
        f"residual_std: {summary['residual_std']:.6f}"
    )
    per_sample = summary["per_sample_pearson"]
    if per_sample["available"]:
        print(
            "per_sample_pearson: "
            f"mean={per_sample['mean']:.6f} std={per_sample['std']:.6f} "
            f"p05={per_sample['p05']:.6f} p50={per_sample['p50']:.6f} p95={per_sample['p95']:.6f}"
        )
    else:
        print("per_sample_pearson: unavailable")
    print(f"scatter_plot: {summary['analysis_files']['scatter_pred_vs_target']}")
    print(f"residual_hist: {summary['analysis_files']['residual_hist']}")


if __name__ == "__main__":
    main()
