from __future__ import annotations

from typing import Dict

import numpy as np


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.square(y_true - y_pred)))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum(np.square(y_true - y_pred)))
    centered = y_true - np.mean(y_true, axis=0, keepdims=True)
    ss_tot = float(np.sum(np.square(centered)))
    if ss_tot <= 1e-12:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def mean_sample_pearson(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_centered = y_true - np.mean(y_true, axis=1, keepdims=True)
    pred_centered = y_pred - np.mean(y_pred, axis=1, keepdims=True)
    numerator = np.sum(true_centered * pred_centered, axis=1)
    denominator = np.sqrt(
        np.sum(np.square(true_centered), axis=1) * np.sum(np.square(pred_centered), axis=1)
    )
    valid = denominator > 1e-12
    correlations = np.zeros(y_true.shape[0], dtype=np.float64)
    correlations[valid] = numerator[valid] / denominator[valid]
    return float(np.mean(correlations))


def _rank_rows(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, axis=1, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    row_indices = np.arange(values.shape[0])[:, None]
    ranks[row_indices, order] = np.arange(values.shape[1], dtype=np.float32)[None, :]
    return ranks


def mean_sample_spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_ranks = _rank_rows(y_true)
    pred_ranks = _rank_rows(y_pred)
    return mean_sample_pearson(true_ranks, pred_ranks)


def mean_topk_overlap(y_true: np.ndarray, y_pred: np.ndarray, top_k: int = 20) -> float:
    if y_true.shape[1] == 0:
        return 0.0
    k = max(1, min(int(top_k), int(y_true.shape[1])))
    true_idx = np.argpartition(-np.abs(y_true), kth=k - 1, axis=1)[:, :k]
    pred_idx = np.argpartition(-np.abs(y_pred), kth=k - 1, axis=1)[:, :k]
    overlaps = []
    for row_idx in range(y_true.shape[0]):
        overlaps.append(len(set(true_idx[row_idx].tolist()) & set(pred_idx[row_idx].tolist())) / float(k))
    return float(np.mean(overlaps))


def uncertainty_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray | None = None) -> Dict[str, float | bool | None]:
    if y_std is None:
        return {
            "available": False,
            "nll": None,
            "rmv": None,
        }
    variance = np.square(np.clip(y_std, 1e-6, None))
    nll = 0.5 * np.mean(np.log(2.0 * np.pi * variance) + np.square(y_true - y_pred) / variance)
    rmv = float(np.sqrt(np.mean(variance)))
    return {
        "available": True,
        "nll": float(nll),
        "rmv": rmv,
    }


def compute_response_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray | None = None,
    top_k: int = 20,
) -> Dict[str, float | Dict[str, float | bool | None]]:
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "pearson": mean_sample_pearson(y_true, y_pred),
        "spearman": mean_sample_spearman(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "topk_overlap": mean_topk_overlap(y_true, y_pred, top_k=top_k),
        "uncertainty": uncertainty_metrics(y_true, y_pred, y_std=y_std),
    }
