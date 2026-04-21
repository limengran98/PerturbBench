from __future__ import annotations

from typing import Dict

import numpy as np

from sci_response.baselines.metrics import compute_response_metrics


def evaluate_predictions(
    y_true: np.ndarray | Dict[str, np.ndarray],
    delta_true: np.ndarray | Dict[str, np.ndarray],
    y_pred: np.ndarray | None = None,
    delta_pred: np.ndarray | None = None,
    delta_std: np.ndarray | None = None,
    top_k: int = 20,
) -> Dict[str, Dict[str, float | Dict[str, float | bool | None]]]:
    if isinstance(y_true, dict) and isinstance(delta_true, dict):
        batch = y_true
        predictions = delta_true
        return {
            "response": compute_response_metrics(batch["post"], predictions["post"], top_k=top_k),
            "delta": compute_response_metrics(batch["delta"], predictions["delta"], top_k=top_k),
        }
    if y_pred is None or delta_pred is None:
        raise ValueError("Explicit y_pred and delta_pred are required for array-based evaluation")
    return {
        "response": compute_response_metrics(y_true, y_pred, top_k=top_k),
        "delta": compute_response_metrics(delta_true, delta_pred, y_std=delta_std, top_k=top_k),
    }
