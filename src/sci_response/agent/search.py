from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from sci_response.data.io import stable_hash


PARAM_PRIORITY = [
    "learning_rate",
    "alpha",
    "l1_ratio",
    "hidden_dim",
    "hidden_layers",
    "latent_dim",
    "perturb_rank",
    "hidden_node",
    "d_token",
    "n_layers",
    "n_heads",
    "depth",
    "max_depth",
    "iterations",
    "n_estimators",
    "epochs",
    "max_iter",
    "dropout",
    "weight_decay",
    "batch_size",
    "transport_weight",
    "mmd_weight",
    "latent_l2_weight",
    "baseline_recon_weight",
    "skip_alpha",
]

NON_MUTATING_KEYS = {
    "n_jobs",
    "task_type",
    "devices",
    "max_steps_per_epoch",
    "max_train_samples",
    "max_val_samples",
    "_runtime_device",
    "_requested_device",
    "_cuda_visible_devices",
}


def params_signature(params: Dict[str, Any]) -> str:
    return stable_hash({"params": params})


def _ordered_keys(params: Dict[str, Any]) -> List[str]:
    ranked = {key: index for index, key in enumerate(PARAM_PRIORITY)}
    return sorted(params.keys(), key=lambda key: (ranked.get(key, len(ranked) + 1000), str(key)))


def _clip_float(value: float, *, lower: float | None = None, upper: float | None = None) -> float:
    clipped = float(value)
    if lower is not None:
        clipped = max(clipped, float(lower))
    if upper is not None:
        clipped = min(clipped, float(upper))
    return clipped


def _dedupe_preserve(items: Iterable[Any]) -> List[Any]:
    seen = set()
    deduped: List[Any] = []
    for item in items:
        marker = repr(item)
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(item)
    return deduped


def _mutate_scalar(key: str, value: Any) -> List[Tuple[Any, str]]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        if key in {"batch_size", "iterations", "epochs", "max_iter", "n_estimators"}:
            candidates = [
                max(1, int(round(value * 0.5))),
                max(1, int(round(value * 1.5))),
                max(1, int(round(value * 2.0))),
            ]
        elif key in {"hidden_dim", "latent_dim", "perturb_rank", "hidden_node", "d_token"}:
            candidates = [
                max(4, int(round(value * 0.75 / 4.0) * 4)),
                max(4, int(round(value * 1.25 / 4.0) * 4)),
                max(4, int(round(value * 1.5 / 4.0) * 4)),
            ]
        elif key in {"n_layers", "n_heads", "depth", "max_depth"}:
            candidates = [max(1, value - 1), value + 1]
        else:
            candidates = [max(1, int(round(value * 0.8))), max(1, int(round(value * 1.25)))]
        return [(candidate, f"{key}={candidate}") for candidate in _dedupe_preserve(candidate for candidate in candidates if candidate != value)]

    if isinstance(value, float):
        if key == "dropout":
            candidates = [
                _clip_float(value - 0.05, lower=0.0, upper=0.5),
                _clip_float(value + 0.05, lower=0.0, upper=0.5),
                _clip_float(value + 0.1, lower=0.0, upper=0.5),
            ]
        elif key == "l1_ratio":
            candidates = [0.2, 0.35, 0.5, 0.65, 0.8]
        elif key in {"learning_rate", "weight_decay", "alpha", "transport_weight", "mmd_weight", "latent_l2_weight", "baseline_recon_weight"}:
            candidates = [value * 0.5, value * 0.8, value * 1.25, value * 2.0]
        elif 0.0 <= value <= 1.0:
            candidates = [
                _clip_float(value - 0.1, lower=0.0, upper=1.0),
                _clip_float(value + 0.1, lower=0.0, upper=1.0),
            ]
        else:
            candidates = [value * 0.8, value * 1.25]
        cleaned = []
        for candidate in candidates:
            if math.isclose(float(candidate), float(value), rel_tol=1e-9, abs_tol=1e-12):
                continue
            cleaned.append(float(candidate))
        return [(candidate, f"{key}={candidate}") for candidate in _dedupe_preserve(cleaned)]

    return []


def _mutate_sequence(key: str, value: Sequence[Any]) -> List[Tuple[Any, str]]:
    if not value:
        return []
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return []
    dims = [int(item) for item in value]
    candidates: List[List[int]] = []
    candidates.append([max(4, int(round(dim * 0.75 / 4.0) * 4)) for dim in dims])
    candidates.append([max(4, int(round(dim * 1.25 / 4.0) * 4)) for dim in dims])
    if len(dims) > 1:
        candidates.append(dims[:-1])
    candidates.append(dims + [dims[-1]])
    cleaned = []
    for candidate in candidates:
        if candidate == dims:
            continue
        cleaned.append(candidate)
    return [(candidate, f"{key}={candidate}") for candidate in _dedupe_preserve(cleaned)]


def generate_neighbor_candidates(params: Dict[str, Any]) -> List[Tuple[Dict[str, Any], str]]:
    candidates: List[Tuple[Dict[str, Any], str]] = []
    for key in _ordered_keys(params):
        if key in NON_MUTATING_KEYS:
            continue
        value = params[key]
        if isinstance(value, (tuple, list)):
            mutations = _mutate_sequence(key, list(value))
        else:
            mutations = _mutate_scalar(key, value)
        for mutated_value, note in mutations:
            candidate = copy.deepcopy(params)
            candidate[key] = mutated_value
            candidates.append((candidate, note))

    # A few simple paired moves for common training knobs.
    if "learning_rate" in params:
        for scale, epoch_scale in [(0.8, 1.25), (1.25, 0.8)]:
            candidate = copy.deepcopy(params)
            candidate["learning_rate"] = float(params["learning_rate"]) * scale
            if "epochs" in candidate:
                candidate["epochs"] = max(1, int(round(int(candidate["epochs"]) * epoch_scale)))
            if "iterations" in candidate:
                candidate["iterations"] = max(1, int(round(int(candidate["iterations"]) * epoch_scale)))
            if "max_iter" in candidate:
                candidate["max_iter"] = max(1, int(round(int(candidate["max_iter"]) * epoch_scale)))
            candidates.append((candidate, f"paired_lr_scale={scale}_epoch_scale={epoch_scale}"))
    return candidates


def propose_next_params(
    *,
    base_params: Dict[str, Any],
    history: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], str]:
    seen = {params_signature(dict(item.get("params", {}))) for item in history}
    completed = [item for item in history if item.get("execution_status") == "completed" and item.get("objective_value") is not None]
    if completed:
        reference_record = min(completed, key=lambda item: float(item["objective_value"]))
    else:
        reference_record = history[0] if history else {"params": base_params}
    reference_params = copy.deepcopy(dict(reference_record.get("params", base_params)))

    for candidate, note in generate_neighbor_candidates(reference_params):
        signature = params_signature(candidate)
        if signature in seen:
            continue
        return candidate, note

    fallback = copy.deepcopy(reference_params)
    numeric_keys = [key for key in _ordered_keys(fallback) if isinstance(fallback[key], (int, float)) and not isinstance(fallback[key], bool)]
    if not numeric_keys:
        return fallback, "no_mutation_available"
    selected_key = numeric_keys[len(history) % len(numeric_keys)]
    selected_value = fallback[selected_key]
    if isinstance(selected_value, int):
        fallback[selected_key] = max(1, int(round(selected_value * 1.1)))
    else:
        fallback[selected_key] = float(selected_value) * 1.1
    return fallback, f"fallback_jitter_{selected_key}"
