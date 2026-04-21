from __future__ import annotations

import json
from typing import Any, Dict, Mapping


def _get_nested(payload: Mapping[str, Any], dotted_path: str) -> Any:
    current: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return None


def _parse_train_log(train_log_text: str) -> Dict[str, Any]:
    text = str(train_log_text or "")
    lowered = text.lower()
    epoch_lines = [line for line in text.splitlines() if "epoch=" in line.lower()]
    has_divergence = "divergence_stop" in lowered or "overflow" in lowered or "nan" in lowered
    has_early_stop = "early_stop" in lowered
    has_scheduler = "lr=" in lowered
    return {
        "epoch_line_count": int(len(epoch_lines)),
        "has_divergence": bool(has_divergence),
        "has_early_stop": bool(has_early_stop),
        "has_scheduler_trace": bool(has_scheduler),
        "text_preview": "\n".join(text.splitlines()[-12:]),
    }


def run_behavior_probes_v8(
    *,
    mechanism_card: Mapping[str, Any],
    review: Mapping[str, Any],
    metrics_payload: Mapping[str, Any],
    train_log_text: str,
    reference_record: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    training_payload = dict(metrics_payload.get("training", {}))
    current_val_mse = _safe_float(_get_nested(metrics_payload, "val.delta.mse"))
    current_test_mse = _safe_float(_get_nested(metrics_payload, "test.delta.mse"))
    current_test_pearson = _safe_float(_get_nested(metrics_payload, "test.delta.pearson"))
    reference_val_mse = _safe_float(reference_record.get("val.delta.mse")) if reference_record else None
    reference_test_mse = _safe_float(reference_record.get("test.delta.mse")) if reference_record else None
    reference_test_pearson = _safe_float(reference_record.get("test.delta.pearson")) if reference_record else None

    log_summary = _parse_train_log(train_log_text)
    best_epoch = _safe_float(training_payload.get("best_epoch"))
    best_val_loss = _safe_float(training_payload.get("best_val_loss"))
    stable_training = bool(
        best_epoch is not None
        and best_epoch >= 0.0
        and not log_summary["has_divergence"]
    )

    feature_flags = dict(review.get("feature_flags", {}))
    results: dict[str, bool] = {}
    details: dict[str, Any] = {}
    for witness in mechanism_card.get("behavioral_witnesses", []):
        witness_name = str(witness)
        passed = False
        detail: Dict[str, Any] = {}
        if witness_name == "stability_improves":
            passed = stable_training and (
                reference_val_mse is None
                or (current_val_mse is not None and current_val_mse <= reference_val_mse * 1.02)
            )
            detail = {
                "stable_training": stable_training,
                "current_val_mse": current_val_mse,
                "reference_val_mse": reference_val_mse,
            }
        elif witness_name == "conditioning_sensitivity":
            passed = bool(feature_flags.get("film_conditioning") or feature_flags.get("context_gate"))
            if passed and reference_test_pearson is not None and current_test_pearson is not None:
                passed = current_test_pearson >= reference_test_pearson - 0.01
            detail = {
                "feature_flags": {
                    "film_conditioning": bool(feature_flags.get("film_conditioning")),
                    "context_gate": bool(feature_flags.get("context_gate")),
                },
                "current_test_pearson": current_test_pearson,
                "reference_test_pearson": reference_test_pearson,
            }
        elif witness_name == "delta_head_specialization":
            passed = bool(feature_flags.get("dual_head") or feature_flags.get("residual_output_head"))
            if passed and current_test_mse is not None and reference_test_mse is not None:
                passed = current_test_mse <= reference_test_mse * 1.03
            detail = {
                "feature_flags": {
                    "dual_head": bool(feature_flags.get("dual_head")),
                    "residual_output_head": bool(feature_flags.get("residual_output_head")),
                },
                "current_test_mse": current_test_mse,
                "reference_test_mse": reference_test_mse,
            }
        elif witness_name == "skip_path_contribution":
            passed = bool(feature_flags.get("global_skip"))
            if passed and current_test_mse is not None and reference_test_mse is not None:
                passed = current_test_mse <= reference_test_mse
            detail = {
                "feature_flags": {"global_skip": bool(feature_flags.get("global_skip"))},
                "current_test_mse": current_test_mse,
                "reference_test_mse": reference_test_mse,
            }
        elif witness_name == "regularization_effect":
            plugin_present = bool(
                feature_flags.get("ema_plugin")
                or feature_flags.get("mixup_plugin")
                or feature_flags.get("input_noise")
                or feature_flags.get("dropout_gate")
            )
            passed = plugin_present and stable_training
            if passed and reference_val_mse is not None and current_val_mse is not None:
                passed = current_val_mse <= reference_val_mse * 1.03
            detail = {
                "plugin_present": plugin_present,
                "stable_training": stable_training,
                "current_val_mse": current_val_mse,
                "reference_val_mse": reference_val_mse,
            }
        elif witness_name == "loss_coupling_effect":
            passed = bool(feature_flags.get("pcc_loss"))
            if passed and current_test_pearson is not None and reference_test_pearson is not None:
                passed = current_test_pearson >= reference_test_pearson
            detail = {
                "feature_flags": {"pcc_loss": bool(feature_flags.get("pcc_loss"))},
                "current_test_pearson": current_test_pearson,
                "reference_test_pearson": reference_test_pearson,
            }
        results[witness_name] = bool(passed)
        details[witness_name] = detail

    passed_count = sum(1 for value in results.values() if value)
    score = float(passed_count / max(1, len(results))) if results else 0.0
    return {
        "passed": bool(score >= 0.5 or (results and results.get("stability_improves") is True)),
        "score": float(score),
        "results": results,
        "details": details,
        "stable_training": bool(stable_training),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "training_payload": training_payload,
        "train_log_summary": log_summary,
    }


def stability_probe_v8(
    *,
    train_log_text: str,
    metrics_payload: Mapping[str, Any],
    reference_record: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    log_summary = _parse_train_log(train_log_text)
    training_payload = dict(metrics_payload.get("training", {}))
    best_epoch = _safe_float(training_payload.get("best_epoch"))
    best_val_loss = _safe_float(training_payload.get("best_val_loss"))
    reference_val_mse = _safe_float(reference_record.get("val.delta.mse")) if reference_record else None
    current_val_mse = _safe_float(_get_nested(metrics_payload, "val.delta.mse"))
    stable = bool(
        best_epoch is not None
        and best_epoch >= 0.0
        and not log_summary["has_divergence"]
        and log_summary["epoch_line_count"] > 0
    )
    if stable and reference_val_mse is not None and current_val_mse is not None:
        stable = current_val_mse <= reference_val_mse * 1.08
    score = 1.0 if stable else 0.0
    if stable and log_summary["has_early_stop"]:
        score = 1.1
    return {
        "passed": bool(stable),
        "score": float(score),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "train_log_summary": log_summary,
    }


def summarize_behavior_probe_for_trace(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        {
            "passed": bool(payload.get("passed")),
            "score": float(payload.get("score") or 0.0),
            "results": dict(payload.get("results", {})),
        },
        ensure_ascii=True,
        sort_keys=True,
    )
