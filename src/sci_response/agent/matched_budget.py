from __future__ import annotations

from typing import Any, Dict, Sequence


MATCHED_BUDGET_STOPPING_RULE = "raw_iteration_budget_v2_attempt_limit"


def completed_evaluation_count(history: Sequence[Dict[str, Any]]) -> int:
    return int(
        sum(
            1
            for row in history
            if str(row.get("execution_status")) == "completed" and row.get("objective_value") is not None
        )
    )


def failed_history_count(history: Sequence[Dict[str, Any]]) -> int:
    return int(
        sum(
            1
            for row in history
            if str(row.get("execution_status")) != "completed"
        )
    )


def candidate_attempt_budget_limit(max_completed_evaluations: int) -> int:
    return int(max(1, int(max_completed_evaluations)))


def should_continue_search(history: Sequence[Dict[str, Any]], max_completed_evaluations: int) -> bool:
    return len(history) < candidate_attempt_budget_limit(int(max_completed_evaluations))


def default_stopping_reason(history: Sequence[Dict[str, Any]], max_completed_evaluations: int) -> str:
    if len(history) >= candidate_attempt_budget_limit(int(max_completed_evaluations)):
        return "raw_iteration_budget_exhausted"
    return "search_stopped_without_budget_exhaustion"


def aggregate_llm_usage(rows: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    summary = {
        "llm_request_count": 0,
        "llm_prompt_tokens": 0,
        "llm_completion_tokens": 0,
        "llm_total_tokens": 0,
        "llm_repair_request_count": 0,
    }
    for row in rows:
        summary["llm_request_count"] += 1
        summary["llm_prompt_tokens"] += int(row.get("prompt_tokens", 0) or 0)
        summary["llm_completion_tokens"] += int(row.get("completion_tokens", 0) or 0)
        summary["llm_total_tokens"] += int(row.get("total_tokens", 0) or 0)
        if bool(row.get("is_repair_request")):
            summary["llm_repair_request_count"] += 1
    return summary
