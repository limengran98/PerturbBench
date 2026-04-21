from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from sci_response.agent.edits import describe_edit
from sci_response.agent.ir import enumerate_editable_sites
from sci_response.agent.prompts import build_mechanism_fidelity_prompts
from sci_response.data.io import load_json, load_yaml


class MalformedLLMResponseError(ValueError):
    def __init__(self, message: str, *, raw_content: str | None = None, raw_payload: Dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.raw_content = raw_content
        self.raw_payload = raw_payload


def _load_structured_config(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".json":
        return load_json(path)
    return load_yaml(path)


def _extract_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        raise ValueError("LLM response does not contain a JSON object")
    payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Expected JSON object from LLM response")
    return payload


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    model: str
    temperature: float
    max_tokens: int
    timeout: int
    retry_attempts: int
    retry_backoff_seconds: float
    compact_retry_max_tokens: int
    prompt_compaction_threshold_chars: int
    base_url: str
    api_key: str


def load_llm_settings(config_path: Path) -> LLMSettings:
    payload = _load_structured_config(config_path)
    llm_payload = dict(payload.get("llm", {}))
    providers_payload = dict(payload.get("providers", {}))
    provider_name = str(llm_payload["provider"])
    provider_payload = dict(providers_payload[provider_name])
    api_key = str(provider_payload.get("api_key") or os.environ.get("SCI_RESPONSE_LLM_API_KEY", "")).strip()
    if not api_key:
        raise ValueError(f"LLM provider {provider_name!r} is missing api_key and SCI_RESPONSE_LLM_API_KEY is not set")
    return LLMSettings(
        provider=provider_name,
        model=str(llm_payload["model"]),
        temperature=float(llm_payload.get("temperature", 0.2)),
        max_tokens=int(llm_payload.get("max_tokens", 4096)),
        timeout=int(llm_payload.get("timeout", 120)),
        retry_attempts=int(llm_payload.get("retry_attempts", 3)),
        retry_backoff_seconds=float(llm_payload.get("retry_backoff_seconds", 2.0)),
        compact_retry_max_tokens=int(llm_payload.get("compact_retry_max_tokens", 1024)),
        prompt_compaction_threshold_chars=int(llm_payload.get("prompt_compaction_threshold_chars", 12000)),
        base_url=str(provider_payload["base_url"]).rstrip("/"),
        api_key=api_key,
    )


class StructuredLLMClient:
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self._usage_events: List[Dict[str, Any]] = []

    def _endpoint(self) -> str:
        return f"{self.settings.base_url}/chat/completions"

    def _chat_content(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens_override: int | None = None,
        force_json_response: bool = False,
        request_label: str = "chat",
        is_repair_request: bool = False,
    ) -> tuple[str, Dict[str, Any]]:
        payload = {
            "model": self.settings.model,
            "temperature": self.settings.temperature,
            "max_tokens": int(max_tokens_override or self.settings.max_tokens),
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if force_json_response:
            payload["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            self._endpoint(),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.api_key}",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt_index in range(1, max(1, int(self.settings.retry_attempts)) + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.settings.timeout) as response:
                    raw = response.read().decode("utf-8")
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                retryable = 500 <= int(exc.code) < 600
                last_error = RuntimeError(f"LLM HTTP error {exc.code}: {body}")
                if retryable and attempt_index < int(self.settings.retry_attempts):
                    time.sleep(float(self.settings.retry_backoff_seconds) * attempt_index)
                    continue
                raise last_error from exc
            except Exception as exc:
                last_error = exc
                retryable = _is_retryable_exception(exc)
                if retryable and attempt_index < int(self.settings.retry_attempts):
                    time.sleep(float(self.settings.retry_backoff_seconds) * attempt_index)
                    continue
                raise RuntimeError(f"LLM request failed: {exc}") from exc
        else:  # pragma: no cover
            raise RuntimeError(f"LLM request failed after retries: {last_error}")
        payload = json.loads(raw)
        try:
            content = payload["choices"][0]["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Unexpected LLM response schema: {payload}") from exc
        usage_payload = payload.get("usage", {})
        if not isinstance(usage_payload, dict):
            usage_payload = {}
        self._usage_events.append(
            {
                "request_label": str(request_label),
                "is_repair_request": bool(is_repair_request),
                "provider": self.settings.provider,
                "model": self.settings.model,
                "max_tokens": int(max_tokens_override or self.settings.max_tokens),
                "prompt_tokens": int(usage_payload.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage_payload.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage_payload.get("total_tokens", 0) or 0),
                "force_json_response": bool(force_json_response),
            }
        )
        return str(content), payload

    def _repair_json_response(self, raw_content: str, *, max_tokens_override: int | None = None) -> Dict[str, Any]:
        repair_system = (
            "You are a JSON repair assistant. Rewrite the provided assistant output into exactly one valid JSON object. "
            "Return JSON only, with no markdown fences and no explanation."
        )
        repair_user = (
            "Convert the following model response into one valid JSON object that follows the same intent. "
            "Preserve fields if present, otherwise infer the minimal valid structure. "
            "Expected top-level fields are proposal_note, selected_mechanism_templates, scientific_hypothesis, "
            "mechanistic_rationale, risk_notes, edits.\n\n"
            "SOURCE RESPONSE:\n"
            f"{raw_content}\n"
        )
        repaired_content, _ = self._chat_content(
            system_prompt=repair_system,
            user_prompt=repair_user,
            max_tokens_override=max_tokens_override
            or min(int(self.settings.compact_retry_max_tokens), int(self.settings.max_tokens)),
            force_json_response=True,
            request_label="json_repair",
            is_repair_request=True,
        )
        return _extract_json_object(repaired_content)

    def chat_text(self, *, system_prompt: str, user_prompt: str, max_tokens_override: int | None = None) -> str:
        content, _ = self._chat_content(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens_override=max_tokens_override,
            force_json_response=False,
            request_label="chat_text",
        )
        return str(content)

    def chat_json(self, *, system_prompt: str, user_prompt: str, max_tokens_override: int | None = None) -> Dict[str, Any]:
        content, raw_payload = self._chat_content(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens_override=max_tokens_override,
            force_json_response=True,
            request_label="chat_json",
        )
        try:
            return _extract_json_object(content)
        except ValueError as exc:
            try:
                return self._repair_json_response(
                    content,
                    max_tokens_override=min(
                        int(max_tokens_override or self.settings.max_tokens),
                        int(self.settings.compact_retry_max_tokens),
                    ),
                )
            except Exception as repair_exc:
                raise MalformedLLMResponseError(
                    f"{exc}; repair_failed={type(repair_exc).__name__}: {repair_exc}",
                    raw_content=content,
                    raw_payload=raw_payload,
                ) from exc

    def drain_usage_events(self) -> List[Dict[str, Any]]:
        events = list(self._usage_events)
        self._usage_events.clear()
        return events


def _is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, socket.timeout, urllib.error.URLError)):
        return True
    message = str(exc).lower()
    return "timed out" in message or "temporarily unavailable" in message or "connection reset" in message


def _retryable_runtime_error(exc: Exception) -> bool:
    return _is_retryable_exception(exc) or "llm http error 5" in str(exc).lower()


def _normalize_llm_edit(edit: Dict[str, Any], *, model_ir: Dict[str, Any], site_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    normalized = dict(edit)
    primitive = str(normalized.get("primitive", "")).strip()
    path = str(normalized.get("path", "")).strip()
    if not primitive or not path:
        raise ValueError(f"LLM edit must contain primitive and path, got {edit!r}")
    normalized["primitive"] = primitive
    normalized["path"] = path

    site = site_lookup.get(path)
    if primitive == "cycle_enum":
        choices = normalized.get("choices")
        if choices is None and site is not None:
            choices = site.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"cycle_enum edit is missing valid choices for {path}")
        normalized["choices"] = [item for item in choices]
        normalized["step"] = int(normalized.get("step", 1))
    elif primitive == "increment":
        normalized["delta"] = int(normalized.get("delta", 1))
    elif primitive == "scale_numeric":
        if "factor" not in normalized:
            raise ValueError(f"scale_numeric edit is missing factor for {path}")
        normalized["factor"] = float(normalized["factor"])
    elif primitive == "set_scalar":
        if "value" not in normalized:
            raise ValueError(f"set_scalar edit is missing value for {path}")
    elif primitive == "toggle_boolean":
        pass
    else:
        raise ValueError(f"Unsupported LLM edit primitive: {primitive}")
    return normalized


def llm_propose_edit_sequence(
    *,
    llm_client: StructuredLLMClient,
    dataset_key: str,
    model_ir: Dict[str, Any],
    history: List[Dict[str, Any]],
    attempt_index: int = 1,
    recent_rejections: List[Dict[str, Any]] | None = None,
    prioritized_templates: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    prompt_variants = []
    full_prompts = build_mechanism_fidelity_prompts(
        dataset_key=dataset_key,
        model_ir=model_ir,
        history=history,
        attempt_index=attempt_index,
        recent_rejections=list(recent_rejections or []),
        prioritized_templates=prioritized_templates,
        compact_level=0,
    )
    prompt_variants.append(
        (
            "full",
            full_prompts,
            int(llm_client.settings.max_tokens),
        )
    )
    if len(full_prompts["user_prompt"]) >= int(llm_client.settings.prompt_compaction_threshold_chars):
        prompt_variants.append(
            (
                "compact",
                build_mechanism_fidelity_prompts(
                    dataset_key=dataset_key,
                    model_ir=model_ir,
                    history=history,
                    attempt_index=attempt_index,
                    recent_rejections=list(recent_rejections or []),
                    prioritized_templates=prioritized_templates,
                    compact_level=1,
                ),
                min(int(llm_client.settings.max_tokens), int(llm_client.settings.compact_retry_max_tokens)),
            )
        )
    prompt_variants.append(
        (
            "minimal",
            build_mechanism_fidelity_prompts(
                dataset_key=dataset_key,
                model_ir=model_ir,
                history=history,
                attempt_index=attempt_index,
                recent_rejections=list(recent_rejections or []),
                prioritized_templates=prioritized_templates,
                compact_level=2,
            ),
            min(int(llm_client.settings.max_tokens), max(512, int(llm_client.settings.compact_retry_max_tokens // 2))),
        )
    )

    response = None
    errors: List[str] = []
    for variant_name, prompts, max_tokens in prompt_variants:
        try:
            response = llm_client.chat_json(
                system_prompt=prompts["system_prompt"],
                user_prompt=prompts["user_prompt"],
                max_tokens_override=max_tokens,
            )
            break
        except Exception as exc:
            errors.append(f"{variant_name}:{type(exc).__name__}:{exc}")
            if not _retryable_runtime_error(exc):
                raise
    if response is None:
        raise RuntimeError("LLM request failed across prompt variants: " + " | ".join(errors))
    edits = response.get("edits", [])
    if not isinstance(edits, list):
        raise ValueError("LLM proposal must contain a list under 'edits'")
    site_lookup = {str(site["path"]): dict(site) for site in enumerate_editable_sites(model_ir)}
    normalized_edits: List[Dict[str, Any]] = []
    for edit in edits:
        if not isinstance(edit, dict):
            continue
        normalized = _normalize_llm_edit(edit, model_ir=model_ir, site_lookup=site_lookup)
        normalized_edits.append(normalized)
    return {
        "proposal_note": str(response.get("proposal_note", "llm_structured_edit")),
        "selected_mechanism_templates": [
            str(item) for item in response.get("selected_mechanism_templates", []) if str(item).strip()
        ] if isinstance(response.get("selected_mechanism_templates", []), list) else [],
        "scientific_hypothesis": str(response.get("scientific_hypothesis", "")),
        "mechanistic_rationale": str(response.get("mechanistic_rationale", "")),
        "risk_notes": list(response.get("risk_notes", [])) if isinstance(response.get("risk_notes", []), list) else [],
        "edits": normalized_edits,
        "llm_preview": [describe_edit(edit) for edit in normalized_edits],
    }
