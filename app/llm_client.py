from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
from dataclasses import dataclass
from typing import Any, Protocol

from openai_messages_client import chat_completion

from .config import Settings
from .errors import AppError






_PROVIDER_REQUEST_SLOTS = threading.BoundedSemaphore(50)
_CONFIGURED_GATES_LOCK = threading.Lock()
_CONFIGURED_GATES: dict[int, threading.BoundedSemaphore] = {}


def _configured_gate(limit: int) -> threading.BoundedSemaphore:
    with _CONFIGURED_GATES_LOCK:
        gate = _CONFIGURED_GATES.get(limit)
        if gate is None:
            gate = threading.BoundedSemaphore(limit)
            _CONFIGURED_GATES[limit] = gate
        return gate


class StructuredLLM(Protocol):
    model: str

    def complete_json(self, messages: list[dict[str, str]], purpose: str) -> dict[str, Any]: ...


def extract_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.I)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:


        raise AppError("LLM_INVALID_JSON", 502) from exc
    if not isinstance(value, dict):
        raise AppError("LLM_INVALID_JSON", 502)
    return value


@dataclass(slots=True)
class LLMClient:
    settings: Settings

    @property
    def model(self) -> str:
        return self.settings.llm_model

    def complete_json(self, messages: list[dict[str, str]], purpose: str) -> dict[str, Any]:
        if not self.settings.llm_api_key:
            raise AppError("LLM_NOT_CONFIGURED", 503)

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                deadline = time.monotonic() + self.settings.llm_timeout
                configured_gate = _configured_gate(self.settings.llm_max_concurrency)
                if not configured_gate.acquire(timeout=self.settings.llm_timeout):
                    raise AppError("LLM_CONCURRENCY_BUSY", 503, details={"purpose": purpose})
                provider_acquired = False
                try:
                    remaining = max(0.0, deadline - time.monotonic())
                    provider_acquired = _PROVIDER_REQUEST_SLOTS.acquire(timeout=remaining)
                    if not provider_acquired:
                        raise AppError("LLM_CONCURRENCY_BUSY", 503, details={"purpose": purpose})
                    response = chat_completion(
                        base_url=self.settings.llm_base_url,
                        api_key=self.settings.llm_api_key,
                        model=self.settings.llm_model,
                        messages=messages,
                        temperature=self.settings.llm_temperature,
                        max_tokens=self.settings.llm_max_tokens,
                        timeout=self.settings.llm_timeout,



                        extra_payload=None,
                    )
                finally:
                    if provider_acquired:
                        _PROVIDER_REQUEST_SLOTS.release()
                    configured_gate.release()
                if not isinstance(response, str) or not response.strip():
                    raise AppError("LLM_EMPTY_RESPONSE", 502)
                return extract_json_object(response)
            except AppError:
                raise
            except (TimeoutError, urllib.error.URLError) as exc:
                reason = getattr(exc, "reason", None)
                last_error = TimeoutError(str(exc)) if isinstance(reason, TimeoutError) else exc
                retryable = True
            except RuntimeError as exc:
                last_error = exc
                message = str(exc)
                retryable = any(code in message for code in ("HTTP 429", "HTTP 500", "HTTP 502", "HTTP 503", "HTTP 504"))
                if "HTTP 401" in message or "HTTP 403" in message:
                    raise AppError("LLM_AUTH_FAILED", 502) from exc
                if not retryable:
                    raise AppError("LLM_REQUEST_FAILED", 502, details={"purpose": purpose}) from exc
            except (KeyError, IndexError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AppError("LLM_RESPONSE_MALFORMED", 502, details={"purpose": purpose}) from exc

            if not retryable or attempt == 2:
                break
            time.sleep(0.5 * (2**attempt))

        code = "LLM_TIMEOUT" if isinstance(last_error, TimeoutError) else "LLM_TEMPORARILY_UNAVAILABLE"
        raise AppError(code, 503, details={"purpose": purpose}) from last_error
