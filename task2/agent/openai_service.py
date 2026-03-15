from __future__ import annotations

import random
import re
import threading
import time
from typing import Any

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from .config import Settings
from .observability import flush_langfuse, langfuse_is_enabled
from .prompts import AGENT_SYSTEM_PROMPT, COMPLETION_SYSTEM_PROMPT, build_completion_input

_SEMAPHORE_LOCK = threading.Lock()
_REQUEST_SEMAPHORES: dict[int, threading.BoundedSemaphore] = {}


class OpenAIService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = self._build_client()

    def _build_client(self) -> OpenAI:
        client_cls: type[OpenAI] = OpenAI
        if langfuse_is_enabled():
            try:
                from langfuse.openai import OpenAI as LangfuseOpenAI
            except ImportError:
                client_cls = OpenAI
            else:
                client_cls = LangfuseOpenAI

        return client_cls(
            api_key=self.settings.openai_api_key,
            max_retries=self.settings.openai_sdk_max_retries,
            timeout=self.settings.openai_timeout_seconds,
        )

    def start_agent_run(
        self,
        task_input: str,
        tools: list[dict[str, Any]],
    ) -> Any:
        response = self._responses_create_with_retries(
            model=self.settings.agent_model,
            instructions=AGENT_SYSTEM_PROMPT,
            input=task_input,
            tools=tools,
            parallel_tool_calls=False,
            prompt_cache_key=f"task2-agent-start:{self.settings.agent_model}",
            prompt_cache_retention=self.settings.openai_prompt_cache_retention,
        )
        flush_langfuse()
        return response

    def continue_agent_run(
        self,
        previous_response_id: str,
        tool_outputs: list[dict[str, str]],
        tools: list[dict[str, Any]],
    ) -> Any:
        response = self._responses_create_with_retries(
            model=self.settings.agent_model,
            instructions=AGENT_SYSTEM_PROMPT,
            previous_response_id=previous_response_id,
            input=tool_outputs,
            tools=tools,
            parallel_tool_calls=False,
            prompt_cache_key=f"task2-agent-continue:{self.settings.agent_model}",
            prompt_cache_retention=self.settings.openai_prompt_cache_retention,
        )
        flush_langfuse()
        return response

    def continue_agent_run_with_input(
        self,
        *,
        previous_response_id: str,
        input_payload: Any,
        tools: list[dict[str, Any]],
    ) -> Any:
        response = self._responses_create_with_retries(
            model=self.settings.agent_model,
            instructions=AGENT_SYSTEM_PROMPT,
            previous_response_id=previous_response_id,
            input=input_payload,
            tools=tools,
            parallel_tool_calls=False,
            prompt_cache_key=f"task2-agent-continue-input:{self.settings.agent_model}",
            prompt_cache_retention=self.settings.openai_prompt_cache_retention,
        )
        flush_langfuse()
        return response

    def generate_completion(
        self,
        *,
        context: str,
        prefix: str,
        suffix: str,
        target_path: str,
        language: str,
        model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> str:
        completion_model = model or self.settings.completion_model
        response = self._responses_create_with_retries(
            model=completion_model,
            instructions=COMPLETION_SYSTEM_PROMPT,
            input=build_completion_input(
                context=context,
                prefix=prefix,
                suffix=suffix,
                target_path=target_path,
                language=language,
            ),
            max_output_tokens=max_output_tokens or self.settings.completion_max_output_tokens,
            prompt_cache_key=f"task2-completion:{completion_model}:{language}",
            prompt_cache_retention=self.settings.openai_prompt_cache_retention,
        )
        flush_langfuse()
        return self.extract_response_text(response)

    def extract_function_calls(self, response: Any) -> list[Any]:
        return [
            item
            for item in getattr(response, "output", []) or []
            if getattr(item, "type", None) == "function_call"
        ]

    def describe_response(self, response: Any) -> dict[str, Any]:
        function_calls = self.extract_function_calls(response)
        return {
            "response_id": getattr(response, "id", None),
            "output_text": self.extract_response_text(response),
            "usage": self._describe_usage(response),
            "output_types": [
                getattr(item, "type", "")
                for item in getattr(response, "output", []) or []
            ],
            "function_calls": [
                {
                    "call_id": getattr(call, "call_id", None),
                    "name": getattr(call, "name", None),
                    "arguments": getattr(call, "arguments", None),
                }
                for call in function_calls
            ],
        }

    def extract_response_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text.strip()

        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    text = getattr(content, "text", "")
                    if text:
                        return text.strip()

        return ""

    def _responses_create_with_retries(self, **kwargs: Any) -> Any:
        semaphore = _get_request_semaphore(self.settings.openai_max_concurrent_requests)
        with semaphore:
            last_exc: Exception | None = None
            for attempt in range(self.settings.openai_max_retries + 1):
                try:
                    return self.client.responses.create(**kwargs)
                except (RateLimitError, APIConnectionError, APITimeoutError) as exc:
                    last_exc = exc
                except APIStatusError as exc:
                    if getattr(exc, "status_code", None) and int(exc.status_code) < 500:
                        raise
                    last_exc = exc

                if attempt >= self.settings.openai_max_retries:
                    break
                time.sleep(self._retry_delay_seconds(last_exc, attempt))

            assert last_exc is not None
            raise last_exc

    def _retry_delay_seconds(self, exc: Exception, attempt: int) -> float:
        retry_after = self._retry_after_seconds(exc)
        if retry_after is not None:
            return min(retry_after, self.settings.openai_retry_max_seconds)

        base = self.settings.openai_retry_base_seconds * (2**attempt)
        jitter = random.uniform(0.0, min(1.0, base * 0.25))
        return min(base + jitter, self.settings.openai_retry_max_seconds)

    def _retry_after_seconds(self, exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            retry_after_ms = headers.get("retry-after-ms")
            if retry_after_ms:
                try:
                    return max(0.0, float(retry_after_ms) / 1000.0)
                except ValueError:
                    pass
            retry_after = headers.get("retry-after")
            if retry_after:
                try:
                    return max(0.0, float(retry_after))
                except ValueError:
                    pass

        message = ""
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error_body = body.get("error")
            if isinstance(error_body, dict):
                message = str(error_body.get("message", ""))
        if not message:
            message = str(exc)

        match = re.search(r"Please try again in (\d+)ms", message)
        if match:
            return max(0.0, float(match.group(1)) / 1000.0)
        return None

    def _describe_usage(self, response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}

        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", input_tokens + output_tokens) or 0)

        input_details = getattr(usage, "input_tokens_details", None)
        cached_tokens = int(getattr(input_details, "cached_tokens", 0) or 0) if input_details else 0

        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cached_tokens": cached_tokens,
        }


def _get_request_semaphore(limit: int) -> threading.BoundedSemaphore:
    bounded_limit = max(1, int(limit))
    with _SEMAPHORE_LOCK:
        semaphore = _REQUEST_SEMAPHORES.get(bounded_limit)
        if semaphore is None:
            semaphore = threading.BoundedSemaphore(bounded_limit)
            _REQUEST_SEMAPHORES[bounded_limit] = semaphore
        return semaphore
