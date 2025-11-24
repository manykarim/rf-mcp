"""OpenAI chat completion helper used by the prompt runner."""

from __future__ import annotations

import json
from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI
from openai._exceptions import OpenAIError


@dataclass
class LlmToolCall:
    """Represents a single tool call returned by the LLM."""

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LlmResponse:
    """LLM response payload consumed by the prompt runner."""

    content: str
    tool_calls: List[LlmToolCall]
    raw_response: Any


class OpenAILlmClient:
    """Thin wrapper that submits chat completion requests with tool support."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        self._mute_http_logging()
        self._client = OpenAI(**client_kwargs)

    def complete_chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> LlmResponse:
        """Call the OpenAI chat completions API and parse tool calls."""

        try:
            completion = self._client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except OpenAIError as exc:  # pragma: no cover - network failure path
            raise RuntimeError(f"OpenAI chat completion failed: {exc}") from exc

        choice = completion.choices[0].message
        content = choice.content or ""
        tool_calls: List[LlmToolCall] = []
        for call in choice.tool_calls or []:
            args_text = call.function.arguments or "{}"
            try:
                parsed_args = json.loads(args_text)
                if not isinstance(parsed_args, dict):
                    parsed_args = {"value": parsed_args}
            except json.JSONDecodeError:
                parsed_args = {"raw": args_text}
            tool_calls.append(
                LlmToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=parsed_args,
                )
            )

        return LlmResponse(content=content, tool_calls=tool_calls, raw_response=completion)

    @staticmethod
    def _mute_http_logging() -> None:
        noisy_loggers = [
            "httpx",
            "httpcore",
            "openai",
            "openai._base_client",
        ]
        try:  # httpx config logger is separate
            import httpx

            noisy_loggers.append(httpx._config.logger.name)  # type: ignore[attr-defined]
            httpx._config.logger.setLevel(logging.WARNING)  # type: ignore[attr-defined]
            httpx._config.logger.propagate = False  # type: ignore[attr-defined]
            if not httpx._config.logger.handlers:  # type: ignore[attr-defined]
                httpx._config.logger.addHandler(logging.NullHandler())  # type: ignore[attr-defined]
        except Exception:
            pass

        for name in noisy_loggers:
            logger = logging.getLogger(name)
            logger.setLevel(logging.WARNING)
            logger.propagate = False
            if not logger.handlers:
                logger.addHandler(logging.NullHandler())
