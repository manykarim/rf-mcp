"""MiniMax model support for the pydantic-ai agentic e2e harness.

Single source of truth for wiring MiniMax models (M2, M2.5, M2.7, M3) into the
in-process pydantic-ai harness (``agent_integration.py``). MiniMax exposes an
OpenAI-compatible Chat Completions endpoint that supports the standard ``tools``
schema FastMCPToolset emits, so we drive it through pydantic-ai's ``OpenAIModel``
pointed at a custom provider.

Verified against pydantic-ai 1.37.0 and ``https://api.minimax.io/v1`` (all four
model IDs live, tool-calling confirmed). See the ``e2e-minimax`` audit and
``experiments/EVAL_*`` for the framing quirks of the weaker tiers.

Environment:
- ``MINIMAX_API_KEY`` — MiniMax platform key (also a repo secret for CI). When set,
  MiniMax models become usable and ``real_llm_available()`` is True.
- ``MINIMAX_MODELS`` — comma-separated subset to test (default: all four).
- ``E2E_AGENT_MODEL`` — force a specific model for the shared autonomous suite.
"""

from __future__ import annotations

import json as _json
import os
from typing import Any, List, Optional

import httpx

# service_tier values the installed openai SDK accepts. MiniMax returns 'standard',
# which is outside this set and trips the SDK's strict Literal validation.
_ALLOWED_SERVICE_TIERS = {None, "auto", "default", "flex", "scale", "priority"}

# MiniMax's OpenAI-compatible base URL. Use api.minimax.io (NOT .com — .com 401s).
MINIMAX_BASE_URL = "https://api.minimax.io/v1"

# Exact model IDs the MiniMax API expects (casing is significant).
MINIMAX_MODELS: List[str] = ["MiniMax-M2", "MiniMax-M2.5", "MiniMax-M2.7", "MiniMax-M3"]

# The most reliable MiniMax tier for a single-model smoke (strongest tool framing).
MINIMAX_DEFAULT_MODEL = "MiniMax-M3"


def minimax_api_key() -> Optional[str]:
    """Return the MiniMax API key from the environment, or None if unset/blank."""
    key = os.getenv("MINIMAX_API_KEY", "").strip()
    return key or None


def is_minimax_model(model_name: str) -> bool:
    """True when ``model_name`` targets a MiniMax model (case-insensitive prefix)."""
    return model_name.lower().startswith("minimax")


class _MiniMaxSanitizingTransport(httpx.AsyncBaseTransport):
    """Strip non-OpenAI-conformant fields from MiniMax chat-completion responses.

    MiniMax's OpenAI-compatible endpoint returns ``service_tier="standard"`` (and a few
    extra top-level keys). The extra keys are ignored by the openai SDK, but
    ``service_tier="standard"`` fails its strict ``Literal`` validation and would abort
    the run before any tool call. We drop that field (leaving it unset/None) so the
    response parses. Non-JSON and streaming bodies pass through untouched.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner

    @staticmethod
    def _rebuilt_headers(response: httpx.Response) -> dict:
        # aread() returns the DECODED body, so strip framing/encoding headers or the
        # downstream client would try to gunzip already-decompressed content ->
        # "Connection error".
        drop = {"content-encoding", "content-length", "transfer-encoding"}
        return {k: v for k, v in response.headers.items() if k.lower() not in drop}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        ctype = response.headers.get("content-type", "")
        if "application/json" not in ctype:
            return response
        body = await response.aread()
        headers = self._rebuilt_headers(response)
        try:
            data = _json.loads(body)
        except Exception:
            return httpx.Response(
                response.status_code, headers=headers, content=body, request=request
            )
        content = body
        if isinstance(data, dict) and data.get("service_tier") not in _ALLOWED_SERVICE_TIERS:
            data.pop("service_tier", None)
            content = _json.dumps(data).encode()
        return httpx.Response(
            response.status_code, headers=headers, content=content, request=request
        )


_SHARED_MINIMAX_CLIENT: Optional[httpx.AsyncClient] = None


def _minimax_http_client() -> httpx.AsyncClient:
    """Process-shared httpx client that sanitizes MiniMax responses for the openai SDK.

    Cached as a single instance (httpx clients are built for reuse) so a multi-model
    sweep does not leak one open connection pool per model — the client outlives all
    the per-model providers and is reclaimed at interpreter exit.
    """
    global _SHARED_MINIMAX_CLIENT
    if _SHARED_MINIMAX_CLIENT is None:
        _SHARED_MINIMAX_CLIENT = httpx.AsyncClient(
            transport=_MiniMaxSanitizingTransport(httpx.AsyncHTTPTransport())
        )
    return _SHARED_MINIMAX_CLIENT


def _openai_model_cls():
    """Return the OpenAI chat-model class across pydantic-ai versions.

    pydantic-ai renamed ``OpenAIModel`` -> ``OpenAIChatModel`` (the Chat Completions
    model). Prefer the new name to avoid the deprecation warning; fall back to the
    legacy name on older installs.
    """
    from pydantic_ai.models import openai as _openai

    return getattr(_openai, "OpenAIChatModel", None) or _openai.OpenAIModel


def resolve_model(model_name: str) -> Any:
    """Build a pydantic-ai model object for ``model_name``.

    MiniMax models are routed to the MiniMax OpenAI-compatible endpoint using
    ``MINIMAX_API_KEY``; everything else uses the default (real OpenAI) provider,
    preserving the existing behaviour for ``gpt-*`` models.

    Raises:
        RuntimeError: if a MiniMax model is requested but ``MINIMAX_API_KEY`` is unset.
    """
    model_cls = _openai_model_cls()
    if is_minimax_model(model_name):
        key = minimax_api_key()
        if not key:
            raise RuntimeError(
                f"MiniMax model '{model_name}' requested but MINIMAX_API_KEY is not set"
            )
        from pydantic_ai.providers.openai import OpenAIProvider

        return model_cls(
            model_name,
            provider=OpenAIProvider(
                base_url=MINIMAX_BASE_URL,
                api_key=key,
                http_client=_minimax_http_client(),
            ),
        )
    return model_cls(model_name)


def default_agent_model() -> str:
    """Pick the model for the shared autonomous suite based on available keys.

    Priority: explicit ``E2E_AGENT_MODEL`` > MiniMax (when its key is present) >
    ``OPENAI_MODEL`` > ``gpt-5-mini``. This lets the same suite run under MiniMax in
    the MiniMax CI job without touching OpenAI-specific env.
    """
    forced = os.getenv("E2E_AGENT_MODEL", "").strip()
    if forced:
        return forced
    if minimax_api_key():
        return os.getenv("MINIMAX_MODEL", MINIMAX_DEFAULT_MODEL).strip() or MINIMAX_DEFAULT_MODEL
    return os.getenv("OPENAI_MODEL", "").strip() or "gpt-5-mini"


def real_llm_available() -> bool:
    """True when a real LLM can drive the agent (OpenAI opt-in OR MiniMax key set)."""
    openai_ready = (
        os.getenv("USE_REAL_LLM", "false").lower() in ("true", "1", "yes")
        and bool(os.getenv("OPENAI_API_KEY", "").strip())
    )
    return openai_ready or bool(minimax_api_key())


def minimax_models_to_test() -> List[str]:
    """Return the MiniMax model IDs to exercise (env ``MINIMAX_MODELS`` or all four)."""
    raw = os.getenv("MINIMAX_MODELS", "").strip()
    if not raw:
        return list(MINIMAX_MODELS)
    return [m.strip() for m in raw.split(",") if m.strip()]
