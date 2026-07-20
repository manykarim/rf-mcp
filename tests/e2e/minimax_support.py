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

# OpenRouter's OpenAI-compatible base URL — routes open-weight (self-hostable) models.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Exact model IDs the MiniMax API expects (casing is significant).
MINIMAX_MODELS: List[str] = ["MiniMax-M2", "MiniMax-M2.5", "MiniMax-M2.7", "MiniMax-M3"]

# The most reliable MiniMax tier for a single-model smoke (strongest tool framing).
MINIMAX_DEFAULT_MODEL = "MiniMax-M3"

# The pinnable, self-hostable REFERENCE model (Apache-2.0). Verified to match MiniMax-M3
# on the calibrated scenario while being pinnable to exact weights -> reproducible
# baselines. See change autonomous-e2e-coverage / design.md.
REFERENCE_MODEL = "qwen/qwen3-coder-30b-a3b-instruct"

# Pin descriptor for the reference — records HOW the weights are pinned so a baseline
# change is attributable to rf-mcp, not a silent model update. Filled in when a
# self-hosted/pinned instance is used; the slug alone identifies the OpenRouter route.
REFERENCE_PIN = {
    "model": REFERENCE_MODEL,
    "license": "Apache-2.0",
    "hf_revision": None,      # HF commit SHA when self-hosted / pinned
    "quant_sha256": None,     # quantized weight file hash when self-hosted
    "chat_template_hash": None,
    "sampling": {"temperature": 0.0},
}


def minimax_api_key() -> Optional[str]:
    """Return the MiniMax API key from the environment, or None if unset/blank."""
    key = os.getenv("MINIMAX_API_KEY", "").strip()
    return key or None


def openrouter_api_key() -> Optional[str]:
    """Return the OpenRouter API key from the environment, or None if unset/blank."""
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    return key or None


def is_minimax_model(model_name: str) -> bool:
    """True when ``model_name`` targets a MiniMax model (case-insensitive prefix)."""
    return model_name.lower().startswith("minimax")


def is_openrouter_model(model_name: str) -> bool:
    """True when ``model_name`` is an OpenRouter slug (``vendor/model`` form).

    OpenRouter model IDs contain a ``/`` (e.g. ``qwen/qwen3-coder-30b-a3b-instruct``),
    which distinguishes them from MiniMax IDs and bare OpenAI IDs (``gpt-5-mini``).
    """
    return "/" in model_name and not is_minimax_model(model_name)


class _ServiceTierSanitizingTransport(httpx.AsyncBaseTransport):
    """Strip non-OpenAI-conformant ``service_tier`` from chat-completion responses.

    MiniMax's OpenAI-compatible endpoint returns ``service_tier="standard"`` (and a few
    extra top-level keys); some OpenRouter backends do likewise. The extra keys are
    ignored by the openai SDK, but a ``service_tier`` outside its strict ``Literal`` set
    aborts the run before any tool call. We drop that field (leaving it unset/None) so the
    response parses. Non-JSON and streaming bodies pass through untouched. Applies to both
    the MiniMax and OpenRouter providers.
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


_SHARED_SANITIZING_CLIENT: Optional[httpx.AsyncClient] = None


def _sanitizing_http_client() -> httpx.AsyncClient:
    """Process-shared httpx client that sanitizes responses for the openai SDK.

    Cached as a single instance (httpx clients are built for reuse) so a multi-model
    sweep does not leak one open connection pool per model — the client outlives all
    the per-model providers and is reclaimed at interpreter exit. Shared by the MiniMax
    and OpenRouter providers.
    """
    global _SHARED_SANITIZING_CLIENT
    if _SHARED_SANITIZING_CLIENT is None:
        _SHARED_SANITIZING_CLIENT = httpx.AsyncClient(
            transport=_ServiceTierSanitizingTransport(httpx.AsyncHTTPTransport())
        )
    return _SHARED_SANITIZING_CLIENT


# Backwards-compatible alias (older callers / experiment scripts import this name).
_minimax_http_client = _sanitizing_http_client


def _openai_model_cls():
    """Return the OpenAI chat-model class across pydantic-ai versions.

    pydantic-ai renamed ``OpenAIModel`` -> ``OpenAIChatModel`` (the Chat Completions
    model). Prefer the new name to avoid the deprecation warning; fall back to the
    legacy name on older installs.
    """
    from pydantic_ai.models import openai as _openai

    return getattr(_openai, "OpenAIChatModel", None) or _openai.OpenAIModel


def provider_for(model_name: str) -> str:
    """Return the provider routing for a model slug: 'minimax' | 'openrouter' | 'openai'."""
    if is_minimax_model(model_name):
        return "minimax"
    if is_openrouter_model(model_name):
        return "openrouter"
    return "openai"


def resolve_model(model_name: str) -> Any:
    """Build a pydantic-ai model object for ``model_name``, routed by provider.

    - MiniMax IDs (``MiniMax-*``)      -> MiniMax OpenAI-compatible endpoint (MINIMAX_API_KEY)
    - OpenRouter slugs (``vendor/id``) -> OpenRouter endpoint (OPENROUTER_API_KEY)
    - bare IDs (``gpt-5-mini``)        -> default OpenAI provider (unchanged behaviour)

    Both custom providers reuse the service_tier sanitizing http client.

    Raises:
        RuntimeError: if a routed provider's API key is unset.
    """
    model_cls = _openai_model_cls()
    provider = provider_for(model_name)
    if provider == "openai":
        return model_cls(model_name)

    from pydantic_ai.providers.openai import OpenAIProvider

    if provider == "minimax":
        key, base = minimax_api_key(), MINIMAX_BASE_URL
        keyname = "MINIMAX_API_KEY"
    else:  # openrouter
        key, base = openrouter_api_key(), OPENROUTER_BASE_URL
        keyname = "OPENROUTER_API_KEY"
    if not key:
        raise RuntimeError(f"Model '{model_name}' requires {keyname} but it is not set")

    settings = None
    # OpenRouter routes the SAME slug across providers at different quantizations/uptimes
    # (non-reproducible; the cause of qwen3-coder's flaky no-tool-call runs). Pin a single
    # provider (and thus quant) via OPENROUTER_PROVIDER for a reproducible reference.
    pin = os.getenv("OPENROUTER_PROVIDER", "").strip() if provider == "openrouter" else ""
    if pin:
        from pydantic_ai.models.openai import OpenAIChatModelSettings

        settings = OpenAIChatModelSettings(
            extra_body={"provider": {"order": [pin], "allow_fallbacks": False}}
        )
    kwargs = {"provider": OpenAIProvider(base_url=base, api_key=key, http_client=_sanitizing_http_client())}
    if settings is not None:
        kwargs["settings"] = settings
    return model_cls(model_name, **kwargs)


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
    """True when a real LLM can drive the agent (OpenAI opt-in, MiniMax, or OpenRouter)."""
    openai_ready = (
        os.getenv("USE_REAL_LLM", "false").lower() in ("true", "1", "yes")
        and bool(os.getenv("OPENAI_API_KEY", "").strip())
    )
    return openai_ready or bool(minimax_api_key()) or bool(openrouter_api_key())


def minimax_models_to_test() -> List[str]:
    """Return the MiniMax model IDs to exercise (env ``MINIMAX_MODELS`` or all four)."""
    raw = os.getenv("MINIMAX_MODELS", "").strip()
    if not raw:
        return list(MINIMAX_MODELS)
    return [m.strip() for m in raw.split(",") if m.strip()]
