"""Configuration helpers for the RobotMCP prompt runner."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

try:  # Optional dependency for loading .env files at runtime
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv is an optional extra
    load_dotenv = None  # type: ignore[assignment]

_DEFAULT_MAX_ITERATIONS = 20
_DEFAULT_MAX_TOKENS = 800
_DEFAULT_TEMPERATURE = 0.2
_ENV_LOADED = False


@dataclass(frozen=True)
class PromptRuntimeConfig:
    """Holds runtime settings for the prompt runner."""

    api_key: str
    model: str
    base_url: Optional[str] = None
    temperature: float = _DEFAULT_TEMPERATURE
    max_iterations: int = _DEFAULT_MAX_ITERATIONS
    max_tokens: int = _DEFAULT_MAX_TOKENS

    def with_overrides(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        max_iterations: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> "PromptRuntimeConfig":
        """Return a copy with the provided overrides applied."""

        cfg = self
        if api_key:
            cfg = replace(cfg, api_key=api_key)
        if model:
            cfg = replace(cfg, model=model)
        if base_url is not None:
            cfg = replace(cfg, base_url=base_url or None)
        if temperature is not None:
            cfg = replace(cfg, temperature=temperature)
        if max_iterations is not None:
            cfg = replace(cfg, max_iterations=max_iterations)
        if max_tokens is not None:
            cfg = replace(cfg, max_tokens=max_tokens)
        return cfg


def load_prompt_config(
    *,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: Optional[float] = None,
    max_iterations: Optional[int] = None,
    max_tokens: Optional[int] = None,
) -> PromptRuntimeConfig:
    """Load runtime configuration from environment variables and overrides."""

    _ensure_env_loaded()
    resolved_key = api_key or os.getenv("ROBOTMCP_PROMPT_API_KEY") or os.getenv("OPENAI_API_KEY", "").strip()
    if not resolved_key:
        raise RuntimeError(
            "OPENAI_API_KEY (or ROBOTMCP_PROMPT_API_KEY) must be set to run MCP prompts"
        )

    resolved_model = (
        model
        or os.getenv("ROBOTMCP_PROMPT_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4o-mini"
    )

    resolved_base_url = base_url
    if resolved_base_url is None:
        resolved_base_url = os.getenv("ROBOTMCP_PROMPT_BASE_URL") or os.getenv(
            "OPENAI_BASE_URL"
        )
        if resolved_base_url:
            resolved_base_url = resolved_base_url.strip() or None

    resolved_temperature = temperature if temperature is not None else _DEFAULT_TEMPERATURE
    resolved_iterations = max_iterations if max_iterations is not None else _DEFAULT_MAX_ITERATIONS
    resolved_tokens = max_tokens if max_tokens is not None else _DEFAULT_MAX_TOKENS

    return PromptRuntimeConfig(
        api_key=resolved_key,
        model=resolved_model,
        base_url=resolved_base_url,
        temperature=resolved_temperature,
        max_iterations=resolved_iterations,
        max_tokens=resolved_tokens,
    )


def _ensure_env_loaded() -> None:
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    if load_dotenv is None:
        return
    dotenv_path = Path.cwd() / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path)  # type: ignore[func-returns-value]
    else:
        load_dotenv()  # Fallback to default search
