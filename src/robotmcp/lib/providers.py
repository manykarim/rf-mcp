"""AI Provider configuration and abstraction.

Supports multiple AI providers through pydantic-ai:
- Anthropic (Claude)
- OpenAI (GPT)
- Ollama (local)
- Azure OpenAI
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import yaml


@dataclass
class ProviderConfig:
    """Configuration for an AI provider."""

    provider: str = "anthropic"
    api_key: Optional[str] = None
    model: str = "claude-sonnet-4-20250514"
    retries: int = 3
    retry_delay: float = 1.0
    log_level: str = "INFO"
    max_tokens: int = 4096
    temperature: float = 0.1
    base_url: Optional[str] = None
    extra_options: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_kwargs(cls, **kwargs) -> "ProviderConfig":
        """Create config from keyword arguments (RF library import args)."""
        config = cls()

        # Handle provider
        if "provider" in kwargs:
            config.provider = kwargs["provider"].lower()

        # Handle API key - support env var syntax
        if "api_key" in kwargs:
            api_key = kwargs["api_key"]
            if api_key.startswith("%{") and api_key.endswith("}"):
                # Robot Framework environment variable syntax
                env_var = api_key[2:-1]
                config.api_key = os.environ.get(env_var)
            elif api_key.startswith("${") and api_key.endswith("}"):
                # Also support ${ENV_VAR} syntax
                env_var = api_key[2:-1]
                config.api_key = os.environ.get(env_var)
            else:
                config.api_key = api_key

        # Handle model
        if "model" in kwargs:
            config.model = kwargs["model"]

        # Handle retries
        if "retries" in kwargs:
            config.retries = int(kwargs["retries"])

        # Handle retry_delay - support RF time format
        if "retry_delay" in kwargs:
            delay = kwargs["retry_delay"]
            if isinstance(delay, str):
                # Parse RF time format (e.g., "1s", "500ms", "2m")
                config.retry_delay = cls._parse_time_string(delay)
            else:
                config.retry_delay = float(delay)

        # Handle log level
        if "log_level" in kwargs:
            config.log_level = kwargs["log_level"].upper()

        # Handle max_tokens
        if "max_tokens" in kwargs:
            config.max_tokens = int(kwargs["max_tokens"])

        # Handle temperature
        if "temperature" in kwargs:
            config.temperature = float(kwargs["temperature"])

        # Handle base_url (for Ollama, Azure)
        if "base_url" in kwargs:
            config.base_url = kwargs["base_url"]

        return config

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ProviderConfig":
        """Load configuration from a YAML file."""
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        config = cls()

        # Top-level settings
        if "provider" in data:
            config.provider = data["provider"].lower()
        if "api_key" in data:
            api_key = data["api_key"]
            if api_key.startswith("${") and api_key.endswith("}"):
                env_var = api_key[2:-1]
                config.api_key = os.environ.get(env_var)
            else:
                config.api_key = api_key
        if "model" in data:
            config.model = data["model"]
        if "retries" in data:
            config.retries = int(data["retries"])
        if "retry_delay" in data:
            config.retry_delay = float(data["retry_delay"])
        if "log_level" in data:
            config.log_level = data["log_level"].upper()

        # Provider-specific settings
        provider_settings = data.get(config.provider, {})
        if "max_tokens" in provider_settings:
            config.max_tokens = int(provider_settings["max_tokens"])
        if "temperature" in provider_settings:
            config.temperature = float(provider_settings["temperature"])
        if "base_url" in provider_settings:
            config.base_url = provider_settings["base_url"]

        # Store extra options
        config.extra_options = provider_settings

        return config

    @staticmethod
    def _parse_time_string(time_str: str) -> float:
        """Parse Robot Framework time string to seconds."""
        time_str = time_str.strip().lower()

        if time_str.endswith("ms"):
            return float(time_str[:-2]) / 1000
        elif time_str.endswith("s"):
            return float(time_str[:-1])
        elif time_str.endswith("m"):
            return float(time_str[:-1]) * 60
        elif time_str.endswith("h"):
            return float(time_str[:-1]) * 3600
        else:
            # Assume seconds if no unit
            return float(time_str)


def get_model_instance(config: ProviderConfig):
    """Get the appropriate pydantic-ai model instance for the configured provider.

    Returns:
        A pydantic-ai model instance configured for the specified provider.

    Raises:
        ImportError: If pydantic-ai is not installed
        ValueError: If provider is not supported
    """
    try:
        from pydantic_ai import Agent
        from pydantic_ai.models import Model
    except ImportError:
        raise ImportError(
            "pydantic-ai is required for AILibrary. "
            "Install with: pip install rf-mcp[lib]"
        )

    provider = config.provider.lower()

    # Set API key in environment if provided (pydantic-ai reads from env)
    if config.api_key:
        if provider == "anthropic":
            os.environ.setdefault("ANTHROPIC_API_KEY", config.api_key)
        elif provider in ("openai", "azure"):
            os.environ.setdefault("OPENAI_API_KEY", config.api_key)

    if provider == "anthropic":
        try:
            from pydantic_ai.models.anthropic import AnthropicModel

            return AnthropicModel(model_name=config.model)
        except ImportError:
            # Fallback to model string format
            return config.model

    elif provider == "openai":
        try:
            from pydantic_ai.models.openai import OpenAIChatModel

            # Use custom provider if base_url is specified
            if config.base_url:
                from openai import AsyncOpenAI
                from pydantic_ai.providers.openai import OpenAIProvider

                client = AsyncOpenAI(
                    api_key=config.api_key or os.environ.get("OPENAI_API_KEY"),
                    base_url=config.base_url,
                )
                openai_provider = OpenAIProvider(openai_client=client)
                return OpenAIChatModel(config.model, provider=openai_provider)

            return OpenAIChatModel(config.model)
        except ImportError:
            return config.model

    elif provider == "ollama":
        try:
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
            from openai import AsyncOpenAI

            base_url = config.base_url or "http://localhost:11434/v1"
            client = AsyncOpenAI(
                api_key="ollama",  # Ollama doesn't need a real key
                base_url=base_url,
            )
            ollama_provider = OpenAIProvider(openai_client=client)
            return OpenAIChatModel(config.model, provider=ollama_provider)
        except ImportError:
            return config.model

    elif provider == "azure":
        try:
            from pydantic_ai.models.openai import OpenAIChatModel
            from pydantic_ai.providers.openai import OpenAIProvider
            from openai import AsyncAzureOpenAI

            if not config.base_url:
                raise ValueError("Azure OpenAI requires base_url configuration")

            client = AsyncAzureOpenAI(
                api_key=config.api_key or os.environ.get("AZURE_OPENAI_KEY"),
                azure_endpoint=config.base_url,
                api_version="2024-02-15-preview",
            )
            azure_provider = OpenAIProvider(openai_client=client)
            return OpenAIChatModel(config.model, provider=azure_provider)
        except ImportError:
            return config.model

    else:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: anthropic, openai, ollama, azure"
        )
