"""Tests for providers module."""

import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from robotmcp.lib.providers import ProviderConfig


class TestProviderConfig:
    """Tests for ProviderConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ProviderConfig()

        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-4-20250514"
        assert config.retries == 3
        assert config.retry_delay == 1.0
        assert config.log_level == "INFO"
        assert config.max_tokens == 4096
        assert config.temperature == 0.1

    def test_from_kwargs_basic(self):
        """Test creating config from keyword arguments."""
        config = ProviderConfig.from_kwargs(
            provider="openai",
            model="gpt-4",
            retries=5,
        )

        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.retries == 5

    def test_from_kwargs_env_var_syntax(self):
        """Test environment variable syntax for API key."""
        with patch.dict(os.environ, {"MY_API_KEY": "secret123"}):
            config = ProviderConfig.from_kwargs(
                api_key="%{MY_API_KEY}",
            )
            assert config.api_key == "secret123"

    def test_from_kwargs_dollar_env_syntax(self):
        """Test ${ENV_VAR} syntax for API key."""
        with patch.dict(os.environ, {"MY_API_KEY": "secret456"}):
            config = ProviderConfig.from_kwargs(
                api_key="${MY_API_KEY}",
            )
            assert config.api_key == "secret456"

    def test_from_kwargs_retry_delay_formats(self):
        """Test parsing different retry delay formats."""
        # Seconds
        config = ProviderConfig.from_kwargs(retry_delay="2s")
        assert config.retry_delay == 2.0

        # Milliseconds
        config = ProviderConfig.from_kwargs(retry_delay="500ms")
        assert config.retry_delay == 0.5

        # Minutes
        config = ProviderConfig.from_kwargs(retry_delay="1m")
        assert config.retry_delay == 60.0

        # No unit (assumes seconds)
        config = ProviderConfig.from_kwargs(retry_delay="3")
        assert config.retry_delay == 3.0

    def test_from_yaml(self):
        """Test loading config from YAML file."""
        yaml_content = """
provider: openai
model: gpt-4-turbo
retries: 5
retry_delay: 2.0

openai:
  max_tokens: 8192
  temperature: 0.2
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            f.flush()

            config = ProviderConfig.from_yaml(f.name)

            assert config.provider == "openai"
            assert config.model == "gpt-4-turbo"
            assert config.retries == 5
            assert config.retry_delay == 2.0
            assert config.max_tokens == 8192
            assert config.temperature == 0.2

        os.unlink(f.name)

    def test_parse_time_string(self):
        """Test time string parsing."""
        assert ProviderConfig._parse_time_string("1s") == 1.0
        assert ProviderConfig._parse_time_string("100ms") == 0.1
        assert ProviderConfig._parse_time_string("2m") == 120.0
        assert ProviderConfig._parse_time_string("1h") == 3600.0
        assert ProviderConfig._parse_time_string("5") == 5.0

    def test_provider_lowercase(self):
        """Test provider is normalized to lowercase."""
        config = ProviderConfig.from_kwargs(provider="ANTHROPIC")
        assert config.provider == "anthropic"

        config = ProviderConfig.from_kwargs(provider="OpenAI")
        assert config.provider == "openai"
