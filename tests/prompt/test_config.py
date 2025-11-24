import os

import pytest

import robotmcp.prompt.config as prompt_config
from robotmcp.prompt.config import PromptRuntimeConfig, load_prompt_config


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("ROBOTMCP_PROMPT_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ROBOTMCP_PROMPT_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("ROBOTMCP_PROMPT_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)


def test_requires_api_key(monkeypatch):
    monkeypatch.setattr(prompt_config, "_ENV_LOADED", False)
    monkeypatch.setattr(prompt_config, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        load_prompt_config()


def test_loads_from_openai_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "key-123")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-test")
    cfg = load_prompt_config()
    assert cfg.api_key == "key-123"
    assert cfg.model == "gpt-test"


def test_overrides_precedence(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fallback-key")
    cfg = load_prompt_config(model="gpt-a", temperature=0.5)
    assert cfg.model == "gpt-a"
    assert cfg.temperature == pytest.approx(0.5)
    updated = cfg.with_overrides(base_url="https://example.com")
    assert isinstance(updated, PromptRuntimeConfig)
    assert updated.base_url == "https://example.com"


def test_loads_from_dotenv(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("OPENAI_API_KEY=dot-env-key\nOPENAI_MODEL=gpt-dot\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(prompt_config, "_ENV_LOADED", False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ROBOTMCP_PROMPT_API_KEY", raising=False)
    cfg = load_prompt_config()
    assert cfg.api_key == "dot-env-key"
    assert cfg.model == "gpt-dot"
