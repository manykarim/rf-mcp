"""ModelTier.from_model_name remap — hosted Claude/GPT-4 → LARGE_CONTEXT.

The pre-remap version mapped every "claude*", "gpt-4*", and "haiku/sonnet"
substring match to ModelTier.HOSTED. That HOSTED tier was then auto-selected
for a small-context tool profile (`browser_exec`, 6 tools) — wrong for
modern Anthropic / OpenAI hosted models which all have 128K-200K context
windows. The fix maps them to LARGE_CONTEXT so the auto-selected profile
is `full`.

Edge case pinned: gpt-4o-mini must classify as STANDARD even though its
name contains the gpt-4o substring (the `mini` suffix wins).
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter

from robotmcp.domains.shared.kernel import ModelTier, SelectMatch


class TestLargeContextHosted:
    """Claude 3+ haiku/sonnet/opus, OpenAI gpt-4o/turbo, Gemini 1.5+ etc."""

    @pytest.mark.parametrize("name", [
        "haiku", "sonnet", "opus",
        "claude-haiku", "claude-sonnet", "claude-opus",
        "claude-3-5-haiku", "claude-3-5-sonnet", "claude-3-opus",
        "claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7",
        "gpt-4o", "gpt-4-turbo",
        "gemini-1.5-pro", "gemini-2.0-flash",
        "o1-preview", "o3-mini",
    ])
    def test_hosted_large_context(self, name):
        assert ModelTier.from_model_name(name) == ModelTier.LARGE_CONTEXT, name


class TestStandardSmallHosted:
    """Cost-optimised micro-hosted models."""

    @pytest.mark.parametrize("name", [
        "gpt-4o-mini",   # explicit override before gpt-4o substring match
        "gemini-flash",
        "mistral-nano",  # generic nano-suffix model
    ])
    def test_small_hosted_standard(self, name):
        assert ModelTier.from_model_name(name) == ModelTier.STANDARD, name


class TestOpenWeightParamCount:
    """Open-weight model families resolve by parameter count."""

    @pytest.mark.parametrize("name, tier", [
        ("mistral-7b-instruct", ModelTier.SMALL_7B),
        ("llama-3.2-3b", ModelTier.SMALL_7B),
        ("qwen-2.5-14b", ModelTier.MEDIUM_13B),
        ("mixtral-8x7b", ModelTier.SMALL_7B),
        ("llama-3.3-70b", ModelTier.LARGE_CONTEXT),
        ("qwen-32b", ModelTier.STANDARD),
    ])
    def test_param_count_resolution(self, name, tier):
        assert ModelTier.from_model_name(name) == tier, name


class TestSmallFamiliesByName:
    """Known small-model families without explicit param count."""

    @pytest.mark.parametrize("name", ["phi-3-mini", "phi-2", "glm-4.5-air", "qwen2.5-coder"])
    def test_named_small_family(self, name):
        assert ModelTier.from_model_name(name) == ModelTier.SMALL_7B


class TestUnknownDefaultsToStandard:
    """Unknown models default to STANDARD, not HOSTED."""

    @pytest.mark.parametrize("name", ["completely-unknown-model", "foo-bar-1.0", ""])
    def test_unknown_standard(self, name):
        assert ModelTier.from_model_name(name) == ModelTier.STANDARD


class TestSelectMatchAlias:
    """The SelectMatch Pydantic alias accepts and normalises the 5 values."""

    @pytest.mark.parametrize("raw,expected", [
        ("label", "label"),
        ("value", "value"),
        ("INDEX", "index"),
        ("  text  ", "text"),
        ("Auto", "auto"),
    ])
    def test_accepts_case_and_whitespace(self, raw, expected):
        ta = TypeAdapter(SelectMatch)
        assert ta.validate_python(raw) == expected

    def test_rejects_unknown(self):
        ta = TypeAdapter(SelectMatch)
        with pytest.raises(Exception):
            ta.validate_python("frobnicate")
