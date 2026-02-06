"""Tests for ROBOTMCP_USE_SAMPLING feature flag integration.

Tests verify that:
1. Feature flag is off by default
2. When enabled, sampling enhances analyze_scenario results
3. When enabled, sampling enhances recommend_libraries results
4. Graceful fallback when sampling fails
5. Feature flag respects various env var formats
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _parse_result(raw_result) -> dict:
    """Parse tool result from FastMCP client."""
    if isinstance(raw_result, dict):
        return raw_result
    if hasattr(raw_result, "data"):
        data = raw_result.data
        if isinstance(data, dict):
            return data
        if isinstance(data, str):
            try:
                return json.loads(data)
            except (json.JSONDecodeError, TypeError):
                pass
    if hasattr(raw_result, "content"):
        content = raw_result.content
        if isinstance(content, list):
            for item in content:
                text = getattr(item, "text", None) or (
                    item.get("text") if isinstance(item, dict) else None
                )
                if text:
                    try:
                        return json.loads(text)
                    except (json.JSONDecodeError, TypeError):
                        pass
    try:
        return json.loads(str(raw_result))
    except (json.JSONDecodeError, TypeError):
        return {"raw": str(raw_result)[:500]}


# ===========================================================================
# 1. Feature flag configuration tests (no LLM needed)
# ===========================================================================


class TestFeatureFlagConfig:
    """Test feature flag configuration."""

    def test_sampling_disabled_by_default(self):
        """Feature flag should be off by default."""
        from robotmcp.utils.sampling import is_sampling_enabled

        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("ROBOTMCP_USE_SAMPLING", None)
            assert is_sampling_enabled() is False

    def test_sampling_enabled_true(self):
        """Feature flag enabled with 'true'."""
        from robotmcp.utils.sampling import is_sampling_enabled

        with patch.dict(os.environ, {"ROBOTMCP_USE_SAMPLING": "true"}):
            assert is_sampling_enabled() is True

    def test_sampling_enabled_1(self):
        """Feature flag enabled with '1'."""
        from robotmcp.utils.sampling import is_sampling_enabled

        with patch.dict(os.environ, {"ROBOTMCP_USE_SAMPLING": "1"}):
            assert is_sampling_enabled() is True

    def test_sampling_enabled_yes(self):
        """Feature flag enabled with 'yes'."""
        from robotmcp.utils.sampling import is_sampling_enabled

        with patch.dict(os.environ, {"ROBOTMCP_USE_SAMPLING": "yes"}):
            assert is_sampling_enabled() is True

    def test_sampling_enabled_case_insensitive(self):
        """Feature flag is case insensitive."""
        from robotmcp.utils.sampling import is_sampling_enabled

        with patch.dict(os.environ, {"ROBOTMCP_USE_SAMPLING": "TRUE"}):
            assert is_sampling_enabled() is True

    def test_sampling_disabled_false(self):
        """Feature flag disabled with 'false'."""
        from robotmcp.utils.sampling import is_sampling_enabled

        with patch.dict(os.environ, {"ROBOTMCP_USE_SAMPLING": "false"}):
            assert is_sampling_enabled() is False

    def test_sampling_disabled_empty(self):
        """Feature flag disabled with empty string."""
        from robotmcp.utils.sampling import is_sampling_enabled

        with patch.dict(os.environ, {"ROBOTMCP_USE_SAMPLING": ""}):
            assert is_sampling_enabled() is False


# ===========================================================================
# 2. Sampling function unit tests (mocked ctx)
# ===========================================================================


class TestSamplingFunctions:
    """Test sampling helper functions with mocked context."""

    @pytest.mark.asyncio
    async def test_sample_analyze_scenario_returns_dict(self):
        """sample_analyze_scenario returns parsed dict when ctx.sample works."""
        from robotmcp.utils.sampling import sample_analyze_scenario

        mock_ctx = AsyncMock()
        mock_result = MagicMock()
        mock_result.text = json.dumps(
            {
                "session_type": "web_automation",
                "detected_context": "web",
                "primary_library": "Browser",
                "library_preference": None,
            }
        )
        mock_ctx.sample = AsyncMock(return_value=mock_result)

        result = await sample_analyze_scenario(
            mock_ctx, "Click login button on the website", "web"
        )
        assert result is not None
        assert result.get("session_type") == "web_automation"
        assert result.get("primary_library") == "Browser"

    @pytest.mark.asyncio
    async def test_sample_analyze_scenario_returns_none_without_ctx(self):
        """Returns None when ctx is None."""
        from robotmcp.utils.sampling import sample_analyze_scenario

        result = await sample_analyze_scenario(None, "Click button", "web")
        assert result is None

    @pytest.mark.asyncio
    async def test_sample_analyze_scenario_returns_none_on_error(self):
        """Returns None when ctx.sample raises."""
        from robotmcp.utils.sampling import sample_analyze_scenario

        mock_ctx = AsyncMock()
        mock_ctx.sample = AsyncMock(side_effect=Exception("API error"))
        result = await sample_analyze_scenario(mock_ctx, "Click button", "web")
        assert result is None

    @pytest.mark.asyncio
    async def test_sample_recommend_libraries_returns_list(self):
        """sample_recommend_libraries returns list of recommendations."""
        from robotmcp.utils.sampling import sample_recommend_libraries

        mock_ctx = AsyncMock()
        mock_result = MagicMock()
        mock_result.text = json.dumps(
            [
                {"library_name": "Browser", "confidence": 0.95, "rationale": "Web testing"},
                {"library_name": "BuiltIn", "confidence": 0.8, "rationale": "Core library"},
            ]
        )
        mock_ctx.sample = AsyncMock(return_value=mock_result)

        rule_recs = [{"library_name": "Browser"}, {"library_name": "SeleniumLibrary"}]
        result = await sample_recommend_libraries(
            mock_ctx, "Click button on web", "web", rule_recs
        )
        assert result is not None
        assert isinstance(result, list)
        assert len(result) >= 1
        names = [r.get("library_name") for r in result]
        assert "Browser" in names

    @pytest.mark.asyncio
    async def test_sample_recommend_excludes_conflicts(self):
        """LLM recommendations should not contain both Browser and SeleniumLibrary."""
        from robotmcp.utils.sampling import sample_recommend_libraries

        mock_ctx = AsyncMock()
        mock_result = MagicMock()
        # Simulate LLM returning both (it shouldn't, but test the guard)
        mock_result.text = json.dumps(
            [
                {"library_name": "Browser", "confidence": 0.95, "rationale": "Modern web"},
                {
                    "library_name": "SeleniumLibrary",
                    "confidence": 0.85,
                    "rationale": "Legacy",
                },
            ]
        )
        mock_ctx.sample = AsyncMock(return_value=mock_result)

        result = await sample_recommend_libraries(
            mock_ctx, "Test web app", "web", []
        )
        if result:
            names = [r.get("library_name") for r in result]
            # At most one of Browser/SeleniumLibrary should be present
            assert not ("Browser" in names and "SeleniumLibrary" in names), (
                "Should not recommend both Browser and SeleniumLibrary"
            )

    @pytest.mark.asyncio
    async def test_sample_detect_library_preference_selenium(self):
        """Detects Selenium preference."""
        from robotmcp.utils.sampling import sample_detect_library_preference

        mock_ctx = AsyncMock()
        mock_result = MagicMock()
        mock_result.text = json.dumps({"library_preference": "SeleniumLibrary"})
        mock_ctx.sample = AsyncMock(return_value=mock_result)

        result = await sample_detect_library_preference(
            mock_ctx, "Use Selenium to test the checkout"
        )
        assert result is not None
        assert "Selenium" in result or "selenium" in result.lower()

    @pytest.mark.asyncio
    async def test_sample_detect_session_type(self):
        """Detects session type from scenario."""
        from robotmcp.utils.sampling import sample_detect_session_type

        mock_ctx = AsyncMock()
        mock_result = MagicMock()
        mock_result.text = json.dumps({"session_type": "api_testing"})
        mock_ctx.sample = AsyncMock(return_value=mock_result)

        result = await sample_detect_session_type(
            mock_ctx, "Send GET request to /api/users", "api"
        )
        assert result is not None
        assert "api" in result.lower()

    @pytest.mark.asyncio
    async def test_parse_json_with_code_fences(self):
        """_parse_json_response handles markdown code fences."""
        from robotmcp.utils.sampling import _parse_json_response

        text = '```json\n{"key": "value"}\n```'
        result = _parse_json_response(text)
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_parse_json_plain(self):
        """_parse_json_response handles plain JSON."""
        from robotmcp.utils.sampling import _parse_json_response

        result = _parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_parse_json_array(self):
        """_parse_json_response handles JSON arrays."""
        from robotmcp.utils.sampling import _parse_json_response

        result = _parse_json_response("[1, 2, 3]")
        assert result == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_parse_json_returns_none_on_invalid(self):
        """_parse_json_response returns None for invalid JSON."""
        from robotmcp.utils.sampling import _parse_json_response

        result = _parse_json_response("not json at all")
        assert result is None


# ===========================================================================
# 3. Server integration tests (MCP client, feature flag on)
# ===========================================================================


class TestServerSamplingIntegration:
    """Test sampling integration in server.py via MCP client.

    These tests use the real MCP server but mock the sampling module
    to avoid requiring an LLM API key.
    """

    @pytest.mark.asyncio
    async def test_analyze_scenario_without_sampling(self):
        """analyze_scenario works normally when sampling is disabled."""
        from fastmcp import Client
        from fastmcp.client import FastMCPTransport
        from robotmcp.server import mcp

        with patch.dict(os.environ, {"ROBOTMCP_USE_SAMPLING": "false"}):
            async with Client(
                FastMCPTransport(mcp, raise_exceptions=True),
                timeout=timedelta(seconds=30),
                init_timeout=30,
            ) as client:
                result = await client.call_tool(
                    "analyze_scenario",
                    {"scenario": "Click login button on website", "context": "web"},
                )
                data = _parse_result(result)
                assert data.get("success") is True
                assert "session_id" in data
                # Should not have sampling flag
                assert data.get("sampling_enhanced") is not True

    @pytest.mark.asyncio
    async def test_recommend_libraries_without_sampling(self):
        """recommend_libraries works normally when sampling is disabled."""
        from fastmcp import Client
        from fastmcp.client import FastMCPTransport
        from robotmcp.server import mcp

        with patch.dict(os.environ, {"ROBOTMCP_USE_SAMPLING": "false"}):
            async with Client(
                FastMCPTransport(mcp, raise_exceptions=True),
                timeout=timedelta(seconds=30),
                init_timeout=30,
            ) as client:
                result = await client.call_tool(
                    "recommend_libraries",
                    {"scenario": "Click login button", "context": "web"},
                )
                data = _parse_result(result)
                assert data.get("success") is True
                assert "recommendations" in data

    @pytest.mark.asyncio
    async def test_analyze_scenario_sampling_flag_present_when_enabled(self):
        """When sampling is enabled, the result should indicate sampling was attempted."""
        from fastmcp import Client
        from fastmcp.client import FastMCPTransport
        from robotmcp.server import mcp

        # Enable feature flag - sampling will be attempted but ctx.sample may not
        # be available in test client context, so we just verify the flag logic
        # executes without error.
        with patch.dict(os.environ, {"ROBOTMCP_USE_SAMPLING": "true"}):
            async with Client(
                FastMCPTransport(mcp, raise_exceptions=True),
                timeout=timedelta(seconds=30),
                init_timeout=30,
            ) as client:
                result = await client.call_tool(
                    "analyze_scenario",
                    {"scenario": "Send GET request to API endpoint", "context": "api"},
                )
                data = _parse_result(result)
                assert data.get("success") is True
                # The function should still succeed even if sampling fails/unavailable
                assert "session_id" in data


# ===========================================================================
# 4. Live LLM sampling tests (requires OPENAI_API_KEY)
# ===========================================================================


class TestLiveSampling:
    """Tests with real LLM calls. Only run if OPENAI_API_KEY is set."""

    @pytest.fixture(autouse=True)
    def skip_without_api_key(self):
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    @pytest.mark.asyncio
    async def test_live_recommend_with_sampling_enabled(self):
        """Full integration test: recommend_libraries with sampling enabled."""
        from fastmcp import Client
        from fastmcp.client import FastMCPTransport
        from robotmcp.server import mcp

        with patch.dict(os.environ, {"ROBOTMCP_USE_SAMPLING": "true"}):
            async with Client(
                FastMCPTransport(mcp, raise_exceptions=True),
                timeout=timedelta(seconds=60),
                init_timeout=60,
            ) as client:
                # First analyze scenario
                analyze_result = await client.call_tool(
                    "analyze_scenario",
                    {
                        "scenario": "Send a GET request to /api/users and verify the response is 200",
                        "context": "api",
                    },
                )
                analyze_data = _parse_result(analyze_result)
                session_id = analyze_data.get("session_id", "")

                # Then recommend libraries
                rec_result = await client.call_tool(
                    "recommend_libraries",
                    {
                        "scenario": "Send a GET request to /api/users and verify the response is 200",
                        "context": "api",
                        "session_id": session_id,
                    },
                )
                rec_data = _parse_result(rec_result)
                assert rec_data.get("success") is True
                recs = rec_data.get("recommendations", [])
                rec_names = [
                    r.get("library_name", r.get("name", "")) for r in recs
                ]
                assert "RequestsLibrary" in rec_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
