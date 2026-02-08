"""E2E keyword routing integration tests via MCP client.

Tests full tool-call sequences that exercise library preference enforcement,
keyword validation, and cross-library routing through the real MCP server.

Run with: uv run pytest tests/integration/test_keyword_routing_e2e.py -v
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from fastmcp import Client

from robotmcp.server import mcp


def _sid(prefix: str = "route") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


class TestSharedKeywordPassesBothSessions:
    """Shared keywords (Go To, Close Browser, etc.) should work regardless of session library."""

    @pytest.mark.asyncio
    async def test_shared_keyword_in_builtin_session(self, mcp_client):
        """Shared BuiltIn keywords work in any session."""
        sid = _sid("shared")
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "init", "libraries": ["BuiltIn"]},
        )

        # Log is a BuiltIn keyword — should always work
        result = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Log", "arguments": ["Shared keyword test"], "session_id": sid},
        )
        assert result.data["success"] is True


class TestNoPreferenceSessionAllowsAllKeywords:
    """Session without explicit library preference should allow all keywords."""

    @pytest.mark.asyncio
    async def test_no_preference_session_allows_builtin_keywords(self, mcp_client):
        """No-preference session allows all BuiltIn keywords."""
        sid = _sid("nopref")
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "init", "libraries": ["BuiltIn"]},
        )

        # Multiple BuiltIn keywords should work
        for kw, args in [
            ("Log", ["test"]),
            ("Set Variable", ["value"]),
            ("Should Be True", ["True"]),
        ]:
            result = await mcp_client.call_tool(
                "execute_step",
                {"keyword": kw, "arguments": args, "session_id": sid},
            )
            assert result.data["success"] is True, f"Keyword '{kw}' failed in no-preference session"


class TestFullToolCallSequence:
    """Full 5-tool sequence: analyze -> recommend -> manage -> execute -> state."""

    @pytest.mark.asyncio
    async def test_full_5_tool_sequence(self, mcp_client):
        """Complete tool call sequence exercises keyword routing at each stage."""
        # 1. Analyze scenario
        analyze = await mcp_client.call_tool(
            "analyze_scenario",
            {"scenario": "Test data processing with BuiltIn and Collections keywords", "context": "api"},
        )
        assert analyze.data["success"] is True
        sid = analyze.data["session_id"]

        # 2. Recommend libraries
        recommend = await mcp_client.call_tool(
            "recommend_libraries",
            {"scenario": "Data processing with lists and dictionaries"},
        )
        assert recommend.data["success"] is True

        # 3. Import additional library
        manage = await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "import_library", "library_name": "Collections"},
        )
        assert manage.data["success"] is True

        # 4. Execute keywords from both libraries
        step1 = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Log", "arguments": ["Starting data processing"], "session_id": sid},
        )
        assert step1.data["success"] is True

        step2 = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Create Dictionary",
                "arguments": ["name=Alice", "age=30"],
                "session_id": sid,
                "assign_to": "USER",
            },
        )
        assert step2.data["success"] is True

        # 5. Get session state
        state = await mcp_client.call_tool(
            "get_session_state",
            {"session_id": sid, "sections": ["summary", "variables", "libraries"]},
        )
        assert state.data["success"] is True


class TestFindKeywordsSessionAware:
    """Test find_keywords respects session library context."""

    @pytest.mark.asyncio
    async def test_find_keywords_with_session_id(self, mcp_client):
        """find_keywords with session_id should use session's library context."""
        sid = _sid("fksess")
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "init", "libraries": ["BuiltIn", "String"]},
        )

        # Search for a String library keyword
        result = await mcp_client.call_tool(
            "find_keywords",
            {"query": "Convert To Upper Case", "session_id": sid},
        )
        assert result.data["success"] is True
        keywords = result.data.get("result", [])
        assert len(keywords) > 0

    @pytest.mark.asyncio
    async def test_find_keywords_without_session_searches_globally(self, mcp_client):
        """find_keywords without session_id searches the global cache."""
        result = await mcp_client.call_tool(
            "find_keywords",
            {"query": "Log"},
        )
        assert result.data["success"] is True
        keywords = result.data.get("result", [])
        assert len(keywords) > 0
