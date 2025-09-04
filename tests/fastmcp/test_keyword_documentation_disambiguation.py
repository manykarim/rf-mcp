"""Disambiguation tests for get_keyword_documentation across libraries with same keyword names.

Validates that:
- When a library is specified, results are restricted to that library (no cross-library fallback).
- When no library is specified, all matching keywords across libraries are returned.
- Suggestions are provided when a keyword is not found in the requested library.
"""

import pytest
import pytest_asyncio
from fastmcp import Client

from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_browser_open_browser_doc(mcp_client):
    """Request Browser's Open Browser and ensure Browser docs are returned."""
    res = await mcp_client.call_tool(
        "get_keyword_documentation",
        {"keyword_name": "Open Browser", "library_name": "Browser"},
    )
    assert res.data.get("success") is True
    kw = res.data.get("keyword", {})
    assert kw.get("name") == "Open Browser"
    assert kw.get("library") == "Browser"


@pytest.mark.asyncio
async def test_selenium_open_browser_doc(mcp_client):
    """Request SeleniumLibrary's Open Browser and ensure Selenium docs are returned."""
    res = await mcp_client.call_tool(
        "get_keyword_documentation",
        {"keyword_name": "Open Browser", "library_name": "SeleniumLibrary"},
    )
    assert res.data.get("success") is True
    kw = res.data.get("keyword", {})
    assert kw.get("name") == "Open Browser"
    assert kw.get("library") == "SeleniumLibrary"


@pytest.mark.asyncio
async def test_open_browser_all_matches(mcp_client):
    """Without library filter, return all matches across libraries."""
    res = await mcp_client.call_tool(
        "get_keyword_documentation",
        {"keyword_name": "Open Browser"},
    )
    # New behavior: matches array with all libraries that define this keyword
    assert res.data.get("success") is True
    matches = res.data.get("matches")
    assert isinstance(matches, list) and len(matches) >= 1
    libs = {m.get("library") for m in matches}
    # Expect at least one; commonly both Browser and SeleniumLibrary are present
    assert "SeleniumLibrary" in libs
    assert "Browser" in libs


@pytest.mark.asyncio
async def test_not_found_in_library_suggestions(mcp_client):
    """When keyword not found in a specific library, return suggestions from that library."""
    res = await mcp_client.call_tool(
        "get_keyword_documentation",
        {"keyword_name": "Open Browsers", "library_name": "Browser"},
    )
    assert res.data.get("success") is False
    # Suggestions should include the close match in Browser: Open Browser
    sugg = res.data.get("suggestions", [])
    assert "Open Browser" in sugg or any("open browser" in s.lower() for s in sugg)

