import asyncio
import types

import pytest

from robotmcp.plugins.builtin.browser_plugin import BrowserLibraryPlugin
from robotmcp.plugins.manager import LibraryPluginManager


@pytest.mark.asyncio
async def test_browser_open_browser_override_rejects_and_guides():
    mgr = LibraryPluginManager()
    plugin = BrowserLibraryPlugin()
    mgr.register_plugin(plugin, source="test")

    override = mgr.get_keyword_override("Browser", "Open Browser")
    assert override is not None

    # Minimal session stub
    session = types.SimpleNamespace(
        session_id="sid",
        explicit_library_preference="",
        browser_state=types.SimpleNamespace(active_library="browser"),
    )

    result = await override(session, "Open Browser", ["http://example.com"], None)
    assert result is not None
    assert result["success"] is False
    assert "Open Browser" in result["error"]
    assert any("New Browser" in g for g in result.get("guidance", []))
