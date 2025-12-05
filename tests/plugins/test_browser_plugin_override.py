import pytest
from types import SimpleNamespace

from robotmcp.plugins.builtin.browser_plugin import BrowserLibraryPlugin
from robotmcp import server


@pytest.mark.asyncio
async def test_open_browser_rejected_with_guidance(monkeypatch):
    plugin = BrowserLibraryPlugin()
    result = await plugin._override_open_browser(
        SimpleNamespace(
            session_id="sid",
            explicit_library_preference="Browser",
            browser_state=SimpleNamespace(
                active_library="browser", browser_id=None, context_id=None
            ),
        ),
        "Open Browser",
        ["https://example.com", "chromium"],
        None,
    )

    assert result and result.get("success") is False
    assert "Open Browser" in result.get("error", "")
    guidance = result.get("guidance", []) or []
    assert any("New Browser" in g for g in guidance)


@pytest.mark.asyncio
async def test_open_browser_override_skips_for_selenium(monkeypatch):
    plugin = BrowserLibraryPlugin()

    async def fake_execute_step(*args, **kwargs):
        raise AssertionError("Should not be called")

    monkeypatch.setattr(server.execution_engine, "execute_step", fake_execute_step)

    session = SimpleNamespace(
        session_id="sid",
        explicit_library_preference="SeleniumLibrary",
        browser_state=SimpleNamespace(active_library="selenium", browser_id=None, context_id=None),
    )

    result = await plugin._override_open_browser(session, "Open Browser", ["https://example.com"], None)
    assert result is None


@pytest.mark.asyncio
async def test_open_browser_reuses_existing_handles(monkeypatch):
    plugin = BrowserLibraryPlugin()
    result = await plugin._override_open_browser(
        SimpleNamespace(
            session_id="sid",
            explicit_library_preference="Browser",
            browser_state=SimpleNamespace(
                active_library="browser", browser_id="b1", context_id="c1"
            ),
        ),
        "Open Browser",
        ["https://example.com"],
        None,
    )

    assert result and result.get("success") is False
