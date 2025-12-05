import pytest
from types import SimpleNamespace

from robotmcp.plugins.builtin.browser_plugin import BrowserLibraryPlugin
from robotmcp import server


@pytest.mark.asyncio
async def test_open_browser_translates_to_playwright(monkeypatch):
    plugin = BrowserLibraryPlugin()
    calls = []

    async def fake_execute_step(keyword, arguments=None, session_id=None, **kwargs):
        calls.append((keyword, arguments or []))
        return {"success": True}

    monkeypatch.setattr(server.execution_engine, "execute_step", fake_execute_step)

    session = SimpleNamespace(
        session_id="sid",
        explicit_library_preference="Browser",
        browser_state=SimpleNamespace(
            active_library="browser", browser_id=None, context_id=None
        ),
    )

    result = await plugin._override_open_browser(
        session, "Open Browser", ["https://example.com", "chromium"], None
    )

    assert result and result.get("success") is True
    assert calls == [
        ("New Browser", ["chromium"]),
        ("New Context", []),
        ("New Page", ["https://example.com"]),
    ]


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
    calls = []

    async def fake_execute_step(keyword, arguments=None, session_id=None, **kwargs):
        calls.append((keyword, arguments or []))
        return {"success": True}

    monkeypatch.setattr(server.execution_engine, "execute_step", fake_execute_step)

    session = SimpleNamespace(
        session_id="sid",
        explicit_library_preference="Browser",
        browser_state=SimpleNamespace(
            active_library="browser", browser_id="b1", context_id="c1"
        ),
    )

    result = await plugin._override_open_browser(
        session, "Open Browser", ["https://example.com"], None
    )

    assert result and result.get("success") is True
    # Only a new page should be opened when browser/context already exist
    assert calls == [("New Page", ["https://example.com"])]

