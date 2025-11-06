import asyncio
from unittest.mock import AsyncMock

import pytest

from robotmcp.components.execution.page_source_service import PageSourceService
from robotmcp.models.session_models import ExecutionSession


def _make_session(session_id: str = "session") -> ExecutionSession:
    session = ExecutionSession(session_id=session_id)
    session.imported_libraries.append("Browser")
    return session


def test_get_page_source_via_rf_context_populates_session(monkeypatch):
    service = PageSourceService()
    session = _make_session()

    class DummyRFManager:
        def __init__(self):
            self.calls = []

        def execute_keyword_with_context(
            self,
            *,
            session_id,
            keyword_name,
            arguments,
            assign_to=None,
            session_variables=None,
        ):
            self.calls.append(keyword_name)
            if keyword_name == "Get Page Source":
                return {"success": True, "output": "<html><title>T</title></html>"}
            if keyword_name == "Get Url":
                return {"success": True, "output": "https://example.test"}
            if keyword_name == "Get Title":
                return {"success": True, "output": "Example"}
            return {"success": False}

    dummy = DummyRFManager()
    monkeypatch.setattr(
        "robotmcp.components.execution.rf_native_context_manager.get_rf_native_context_manager",
        lambda: dummy,
    )

    html = service._get_page_source_via_rf_context(session)

    assert html.startswith("<html>")
    assert session.browser_state.current_url == "https://example.test"
    assert session.browser_state.page_title == "Example"


def test_get_page_source_via_rf_context_fallback_on_error(monkeypatch):
    service = PageSourceService()
    session = _make_session()
    session.browser_state.page_source = "<cached></cached>"

    class ExplodingManager:
        def execute_keyword_with_context(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "robotmcp.components.execution.rf_native_context_manager.get_rf_native_context_manager",
        lambda: ExplodingManager(),
    )

    html = service._get_page_source_via_rf_context(session)
    assert html == "<cached></cached>"


@pytest.mark.asyncio
async def test_get_page_source_from_plugin_success(monkeypatch):
    service = PageSourceService()
    session = _make_session()

    class DummyProvider:
        def __init__(self):
            self.calls = []

        async def get_page_source(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return {"success": True, "page_source": "<html>plugin</html>"}

    provider = DummyProvider()

    class DummyPluginManager:
        def get_state_provider(self, active_library):
            assert active_library == "Browser"
            return provider

        def list_plugin_names(self):
            return ["dummy"]

    monkeypatch.setattr(
        "robotmcp.components.execution.page_source_service.library_registry.get_all_libraries",
        lambda: None,
    )
    monkeypatch.setattr(
        "robotmcp.components.execution.page_source_service.get_library_plugin_manager",
        lambda: DummyPluginManager(),
    )

    result = await service._get_page_source_from_plugin(
        session,
        browser_library_manager=None,
        full_source=True,
        filtered=False,
        filtering_level="standard",
        include_reduced_dom=True,
    )

    assert result == {"success": True, "page_source": "<html>plugin</html>"}
    assert provider.calls


@pytest.mark.asyncio
async def test_get_page_source_from_plugin_handles_missing_provider(monkeypatch):
    service = PageSourceService()
    session = _make_session()

    class DummyPluginManager:
        def get_state_provider(self, active_library):
            return None

    monkeypatch.setattr(
        "robotmcp.components.execution.page_source_service.library_registry.get_all_libraries",
        lambda: None,
    )
    monkeypatch.setattr(
        "robotmcp.components.execution.page_source_service.get_library_plugin_manager",
        lambda: DummyPluginManager(),
    )

    result = await service._get_page_source_from_plugin(
        session,
        browser_library_manager=None,
        full_source=False,
        filtered=False,
        filtering_level="standard",
        include_reduced_dom=False,
    )

    assert result is None


def test_filter_page_source_levels():
    html = """
    <html>
        <head>
            <title>Sample</title>
            <script>var x = 1;</script>
            <style>.hidden { display:none; }</style>
        </head>
        <body>
            <div class="visible">Hello</div>
            <div class="hidden">Secret</div>
            <button onclick="alert(1)">Click</button>
        </body>
    </html>
    """
    service = PageSourceService()
    standard = service.filter_page_source(html, "standard")
    aggressive = service.filter_page_source(html, "aggressive")

    assert "<script>" not in standard
    assert "onclick" not in standard
    # Hidden element stays for standard filtering
    assert "Secret" in standard
    # Aggressive filtering removes hidden node entirely
    assert "Secret" not in aggressive


@pytest.mark.asyncio
async def test_get_page_source_fallback_to_rf(monkeypatch):
    service = PageSourceService()
    session = _make_session()

    session.browser_state.current_url = None
    session.browser_state.page_title = None

    service._get_page_source_from_plugin = AsyncMock(return_value=None)
    service._get_page_source_via_rf_context = lambda s: "<html><body>data</body></html>"

    result = await service.get_page_source(
        session=session,
        browser_library_manager=None,
        full_source=False,
        filtered=True,
        filtering_level="minimal",
        include_reduced_dom=False,
    )

    assert result["success"] is True
    assert result["filtering_applied"] is True
    assert "page_source_preview" in result


@pytest.mark.asyncio
async def test_extract_page_context(monkeypatch):
    service = PageSourceService()
    html = """
    <html>
        <head><title>Demo Page</title></head>
        <body>
            <form action="/submit" method="post">
                <input type="text" name="username" id="user" placeholder="Name"/>
                <button type="submit">Send</button>
            </form>
            <a href="/more" title="More info">More Info</a>
            <h1>Main Heading</h1>
            <img src="img.png" alt="demo" />
        </body>
    </html>
    """
    context = await service.extract_page_context(html)

    assert context["page_title"] == "Demo Page"
    assert context["forms"][0]["method"] == "POST"
    assert context["buttons"][0]["text"] == "Send"
    assert context["links"][0]["href"] == "/more"
    assert context["headings"][0]["level"] == "h1"
    assert context["images"][0]["src"] == "img.png"


def test_get_filtered_source_stats(monkeypatch):
    service = PageSourceService()
    original = "<html><body><div>keep</div><script>drop()</script></body></html>"
    filtered = "<html><body><div>keep</div></body></html>"
    stats = service.get_filtered_source_stats(original, filtered)

    assert stats["original_size"] > stats["filtered_size"]
    assert stats["elements_removed"] >= 1


def test_validate_and_supported_levels():
    service = PageSourceService()
    assert service.validate_filtering_level("minimal") is True
    assert service.validate_filtering_level("invalid") is False
    assert set(service.get_supported_filtering_levels()) == {"minimal", "standard", "aggressive"}
