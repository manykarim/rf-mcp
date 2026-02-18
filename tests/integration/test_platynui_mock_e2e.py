"""Integration tests for PlatynUI desktop automation support (ADR-012 Phase 2 & 5).

Tests cover:
  - PlatynUIPlugin class: metadata, capabilities, state provider, keyword map
  - PlatynUIStateProvider: get_page_source, get_application_state, get_ui_tree
  - server.py get_session_state "ui_tree" section handler
  - _is_desktop_session helper (via MCP round-trip)
  - Plugin registration via get_library_plugin_manager

All tests run WITHOUT the PlatynUI native backend.

Run with:
    uv run pytest tests/integration/test_platynui_mock_e2e.py -v
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from fastmcp import Client

from robotmcp.server import mcp


def _sid(prefix: str = "platynui") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


# =============================================================================
# 1. PlatynUIPlugin class (direct import from untracked file)
# =============================================================================


class TestPlatynUIPluginDirect:
    """Test PlatynUIPlugin class directly."""

    def test_plugin_instantiation(self):
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin

        plugin = PlatynUIPlugin()
        assert plugin.metadata.name == "PlatynUI"

    def test_plugin_metadata_fields(self):
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin

        plugin = PlatynUIPlugin()
        meta = plugin.metadata
        assert meta.library_type == "external"
        assert "desktop" in meta.categories
        assert meta.import_path == "PlatynUI.BareMetal"
        assert meta.package_name == "robotframework-platynui"

    def test_plugin_capabilities(self):
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin

        plugin = PlatynUIPlugin()
        caps = plugin.capabilities
        assert "desktop" in caps.contexts
        assert caps.supports_page_source is False
        assert caps.supports_application_state is True

    def test_plugin_has_state_provider(self):
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin

        plugin = PlatynUIPlugin()
        provider = plugin.get_state_provider()
        assert provider is not None
        assert hasattr(provider, "get_ui_tree")
        assert hasattr(provider, "get_page_source")
        assert hasattr(provider, "get_application_state")

    def test_plugin_keyword_library_map(self):
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin

        plugin = PlatynUIPlugin()
        kw_map = plugin.get_keyword_library_map()
        assert kw_map["query"] == "PlatynUI"
        assert kw_map["pointer click"] == "PlatynUI"
        assert kw_map["keyboard type"] == "PlatynUI"
        assert kw_map["activate"] == "PlatynUI"
        assert len(kw_map) == len(PlatynUIPlugin._ALL_KEYWORDS)

    def test_plugin_keyword_alternatives(self):
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin

        plugin = PlatynUIPlugin()
        alts = plugin.get_keyword_alternatives()
        assert "click element" in alts
        assert alts["click element"]["alternative"] == "Pointer Click"
        assert "input text" in alts
        assert alts["input text"]["alternative"] == "Keyboard Type"

    def test_plugin_no_incompatible_libraries(self):
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin

        plugin = PlatynUIPlugin()
        assert plugin.get_incompatible_libraries() == []

    def test_plugin_hints(self):
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin

        plugin = PlatynUIPlugin()
        hints = plugin.hints
        assert hints is not None
        assert len(hints.standard_keywords) > 0
        assert "Query" in hints.standard_keywords
        assert len(hints.usage_examples) > 0

    def test_plugin_all_keywords_complete(self):
        """Verify the keyword set includes all expected PlatynUI keywords."""
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin

        expected = {
            "query", "pointer click", "pointer multi click",
            "pointer press", "pointer release", "pointer move to",
            "get pointer position", "focus", "activate", "restore",
            "maximize", "minimize", "close", "get attribute",
            "keyboard type", "keyboard press", "keyboard release",
            "take screenshot", "highlight",
        }
        assert PlatynUIPlugin._ALL_KEYWORDS == expected

    def test_plugin_browser_shared_keywords(self):
        """Verify the browser-shared keyword set."""
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin

        expected_shared = {"focus", "get attribute", "take screenshot"}
        assert PlatynUIPlugin._BROWSER_SHARED_KEYWORDS == expected_shared


# =============================================================================
# 2. PlatynUIStateProvider (no native backend needed)
# =============================================================================


class TestPlatynUIStateProvider:
    """Direct tests for PlatynUIStateProvider methods."""

    @pytest.mark.asyncio
    async def test_get_page_source_returns_none(self):
        """Desktop apps have no HTML DOM; get_page_source returns None."""
        from robotmcp.models.session_models import ExecutionSession
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIStateProvider

        provider = PlatynUIStateProvider()
        session = ExecutionSession(session_id=_sid())
        result = await provider.get_page_source(
            session,
            full_source=True,
            filtered=False,
            filtering_level="standard",
            include_reduced_dom=False,
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_get_application_state_returns_stub(self):
        from robotmcp.models.session_models import ExecutionSession
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIStateProvider

        provider = PlatynUIStateProvider()
        session = ExecutionSession(session_id=_sid())
        result = await provider.get_application_state(session)
        assert result is not None
        assert result["provider"] == "PlatynUI"
        assert result["session_id"] == session.session_id
        assert "Query" in result["note"]

    @pytest.mark.asyncio
    async def test_get_ui_tree_returns_stub(self):
        from robotmcp.models.session_models import ExecutionSession
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIStateProvider

        provider = PlatynUIStateProvider()
        session = ExecutionSession(session_id=_sid())
        result = await provider.get_ui_tree(session, max_depth=5)
        assert result["success"] is False
        assert result["provider"] == "PlatynUI"
        assert result["max_depth"] == 5
        assert result["session_id"] == session.session_id
        assert "native backend" in result["note"].lower() or "Query" in result["note"]

    @pytest.mark.asyncio
    async def test_get_ui_tree_default_params(self):
        from robotmcp.models.session_models import ExecutionSession
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIStateProvider

        provider = PlatynUIStateProvider()
        session = ExecutionSession(session_id=_sid())
        result = await provider.get_ui_tree(session)
        assert result["max_depth"] == 3
        assert result["format"] == "text"

    @pytest.mark.asyncio
    async def test_get_ui_tree_format_parameter(self):
        from robotmcp.models.session_models import ExecutionSession
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIStateProvider

        provider = PlatynUIStateProvider()
        session = ExecutionSession(session_id=_sid())
        result = await provider.get_ui_tree(session, max_depth=2, format="json")
        assert result["format"] == "json"
        assert result["max_depth"] == 2


# =============================================================================
# 3. Plugin registration via plugin manager
# =============================================================================


class TestPluginManagerRegistration:
    """Test that PlatynUIPlugin can be registered and discovered."""

    def test_register_platynui_plugin(self):
        from robotmcp.plugins import get_library_plugin_manager
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin

        manager = get_library_plugin_manager()
        plugin = PlatynUIPlugin()
        manager.register_plugin(plugin, source="test")

        retrieved = manager.get_plugin("PlatynUI")
        assert retrieved is not None
        assert retrieved.metadata.name == "PlatynUI"

    def test_state_provider_via_manager(self):
        from robotmcp.plugins import get_library_plugin_manager
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin

        manager = get_library_plugin_manager()
        plugin = PlatynUIPlugin()
        manager.register_plugin(plugin, source="test")

        provider = manager.get_state_provider("PlatynUI")
        assert provider is not None
        assert hasattr(provider, "get_ui_tree")


# =============================================================================
# 4. Keyword validation for PlatynUI sessions
# =============================================================================


class TestPlatynUIKeywordValidation:
    """Test PlatynUIPlugin.validate_keyword_for_session behavior."""

    def test_validation_rejects_web_keyword_in_platynui_session(self):
        from robotmcp.models.session_models import ExecutionSession
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin

        plugin = PlatynUIPlugin()
        session = ExecutionSession(session_id=_sid())
        session.imported_libraries.append("PlatynUI.BareMetal")

        result = plugin.validate_keyword_for_session(
            session, "Click Element", "Browser"
        )
        assert result is not None
        assert result["success"] is False
        assert "Pointer Click" in result.get("alternative", "")

    def test_validation_allows_shared_keyword(self):
        from robotmcp.models.session_models import ExecutionSession
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin

        plugin = PlatynUIPlugin()
        session = ExecutionSession(session_id=_sid())
        session.imported_libraries.append("PlatynUI.BareMetal")

        result = plugin.validate_keyword_for_session(
            session, "Take Screenshot", "Browser"
        )
        assert result is None  # Shared keyword, allowed

    def test_validation_allows_web_keywords_when_web_library_imported(self):
        from robotmcp.models.session_models import ExecutionSession
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin

        plugin = PlatynUIPlugin()
        session = ExecutionSession(session_id=_sid())
        session.imported_libraries.append("PlatynUI.BareMetal")
        session.imported_libraries.append("Browser")

        result = plugin.validate_keyword_for_session(
            session, "Click Element", "Browser"
        )
        assert result is None  # Web library also imported, allowed

    def test_validation_passes_for_non_platynui_session(self):
        from robotmcp.models.session_models import ExecutionSession
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin

        plugin = PlatynUIPlugin()
        session = ExecutionSession(session_id=_sid())
        session.imported_libraries.append("Browser")

        result = plugin.validate_keyword_for_session(
            session, "Click Element", "Browser"
        )
        assert result is None  # Not a PlatynUI session


# =============================================================================
# 5. Failure hints generation
# =============================================================================


class TestPlatynUIFailureHints:
    """Test PlatynUIPlugin.generate_failure_hints."""

    def test_hint_for_element_not_found(self):
        from robotmcp.models.session_models import ExecutionSession
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin

        plugin = PlatynUIPlugin()
        session = ExecutionSession(session_id=_sid())
        hints = plugin.generate_failure_hints(
            session, "Pointer Click", ["/Window/Button[@Name='Missing']"],
            "Query returned no results for the given locator"
        )
        assert len(hints) > 0
        assert "not found" in hints[0]["title"].lower() or "no results" in hints[0]["title"].lower()

    def test_hint_for_native_backend_missing(self):
        from robotmcp.models.session_models import ExecutionSession
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin

        plugin = PlatynUIPlugin()
        session = ExecutionSession(session_id=_sid())
        hints = plugin.generate_failure_hints(
            session, "Query", ["/Window"],
            "No module named 'platynui_native'"
        )
        assert len(hints) > 0
        assert "native" in hints[0]["title"].lower() or "not installed" in hints[0]["title"].lower()

    def test_no_hints_for_unrelated_error(self):
        from robotmcp.models.session_models import ExecutionSession
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin

        plugin = PlatynUIPlugin()
        session = ExecutionSession(session_id=_sid())
        hints = plugin.generate_failure_hints(
            session, "Log", ["hello"], "Some unrelated error"
        )
        assert hints == []


# =============================================================================
# 6. get_session_state ui_tree section -- non-desktop session
# =============================================================================


class TestGetSessionStateUiTreeNonDesktop:
    """Test ui_tree section returns guidance for non-desktop sessions."""

    @pytest.mark.asyncio
    async def test_ui_tree_non_desktop_session_returns_error(self, mcp_client):
        """Requesting ui_tree on a non-desktop session returns guidance."""
        sid = _sid("web")
        init = await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "init"},
        )
        assert init.data["success"] is True

        state = await mcp_client.call_tool(
            "get_session_state",
            {"session_id": sid, "sections": ["ui_tree"]},
        )
        assert state.data["success"] is True
        ui_tree = state.data["sections"]["ui_tree"]
        assert ui_tree["success"] is False
        assert "PlatynUI" in ui_tree["error"] or "desktop" in ui_tree["error"]
        assert "guidance" in ui_tree

    @pytest.mark.asyncio
    async def test_ui_tree_unknown_session_returns_error(self, mcp_client):
        """Requesting ui_tree for an unknown session returns guidance."""
        sid = _sid("nonexistent")
        state = await mcp_client.call_tool(
            "get_session_state",
            {"session_id": sid, "sections": ["ui_tree"]},
        )
        assert state.data["success"] is True
        ui_tree = state.data["sections"]["ui_tree"]
        assert ui_tree["success"] is False
        assert "guidance" in ui_tree

    @pytest.mark.asyncio
    async def test_ui_tree_combined_with_other_sections(self, mcp_client):
        """ui_tree can be requested alongside other sections."""
        sid = _sid("combo")
        init = await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "init"},
        )
        assert init.data["success"] is True

        state = await mcp_client.call_tool(
            "get_session_state",
            {"session_id": sid, "sections": ["summary", "ui_tree", "variables"]},
        )
        assert state.data["success"] is True
        sections = state.data["sections"]
        assert "summary" in sections
        assert "ui_tree" in sections
        assert "variables" in sections


# =============================================================================
# 7. get_session_state ui_tree section -- desktop session (PlatynUI registered)
# =============================================================================


class TestGetSessionStateUiTreeDesktop:
    """Test ui_tree section for desktop sessions with PlatynUI plugin registered.

    These tests manually register the PlatynUI plugin and configure a session
    with PlatynUI imported so _is_desktop_session returns True, then verify
    the ui_tree section invokes the state provider stub.
    """

    @pytest.mark.asyncio
    async def test_ui_tree_desktop_session_via_direct_setup(self):
        """Manually configure a desktop session and verify ui_tree stub."""
        from robotmcp.models.session_models import ExecutionSession
        from robotmcp.plugins import get_library_plugin_manager
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin
        from robotmcp.server import execution_engine

        # Register PlatynUI plugin
        manager = get_library_plugin_manager()
        manager.register_plugin(PlatynUIPlugin(), source="test")

        # Create session and mark as desktop
        sid = _sid("desktop-direct")
        session = execution_engine.session_manager.create_session(sid)
        session.imported_libraries.append("PlatynUI.BareMetal")

        # Now call get_session_state via MCP
        async with Client(mcp) as client:
            state = await client.call_tool(
                "get_session_state",
                {"session_id": sid, "sections": ["ui_tree"]},
            )
            assert state.data["success"] is True
            ui_tree = state.data["sections"]["ui_tree"]
            # Should get the stub response from PlatynUIStateProvider
            assert ui_tree["provider"] == "PlatynUI"
            assert ui_tree["success"] is False  # stub returns False
            assert "native backend" in ui_tree["note"].lower() or "Query" in ui_tree["note"]

    @pytest.mark.asyncio
    async def test_ui_tree_desktop_session_includes_session_id(self):
        """Desktop session ui_tree stub includes session_id."""
        from robotmcp.plugins import get_library_plugin_manager
        from robotmcp.plugins.builtin.platynui_plugin import PlatynUIPlugin
        from robotmcp.server import execution_engine

        manager = get_library_plugin_manager()
        manager.register_plugin(PlatynUIPlugin(), source="test")

        sid = _sid("desktop-sid")
        session = execution_engine.session_manager.create_session(sid)
        session.imported_libraries.append("PlatynUI")

        async with Client(mcp) as client:
            state = await client.call_tool(
                "get_session_state",
                {"session_id": sid, "sections": ["ui_tree"]},
            )
            ui_tree = state.data["sections"]["ui_tree"]
            assert ui_tree["session_id"] == sid


# =============================================================================
# 8. _is_desktop_session helper (tested indirectly via MCP round-trip)
# =============================================================================


class TestIsDesktopSessionHelper:
    """Verify _is_desktop_session detects desktop sessions correctly."""

    def test_helper_with_platynui_imported(self):
        from robotmcp.models.session_models import ExecutionSession
        from robotmcp.server import _is_desktop_session

        session = ExecutionSession(session_id=_sid())
        session.imported_libraries.append("PlatynUI.BareMetal")
        assert _is_desktop_session(session) is True

    def test_helper_with_platynui_name(self):
        from robotmcp.models.session_models import ExecutionSession
        from robotmcp.server import _is_desktop_session

        session = ExecutionSession(session_id=_sid())
        session.imported_libraries.append("PlatynUI")
        assert _is_desktop_session(session) is True

    def test_helper_web_session_not_desktop(self):
        from robotmcp.models.session_models import ExecutionSession
        from robotmcp.server import _is_desktop_session

        session = ExecutionSession(session_id=_sid())
        session.imported_libraries.append("Browser")
        assert _is_desktop_session(session) is False

    def test_helper_empty_session_not_desktop(self):
        from robotmcp.models.session_models import ExecutionSession
        from robotmcp.server import _is_desktop_session

        session = ExecutionSession(session_id=_sid())
        assert _is_desktop_session(session) is False

    def test_helper_with_desktop_platform_type(self):
        from robotmcp.models.session_models import ExecutionSession, PlatformType
        from robotmcp.server import _is_desktop_session

        session = ExecutionSession(session_id=_sid())
        session.platform_type = PlatformType.DESKTOP
        assert _is_desktop_session(session) is True
