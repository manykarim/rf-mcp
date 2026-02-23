"""Unit tests for the fastmcp compatibility layer.

Tests cover: version detection, DISABLED_TOOL_KWARGS, finalize_disabled_tools,
get_tool_fn, ToolManagerCompat, set_server_lifespan.

Run with: uv run pytest tests/unit/test_fastmcp_compat.py -v
"""

__test__ = True

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from robotmcp.compat.fastmcp_compat import (
    DISABLED_TOOL_KWARGS,
    FASTMCP_V3,
    FASTMCP_VERSION,
    ToolManagerCompat,
    finalize_disabled_tools,
    get_tool_fn,
    set_server_lifespan,
)


# =============================================================================
# Version detection
# =============================================================================


class TestVersionDetection:
    """Test fastmcp version detection."""

    def test_version_string_not_empty(self):
        assert FASTMCP_VERSION != ""
        assert "." in FASTMCP_VERSION

    def test_v3_flag_matches_version(self):
        major = int(FASTMCP_VERSION.split(".")[0])
        assert FASTMCP_V3 == (major >= 3)

    def test_current_version_is_v2(self):
        """We're running on fastmcp 2.x in the test environment."""
        assert not FASTMCP_V3
        assert FASTMCP_VERSION.startswith("2.")


# =============================================================================
# DISABLED_TOOL_KWARGS
# =============================================================================


class TestDisabledToolKwargs:
    """Test DISABLED_TOOL_KWARGS behavior."""

    def test_on_v2_has_enabled_false(self):
        """On v2, DISABLED_TOOL_KWARGS should contain enabled=False."""
        if not FASTMCP_V3:
            assert DISABLED_TOOL_KWARGS == {"enabled": False}

    def test_on_v3_is_empty(self):
        """On v3, DISABLED_TOOL_KWARGS should be empty (tools disabled later)."""
        if FASTMCP_V3:
            assert DISABLED_TOOL_KWARGS == {}

    def test_can_unpack_into_tool_decorator(self):
        """Verify kwargs can be unpacked without error."""
        d = {**DISABLED_TOOL_KWARGS}
        assert isinstance(d, dict)


# =============================================================================
# finalize_disabled_tools
# =============================================================================


class TestFinalizeDisabledTools:
    """Test finalize_disabled_tools function."""

    def test_noop_on_v2(self):
        """On v2, finalize is a no-op (enabled=False already handled)."""
        mock_server = MagicMock()
        finalize_disabled_tools(mock_server, {"tool_a", "tool_b"})
        if not FASTMCP_V3:
            mock_server.disable.assert_not_called()

    def test_empty_names_skipped(self):
        """Empty name set should be skipped."""
        mock_server = MagicMock()
        finalize_disabled_tools(mock_server, set())
        mock_server.disable.assert_not_called()

    @patch("robotmcp.compat.fastmcp_compat.FASTMCP_V3", True)
    def test_calls_disable_on_v3(self):
        """On v3, should call mcp.disable(names=...)."""
        mock_server = MagicMock()
        names = {"tool_a", "tool_b"}
        finalize_disabled_tools(mock_server, names)
        mock_server.disable.assert_called_once_with(names=names)

    @patch("robotmcp.compat.fastmcp_compat.FASTMCP_V3", True)
    def test_handles_missing_disable_method(self):
        """On v3 without disable(), should log warning not crash."""
        mock_server = MagicMock(spec=[])  # no disable attribute
        finalize_disabled_tools(mock_server, {"tool_a"})
        # Should not raise


# =============================================================================
# get_tool_fn
# =============================================================================


class TestGetToolFn:
    """Test get_tool_fn helper."""

    def test_returns_fn_attr_when_present(self):
        """On v2, tool objects have .fn attribute."""
        async def real_fn():
            pass
        tool_obj = MagicMock()
        tool_obj.fn = real_fn
        assert get_tool_fn(tool_obj) is real_fn

    def test_returns_object_when_no_fn(self):
        """On v3, tool IS the function (no .fn attribute)."""
        async def real_fn():
            pass
        assert get_tool_fn(real_fn) is real_fn

    def test_returns_callable(self):
        """Result should always be callable."""
        async def real_fn():
            pass

        # v2 style
        tool_obj = MagicMock()
        tool_obj.fn = real_fn
        assert callable(get_tool_fn(tool_obj))

        # v3 style
        assert callable(get_tool_fn(real_fn))


# =============================================================================
# ToolManagerCompat
# =============================================================================


class TestToolManagerCompat:
    """Test ToolManagerCompat version-aware wrapper."""

    def _make_v2_server(self):
        """Create a mock FastMCP v2 server with _tool_manager."""
        server = MagicMock()
        server._tool_manager = MagicMock()
        server._tool_manager.get_tools = AsyncMock(return_value={
            "tool_a": MagicMock(name="tool_a", enabled=True),
            "tool_b": MagicMock(name="tool_b", enabled=False),
        })
        server._tool_manager.has_tool = AsyncMock(return_value=True)
        return server

    @pytest.mark.asyncio
    async def test_get_tools_v2(self):
        server = self._make_v2_server()
        compat = ToolManagerCompat(server)
        tools = await compat.get_tools()
        assert "tool_a" in tools
        assert "tool_b" in tools

    @pytest.mark.asyncio
    async def test_has_tool_v2(self):
        server = self._make_v2_server()
        compat = ToolManagerCompat(server)
        assert await compat.has_tool("tool_a") is True

    def test_remove_tool_v2(self):
        server = self._make_v2_server()
        compat = ToolManagerCompat(server)
        compat.remove_tool("tool_a")
        server._tool_manager.remove_tool.assert_called_once_with("tool_a")

    def test_add_tool_v2(self):
        server = self._make_v2_server()
        compat = ToolManagerCompat(server)
        tool = MagicMock()
        compat.add_tool(tool)
        server._tool_manager.add_tool.assert_called_once_with(tool)


# =============================================================================
# set_server_lifespan
# =============================================================================


class TestSetServerLifespan:
    """Test set_server_lifespan compatibility helper."""

    def test_sets_on_mcp_server(self):
        """v2: sets via _mcp_server.lifespan."""
        server = MagicMock()
        server._mcp_server = MagicMock()
        lifespan = MagicMock()
        set_server_lifespan(server, lifespan)
        assert server._mcp_server.lifespan is lifespan

    def test_sets_on_server_directly(self):
        """v3 fallback: sets .lifespan directly on server."""
        server = MagicMock(spec=["lifespan"])
        lifespan = MagicMock()
        set_server_lifespan(server, lifespan)
        assert server.lifespan is lifespan

    def test_no_crash_when_no_attribute(self):
        """Should log warning, not crash, if no compatible attribute."""
        server = MagicMock(spec=[])
        lifespan = MagicMock()
        set_server_lifespan(server, lifespan)
        # Should not raise


# =============================================================================
# ToolManagerAdapter integration
# =============================================================================


class TestToolManagerAdapterFiltersDisabled:
    """Test that ToolManagerAdapter correctly filters disabled tools."""

    @pytest.mark.asyncio
    async def test_initialize_excludes_disabled(self):
        """Snapshot should only include enabled tools."""
        from robotmcp.domains.tool_profile.adapters.fastmcp_adapter import (
            ToolManagerAdapter,
        )

        mock_tool_enabled = MagicMock()
        mock_tool_enabled.enabled = True
        mock_tool_enabled.name = "enabled_tool"

        mock_tool_disabled = MagicMock()
        mock_tool_disabled.enabled = False
        mock_tool_disabled.name = "disabled_tool"

        server = MagicMock()
        server._tool_manager = MagicMock()
        server._tool_manager.get_tools = AsyncMock(return_value={
            "enabled_tool": mock_tool_enabled,
            "disabled_tool": mock_tool_disabled,
        })

        adapter = ToolManagerAdapter(server)
        await adapter.initialize()

        assert "enabled_tool" in adapter._original_tools
        assert "disabled_tool" not in adapter._original_tools

    @pytest.mark.asyncio
    async def test_get_visible_tool_names_excludes_disabled(self):
        """get_visible_tool_names should filter out disabled tools."""
        from robotmcp.domains.tool_profile.adapters.fastmcp_adapter import (
            ToolManagerAdapter,
        )

        mock_enabled = MagicMock()
        mock_enabled.enabled = True

        mock_disabled = MagicMock()
        mock_disabled.enabled = False

        server = MagicMock()
        server._tool_manager = MagicMock()
        server._tool_manager.get_tools = AsyncMock(return_value={
            "visible_a": mock_enabled,
            "visible_b": mock_enabled,
            "hidden_c": mock_disabled,
        })

        adapter = ToolManagerAdapter(server)
        names = await adapter.get_visible_tool_names()

        assert "visible_a" in names
        assert "visible_b" in names
        assert "hidden_c" not in names


# =============================================================================
# Server.py disabled tool names consistency
# =============================================================================


class TestDisabledToolNamesConsistency:
    """Verify _DISABLED_TOOL_NAMES in server.py matches actual decorators."""

    def test_disabled_names_set_exists(self):
        """The _DISABLED_TOOL_NAMES set should be importable from server."""
        import ast
        import pathlib

        server_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "src"
            / "robotmcp"
            / "server.py"
        )
        source = server_path.read_text()

        # Find the frozenset assignment
        assert "_DISABLED_TOOL_NAMES" in source
        assert "finalize_disabled_tools" in source

    def test_disabled_names_match_decorators(self):
        """All DISABLED_TOOL_KWARGS decorators should have names in the set."""
        import ast
        import pathlib

        server_path = (
            pathlib.Path(__file__).parent.parent.parent
            / "src"
            / "robotmcp"
            / "server.py"
        )
        source = server_path.read_text()
        tree = ast.parse(source)

        # Extract tool names from decorators using DISABLED_TOOL_KWARGS
        decorator_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.AsyncFunctionDef):
                for deco in node.decorator_list:
                    if isinstance(deco, ast.Call):
                        src_segment = ast.get_source_segment(source, deco)
                        if src_segment and "DISABLED_TOOL_KWARGS" in src_segment:
                            tool_name = node.name
                            for kw in deco.keywords:
                                if kw.arg == "name" and isinstance(
                                    kw.value, ast.Constant
                                ):
                                    tool_name = kw.value.value
                                    break
                            decorator_names.add(tool_name)

        # Extract names from _DISABLED_TOOL_NAMES frozenset
        # Handles both ast.Assign and ast.AnnAssign (type-annotated)
        set_names = set()
        for node in ast.walk(tree):
            target_name = None
            value = None
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                t = node.targets[0]
                if isinstance(t, ast.Name):
                    target_name = t.id
                    value = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(
                node.target, ast.Name
            ):
                target_name = node.target.id
                value = node.value

            if target_name == "_DISABLED_TOOL_NAMES" and value is not None:
                if isinstance(value, ast.Call):
                    for arg in value.args:
                        if isinstance(arg, ast.Set):
                            for elt in arg.elts:
                                if isinstance(elt, ast.Constant):
                                    set_names.add(elt.value)

        assert decorator_names, "Should find decorated tools"
        assert set_names, "Should find _DISABLED_TOOL_NAMES entries"
        assert decorator_names == set_names, (
            f"Mismatch!\n"
            f"  In decorators but not in set: {decorator_names - set_names}\n"
            f"  In set but not in decorators: {set_names - decorator_names}"
        )
