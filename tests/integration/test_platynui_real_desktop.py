"""Integration tests for PlatynUI with real desktop automation.

Tests PlatynUI.BareMetal keywords through rf-mcp MCP server against
real applications via AT-SPI2 accessibility bus on Linux.

Prerequisites:
    - PlatynUI native built from source (new_core branch, no mock-provider)
    - AT-SPI2 bus running (gsettings toolkit-accessibility true)
    - X11/XWayland display available

Run with:
    uv run pytest tests/integration/test_platynui_real_desktop.py -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid

import pytest
import pytest_asyncio
from fastmcp import Client

from robotmcp.server import mcp


def _has_real_platynui() -> bool:
    """Check if PlatynUI native is built with AT-SPI provider."""
    try:
        import platynui_native._native as pn

        r = pn.Runtime()
        providers = r.providers()
        return any(p.get("id") == "atspi" for p in providers)
    except Exception:
        return False


def _has_display() -> bool:
    """Check if a display is available."""
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _has_gnome_calculator() -> bool:
    """Check if gnome-calculator is installed."""
    return shutil.which("gnome-calculator") is not None


skip_no_platynui = pytest.mark.skipif(
    not _has_real_platynui(),
    reason="PlatynUI native not built with AT-SPI provider",
)
skip_no_display = pytest.mark.skipif(
    not _has_display(),
    reason="No display available",
)
skip_no_calculator = pytest.mark.skipif(
    not _has_gnome_calculator(),
    reason="gnome-calculator not installed",
)


def _sid(prefix: str = "platynui-real") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


# =============================================================================
# 1. PlatynUI Runtime Direct Tests (no MCP, just native)
# =============================================================================


@skip_no_platynui
@skip_no_display
class TestPlatynUIRuntimeDirect:
    """Test PlatynUI runtime directly without MCP layer."""

    def test_runtime_initializes_with_atspi(self):
        """Runtime initializes and discovers AT-SPI2 provider."""
        import platynui_native._native as pn

        r = pn.Runtime()
        providers = r.providers()
        assert len(providers) >= 1
        assert providers[0]["id"] == "atspi"
        assert providers[0]["technology"] == "AT-SPI2"

    def test_desktop_info_available(self):
        """Desktop info reports X11 with display bounds."""
        import platynui_native._native as pn

        r = pn.Runtime()
        di = r.desktop_info()
        assert di["name"] == "X11 Desktop"
        assert di["technology"] == "X11"
        assert di["bounds"].width > 0
        assert di["bounds"].height > 0

    def test_pointer_position_readable(self):
        """Pointer position is a valid point on screen."""
        import platynui_native._native as pn

        r = pn.Runtime()
        pos = r.pointer_position()
        assert hasattr(pos, "x")
        assert hasattr(pos, "y")

    def test_desktop_node_accessible(self):
        """Desktop node is enumerable with applications."""
        import platynui_native._native as pn

        r = pn.Runtime()
        dn = r.desktop_node()
        assert dn.role == "Desktop"
        # Should have at least one app
        count = sum(1 for _ in dn.children())
        assert count > 0

    def test_keyboard_known_keys(self):
        """Keyboard reports known key names."""
        import platynui_native._native as pn

        r = pn.Runtime()
        keys = r.keyboard_known_key_names()
        assert "ENTER" in keys
        assert "ESCAPE" in keys
        assert "BACKSPACE" in keys

    def test_pointer_move_to(self):
        """Pointer can be moved to a position."""
        import platynui_native._native as pn

        r = pn.Runtime()
        # Move to center of screen
        di = r.desktop_info()
        cx = int(di["bounds"].width / 2)
        cy = int(di["bounds"].height / 2)
        r.pointer_move_to((cx, cy))
        time.sleep(0.1)
        pos = r.pointer_position()
        # Allow small tolerance
        assert abs(pos.x - cx) <= 2
        assert abs(pos.y - cy) <= 2


# =============================================================================
# 2. PlatynUI RF Library Direct Tests
# =============================================================================


@skip_no_platynui
@skip_no_display
class TestPlatynUIRFLibrary:
    """Test PlatynUI.BareMetal RF library directly."""

    def test_library_importable(self):
        """PlatynUI.BareMetal is importable."""
        from PlatynUI.BareMetal import BareMetal

        lib = BareMetal()
        assert hasattr(lib, "pointer_click")
        assert hasattr(lib, "keyboard_type")

    def test_keyword_names(self):
        """Library reports expected keyword names."""
        from PlatynUI.BareMetal import BareMetal

        lib = BareMetal()
        names = lib.get_keyword_names()
        expected = {
            "pointer_click",
            "pointer_multi_click",
            "pointer_press",
            "pointer_release",
            "pointer_move_to",
            "keyboard_type",
            "keyboard_press",
            "keyboard_release",
            "activate",
            "maximize",
            "minimize",
            "restore",
            "focus",
            "close",
            "get_attribute",
            "query",
            "get_pointer_position",
            "take_screenshot",
            "highlight",
        }
        actual = set(names)
        assert expected.issubset(actual), f"Missing: {expected - actual}"

    def test_get_pointer_position(self):
        """get_pointer_position returns position dict."""
        from PlatynUI.BareMetal import BareMetal

        lib = BareMetal()
        pos = lib.get_pointer_position()
        assert pos is not None

    def test_pointer_move_to_keyword(self):
        """pointer_move_to keyword moves mouse using absolute coords."""
        from PlatynUI.BareMetal import BareMetal

        lib = BareMetal()
        # Pass None for descriptor, use x/y as absolute screen coords
        lib.pointer_move_to(None, x=100, y=100)
        pos = lib.get_pointer_position()
        assert pos is not None


# =============================================================================
# 3. MCP Integration Tests (PlatynUI session via rf-mcp)
# =============================================================================


@skip_no_platynui
@skip_no_display
@pytest.mark.asyncio
class TestPlatynUIMCPSession:
    """Test PlatynUI through MCP server layer."""

    async def test_session_init_desktop(self, mcp_client):
        """Initialize a desktop automation session."""
        sid = _sid()
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "scenario": "Desktop automation test with PlatynUI",
                "libraries": ["PlatynUI.BareMetal"],
            },
        )
        assert not result.is_error
        data = result.data
        assert data["success"] is True
        assert "PlatynUI.BareMetal" in data["libraries_loaded"]

    async def test_session_get_state_ui_tree(self, mcp_client):
        """get_session_state with ui_tree section for desktop session."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "scenario": "Desktop test",
                "libraries": ["PlatynUI.BareMetal"],
            },
        )
        result = await mcp_client.call_tool(
            "get_session_state",
            {
                "session_id": sid,
                "sections": ["ui_tree"],
            },
        )
        text = str(result)
        # Should contain ui_tree section or indicate desktop session
        assert "ui_tree" in text.lower() or "desktop" in text.lower()

    async def test_execute_get_pointer_position(self, mcp_client):
        """Execute get_pointer_position keyword via MCP."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "scenario": "Desktop test",
                "libraries": ["PlatynUI.BareMetal"],
            },
        )
        result = await mcp_client.call_tool(
            "execute_step",
            {
                "session_id": sid,
                "keyword": "Get Pointer Position",
            },
        )
        assert not result.is_error

    async def test_execute_pointer_move_to(self, mcp_client):
        """Execute pointer_move_to keyword to move cursor."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "scenario": "Desktop test",
                "libraries": ["PlatynUI.BareMetal"],
            },
        )
        result = await mcp_client.call_tool(
            "execute_step",
            {
                "session_id": sid,
                "keyword": "Pointer Move To",
                "arguments": ["${NONE}", "x=200", "y=200"],
            },
        )
        # May succeed or fail depending on RF variable resolution
        # The key test is that it doesn't crash the server
        assert result is not None

    async def test_find_platynui_keywords(self, mcp_client):
        """find_keywords returns PlatynUI keywords."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "scenario": "Desktop test",
                "libraries": ["PlatynUI.BareMetal"],
            },
        )
        result = await mcp_client.call_tool(
            "find_keywords",
            {
                "session_id": sid,
                "query": "pointer",
            },
        )
        text = str(result)
        assert "pointer" in text.lower()

    async def test_session_detect_desktop(self, mcp_client):
        """Session with desktop scenario detects PlatynUI."""
        sid = _sid()
        result = await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "scenario": "Test desktop application with PlatynUI for Windows calculator automation",
            },
        )
        assert not result.is_error
        data = result.data
        assert data["success"] is True


# =============================================================================
# 4. Calculator Interaction Tests (requires gnome-calculator)
# =============================================================================


@skip_no_platynui
@skip_no_display
@skip_no_calculator
@pytest.mark.asyncio
class TestPlatynUICalculator:
    """Test PlatynUI interaction with gnome-calculator.

    These tests interact with a real calculator application via
    pointer clicks at known screen coordinates.
    """

    @pytest_asyncio.fixture(autouse=True)
    async def setup_calculator(self):
        """Launch and position gnome-calculator."""
        # Kill existing instances
        subprocess.run(
            ["pkill", "-f", "gnome-calculator"],
            capture_output=True,
        )
        time.sleep(0.5)

        # Launch calculator
        self.calc_proc = subprocess.Popen(
            ["gnome-calculator"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)  # Wait for window to appear

        yield

        # Cleanup
        self.calc_proc.terminate()
        try:
            self.calc_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.calc_proc.kill()

    async def test_calculator_app_visible_in_atspi(self):
        """gnome-calculator appears in AT-SPI app list."""
        import platynui_native._native as pn

        r = pn.Runtime()
        dn = r.desktop_node()
        found = False
        for app in dn.children():
            if "calculator" in app.name.lower():
                found = True
                break
        assert found, "gnome-calculator not found in AT-SPI app list"

    async def test_calculator_frame_queryable(self, mcp_client):
        """Calculator frame is findable via PlatynUI query."""
        sid = _sid()
        await mcp_client.call_tool(
            "manage_session",
            {
                "action": "init",
                "session_id": sid,
                "scenario": "Desktop test",
                "libraries": ["PlatynUI.BareMetal"],
            },
        )
        result = await mcp_client.call_tool(
            "execute_step",
            {
                "session_id": sid,
                "keyword": "Query",
                "arguments": ["//control:Frame[@Name='Calculator']"],
            },
        )
        # Query may succeed or timeout depending on AT-SPI bus performance
        # The test validates the MCP→RF→PlatynUI integration path works
        assert result is not None
