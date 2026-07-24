"""Live E2E tests for the PlatynUI new-core integration (ADR-025).

Drives the REAL MCP server in-process (fastmcp.Client) against the REAL
desktop via platynui-native. Requires:
* a display (X11 or XWayland — the env shim forces the X11 backend)
* platynui-native + the matched-set RF library installed (see ADR-025)

All tests are skipped automatically when the prerequisites are missing.

Run with: uv run --no-sync pytest tests/integration/test_platynui_newcore_e2e.py -v
"""

from __future__ import annotations

import os
import sys
import uuid

import pytest
import pytest_asyncio


def _platynui_available() -> bool:
    try:
        from PlatynUI.BareMetal import BareMetal  # noqa: F401
        return True
    except Exception:
        return False


def _x_server_reachable(display: str) -> bool:
    """True only when an X server for ``display`` actually accepts a connection.

    A set-but-dangling ``DISPLAY`` (e.g. on a headless CI runner where the env
    var is present but no X server listens) passes a bare ``DISPLAY`` check yet
    fails at runtime with ``x11 connect: Connection refused (os error 111)``.
    Probe the real endpoint so those hosts skip instead of failing."""
    import re
    import socket

    m = re.match(r"^(.*):(\d+)(?:\.\d+)?$", display)
    if not m:
        return False
    host, dnum = m.group(1), int(m.group(2))
    try:
        if host in ("", "unix"):  # local display -> X11 unix socket
            path = f"/tmp/.X11-unix/X{dnum}"
            if not os.path.exists(path):
                return False
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            sock.connect(path)
            sock.close()
            return True
        sock = socket.create_connection((host, 6000 + dnum), timeout=0.5)  # TCP X11
        sock.close()
        return True
    except OSError:
        return False


def _display_available() -> bool:
    display = os.environ.get("DISPLAY")
    return (
        sys.platform == "linux"
        and bool(display)
        and _x_server_reachable(display)
    )


pytestmark = [
    pytest.mark.skipif(
        not _platynui_available(),
        reason="PlatynUI.BareMetal not importable (matched-set install required)",
    ),
    pytest.mark.skipif(
        not _display_available(),
        reason="No DISPLAY available (X11/XWayland required)",
    ),
]

from fastmcp import Client  # noqa: E402

from robotmcp.server import mcp  # noqa: E402


def _sid() -> str:
    return f"platynui-e2e-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


class TestPlatynUINewCoreE2E:
    """Full MCP-tool workflow against the live desktop."""

    @pytest.mark.asyncio
    async def test_full_desktop_workflow(self, mcp_client):
        sid = _sid()

        # 1. Init desktop session with explicit PlatynUI library.
        #    The plugin's on_session_start shim must force the X11 backend
        #    BEFORE any Runtime exists (Wayland portal hang, ADR-025 E2).
        init = await mcp_client.call_tool(
            "manage_session",
            {
                "session_id": sid,
                "action": "init",
                "scenario": "Automate a native desktop application using PlatynUI",
                "libraries": ["PlatynUI.BareMetal", "BuiltIn"],
            },
        )
        assert init.data["success"] is True

        # 2. Execute a real read keyword (no RF-context dependency)
        pos = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Get Pointer Position",
                "arguments": [],
                "session_id": sid,
                "assign_to": "pos",
            },
        )
        assert pos.data["success"] is True, pos.data

        # Env shim fires at the first desktop keyword execution (executor
        # chokepoint, ADR-025): wayland must never survive when DISPLAY is
        # set, or the first Runtime would block on the portal handshake.
        assert os.environ.get("XDG_SESSION_TYPE", "").lower() != "wayland"

        # 3. Query the application list through RF context (fast, scoped)
        apps = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Query",
                "arguments": ["/app:*"],
                "session_id": sid,
                "assign_to": "apps",
            },
        )
        assert apps.data["success"] is True, apps.data

        # 4. ui_tree section lists applications
        state = await mcp_client.call_tool(
            "get_session_state",
            {"session_id": sid, "sections": ["ui_tree"]},
        )
        ui_tree = state.data["sections"]["ui_tree"]
        assert ui_tree["success"] is True, ui_tree
        assert ui_tree["application_count"] >= 1
        names = [a.get("name", "") for a in ui_tree["applications"]]
        assert any(names), names

        # 5. Suite generation includes the executed PlatynUI steps
        suite = await mcp_client.call_tool(
            "build_test_suite",
            {"session_id": sid, "test_name": "PlatynUI Desktop Smoke"},
        )
        assert suite.data["success"] is True
        rf_text = suite.data.get("rf_text", "")
        assert "PlatynUI.BareMetal" in rf_text
        assert "Get Pointer Position" in rf_text

    @pytest.mark.asyncio
    async def test_locator_guidance_dispatch(self, mcp_client):
        guidance = await mcp_client.call_tool(
            "get_locator_guidance",
            {"library": "platynui", "error_message": "element not found"},
        )
        assert guidance.data["success"] is True
        assert guidance.data["library"] == "PlatynUI.BareMetal"
        assert "performance_rules" in guidance.data
        assert "element_not_found_suggestions" in guidance.data

    @pytest.mark.asyncio
    async def test_ui_tree_expansion_with_filter(self, mcp_client):
        """Expanding a known-present application (gnome-shell on GNOME)."""
        sid = _sid()
        init = await mcp_client.call_tool(
            "manage_session",
            {
                "session_id": sid,
                "action": "init",
                "scenario": "Inspect desktop applications with PlatynUI",
                "libraries": ["PlatynUI.BareMetal", "BuiltIn"],
            },
        )
        assert init.data["success"] is True

        state = await mcp_client.call_tool(
            "get_session_state",
            {"session_id": sid, "sections": ["ui_tree"]},
        )
        ui_tree = state.data["sections"]["ui_tree"]
        assert ui_tree["success"] is True
        if not ui_tree["applications"]:
            pytest.skip("No applications on AT-SPI bus")
        target = ui_tree["applications"][0]["name"]

        expanded = await mcp_client.call_tool(
            "get_session_state",
            {
                "session_id": sid,
                "sections": ["ui_tree"],
                "elements_of_interest": [target],
            },
        )
        tree2 = expanded.data["sections"]["ui_tree"]
        assert tree2["success"] is True
        assert tree2.get("expanded_applications", 0) >= 1
        match = [a for a in tree2["applications"] if a.get("expanded")]
        assert match, tree2
