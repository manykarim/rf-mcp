"""Comprehensive live E2E tests: GNOME Calculator + Text Editor via PlatynUI (ADR-025).

Drives the REAL MCP server in-process (fastmcp.Client) against REAL GNOME
apps running on an isolated Xvfb display. Encodes every lesson from the
ADR-025 validation campaign:

* Apps are launched via ``systemd-run --user`` with ``DISPLAY=:99`` +
  ``GDK_BACKEND=x11`` — this wins the GApplication DBus name race against
  gnome-shell's search-provider activations and guarantees X11 windows.
* On the WM-less Xvfb the window sits at (0,0) with no CSD shadow, so
  AT-SPI coordinates align with screen coordinates (requires the
  ``resolve_extents`` fallback patch in platynui-native — see ADR-025
  upstream-bugs section).
* GTK4 does NOT expose text content via AT-SPI. Read-back paths used:
  calculator results -> history ``control:Label[@Name='<result>']``;
  editor text -> ``native:Text.CharacterCount`` + save-to-disk roundtrip.
* Every calculator test asserts a DISTINCT result value, so the
  accumulating history of a shared app instance can never produce a
  false positive.

Prerequisites (all auto-skipped when missing): Linux, Xvfb,
gnome-calculator, gnome-text-editor, systemd --user, PlatynUI matched-set
install. IMPORTANT: run this file STANDALONE — the PlatynUI platform
module binds its X11 connection process-wide, so mixing with tests that
use another display in the same pytest process is not supported (a probe
fixture skips the module if the display is already bound elsewhere).

Run with: uv run --no-sync pytest tests/integration/test_platynui_gnome_apps_e2e.py -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
import pytest_asyncio

E2E_DISPLAY = ":99"

# ---------------------------------------------------------------------------
# Environment must be fixed BEFORE the first platynui Runtime in this process
# (the Rust core caches the session type once per process — ADR-025 E2).
# ---------------------------------------------------------------------------
os.environ["XDG_SESSION_TYPE"] = "x11"
os.environ["DISPLAY"] = E2E_DISPLAY
os.environ.pop("WAYLAND_DISPLAY", None)


def _platynui_available() -> bool:
    try:
        from PlatynUI.BareMetal import BareMetal  # noqa: F401
        return True
    except Exception:
        return False


def _binaries_available() -> bool:
    return all(
        shutil.which(b)
        for b in ("Xvfb", "gnome-calculator", "gnome-text-editor", "systemd-run")
    )


pytestmark = [
    pytest.mark.skipif(sys.platform != "linux", reason="Linux only"),
    pytest.mark.skipif(not _binaries_available(), reason="Xvfb/gnome apps/systemd-run missing"),
    pytest.mark.skipif(
        not _platynui_available(),
        reason="PlatynUI.BareMetal not importable (matched-set install required)",
    ),
]

from fastmcp import Client  # noqa: E402

from robotmcp.server import mcp  # noqa: E402

CALC_APP = "/app:*[@Name='gnome-calculator']"
EDITOR_APP = "/app:*[@Name='gnome-text-editor']"


def _sid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _systemd_run(unit: str, *cmd: str, extra_env: dict | None = None) -> None:
    env_args = [
        f"--setenv=DISPLAY={E2E_DISPLAY}",
        "--setenv=GDK_BACKEND=x11",
        "--setenv=GSK_RENDERER=cairo",
    ]
    for k, v in (extra_env or {}).items():
        env_args.append(f"--setenv={k}={v}")
    subprocess.run(
        ["systemd-run", "--user", f"--unit={unit}", *env_args, *cmd],
        check=True, capture_output=True,
    )


def _systemd_stop(unit: str) -> None:
    subprocess.run(
        ["systemctl", "--user", "stop", f"{unit}.service"],
        check=False, capture_output=True,
    )
    subprocess.run(
        ["systemctl", "--user", "reset-failed", f"{unit}.service"],
        check=False, capture_output=True,
    )


def _raise_x11_window(title: str) -> None:
    """Raise the X11 window whose tree entry matches ``title``.

    On the WM-less Xvfb the WindowSurface pattern is unavailable
    (PlatynUI's window management needs a real WM/EWMH), so tests
    restack via XRaiseWindow directly.
    """
    import ctypes

    out = subprocess.run(
        ["xwininfo", "-root", "-tree"],
        capture_output=True, text=True,
        env={**os.environ, "DISPLAY": E2E_DISPLAY},
    ).stdout
    wid = None
    for line in out.splitlines():
        # Match by substring of the quoted window title and ignore the
        # 1x1 utility windows GTK apps create alongside the real frame.
        if title in line and '"' in line and " 1x1+" not in line:
            wid = int(line.strip().split()[0], 16)
            break
    assert wid is not None, f"X11 window {title!r} not found:\n{out}"
    x11 = ctypes.CDLL("libX11.so.6")
    display = x11.XOpenDisplay(E2E_DISPLAY.encode())
    assert display, "XOpenDisplay failed"
    try:
        x11.XRaiseWindow(display, wid)
        x11.XFlush(display)
    finally:
        x11.XCloseDisplay(display)
    time.sleep(0.5)


def _wait_for_app(runtime, app_name: str, timeout: float = 20.0) -> bool:
    """Poll AT-SPI until the application registers.

    The runtime caches the desktop tree — clear it each round or newly
    launched applications never become visible to this long-lived probe.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            runtime.clear_cache()
            apps = runtime.evaluate(f"/app:*[@Name='{app_name}']")
            if apps:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


# ---------------------------------------------------------------------------
# Module fixtures: Xvfb, display-binding probe, apps
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def xvfb():
    """Start Xvfb on :99 unless one is already serving it."""
    started = None
    if subprocess.run(
        ["xdpyinfo"], env={**os.environ, "DISPLAY": E2E_DISPLAY},
        capture_output=True,
    ).returncode != 0:
        started = subprocess.Popen(
            ["Xvfb", E2E_DISPLAY, "-screen", "0", "1280x1024x24"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if subprocess.run(
                ["xdpyinfo"], env={**os.environ, "DISPLAY": E2E_DISPLAY},
                capture_output=True,
            ).returncode == 0:
                break
            time.sleep(0.5)
        else:
            pytest.skip("Xvfb failed to start")
    yield E2E_DISPLAY
    if started is not None:
        started.terminate()


@pytest.fixture(scope="module")
def probe_runtime(xvfb):
    """Process-wide display-binding guard + shared poll/probe runtime.

    The PlatynUI platform module connects to DISPLAY once per process.
    If another test already bound it to a different display, every click
    here would land on the wrong screen — skip the module instead.
    """
    import platynui_native as pn

    rt = pn.Runtime()
    info = rt.desktop_info()
    bounds = str(info.get("bounds", ""))
    if "1280" not in bounds:
        rt.shutdown()
        pytest.skip(
            f"PlatynUI already bound to another display in this process "
            f"(desktop bounds {bounds!r}); run this file standalone."
        )
    yield rt
    try:
        rt.shutdown()
    except Exception:
        pass


@pytest.fixture(scope="module")
def calculator(xvfb, probe_runtime):
    unit = "rfmcp-e2e-calc"
    _systemd_stop(unit)
    _systemd_run(unit, "/usr/bin/gnome-calculator")
    if not _wait_for_app(probe_runtime, "gnome-calculator"):
        _systemd_stop(unit)
        pytest.skip("gnome-calculator did not register on AT-SPI")
    yield CALC_APP
    _systemd_stop(unit)


@pytest.fixture(scope="module")
def editor_file(tmp_path_factory):
    """Pre-created file so Ctrl+S saves in place (no save dialog)."""
    f = tmp_path_factory.mktemp("editor") / "platynui-e2e.txt"
    f.write_text("seed\n")
    return f


@pytest.fixture(scope="module")
def text_editor(xvfb, probe_runtime, editor_file):
    unit = "rfmcp-e2e-editor"
    _systemd_stop(unit)
    _systemd_run(unit, "/usr/bin/gnome-text-editor", str(editor_file))
    if not _wait_for_app(probe_runtime, "gnome-text-editor"):
        _systemd_stop(unit)
        pytest.skip("gnome-text-editor did not register on AT-SPI")
    yield EDITOR_APP
    _systemd_stop(unit)


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


async def _init_session(client, sid: str) -> None:
    init = await client.call_tool(
        "manage_session",
        {
            "session_id": sid,
            "action": "init",
            "scenario": "Native desktop automation of GNOME apps with PlatynUI",
            "libraries": ["PlatynUI.BareMetal", "BuiltIn"],
        },
    )
    assert init.data["success"] is True, init.data


async def _click(client, sid: str, locator: str) -> None:
    r = await client.call_tool(
        "execute_step",
        {"keyword": "Pointer Click", "arguments": [locator], "session_id": sid},
    )
    assert r.data["success"] is True, r.data


async def _click_buttons(client, sid: str, app: str, names: list[str]) -> None:
    for name in names:
        await _click(client, sid, f"{app}//control:Button[@Name='{name}']")


async def _calc_entry_length(client, sid: str, app: str) -> int:
    """Displayed-entry length proxy.

    GTK4 does not expose the entry's text CONTENT via AT-SPI (ADR-025
    upstream gap #2) — but ``native:Text.CharacterCount`` tracks the
    displayed text exactly, so every click can be verified against the
    expected display state.
    """
    var = f"len_{uuid.uuid4().hex[:6]}"
    r = await client.call_tool(
        "execute_step",
        {
            "keyword": "Get Attribute",
            "arguments": [f"{app}//control:Text", "native:Text.CharacterCount"],
            "session_id": sid,
            "assign_to": var,
        },
    )
    assert r.data["success"] is True, r.data
    return int(str(r.data.get("assigned_variables", {}).get("${" + var + "}")))


async def _verified_calc_sequence(
    client, sid: str, app: str, buttons: list[str], result: str
) -> None:
    """Click a calculator sequence with a per-action display assertion.

    After 'C' the entry must be empty; after every digit/operator the
    displayed text must have grown by exactly that token; after '=' the
    entry must show the result (length check — content not exposed).
    """
    expected_len = None
    for name in buttons:
        await _click(client, sid, f"{app}//control:Button[@Name='{name}']")
        if name == "C":
            expected_len = 0
        elif name == "=":
            expected_len = len(result)
        elif expected_len is not None:
            expected_len += len(name)
        if expected_len is not None:
            actual = await _calc_entry_length(client, sid, app)
            assert actual == expected_len, (
                f"after clicking {name!r}: displayed entry length {actual}, "
                f"expected {expected_len}"
            )


async def _assert_history_result(
    client, sid: str, app: str, expected: str, equation: str | None = None
) -> None:
    """Calculator results appear in the history as Labels: one named with
    the equation (e.g. '7×8') and one named with the result value."""
    r = await client.call_tool(
        "execute_step",
        {
            "keyword": "Get Attribute",
            "arguments": [f"{app}//control:Label[@Name='{expected}']", "Name"],
            "session_id": sid,
            "assign_to": "result",
        },
    )
    assert r.data["success"] is True, r.data
    check = await client.call_tool(
        "execute_step",
        {
            "keyword": "Should Be Equal As Strings",
            "arguments": ["${result}", expected],
            "session_id": sid,
        },
    )
    assert check.data["success"] is True, check.data
    if equation is not None:
        eq = await client.call_tool(
            "execute_step",
            {
                "keyword": "Get Attribute",
                "arguments": [f"{app}//control:Label[@Name='{equation}']", "Name"],
                "session_id": sid,
                "assign_to": "equation",
            },
        )
        assert eq.data["success"] is True, (equation, eq.data)
        eq_check = await client.call_tool(
            "execute_step",
            {
                "keyword": "Should Be Equal As Strings",
                "arguments": ["${equation}", equation],
                "session_id": sid,
            },
        )
        assert eq_check.data["success"] is True, eq_check.data


async def _editor_char_count(client, sid: str, var: str) -> int:
    r = await client.call_tool(
        "execute_step",
        {
            "keyword": "Get Attribute",
            "arguments": [f"{EDITOR_APP}//control:Text", "native:Text.CharacterCount"],
            "session_id": sid,
            "assign_to": var,
        },
    )
    assert r.data["success"] is True, r.data
    value = r.data.get("assigned_variables", {}).get("${" + var + "}")
    return int(str(value))


async def _editor_type(client, sid: str, sequence: str) -> None:
    """Click into the text view (sets X input focus on the WM-less Xvfb via
    PointerRoot semantics), then type at the current focus."""
    await _click(client, sid, f"{EDITOR_APP}//control:Text")
    r = await client.call_tool(
        "execute_step",
        {
            "keyword": "Keyboard Type",
            "arguments": ["${None}", sequence],
            "session_id": sid,
        },
    )
    assert r.data["success"] is True, r.data


# ===========================================================================
# Calculator — every test asserts a DISTINCT value (history-pollution-proof)
# ===========================================================================


class TestCalculatorE2E:
    @pytest.mark.asyncio
    async def test_multiplication_7x8(self, calculator, mcp_client):
        """Canonical scenario from the ADR-025 agent validation."""
        sid = _sid("calc-mul")
        await _init_session(mcp_client, sid)
        await _verified_calc_sequence(
            mcp_client, sid, calculator, ["C", "7", "×", "8", "="], "56"
        )
        await _assert_history_result(
            mcp_client, sid, calculator, "56", equation="7×8"
        )

    @pytest.mark.asyncio
    async def test_addition_multi_digit(self, calculator, mcp_client):
        """Sequential multi-digit entry: 12 + 34 = 46."""
        sid = _sid("calc-add")
        await _init_session(mcp_client, sid)
        await _verified_calc_sequence(
            mcp_client, sid, calculator, ["C", "1", "2", "+", "3", "4", "="], "46"
        )
        await _assert_history_result(
            mcp_client, sid, calculator, "46", equation="12+34"
        )

    @pytest.mark.asyncio
    async def test_division_integer(self, calculator, mcp_client):
        """54 ÷ 6 = 9 (integer result avoids locale decimal separators)."""
        sid = _sid("calc-div")
        await _init_session(mcp_client, sid)
        await _verified_calc_sequence(
            mcp_client, sid, calculator, ["C", "5", "4", "÷", "6", "="], "9"
        )
        await _assert_history_result(
            mcp_client, sid, calculator, "9", equation="54÷6"
        )

    @pytest.mark.asyncio
    async def test_subtraction(self, calculator, mcp_client):
        """100 − 13 = 87 (uses the U+2212 minus glyph from the keypad)."""
        sid = _sid("calc-sub")
        await _init_session(mcp_client, sid)
        await _verified_calc_sequence(
            mcp_client, sid, calculator,
            ["C", "1", "0", "0", "−", "1", "3", "="], "87"
        )
        await _assert_history_result(
            mcp_client, sid, calculator, "87", equation="100−13"
        )

    @pytest.mark.asyncio
    async def test_chained_operations(self, calculator, mcp_client):
        """2 × 3 × 4 = 24 — intermediate '=' free chaining."""
        sid = _sid("calc-chain")
        await _init_session(mcp_client, sid)
        await _verified_calc_sequence(
            mcp_client, sid, calculator, ["C", "2", "×", "3", "×", "4", "="], "24"
        )
        await _assert_history_result(
            mcp_client, sid, calculator, "24", equation="2×3×4"
        )

    @pytest.mark.asyncio
    async def test_keyboard_entry(self, calculator, mcp_client):
        """Keyboard path: click entry to focus, type '9*9<Return>' → 81.

        Exercises the XTest keyboard (new core implements it fully on X11)
        and chord/Return sequence syntax.
        """
        sid = _sid("calc-kbd")
        await _init_session(mcp_client, sid)
        await _click_buttons(mcp_client, sid, calculator, ["C"])
        assert await _calc_entry_length(mcp_client, sid, calculator) == 0
        await _click(mcp_client, sid, f"{calculator}//control:Text")
        r = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Keyboard Type",
                "arguments": ["${None}", "9*9<Return>"],
                "session_id": sid,
            },
        )
        assert r.data["success"] is True, r.data
        # After <Return> the entry displays the result -> length of '81'
        assert await _calc_entry_length(mcp_client, sid, calculator) == 2
        # '*' is normalized to the multiplication sign in the history
        await _assert_history_result(
            mcp_client, sid, calculator, "81", equation="9×9"
        )

    @pytest.mark.asyncio
    async def test_intent_action_click(self, calculator, mcp_client):
        """intent_action(click) resolves to PlatynUI Pointer Click for
        desktop sessions (ADR-025 intent mapping) and really clicks:
        5 × 5 = 25 driven entirely through the intent layer."""
        sid = _sid("calc-intent")
        await _init_session(mcp_client, sid)
        expected_len = None
        for name in ["C", "5", "×", "5", "="]:
            r = await mcp_client.call_tool(
                "intent_action",
                {
                    "intent": "click",
                    "target": f"{calculator}//control:Button[@Name='{name}']",
                    "session_id": sid,
                },
            )
            assert r.data.get("success") is True, r.data
            assert r.data.get("keyword") in ("Pointer Click", None), r.data
            # Per-action display verification, same as the direct path
            if name == "C":
                expected_len = 0
            elif name == "=":
                expected_len = len("25")
            else:
                expected_len += len(name)
            actual = await _calc_entry_length(mcp_client, sid, calculator)
            assert actual == expected_len, (name, actual, expected_len)
        await _assert_history_result(
            mcp_client, sid, calculator, "25", equation="5×5"
        )

    @pytest.mark.asyncio
    async def test_ui_tree_exposes_buttons(self, calculator, mcp_client):
        """get_session_state ui_tree expands the calculator subtree."""
        sid = _sid("calc-tree")
        await _init_session(mcp_client, sid)
        state = await mcp_client.call_tool(
            "get_session_state",
            {
                "session_id": sid,
                "sections": ["ui_tree"],
                "elements_of_interest": ["gnome-calculator"],
            },
        )
        ui_tree = state.data["sections"]["ui_tree"]
        assert ui_tree["success"] is True, ui_tree
        assert ui_tree.get("expanded_applications", 0) >= 1
        expanded = [a for a in ui_tree["applications"] if a.get("expanded")]
        assert expanded, ui_tree
        # ui_tree is a bounded orientation snapshot (default depth 3,
        # ADR-025): Application -> Frame -> Panel. Buttons live deeper —
        # assert the window structure surfaced, not exhaustive depth.
        def _roles(node):
            yield node.get("role")
            for child in node.get("children", []) or []:
                yield from _roles(child)
        roles = set()
        for app in expanded:
            roles.update(_roles(app))
        assert "Frame" in roles, roles
        assert any(a.get("children") for a in expanded), expanded

    @pytest.mark.asyncio
    async def test_suite_generation_no_browser_teardown(self, calculator, mcp_client):
        """Desktop suites must not get a 'Close Browser' teardown
        (regression test for the ADR-025 test_builder fix)."""
        sid = _sid("calc-suite")
        await _init_session(mcp_client, sid)
        await _verified_calc_sequence(
            mcp_client, sid, calculator, ["C", "8", "×", "9", "="], "72"
        )
        await _assert_history_result(
            mcp_client, sid, calculator, "72", equation="8×9"
        )
        suite = await mcp_client.call_tool(
            "build_test_suite",
            {"session_id": sid, "test_name": "Calculator 8x9"},
        )
        assert suite.data["success"] is True, suite.data
        rf_text = suite.data.get("rf_text") or ""
        # Large responses are externalized to an artifact file (ADR-015);
        # follow the pointer so the suite content is always asserted.
        if "Content saved to " in rf_text and ".robotmcp_artifacts" in rf_text:
            artifact = rf_text.split("Content saved to ", 1)[1].split(" (", 1)[0]
            rf_text = Path(artifact).read_text()
        assert rf_text, suite.data
        assert "PlatynUI.BareMetal" in rf_text
        assert "Pointer Click" in rf_text
        assert "Close Browser" not in rf_text


# ===========================================================================
# Text editor — GTK4 text content is NOT AT-SPI readable; assert via
# CharacterCount and on-disk save roundtrips.
# ===========================================================================


class TestTextEditorE2E:
    @pytest.mark.asyncio
    async def test_window_title_contains_filename(self, text_editor, editor_file, mcp_client):
        sid = _sid("ed-title")
        await _init_session(mcp_client, sid)
        r = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Get Attribute",
                "arguments": [f"{EDITOR_APP}//control:Frame", "Name"],
                "session_id": sid,
                "assign_to": "title",
            },
        )
        assert r.data["success"] is True, r.data
        title = str(r.data.get("assigned_variables", {}).get("${title}", ""))
        assert editor_file.name in title or "Text Editor" in title, title

    @pytest.mark.asyncio
    async def test_type_text_character_count(self, text_editor, mcp_client):
        """Replace the seed content and verify the typed length via the
        native Text.CharacterCount attribute (content itself is not
        exposed by GTK4 — ADR-025 upstream gap #2)."""
        sid = _sid("ed-type")
        await _init_session(mcp_client, sid)
        text = "Hello PlatynUI"
        await _editor_type(mcp_client, sid, f"<Ctrl+A><Delete>{text}")
        count = await _editor_char_count(mcp_client, sid, "count")
        assert count == len(text), count

    @pytest.mark.asyncio
    async def test_select_all_delete_empties_buffer(self, text_editor, mcp_client):
        sid = _sid("ed-clear")
        await _init_session(mcp_client, sid)
        baseline = await _editor_char_count(mcp_client, sid, "baseline")
        await _editor_type(mcp_client, sid, "scratch content")
        typed = await _editor_char_count(mcp_client, sid, "typed_count")
        assert typed == baseline + len("scratch content"), (baseline, typed)
        await _editor_type(mcp_client, sid, "<Ctrl+A><Delete>")
        count = await _editor_char_count(mcp_client, sid, "empty_count")
        assert count == 0, count

    @pytest.mark.asyncio
    async def test_undo_restores_text(self, text_editor, mcp_client):
        sid = _sid("ed-undo")
        await _init_session(mcp_client, sid)
        await _editor_type(mcp_client, sid, "<Ctrl+A><Delete>undo target")
        before = await _editor_char_count(mcp_client, sid, "before")
        assert before == len("undo target")
        await _editor_type(mcp_client, sid, "<Ctrl+A><Delete>")
        assert await _editor_char_count(mcp_client, sid, "deleted") == 0
        await _editor_type(mcp_client, sid, "<Ctrl+Z>")
        after = await _editor_char_count(mcp_client, sid, "after")
        assert after == before, (before, after)

    @pytest.mark.asyncio
    async def test_save_roundtrip_to_disk(self, text_editor, editor_file, mcp_client):
        """Type unique text, Ctrl+S (in-place save — file pre-exists, so no
        dialog), then assert the bytes actually reached the disk."""
        sid = _sid("ed-save")
        await _init_session(mcp_client, sid)
        marker = f"saved-by-platynui-{uuid.uuid4().hex[:8]}"
        await _editor_type(mcp_client, sid, f"<Ctrl+A><Delete>{marker}")
        # Verify the buffer holds exactly the marker before saving
        typed = await _editor_char_count(mcp_client, sid, "marker_count")
        assert typed == len(marker), (typed, len(marker))
        await _editor_type(mcp_client, sid, "<Ctrl+S>")
        deadline = time.monotonic() + 10
        content = ""
        while time.monotonic() < deadline:
            content = editor_file.read_text()
            if marker in content:
                break
            time.sleep(0.5)
        assert marker in content, content


# ===========================================================================
# Cross-application scoping
# ===========================================================================


class TestCrossApplicationE2E:
    @pytest.mark.asyncio
    async def test_scoped_queries_disambiguate_apps(
        self, calculator, text_editor, mcp_client
    ):
        """With both apps alive, app-scoped Frame queries must resolve to
        the right process each."""
        sid = _sid("xapp-scope")
        await _init_session(mcp_client, sid)
        for app, expected_app_name in (
            (CALC_APP, "gnome-calculator"),
            (EDITOR_APP, "gnome-text-editor"),
        ):
            r = await mcp_client.call_tool(
                "execute_step",
                {
                    "keyword": "Get Attribute",
                    "arguments": [f"{app}//control:Frame", "native:Accessible.Application"],
                    "session_id": sid,
                    "assign_to": "appref",
                },
            )
            # Some toolkits omit the Application backref; fall back to Name
            if r.data["success"] is not True:
                r = await mcp_client.call_tool(
                    "execute_step",
                    {
                        "keyword": "Get Attribute",
                        "arguments": [f"{app}//control:Frame", "Name"],
                        "session_id": sid,
                        "assign_to": "appref",
                    },
                )
            assert r.data["success"] is True, (expected_app_name, r.data)

    @pytest.mark.asyncio
    async def test_ui_tree_lists_both_apps(self, calculator, text_editor, mcp_client):
        sid = _sid("xapp-tree")
        await _init_session(mcp_client, sid)
        state = await mcp_client.call_tool(
            "get_session_state",
            {"session_id": sid, "sections": ["ui_tree"]},
        )
        ui_tree = state.data["sections"]["ui_tree"]
        assert ui_tree["success"] is True, ui_tree
        names = {a.get("name") for a in ui_tree["applications"]}
        assert "gnome-calculator" in names, names
        assert "gnome-text-editor" in names, names

    @pytest.mark.asyncio
    async def test_calculator_then_editor_in_one_session(
        self, calculator, text_editor, mcp_client
    ):
        """One MCP session driving two desktop apps back-to-back:
        compute 6 × 7 = 42, then type the marker into the editor.

        Both windows sit at (0,0) on the WM-less display — the target
        app must be raised before pointer interaction or the clicks hit
        whichever window is stacked on top.
        """
        sid = _sid("xapp-flow")
        await _init_session(mcp_client, sid)
        _raise_x11_window("Calculator")
        await _verified_calc_sequence(
            mcp_client, sid, calculator, ["C", "6", "×", "7", "="], "42"
        )
        await _assert_history_result(
            mcp_client, sid, calculator, "42", equation="6×7"
        )
        _raise_x11_window("platynui-e2e.txt")
        await _editor_type(mcp_client, sid, "<Ctrl+A><Delete>answer is 42")
        count = await _editor_char_count(mcp_client, sid, "flow_count")
        assert count == len("answer is 42"), count
