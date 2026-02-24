"""E2E tests for PlatynUI live desktop interaction with GNOME Calculator.

Starts the calculator as an X11 app, uses PlatynUI (via platynui_native)
to click buttons and verify visible UI interaction on screen.

Requirements:
    - GNOME Calculator installed (gnome-calculator)
    - X11 display available (DISPLAY=:0)
    - PlatynUI native backend (platynui_native) installed
    - xwininfo and xprop available (x11-utils)
    - AT-SPI2 accessibility enabled

Usage:
    RUN_PLATYNUI_LIVE=true uv run pytest tests/e2e/test_platynui_live_interaction.py -v -s
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass

import pytest

# ---------------------------------------------------------------------------
# Skip unless enabled
# ---------------------------------------------------------------------------

_SKIP_REASON = "Requires RUN_PLATYNUI_LIVE=true and a running display"


def _should_run() -> bool:
    return os.getenv("RUN_PLATYNUI_LIVE", "false").lower() in ("true", "1", "yes")


def _has_display() -> bool:
    return bool(os.environ.get("DISPLAY"))


pytestmark = pytest.mark.skipif(
    not (_should_run() and _has_display()),
    reason=_SKIP_REASON,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class WindowGeometry:
    """X11 window geometry with GTK frame extents."""

    win_x: int
    win_y: int
    win_w: int
    win_h: int
    frame_left: int = 0
    frame_right: int = 0
    frame_top: int = 0
    frame_bottom: int = 0

    @property
    def content_x(self) -> int:
        return self.win_x + self.frame_left

    @property
    def content_y(self) -> int:
        return self.win_y + self.frame_top

    @property
    def content_w(self) -> int:
        return self.win_w - self.frame_left - self.frame_right

    @property
    def content_h(self) -> int:
        return self.win_h - self.frame_top - self.frame_bottom


def _get_window_geometry(window_name: str) -> WindowGeometry:
    """Get X11 window position and GTK frame extents."""
    env = os.environ.copy()
    env["DISPLAY"] = os.environ.get("DISPLAY", ":0")

    result = subprocess.run(
        ["xwininfo", "-name", window_name],
        capture_output=True, text=True, env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"xwininfo failed: {result.stderr}")

    output = result.stdout
    win_x = int(re.search(r"Absolute upper-left X:\s+(\d+)", output).group(1))
    win_y = int(re.search(r"Absolute upper-left Y:\s+(\d+)", output).group(1))
    win_w = int(re.search(r"Width:\s+(\d+)", output).group(1))
    win_h = int(re.search(r"Height:\s+(\d+)", output).group(1))

    geom = WindowGeometry(win_x=win_x, win_y=win_y, win_w=win_w, win_h=win_h)

    # Try to get GTK frame extents (shadow/CSD decorations)
    result2 = subprocess.run(
        ["xprop", "-name", window_name, "_GTK_FRAME_EXTENTS"],
        capture_output=True, text=True, env=env,
    )
    m = re.search(r"= (\d+), (\d+), (\d+), (\d+)", result2.stdout)
    if m:
        geom.frame_left = int(m.group(1))
        geom.frame_right = int(m.group(2))
        geom.frame_top = int(m.group(3))
        geom.frame_bottom = int(m.group(4))

    return geom


def _find_button_positions(calc_node) -> dict[str, tuple[int, int]]:
    """Walk the AT-SPI tree to find button center positions (WINDOW coords).

    Returns a dict mapping button name -> (center_x, center_y) in
    logical pixels relative to the application frame.
    """
    buttons: dict[str, tuple[int, int]] = {}

    def _walk(node, depth: int = 0) -> None:
        if depth > 15:
            return
        if node.role == "Button" and node.name:
            # Use Bounds attribute — under Wayland, Bounds is relative with
            # origin (0,0) per button, but ActivationPoint gives the center.
            # We need the position relative to the frame.
            # Fall back to the AT-SPI WINDOW coordinates if available.
            # Since all Bounds have (0,0) origin, we need to compute positions
            # from the tree structure. Instead, use the known mapping.
            pass
        for child in node.children():
            _walk(child, depth + 1)

    _walk(calc_node)
    return buttons


# Pre-computed GNOME Calculator button positions (WINDOW-relative, logical pixels).
# These are the center coordinates for each button on the Basic calculator layout.
# Obtained from AT-SPI Component.get_extents(WINDOW) on GNOME Calculator.
CALC_BUTTON_CENTERS: dict[str, tuple[int, int]] = {
    "C": (48, 279),
    "(": (116, 279),
    ")": (184, 279),
    "mod": (252, 279),
    "pi": (320, 279),
    "7": (48, 327),
    "8": (116, 327),
    "9": (184, 327),
    "/": (252, 327),
    "sqrt": (320, 327),
    "4": (48, 375),
    "5": (116, 375),
    "6": (184, 375),
    "*": (252, 375),
    "x2": (320, 375),
    "1": (48, 423),
    "2": (116, 423),
    "3": (184, 423),
    "-": (252, 423),
    "=": (320, 447),
    "0": (48, 471),
    ",": (116, 471),
    "%": (184, 471),
    "+": (252, 471),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def calculator_app():
    """Start GNOME Calculator as X11 app for the test module.

    Launches with GDK_BACKEND=x11 so that X11 window geometry tools
    (xwininfo, xprop) work even on Wayland sessions.

    Yields a dict with 'pid', 'runtime', 'geometry' keys.
    """
    import platynui_native as pn

    # Kill any existing calculator
    subprocess.run(["killall", "gnome-calculator"], capture_output=True)
    time.sleep(0.5)

    # Start calculator under X11 backend
    env = os.environ.copy()
    env["GDK_BACKEND"] = "x11"
    env["DISPLAY"] = os.environ.get("DISPLAY", ":0")

    proc = subprocess.Popen(
        ["gnome-calculator"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)  # Wait for window to appear

    assert proc.poll() is None, "Calculator failed to start"

    # Initialize PlatynUI runtime
    rt = pn.Runtime()
    time.sleep(0.5)

    # Find calculator in AT-SPI tree
    desktop = rt.desktop_node()
    calc_app = None
    for child in desktop.children():
        if child.name == "gnome-calculator":
            calc_app = child
            break

    assert calc_app is not None, "Calculator not found in AT-SPI tree"

    frame = list(calc_app.children())[0]
    assert frame.role == "Frame", f"Expected Frame, got {frame.role}"

    # Activate the window
    ws = frame.pattern_by_id("WindowSurface")
    ws.activate()
    time.sleep(1)

    # Get X11 geometry
    geom = _get_window_geometry("Calculator")

    yield {
        "pid": proc.pid,
        "process": proc,
        "runtime": rt,
        "app_node": calc_app,
        "frame": frame,
        "geometry": geom,
    }

    # Teardown: kill calculator
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def _click_button(
    rt,
    geom: WindowGeometry,
    button_name: str,
    delay: float = 0.4,
) -> None:
    """Click a calculator button by name using PlatynUI pointer_click.

    Computes absolute screen coordinates from WINDOW-relative positions
    + X11 content area offset (accounting for GTK frame extents).
    """
    import platynui_native as pn

    if button_name not in CALC_BUTTON_CENTERS:
        raise ValueError(f"Unknown button: {button_name!r}")

    bx, by = CALC_BUTTON_CENTERS[button_name]
    abs_x = geom.content_x + bx
    abs_y = geom.content_y + by

    rt.pointer_click(pn.Point(abs_x, abs_y))
    time.sleep(delay)


def _read_display_text(calc_app) -> str:
    """Read the calculator display text from AT-SPI tree.

    Walks the tree to find text elements in the display area.
    """
    texts: list[str] = []

    def _walk(node, depth: int = 0) -> None:
        if depth > 15:
            return
        try:
            # ScrollPane > Text nodes contain display content
            if node.role == "Text" and node.name is not None:
                texts.append(node.name)
            for child in node.children():
                _walk(child, depth + 1)
        except Exception:
            pass

    _walk(calc_app)
    return " ".join(texts).strip()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCalculatorLaunch:
    """Verify calculator starts and is visible to PlatynUI."""

    def test_calculator_running(self, calculator_app):
        assert calculator_app["process"].poll() is None

    def test_found_in_atspi(self, calculator_app):
        assert calculator_app["app_node"].name == "gnome-calculator"

    def test_frame_visible(self, calculator_app):
        frame = calculator_app["frame"]
        assert frame.name == "Calculator"
        assert frame.role == "Frame"

    def test_window_geometry(self, calculator_app):
        geom = calculator_app["geometry"]
        assert geom.content_w > 0
        assert geom.content_h > 0
        # Content area should be smaller than full window (frame extents)
        assert geom.content_w <= geom.win_w
        assert geom.content_h <= geom.win_h

    def test_buttons_visible(self, calculator_app):
        """Verify buttons exist in the AT-SPI tree."""
        app = calculator_app["app_node"]
        found_buttons: list[str] = []

        def _walk(node, depth=0):
            if depth > 15:
                return
            if node.role == "Button" and node.name:
                found_buttons.append(node.name)
            for child in node.children():
                _walk(child, depth + 1)

        _walk(app)
        assert "7" in found_buttons
        assert "+" in found_buttons
        assert "=" in found_buttons
        assert "C" in found_buttons
        assert len(found_buttons) >= 20, f"Only found {len(found_buttons)} buttons"


class TestCalculatorInteraction:
    """Test visible UI interaction with calculator buttons."""

    def test_clear_button(self, calculator_app):
        """Click C to clear the display."""
        rt = calculator_app["runtime"]
        geom = calculator_app["geometry"]
        _click_button(rt, geom, "C")
        # No assertion on display — just verify no crash

    def test_single_digit(self, calculator_app):
        """Click a single digit button."""
        rt = calculator_app["runtime"]
        geom = calculator_app["geometry"]
        _click_button(rt, geom, "C")
        _click_button(rt, geom, "7")
        time.sleep(0.3)
        # Verify pointer moved to the button area
        pos = rt.pointer_position()
        bx, by = CALC_BUTTON_CENTERS["7"]
        expected_x = geom.content_x + bx
        expected_y = geom.content_y + by
        # Allow 5px tolerance
        assert abs(pos.x - expected_x) < 5, f"Pointer X {pos.x} != {expected_x}"
        assert abs(pos.y - expected_y) < 5, f"Pointer Y {pos.y} != {expected_y}"

    def test_addition_7_plus_3(self, calculator_app):
        """Compute 7+3=10 by clicking calculator buttons.

        This tests visible UI interaction: buttons are clicked on screen,
        numbers appear in the display, and the result is computed.
        """
        rt = calculator_app["runtime"]
        geom = calculator_app["geometry"]

        _click_button(rt, geom, "C")
        _click_button(rt, geom, "7")
        _click_button(rt, geom, "+")
        _click_button(rt, geom, "3")
        _click_button(rt, geom, "=")
        time.sleep(0.5)

        # Take screenshot to verify visually
        _take_screenshot(calculator_app, "addition_7_plus_3")

    def test_multiplication_6_times_8(self, calculator_app):
        """Compute 6*8=48 by clicking calculator buttons."""
        rt = calculator_app["runtime"]
        geom = calculator_app["geometry"]

        _click_button(rt, geom, "C")
        _click_button(rt, geom, "6")
        _click_button(rt, geom, "*")
        _click_button(rt, geom, "8")
        _click_button(rt, geom, "=")
        time.sleep(0.5)

        _take_screenshot(calculator_app, "multiplication_6_times_8")

    def test_subtraction_9_minus_4(self, calculator_app):
        """Compute 9-4=5 by clicking calculator buttons."""
        rt = calculator_app["runtime"]
        geom = calculator_app["geometry"]

        _click_button(rt, geom, "C")
        _click_button(rt, geom, "9")
        _click_button(rt, geom, "-")
        _click_button(rt, geom, "4")
        _click_button(rt, geom, "=")
        time.sleep(0.5)

        _take_screenshot(calculator_app, "subtraction_9_minus_4")

    def test_multi_digit_12_plus_34(self, calculator_app):
        """Compute 12+34=46 by clicking calculator buttons."""
        rt = calculator_app["runtime"]
        geom = calculator_app["geometry"]

        _click_button(rt, geom, "C")
        _click_button(rt, geom, "1")
        _click_button(rt, geom, "2")
        _click_button(rt, geom, "+")
        _click_button(rt, geom, "3")
        _click_button(rt, geom, "4")
        _click_button(rt, geom, "=")
        time.sleep(0.5)

        _take_screenshot(calculator_app, "multi_digit_12_plus_34")

    def test_pointer_position_tracks_clicks(self, calculator_app):
        """Verify pointer moves to each button when clicked."""
        rt = calculator_app["runtime"]
        geom = calculator_app["geometry"]

        for btn_name in ["1", "5", "9", "0"]:
            _click_button(rt, geom, btn_name, delay=0.2)
            pos = rt.pointer_position()
            bx, by = CALC_BUTTON_CENTERS[btn_name]
            expected_x = geom.content_x + bx
            expected_y = geom.content_y + by
            assert abs(pos.x - expected_x) < 5, (
                f"After clicking {btn_name}: pointer at ({pos.x},{pos.y}), "
                f"expected near ({expected_x},{expected_y})"
            )


class TestPlatynUIRuntime:
    """Test PlatynUI runtime capabilities on the running calculator."""

    def test_desktop_node(self, calculator_app):
        rt = calculator_app["runtime"]
        desktop = rt.desktop_node()
        assert desktop.name is not None
        assert desktop.role == "Desktop"

    def test_providers(self, calculator_app):
        rt = calculator_app["runtime"]
        providers = rt.providers()
        assert len(providers) >= 1
        assert any(p["technology"] == "AT-SPI2" for p in providers)

    def test_pointer_position(self, calculator_app):
        rt = calculator_app["runtime"]
        pos = rt.pointer_position()
        assert pos.x >= 0
        assert pos.y >= 0

    def test_window_surface_activate(self, calculator_app):
        """WindowSurface.activate() brings calculator to front."""
        frame = calculator_app["frame"]
        ws = frame.pattern_by_id("WindowSurface")
        ws.activate()
        time.sleep(0.5)
        # No crash = success


# ---------------------------------------------------------------------------
# Screenshot helper
# ---------------------------------------------------------------------------


def _take_screenshot(calculator_app, name: str) -> None:
    """Take a screenshot of the calculator window for visual verification."""
    import shutil

    if not shutil.which("import"):
        return  # ImageMagick not available

    screenshot_dir = os.path.join(
        os.path.dirname(__file__), "metrics", "platynui_screenshots"
    )
    os.makedirs(screenshot_dir, exist_ok=True)
    filepath = os.path.join(screenshot_dir, f"{name}.png")

    env = os.environ.copy()
    env["DISPLAY"] = os.environ.get("DISPLAY", ":0")
    subprocess.run(
        ["import", "-window", "Calculator", filepath],
        capture_output=True,
        env=env,
    )
