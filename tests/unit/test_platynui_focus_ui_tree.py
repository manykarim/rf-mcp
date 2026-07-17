"""Unit tests for ui_tree_service visibility helpers
(change: platynui-focused-execution, task 4.1).

Covers ``_window_visibility(app_node, desktop_bounds)`` and
``_desktop_bounds(runtime)``. These reuse PlatynUIFocusManager's
visibility logic, so we drive them with fake plain-Python nodes.
"""

from __future__ import annotations

from robotmcp.components.execution.ui_tree_service import (
    _desktop_bounds,
    _window_visibility,
)


class FakeRect:
    def __init__(self, x, y, w, h):
        self._x, self._y, self._w, self._h = x, y, w, h

    def x(self):
        return self._x

    def y(self):
        return self._y

    def width(self):
        return self._w

    def height(self):
        return self._h


class FakeWindow:
    def __init__(self, attrs):
        self._attrs = attrs

    def attribute(self, name):
        if name not in self._attrs:
            raise KeyError(name)
        return self._attrs[name]


class FakeAppNode:
    """An application node whose .children() yields one window."""

    def __init__(self, window):
        self._window = window

    def children(self):
        return [self._window]


class FakeRuntime:
    def __init__(self, desktop):
        self._desktop = desktop

    def desktop_info(self):
        return self._desktop


# --------------------------------------------------------------------------
# _window_visibility
# --------------------------------------------------------------------------


def test_window_visibility_visible_dict():
    win = FakeWindow(
        {"IsVisible": True, "IsInView": True, "Bounds": FakeRect(0, 0, 800, 600)}
    )
    app = FakeAppNode(win)
    out = _window_visibility(app, (0.0, 0.0, 1920.0, 1080.0))
    assert isinstance(out, dict)
    assert out["visible"] is True
    assert "reasons" not in out


def test_window_visibility_not_visible_has_reasons():
    win = FakeWindow(
        {"IsVisible": False, "IsInView": True, "Bounds": FakeRect(0, 0, 800, 600)}
    )
    app = FakeAppNode(win)
    out = _window_visibility(app, (0.0, 0.0, 1920.0, 1080.0))
    assert out["visible"] is False
    assert isinstance(out["reasons"], list)
    assert any("IsVisible=false" in r for r in out["reasons"])


def test_window_visibility_zero_size_not_visible():
    win = FakeWindow(
        {"IsVisible": True, "IsInView": True, "Bounds": FakeRect(0, 0, 0, 0)}
    )
    app = FakeAppNode(win)
    out = _window_visibility(app, (0.0, 0.0, 1920.0, 1080.0))
    assert out["visible"] is False
    assert any("zero size" in r for r in out["reasons"])


def test_window_visibility_all_none_returns_none():
    win = FakeWindow({})  # attribute() always raises -> all attrs None
    app = FakeAppNode(win)
    out = _window_visibility(app, (0.0, 0.0, 1920.0, 1080.0))
    assert out is None


def test_window_visibility_no_children_falls_back_to_app_node():
    # app_node with no children -> window = app_node itself (which has no attrs)
    class EmptyApp:
        def children(self):
            return []

    out = _window_visibility(EmptyApp(), (0.0, 0.0, 1920.0, 1080.0))
    # app_node used as window has no attribute() -> all None -> None
    assert out is None


# --------------------------------------------------------------------------
# _desktop_bounds
# --------------------------------------------------------------------------


def test_desktop_bounds_dict_form():
    rt = FakeRuntime({"bounds": {"x": 0, "y": 0, "width": 1920, "height": 1080}})
    assert _desktop_bounds(rt) == (0.0, 0.0, 1920.0, 1080.0)


def test_desktop_bounds_rect_form():
    rt = FakeRuntime({"bounds": FakeRect(10, 20, 1280, 720)})
    assert _desktop_bounds(rt) == (10.0, 20.0, 1280.0, 720.0)


def test_desktop_bounds_no_bounds_key_none():
    rt = FakeRuntime({"other": 1})
    assert _desktop_bounds(rt) is None


def test_desktop_bounds_desktop_info_raises_none():
    class BadRuntime:
        def desktop_info(self):
            raise RuntimeError("no display")

    assert _desktop_bounds(BadRuntime()) is None
