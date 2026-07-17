"""Unit tests for the PlatynUI focus core (change: platynui-focused-execution).

Covers ``src/robotmcp/components/execution/platynui_focus.py``:
pure helper functions, FocusOutcome serialization, and PlatynUIFocusManager
orchestration driven by a fake (plain-Python) platynui runtime. No real
``platynui_native`` runtime or display is required.
"""

from __future__ import annotations

import pytest

from robotmcp.components.execution.platynui_focus import (
    FocusError,
    FocusOutcome,
    PlatynUIFocusManager,
    app_scope_of,
    extract_descriptor,
    focus_disabled_by_env,
    is_interaction_keyword,
    is_unscoped,
    normalize_keyword,
)


# --------------------------------------------------------------------------
# Fake plain-Python platynui objects
# --------------------------------------------------------------------------


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


class _FakeWindowSurface:
    """Fake WindowSurface pattern (the focus mechanism, tier 1)."""

    def __init__(self, calls):
        self._calls = calls

    def activate(self):
        self._calls.append("activate")


class _FakeWindowSurfaceType:
    """Stand-in for ``platynui_native.WindowSurface`` (used as a key)."""


class FakeWindow:
    """A fake top-level window node exposing .attribute() and the
    WindowSurface pattern via get_pattern() (the real focus contract)."""

    def __init__(self, attrs=None, runtime_id="win-1"):
        self._attrs = attrs or {}
        self.runtime_id = runtime_id
        self.calls = []  # records pattern-method invocations

    def attribute(self, name):
        if name not in self._attrs:
            raise KeyError(name)
        return self._attrs[name]

    def top_level_or_self(self):
        return self

    def get_pattern(self, pattern_type):
        # Real nodes return the WindowSurface pattern object; activate() lives
        # on it (PlatynUIFocusManager._window_surface -> .activate()).
        return _FakeWindowSurface(self.calls)


class FakeNode:
    """A fake element node whose top-level resolves to a given window."""

    def __init__(self, window, runtime_id="node-1"):
        self._window = window
        self.runtime_id = runtime_id

    def top_level_or_self(self):
        return self._window


class FakeRuntime:
    def __init__(self, node=None, desktop=None):
        self._node = node
        self._desktop = desktop

    def evaluate_single(self, descriptor):
        return self._node

    def desktop_info(self):
        return self._desktop


# --------------------------------------------------------------------------
# normalize_keyword
# --------------------------------------------------------------------------


def test_normalize_keyword_strips_library_prefix():
    assert normalize_keyword("PlatynUI.BareMetal.Pointer Click") == "pointer click"


def test_normalize_keyword_collapses_separators():
    assert normalize_keyword("Pointer_Multi-Click") == "pointer multi click"


def test_normalize_keyword_handles_empty():
    assert normalize_keyword("") == ""
    assert normalize_keyword(None) == ""


def test_normalize_keyword_lowercases_and_trims():
    assert normalize_keyword("  Keyboard Type  ") == "keyboard type"


# --------------------------------------------------------------------------
# is_interaction_keyword
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "keyword",
    [
        "Pointer Click",
        "PlatynUI.BareMetal.Pointer Click",
        "pointer_press",
        "Keyboard Type",
        "keyboard-press",
    ],
)
def test_is_interaction_keyword_true(keyword):
    assert is_interaction_keyword(keyword) is True


@pytest.mark.parametrize(
    "keyword",
    [
        "Activate Window",
        "Maximize Window",
        "Close Window",
        "Log",
        "Take Screenshot",
    ],
)
def test_is_interaction_keyword_false(keyword):
    assert is_interaction_keyword(keyword) is False


# --------------------------------------------------------------------------
# extract_descriptor
# --------------------------------------------------------------------------


def test_extract_descriptor_pointer_first_positional():
    desc = "/app:*[@Name='X']//control:Button"
    assert extract_descriptor("Pointer Click", [desc]) == desc


def test_extract_descriptor_named_arg_returns_none():
    assert extract_descriptor("Pointer Click", ["only_first=True"]) is None


def test_extract_descriptor_predicate_equals_is_not_named_arg():
    desc = "//control:Button[@Name='7']"
    assert extract_descriptor("Pointer Click", [desc]) == desc


def test_extract_descriptor_empty_args_none():
    assert extract_descriptor("Pointer Click", []) is None


def test_extract_descriptor_blank_first_arg_none():
    assert extract_descriptor("Pointer Click", ["   "]) is None


def test_extract_descriptor_non_string_first_arg_none():
    assert extract_descriptor("Pointer Click", [123]) is None


@pytest.mark.parametrize("sentinel", ["${None}", "none", "", "  NONE  "])
def test_extract_descriptor_keyboard_sentinel_none(sentinel):
    assert extract_descriptor("Keyboard Type", [sentinel, "hello"]) is None


def test_extract_descriptor_keyboard_real_descriptor():
    desc = "/app:*[@Name='X']//control:Edit"
    assert extract_descriptor("Keyboard Type", [desc, "hello"]) == desc


def test_extract_descriptor_non_descriptor_keyword_none():
    # Not a pointer/keyboard/window-descriptor keyword
    assert extract_descriptor("Log", ["message"]) is None


# --------------------------------------------------------------------------
# app_scope_of / is_unscoped
# --------------------------------------------------------------------------


def test_app_scope_of_extracts_prefix():
    desc = "/app:*[@Name='X']//control:Button"
    assert app_scope_of(desc) == "/app:*[@Name='X']"


def test_app_scope_of_whole_when_no_subpath():
    assert app_scope_of("/app:*[@Name='X']") == "/app:*[@Name='X']"


def test_app_scope_of_non_app_returns_none():
    assert app_scope_of("//control:Button") is None
    assert app_scope_of("") is None


def test_is_unscoped_double_slash_true():
    assert is_unscoped("//control:Button") is True


def test_is_unscoped_app_scoped_false():
    assert is_unscoped("/app:*[@Name='X']//control:Button") is False


# --------------------------------------------------------------------------
# focus_disabled_by_env
# --------------------------------------------------------------------------


def test_focus_disabled_by_env_true():
    assert focus_disabled_by_env({"ROBOTMCP_PLATYNUI_NO_FOCUS": "1"}) is True
    assert focus_disabled_by_env({"ROBOTMCP_PLATYNUI_NO_FOCUS": "true"}) is True
    assert focus_disabled_by_env({"ROBOTMCP_PLATYNUI_NO_FOCUS": "YES"}) is True


def test_focus_disabled_by_env_false():
    assert focus_disabled_by_env({}) is False
    assert focus_disabled_by_env({"ROBOTMCP_PLATYNUI_NO_FOCUS": "0"}) is False
    assert focus_disabled_by_env({"ROBOTMCP_PLATYNUI_NO_FOCUS": ""}) is False


# --------------------------------------------------------------------------
# FocusOutcome.to_dict
# --------------------------------------------------------------------------


def test_focus_outcome_default_shape():
    out = FocusOutcome().to_dict()
    assert out == {"attempted": False, "focused": False}


def test_focus_outcome_optional_keys_only_when_set():
    oc = FocusOutcome()
    oc.attempted = True
    oc.focused = True
    oc.bypassed = True
    oc.strategy = "window_surface:activate"
    oc.visible = True
    oc.in_scope = False
    oc.warnings = ["w1"]
    oc.error = "boom"
    out = oc.to_dict()
    assert out == {
        "attempted": True,
        "focused": True,
        "bypassed": True,
        "strategy": "window_surface:activate",
        "visible": True,
        "in_scope": False,
        "warnings": ["w1"],
        "error": "boom",
    }


def test_focus_outcome_visible_false_serialized():
    oc = FocusOutcome()
    oc.visible = False
    out = oc.to_dict()
    assert out["visible"] is False


# --------------------------------------------------------------------------
# ensure_focused short-circuits (no runtime needed)
# --------------------------------------------------------------------------


def test_ensure_focused_bypass_when_focus_false():
    mgr = PlatynUIFocusManager()
    oc = mgr.ensure_focused("Pointer Click", ["/app:*//control:Button"], focus=False)
    assert oc.bypassed is True
    assert oc.attempted is False


def test_ensure_focused_non_interaction_unattempted():
    mgr = PlatynUIFocusManager()
    oc = mgr.ensure_focused("Activate Window", ["/app:*//control:Frame"])
    assert oc.attempted is False
    assert oc.bypassed is False


# --------------------------------------------------------------------------
# window_visibility (fake window)
# --------------------------------------------------------------------------


def test_window_visibility_visible():
    mgr = PlatynUIFocusManager()
    mgr._desktop_bounds = lambda: (0.0, 0.0, 1920.0, 1080.0)
    win = FakeWindow(
        {"IsVisible": True, "IsInView": True, "Bounds": FakeRect(10, 10, 200, 100)}
    )
    visible, warnings = mgr.window_visibility(win)
    assert visible is True
    assert warnings == []


def test_window_visibility_zero_size_not_visible():
    mgr = PlatynUIFocusManager()
    mgr._desktop_bounds = lambda: (0.0, 0.0, 1920.0, 1080.0)
    win = FakeWindow(
        {"IsVisible": True, "IsInView": True, "Bounds": FakeRect(10, 10, 0, 0)}
    )
    visible, warnings = mgr.window_visibility(win)
    assert visible is False
    assert any("zero size" in w for w in warnings)


def test_window_visibility_isvisible_false():
    mgr = PlatynUIFocusManager()
    mgr._desktop_bounds = lambda: (0.0, 0.0, 1920.0, 1080.0)
    win = FakeWindow(
        {"IsVisible": False, "IsInView": True, "Bounds": FakeRect(10, 10, 200, 100)}
    )
    visible, warnings = mgr.window_visibility(win)
    assert visible is False
    assert any("IsVisible=false" in w for w in warnings)


def test_window_visibility_off_screen():
    mgr = PlatynUIFocusManager()
    mgr._desktop_bounds = lambda: (0.0, 0.0, 1920.0, 1080.0)
    # Window far off the right/bottom of the desktop.
    win = FakeWindow(
        {"IsVisible": True, "IsInView": True, "Bounds": FakeRect(5000, 5000, 100, 100)}
    )
    visible, warnings = mgr.window_visibility(win)
    assert visible is False
    assert any("off-screen" in w for w in warnings)


def test_window_visibility_all_none_returns_none():
    mgr = PlatynUIFocusManager()
    win = FakeWindow({})  # attribute() raises -> all None
    visible, warnings = mgr.window_visibility(win)
    assert visible is None


def test_window_visibility_none_window():
    mgr = PlatynUIFocusManager()
    visible, warnings = mgr.window_visibility(None)
    assert visible is None
    assert warnings == []


# --------------------------------------------------------------------------
# resolve_window / focus_window / target_in_window with fake runtime
# --------------------------------------------------------------------------


def test_resolve_window_returns_top_level():
    win = FakeWindow({}, runtime_id="win-A")
    node = FakeNode(win, runtime_id="node-A")
    mgr = PlatynUIFocusManager()
    mgr._runtime = FakeRuntime(node=node)
    resolved = mgr.resolve_window("/app:*//control:Button")
    assert resolved is win


def test_focus_window_activate_first(monkeypatch):
    win = FakeWindow({})  # no IsActive -> not short-circuited
    mgr = PlatynUIFocusManager()
    mgr._runtime = FakeRuntime()
    # The real _window_surface imports platynui_native.WindowSurface as the
    # get_pattern key; stub it to the fake's pattern object so the test is
    # environment-independent (no native package / display required).
    monkeypatch.setattr(
        PlatynUIFocusManager, "_window_surface",
        staticmethod(lambda window: _FakeWindowSurface(window.calls)),
    )
    focused, strategy, input_ready = mgr.focus_window(win, "/app:*[@Name='X']")
    assert focused is True
    assert strategy == "window_surface:activate"
    assert win.calls[0] == "activate"
    assert input_ready is None  # fake surface has no accepts_user_input


def test_focus_window_already_active_short_circuits():
    win = FakeWindow({"IsActive": True})
    mgr = PlatynUIFocusManager()
    mgr._runtime = FakeRuntime()
    focused, strategy, input_ready = mgr.focus_window(win, "/app:*")
    assert focused is True
    assert strategy == "already_active"
    assert win.calls == []  # no activate called


def test_target_in_window_same_runtime_id_true():
    win = FakeWindow({}, runtime_id="win-same")
    target_node = FakeNode(win)  # top-level resolves to win
    mgr = PlatynUIFocusManager()
    mgr._runtime = FakeRuntime(node=target_node)
    assert mgr.target_in_window("/app:*//control:Button", win) is True


def test_target_in_window_different_runtime_id_false():
    win = FakeWindow({}, runtime_id="win-A")
    other_win = FakeWindow({}, runtime_id="win-B")
    target_node = FakeNode(other_win)
    mgr = PlatynUIFocusManager()
    mgr._runtime = FakeRuntime(node=target_node)
    assert mgr.target_in_window("/app:*//control:Button", win) is False


# --------------------------------------------------------------------------
# ensure_focused full path with fake runtime
# --------------------------------------------------------------------------


def _mgr_with_window(window, target_node=None, desktop=None):
    """Build a manager whose runtime resolves descriptors to ``window``.

    ``target_node`` is what target_in_window's evaluate_single returns; by
    default the same node so scope check passes.
    """
    node = FakeNode(window)
    mgr = PlatynUIFocusManager()

    class _RT:
        def evaluate_single(self_inner, descriptor):
            return node if target_node is None else target_node

        def desktop_info(self_inner):
            return desktop

    mgr._runtime = _RT()
    return mgr


def test_ensure_focused_full_success(monkeypatch):
    win = FakeWindow(
        {"IsVisible": True, "IsInView": True, "Bounds": FakeRect(10, 10, 200, 100)},
        runtime_id="win-X",
    )
    node = FakeNode(win)
    mgr = PlatynUIFocusManager()
    mgr._runtime = FakeRuntime(node=node)
    mgr._desktop_bounds = lambda: (0.0, 0.0, 1920.0, 1080.0)
    monkeypatch.setattr(
        PlatynUIFocusManager, "_window_surface",
        staticmethod(lambda window: _FakeWindowSurface(window.calls)),
    )
    oc = mgr.ensure_focused(
        "Pointer Click", ["/app:*[@Name='X']//control:Button"], focus=True
    )
    assert oc.attempted is True
    assert oc.focused is True
    assert oc.visible is True
    assert oc.in_scope is True


def test_ensure_focused_hidden_fail_fast_raises():
    win = FakeWindow(
        {"IsVisible": False, "IsInView": True, "Bounds": FakeRect(10, 10, 200, 100)},
        runtime_id="win-X",
    )
    node = FakeNode(win)
    mgr = PlatynUIFocusManager()
    mgr._runtime = FakeRuntime(node=node)
    mgr._desktop_bounds = lambda: (0.0, 0.0, 1920.0, 1080.0)
    with pytest.raises(FocusError):
        mgr.ensure_focused(
            "Pointer Click",
            ["/app:*[@Name='X']//control:Button"],
            focus=True,
            fail_on_hidden=True,
        )


def test_ensure_focused_cross_window_strict_raises():
    win = FakeWindow(
        {"IsVisible": True, "IsInView": True, "Bounds": FakeRect(10, 10, 200, 100)},
        runtime_id="win-A",
    )
    other = FakeWindow({}, runtime_id="win-B")
    # resolve_window returns win; target_in_window resolves to other window.
    resolve_node = FakeNode(win)
    target_node = FakeNode(other)

    mgr = PlatynUIFocusManager()
    mgr._desktop_bounds = lambda: (0.0, 0.0, 1920.0, 1080.0)

    class _RT:
        def __init__(self):
            self._n = 0

        def evaluate_single(self_inner, descriptor):
            # First call (resolve_window) -> resolve_node;
            # second call (target_in_window) -> target_node.
            self_inner._n += 1
            return resolve_node if self_inner._n == 1 else target_node

        def desktop_info(self_inner):
            return None

    mgr._runtime = _RT()
    with pytest.raises(FocusError):
        mgr.ensure_focused(
            "Pointer Click",
            ["/app:*[@Name='A']//control:Button"],
            focus=True,
            strict_scope=True,
        )


def test_ensure_focused_cross_window_non_strict_warns():
    win = FakeWindow(
        {"IsVisible": True, "IsInView": True, "Bounds": FakeRect(10, 10, 200, 100)},
        runtime_id="win-A",
    )
    other = FakeWindow({}, runtime_id="win-B")
    resolve_node = FakeNode(win)
    target_node = FakeNode(other)

    mgr = PlatynUIFocusManager()
    mgr._desktop_bounds = lambda: (0.0, 0.0, 1920.0, 1080.0)

    class _RT:
        def __init__(self):
            self._n = 0

        def evaluate_single(self_inner, descriptor):
            self_inner._n += 1
            return resolve_node if self_inner._n == 1 else target_node

        def desktop_info(self_inner):
            return None

    mgr._runtime = _RT()
    oc = mgr.ensure_focused(
        "Pointer Click",
        ["/app:*[@Name='A']//control:Button"],
        focus=True,
        strict_scope=False,
    )
    assert oc.in_scope is False
    assert any("cross-window" in w for w in oc.warnings)
    assert oc.error is None
