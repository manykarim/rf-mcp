"""Blind type-at-focus warning
(change: desktop-evidence-and-display-scoping, D7).

On the 2026-06-11 rerun, 20+ blind ``Keyboard Type`` calls (no descriptor)
were robot-level "successes" into an empty display: type-at-focus has no
descriptor, so the focus gate returned early with no warning. Now a
keyboard step with no descriptor warns when no AUT window focus was ever
verified — once per session, never failing the step.
"""

from __future__ import annotations

import types

import pytest

import robotmcp.components.execution.platynui_focus as focus_mod
from robotmcp.components.execution.platynui_focus import (
    UNFOCUSED_TYPING_WARNING,
    PlatynUIFocusManager,
)


class StubSurface:
    def activate(self):
        pass

    def accepts_user_input(self):
        return True


class StubWindow:
    runtime_id = "win-1"

    def supported_patterns(self):
        return ["org.platynui.patterns.WindowSurface"]

    def attribute(self, name):
        raise KeyError(name)

    def top_level_or_self(self):
        return self


class StubRuntime:
    def __init__(self, node=None):
        self._node = node

    def evaluate_single(self, descriptor):
        return self._node

    def desktop_info(self):
        return {}

    def clear_cache(self):
        pass

    def bring_to_front(self, node, **kwargs):
        pass


def _manager(runtime=None):
    mgr = PlatynUIFocusManager()
    mgr._runtime = runtime if runtime is not None else StubRuntime()
    mgr._desktop_bounds = lambda: None
    mgr._window_surface = staticmethod(lambda w: StubSurface())
    mgr._x11_raise = lambda w: False
    return mgr


class TestUnfocusedTypingWarning:
    def test_blind_typing_without_verified_focus_warns(self):
        mgr = _manager()
        oc = mgr.ensure_focused("Keyboard Type", ["hello"])
        # "hello" is not a descriptor (keyboard first-arg heuristic treats
        # non-sentinel strings as descriptors)... use the None sentinel.
        oc = mgr.ensure_focused("Keyboard Type", ["${None}", "hello"])
        assert UNFOCUSED_TYPING_WARNING in oc.warnings
        assert oc.attempted is False  # step itself proceeds untouched

    def test_typing_after_verified_focus_is_silent(self):
        win = StubWindow()
        mgr = _manager(StubRuntime(node=win))
        # Establish verified focus via a targeted interaction first.
        oc1 = mgr.ensure_focused(
            "Pointer Click", ["/app:*[@Name='X']//control:Button"]
        )
        assert mgr.has_verified_focus is True
        oc2 = mgr.ensure_focused("Keyboard Type", ["${None}", "hello"])
        assert UNFOCUSED_TYPING_WARNING not in oc2.warnings

    def test_descriptor_targeted_typing_unaffected(self):
        win = StubWindow()
        mgr = _manager(StubRuntime(node=win))
        oc = mgr.ensure_focused(
            "Keyboard Type", ["/app:*[@Name='X']//control:Edit", "hello"]
        )
        assert UNFOCUSED_TYPING_WARNING not in oc.warnings
        assert oc.attempted is True

    def test_x11_raise_does_not_count_as_verified(self):
        win = StubWindow()
        mgr = _manager(StubRuntime(node=win))
        mgr._window_surface = staticmethod(lambda w: None)
        mgr._runtime = types.SimpleNamespace(
            evaluate_single=lambda d: win,
            desktop_info=lambda: {},
            clear_cache=lambda: None,
        )
        mgr._x11_raise = lambda w: True
        mgr.ensure_focused("Pointer Click", ["/app:*//control:Button"])
        assert mgr.has_verified_focus is False
        oc = mgr.ensure_focused("Keyboard Type", ["${None}", "hi"])
        assert UNFOCUSED_TYPING_WARNING in oc.warnings


class TestExecutorOneShot:
    def _executor(self, monkeypatch, manager):
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        executor = KeywordExecutor()
        executor._platynui_focus_manager = manager
        return executor

    def test_warning_once_per_session(self, monkeypatch):
        mgr = _manager()
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        executor = KeywordExecutor()
        executor._platynui_focus_manager = mgr
        session = types.SimpleNamespace()

        oc1 = executor._platynui_focus_before_act(
            session, "Keyboard Type", ["${None}", "a"]
        )
        oc2 = executor._platynui_focus_before_act(
            session, "Keyboard Type", ["${None}", "b"]
        )
        oc3 = executor._platynui_focus_before_act(
            session, "Keyboard Type", ["${None}", "c"]
        )
        assert UNFOCUSED_TYPING_WARNING in oc1.warnings
        assert UNFOCUSED_TYPING_WARNING not in oc2.warnings
        assert UNFOCUSED_TYPING_WARNING not in oc3.warnings
        assert session.desktop_unfocused_typing_warned is True
