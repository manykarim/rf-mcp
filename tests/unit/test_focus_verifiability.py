"""Upstream-verified focus + focus-unverifiable warning
(change: platynui-visible-safe-targeting, I-2).

Focus-before-act is rebuilt on PlatynUI new-core primitives:
``supported_patterns()`` introspection, ``Runtime.bring_to_front(node,
wait_ms=...)`` verified activation, ``accepts_user_input()`` readiness.
When none of those can verify focus (the LibreOffice Writer frame case),
an explicit warning is emitted instead of silent keystroke loss.

All runtimes here are plain-Python stubs — no platynui_native required.
"""

from __future__ import annotations

import pytest

from robotmcp.components.execution.platynui_focus import (
    BRING_TO_FRONT_WAIT_MS,
    FOCUS_UNVERIFIABLE_PREFIX,
    PlatynUIFocusManager,
)


WS_PATTERN = "org.platynui.patterns.WindowSurface"
FOCUSABLE_PATTERN = "org.platynui.patterns.Focusable"


class StubSurface:
    def __init__(self, accepts=None, calls=None):
        self._accepts = accepts
        self.calls = calls if calls is not None else []

    def activate(self):
        self.calls.append("activate")

    def accepts_user_input(self):
        self.calls.append("accepts_user_input")
        return self._accepts


class StubWindow:
    def __init__(
        self,
        *,
        patterns=None,
        surface=None,
        runtime_id="win-1",
        attrs=None,
        has_supported_patterns=True,
    ):
        self._patterns = patterns
        self._surface = surface
        self.runtime_id = runtime_id
        self._attrs = attrs or {}
        self.calls = []
        if not has_supported_patterns:
            # Simulate an older platynui_native: no introspection API at all.
            self.supported_patterns = None  # not callable

    def supported_patterns(self):  # may be shadowed by None above
        return self._patterns

    def attribute(self, name):
        if name not in self._attrs:
            raise KeyError(name)
        return self._attrs[name]

    def top_level_or_self(self):
        return self


class StubRuntime:
    def __init__(self, *, node=None, accepts_wait_ms=True, bring_fails=False):
        self._node = node
        self._accepts_wait_ms = accepts_wait_ms
        self._bring_fails = bring_fails
        self.bring_calls = []
        self.focus_calls = []

    def evaluate_single(self, descriptor):
        return self._node

    def desktop_info(self):
        return {}

    def clear_cache(self):
        pass

    def bring_to_front(self, node, **kwargs):
        if "wait_ms" in kwargs and not self._accepts_wait_ms:
            raise TypeError("bring_to_front() got an unexpected keyword 'wait_ms'")
        self.bring_calls.append(kwargs)
        if self._bring_fails:
            raise RuntimeError("BringToFrontError: PatternMissing")

    def focus(self, node):
        self.focus_calls.append(node)


def _manager(window, runtime=None):
    mgr = PlatynUIFocusManager()
    mgr._runtime = runtime if runtime is not None else StubRuntime(node=window)
    mgr._desktop_bounds = lambda: None
    return mgr


def _ensure(mgr, **kwargs):
    return mgr.ensure_focused(
        "Keyboard Type", ["/app:*[@Name='X']//control:Edit", "hello"], **kwargs
    )


class TestVerifiedActivation:
    def test_bring_to_front_with_input_ready_true_no_warning(self):
        surface = StubSurface(accepts=True)
        win = StubWindow(patterns=[WS_PATTERN, FOCUSABLE_PATTERN], surface=surface)
        rt = StubRuntime(node=win)
        mgr = _manager(win, rt)
        mgr._window_surface = staticmethod(lambda w: surface)
        oc = _ensure(mgr)
        assert oc.attempted is True
        assert oc.strategy == "bring_to_front"
        assert oc.input_ready is True
        assert oc.patterns == [WS_PATTERN, FOCUSABLE_PATTERN]
        assert not any(FOCUS_UNVERIFIABLE_PREFIX in w for w in oc.warnings)
        # wait_ms was passed to the upstream API
        assert rt.bring_calls == [{"wait_ms": BRING_TO_FRONT_WAIT_MS}]

    def test_input_ready_false_warns(self):
        surface = StubSurface(accepts=False)
        win = StubWindow(patterns=[WS_PATTERN], surface=surface)
        mgr = _manager(win, StubRuntime(node=win))
        mgr._window_surface = staticmethod(lambda w: surface)
        oc = _ensure(mgr)
        assert oc.input_ready is False
        assert any(
            FOCUS_UNVERIFIABLE_PREFIX in w and "not accepting user input" in w
            for w in oc.warnings
        )

    def test_input_ready_unavailable_warns(self):
        # WindowSurface exists but has no accepts_user_input → readiness
        # cannot be confirmed.
        class BareSurface:
            def activate(self):
                pass

        win = StubWindow(patterns=[WS_PATTERN])
        mgr = _manager(win, StubRuntime(node=win))
        mgr._window_surface = staticmethod(lambda w: BareSurface())
        oc = _ensure(mgr)
        assert oc.input_ready is None
        assert any(FOCUS_UNVERIFIABLE_PREFIX in w for w in oc.warnings)

    def test_wait_ms_typeerror_falls_back_to_positional(self):
        win = StubWindow(patterns=[WS_PATTERN])
        rt = StubRuntime(node=win, accepts_wait_ms=False)
        mgr = _manager(win, rt)
        mgr._window_surface = staticmethod(lambda w: StubSurface(accepts=True))
        oc = _ensure(mgr)
        assert oc.strategy == "bring_to_front"
        # Called without kwargs after the TypeError
        assert rt.bring_calls == [{}]


class TestMissingPatternWarning:
    def test_no_focus_patterns_emits_verbatim_warning(self):
        # The LibreOffice Writer frame: patterns exposed, but neither
        # WindowSurface nor Focusable among them.
        win = StubWindow(patterns=["org.platynui.patterns.Element"])
        rt = StubRuntime(node=win, bring_fails=True)
        mgr = _manager(win, rt)
        mgr._window_surface = staticmethod(lambda w: None)
        mgr._x11_raise = lambda w: False
        oc = _ensure(mgr)
        assert oc.attempted is True
        expected = f"{FOCUS_UNVERIFIABLE_PREFIX} (no WindowSurface/Focusable pattern)"
        assert expected in oc.warnings
        # Pattern names surfaced for diagnostics
        assert oc.patterns == ["org.platynui.patterns.Element"]
        assert oc.to_dict()["patterns"] == ["org.platynui.patterns.Element"]

    def test_short_pattern_names_recognized(self):
        # Pattern names may come short-form; suffix matching must accept both.
        surface = StubSurface(accepts=True)
        win = StubWindow(patterns=["WindowSurface"], surface=surface)
        mgr = _manager(win, StubRuntime(node=win))
        mgr._window_surface = staticmethod(lambda w: surface)
        oc = _ensure(mgr)
        assert not any("no WindowSurface/Focusable pattern" in w for w in oc.warnings)


class TestFallbackTiers:
    def test_runtime_focus_used_when_bring_to_front_fails(self):
        win = StubWindow(patterns=[FOCUSABLE_PATTERN])
        rt = StubRuntime(node=win, bring_fails=True)
        mgr = _manager(win, rt)
        mgr._window_surface = staticmethod(lambda w: None)
        oc = _ensure(mgr)
        assert oc.strategy == "focus"
        assert rt.focus_calls  # upstream Runtime.focus reached

    def test_x11_raise_is_flagged_and_warned(self):
        win = StubWindow(patterns=[WS_PATTERN])
        rt = StubRuntime(node=win, bring_fails=True)
        # Runtime without focus()
        del StubRuntime.focus
        try:
            mgr = _manager(win, rt)
            mgr._window_surface = staticmethod(lambda w: None)
            mgr._x11_raise = lambda w: True
            oc = _ensure(mgr)
            assert oc.strategy == "x11_raise"
            assert any(
                FOCUS_UNVERIFIABLE_PREFIX in w and "X11 raise" in w
                for w in oc.warnings
            )
        finally:
            StubRuntime.focus = lambda self, node: self.focus_calls.append(node)


class TestGracefulDegradation:
    def test_runtime_without_supported_patterns(self):
        # Older platynui_native: nodes have no supported_patterns API.
        win = StubWindow(patterns=None, has_supported_patterns=False)
        mgr = _manager(win, StubRuntime(node=win))
        mgr._window_surface = staticmethod(lambda w: None)
        oc = _ensure(mgr)  # must not raise
        assert oc.attempted is True
        assert oc.patterns is None
        assert any(
            FOCUS_UNVERIFIABLE_PREFIX in w and "pattern introspection" in w
            for w in oc.warnings
        )

    def test_supported_patterns_raising_is_treated_as_unavailable(self):
        class RaisingWindow(StubWindow):
            def supported_patterns(self):
                raise RuntimeError("provider gone")

        win = RaisingWindow()
        mgr = _manager(win, StubRuntime(node=win))
        mgr._window_surface = staticmethod(lambda w: None)
        oc = _ensure(mgr)
        assert oc.patterns is None


class TestPidScopeCheck:
    """Lineage contract (change: desktop-aut-process-lineage): warn only on
    a CONFIRMED foreign process, never on bare pid inequality."""

    def test_confirmed_foreign_warns_with_lineage(self, monkeypatch):
        import robotmcp.components.execution.platynui_focus as focus_mod

        monkeypatch.setattr(
            focus_mod, "pid_in_aut_lineage", lambda *a, **k: False
        )
        monkeypatch.setattr(focus_mod, "_read_pid_sid", lambda p: 999)
        win = StubWindow(patterns=[WS_PATTERN], attrs={"ProcessId": 5678})
        mgr = _manager(win, StubRuntime(node=win))
        mgr._window_surface = staticmethod(lambda w: StubSurface(accepts=True))
        oc = _ensure(mgr, aut_pid=1234, aut_sid=111)
        assert any(
            "PID 5678" in w and "PID 1234" in w and "no lineage relation" in w
            for w in oc.warnings
        )

    def test_related_lineage_is_silent(self, monkeypatch):
        # Run-5 shape: wrapper/daemonized pid differs but lineage relates.
        import robotmcp.components.execution.platynui_focus as focus_mod

        monkeypatch.setattr(
            focus_mod, "pid_in_aut_lineage", lambda *a, **k: True
        )
        win = StubWindow(patterns=[WS_PATTERN], attrs={"ProcessId": 5678})
        mgr = _manager(win, StubRuntime(node=win))
        mgr._window_surface = staticmethod(lambda w: StubSurface(accepts=True))
        oc = _ensure(mgr, aut_pid=1234, aut_sid=111)
        assert not any("PID" in w for w in oc.warnings)

    def test_indeterminate_lineage_is_silent(self, monkeypatch):
        import robotmcp.components.execution.platynui_focus as focus_mod

        monkeypatch.setattr(
            focus_mod, "pid_in_aut_lineage", lambda *a, **k: None
        )
        win = StubWindow(patterns=[WS_PATTERN], attrs={"ProcessId": 5678})
        mgr = _manager(win, StubRuntime(node=win))
        mgr._window_surface = staticmethod(lambda w: StubSurface(accepts=True))
        oc = _ensure(mgr, aut_pid=1234)
        assert not any("PID" in w for w in oc.warnings)

    def test_no_aut_pid_skips_check(self):
        win = StubWindow(patterns=[WS_PATTERN], attrs={"ProcessId": 5678})
        mgr = _manager(win, StubRuntime(node=win))
        mgr._window_surface = staticmethod(lambda w: StubSurface(accepts=True))
        oc = _ensure(mgr)
        assert not any("PID" in w for w in oc.warnings)


class TestActivationCache:
    def test_second_focus_uses_cache(self):
        win = StubWindow(patterns=[WS_PATTERN], runtime_id="win-C")
        rt = StubRuntime(node=win)
        mgr = _manager(win, rt)
        mgr._window_surface = staticmethod(lambda w: StubSurface(accepts=True))
        focused1, strategy1, _ = mgr.focus_window(win, "/app:*")
        focused2, strategy2, _ = mgr.focus_window(win, "/app:*")
        assert (focused1, strategy1) == (True, "bring_to_front")
        assert (focused2, strategy2) == (True, "cached")
        assert len(rt.bring_calls) == 1  # no second upstream call

    def test_invalidate_focus_cache_forces_reactivation(self):
        win = StubWindow(patterns=[WS_PATTERN], runtime_id="win-D")
        rt = StubRuntime(node=win)
        mgr = _manager(win, rt)
        mgr._window_surface = staticmethod(lambda w: StubSurface(accepts=True))
        mgr.focus_window(win, "/app:*")
        mgr.invalidate_focus_cache()
        _, strategy, _ = mgr.focus_window(win, "/app:*")
        assert strategy == "bring_to_front"
        assert len(rt.bring_calls) == 2


class TestOutcomeSerialization:
    def test_to_dict_carries_patterns_and_input_ready(self):
        surface = StubSurface(accepts=True)
        win = StubWindow(patterns=[WS_PATTERN], surface=surface)
        mgr = _manager(win, StubRuntime(node=win))
        mgr._window_surface = staticmethod(lambda w: surface)
        oc = _ensure(mgr)
        d = oc.to_dict()
        assert d["patterns"] == [WS_PATTERN]
        assert d["input_ready"] is True
        # The keyword_executor hint channel maps every entry of
        # to_dict()["warnings"] to a platynui_focus_warning hint — the
        # I-2 message only needs to be present here to reach the agent.
        assert "warnings" not in d or all(isinstance(w, str) for w in d["warnings"])
