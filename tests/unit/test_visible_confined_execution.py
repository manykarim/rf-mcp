"""Visible-and-confined desktop execution
(change: platynui-visible-safe-targeting, Group 3).

- Upstream Runtime.highlight() marks the interaction target on the
  (visible) display before input dispatch — default on, kill-switchable,
  strictly soft-fail.
- get_session_state's desktop_environment section proves display identity,
  isolation classification (with provenance), and upstream desktop_info().
- build_isolation_recipe() leads with the visible Xephyr mode and includes
  platynui-cli-rs verification commands + the EWMH-WM requirement.
"""

from __future__ import annotations

import pytest

from robotmcp.components.execution.platynui_focus import (
    HIGHLIGHT_DURATION_MS,
    HIGHLIGHT_ENV,
    PlatynUIFocusManager,
    highlight_disabled_by_env,
)


class StubRect:
    def x(self):
        return 10.0

    def y(self):
        return 20.0

    def width(self):
        return 100.0

    def height(self):
        return 30.0


_UNSET = object()


class StubNode:
    def __init__(self, runtime_id="n1", bounds=_UNSET):
        self.runtime_id = runtime_id
        self._bounds = StubRect() if bounds is _UNSET else bounds

    def attribute(self, name):
        if name == "Bounds" and self._bounds is not None:
            return self._bounds
        raise KeyError(name)

    def top_level_or_self(self):
        return self


class StubRuntime:
    def __init__(self, node=None, highlight_raises=False):
        self._node = node
        self._highlight_raises = highlight_raises
        self.highlight_calls = []
        self.clear_calls = 0

    def evaluate_single(self, descriptor):
        return self._node

    def desktop_info(self):
        return {}

    def clear_cache(self):
        pass

    def highlight(self, rect, **kwargs):
        if self._highlight_raises:
            raise RuntimeError("no highlight provider")
        self.highlight_calls.append((rect, kwargs))

    def clear_highlight(self):
        self.clear_calls += 1


def _manager(runtime):
    mgr = PlatynUIFocusManager()
    mgr._runtime = runtime
    mgr._desktop_bounds = lambda: None
    mgr._window_surface = staticmethod(lambda w: None)
    mgr._x11_raise = lambda w: False
    return mgr


class TestHighlightTarget:
    def test_highlight_called_with_bounds_and_duration(self):
        node = StubNode()
        rt = StubRuntime(node=node)
        mgr = _manager(rt)
        mgr.highlight_target("/app:*//control:Button", node)
        assert len(rt.highlight_calls) == 1
        rect, kwargs = rt.highlight_calls[0]
        assert rect is node._bounds
        assert kwargs == {"duration_ms": HIGHLIGHT_DURATION_MS}

    def test_highlight_duration_typeerror_falls_back(self):
        class OldRuntime(StubRuntime):
            def highlight(self, rect, **kwargs):
                if kwargs:
                    raise TypeError("no duration_ms")
                self.highlight_calls.append((rect, kwargs))

        node = StubNode()
        rt = OldRuntime(node=node)
        mgr = _manager(rt)
        mgr.highlight_target("/x", node)
        assert rt.highlight_calls == [(node._bounds, {})]

    def test_highlight_soft_fails(self):
        node = StubNode()
        rt = StubRuntime(node=node, highlight_raises=True)
        mgr = _manager(rt)
        mgr.highlight_target("/x", node)  # must not raise

    def test_no_bounds_no_call(self):
        node = StubNode(bounds=None)
        rt = StubRuntime(node=node)
        mgr = _manager(rt)
        mgr.highlight_target("/x", node)
        assert rt.highlight_calls == []

    def test_ensure_focused_highlights_before_dispatch(self):
        node = StubNode()
        rt = StubRuntime(node=node)
        mgr = _manager(rt)
        oc = mgr.ensure_focused(
            "Pointer Click", ["/app:*[@Name='X']//control:Button"],
            highlight=True,
        )
        assert oc.attempted is True
        assert len(rt.highlight_calls) == 1
        # The step outcome is unaffected by highlight presence.

    def test_ensure_focused_highlight_default_off_at_module_level(self):
        # ensure_focused only highlights when asked — the default-ON policy
        # lives at the executor gate (session config).
        node = StubNode()
        rt = StubRuntime(node=node)
        mgr = _manager(rt)
        mgr.ensure_focused("Pointer Click", ["/app:*//control:Button"])
        assert rt.highlight_calls == []

    def test_env_kill_switch(self, monkeypatch):
        monkeypatch.setenv(HIGHLIGHT_ENV, "0")
        assert highlight_disabled_by_env() is True
        node = StubNode()
        rt = StubRuntime(node=node)
        mgr = _manager(rt)
        mgr.ensure_focused(
            "Pointer Click", ["/app:*//control:Button"], highlight=True
        )
        assert rt.highlight_calls == []

    def test_env_default_enabled(self, monkeypatch):
        monkeypatch.delenv(HIGHLIGHT_ENV, raising=False)
        assert highlight_disabled_by_env() is False


class TestClearHighlight:
    def test_clear_highlight_invokes_runtime(self):
        rt = StubRuntime()
        mgr = _manager(rt)
        mgr.clear_highlight()
        assert rt.clear_calls == 1

    def test_clear_highlight_without_runtime_is_noop(self):
        mgr = PlatynUIFocusManager()
        mgr._runtime = None
        mgr._get_runtime = lambda: None
        mgr.clear_highlight()  # must not raise


class TestExecutorHighlightPolicy:
    """The executor gate reads platynui_highlight (default True)."""

    def test_session_default_enables_highlight(self, monkeypatch):
        import types

        import robotmcp.components.execution.platynui_focus as focus_mod
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        captured = {}

        class FakeManager:
            def ensure_focused(self, keyword, arguments, **kwargs):
                captured.update(kwargs)
                from robotmcp.components.execution.platynui_focus import FocusOutcome

                return FocusOutcome()

            def invalidate_focus_cache(self):
                pass

        monkeypatch.setattr(focus_mod, "PlatynUIFocusManager", FakeManager)
        executor = KeywordExecutor()
        session = types.SimpleNamespace()
        executor._platynui_focus_before_act(
            session, "Pointer Click", ["/app:*//control:Button"]
        )
        assert captured.get("highlight") is True

    def test_session_opt_out(self, monkeypatch):
        import types

        import robotmcp.components.execution.platynui_focus as focus_mod
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        captured = {}

        class FakeManager:
            def ensure_focused(self, keyword, arguments, **kwargs):
                captured.update(kwargs)
                from robotmcp.components.execution.platynui_focus import FocusOutcome

                return FocusOutcome()

            def invalidate_focus_cache(self):
                pass

        monkeypatch.setattr(focus_mod, "PlatynUIFocusManager", FakeManager)
        executor = KeywordExecutor()
        session = types.SimpleNamespace(platynui_highlight=False)
        executor._platynui_focus_before_act(
            session, "Pointer Click", ["/app:*//control:Button"]
        )
        assert captured.get("highlight") is False


class TestDesktopEnvironmentSection:
    def test_marker_isolated_display_reported(self, monkeypatch):
        import robotmcp.plugins.builtin.platynui_plugin as plugin
        from robotmcp.components.execution.ui_tree_service import (
            get_desktop_environment,
        )

        monkeypatch.setenv("DISPLAY", ":100")
        monkeypatch.setenv("ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY", ":100")
        # Strict guard: the marker must be ownership-corroborated to grant
        # isolated (change: desktop-isolation-marker-hardening).
        import robotmcp.components.execution.desktop_display_safety as _dds
        monkeypatch.setattr(_dds, "_marker_ownership_status", lambda e, d: "verified")

        class RT:
            def desktop_info(self):
                return {
                    "technology": "Linux.AtSpi",
                    "name": "Desktop",
                    "os_name": "Linux",
                    "os_version": "6.8",
                    "bounds": {"x": 0, "y": 0, "width": 1280, "height": 1024},
                    "monitors": [
                        {"id": "m1", "name": "Xephyr", "bounds": StubRect()},
                    ],
                }

        monkeypatch.setattr(plugin, "get_runtime", lambda: RT())
        out = get_desktop_environment(session=object())
        assert out["display"] == ":100"
        assert out["isolation"] == "isolated"
        assert out["isolation_source"] == "marker"
        info = out["desktop_info"]
        assert info["technology"] == "Linux.AtSpi"
        assert info["bounds"] == {"x": 0.0, "y": 0.0, "width": 1280.0, "height": 1024.0}
        assert info["monitors"][0]["bounds"] == {
            "x": 10.0, "y": 20.0, "width": 100.0, "height": 30.0,
        }

    def test_runtime_unavailable_still_reports_classification(self, monkeypatch):
        import robotmcp.plugins.builtin.platynui_plugin as plugin
        from robotmcp.components.execution.ui_tree_service import (
            get_desktop_environment,
        )

        monkeypatch.setenv("DISPLAY", ":100")
        monkeypatch.setenv("ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY", ":100")
        import robotmcp.components.execution.desktop_display_safety as _dds
        monkeypatch.setattr(_dds, "_marker_ownership_status", lambda e, d: "verified")

        def _boom():
            raise RuntimeError("native unavailable")

        monkeypatch.setattr(plugin, "get_runtime", _boom)
        out = get_desktop_environment(session=object())
        assert out["isolation"] == "isolated"
        assert "desktop_info" not in out


class TestDetailedClassification:
    def test_marker_source(self, monkeypatch):
        import robotmcp.components.execution.desktop_display_safety as _dds
        from robotmcp.components.execution.desktop_display_safety import (
            classify_bound_display_detailed,
        )

        # Corroborated marker -> isolated/marker (strict guard).
        monkeypatch.setattr(_dds, "_marker_ownership_status", lambda e, d: "verified")
        out = classify_bound_display_detailed(
            {"DISPLAY": ":42", "ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY": ":42"}
        )
        assert out == {
            "display": ":42",
            "isolation": "isolated",
            "isolation_source": "marker",
        }

    def test_no_display_unknown(self):
        from robotmcp.components.execution.desktop_display_safety import (
            classify_bound_display_detailed,
        )

        out = classify_bound_display_detailed({})
        assert out["display"] is None
        assert out["isolation"] == "unknown"
        assert out["isolation_source"] == "none"


class TestIsolationRecipe:
    def _recipe(self):
        from robotmcp.components.execution.desktop_display_safety import (
            build_isolation_recipe,
        )

        return build_isolation_recipe()

    def test_visible_mode_first_and_recommended(self):
        r = self._recipe()
        assert r["recommended_mode"] == "visible"
        assert r["modes"][0]["mode"] == "visible"
        assert r["modes"][0].get("recommended") is True
        assert "visible on your screen" in r["modes"][0]["description"].lower() or \
            "watch" in r["modes"][0]["description"].lower()

    def test_steps_lead_with_xephyr_and_keep_xvfb_alternative(self):
        r = self._recipe()
        joined = " ".join(r["steps"]).lower()
        assert "xephyr" in joined
        assert "xvfb" in joined  # CI alternative retained
        assert joined.index("xephyr") < joined.index("xvfb")
        assert r["marker_env"] in " ".join(r["steps"])

    def test_cli_verification_commands_present(self):
        r = self._recipe()
        cmds = " ".join(r["verification_commands"])
        for sub in ("info", "window --list", "highlight", "snapshot"):
            assert sub in cmds
        assert "platynui-cli-rs" in cmds

    def test_ewmh_wm_requirement_stated(self):
        r = self._recipe()
        joined = " ".join(r["steps"]).lower()
        assert "ewmh" in joined
        assert "warning" in joined  # degradation consequence stated

    def test_bypass_still_escape_hatch(self):
        r = self._recipe()
        assert "escape hatch" in r["bypass_note"].lower()
