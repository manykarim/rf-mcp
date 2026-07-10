"""Unit tests for desktop ui_tree retrieval + page-source short-circuit (ADR-025).

Covers ui_tree_service.get_ui_tree:
- non-desktop session -> success=False with hint
- desktop session with platynui_native mocked at module boundary:
  application list, app_filters expansion (expanded=True), budget limits,
  runtime.shutdown() always called, ImportError path.
- call-shape: ensure_x11_session_env invoked before Runtime construction.

Plus page_source_service._get_page_source_via_rf_context short-circuit for
desktop sessions.

Run with: uv run pytest tests/unit/test_platynui_newcore_ui_tree.py -q
"""

__test__ = True

import asyncio
import sys
import types

import pytest

from robotmcp.components.execution.ui_tree_service import get_ui_tree


# =============================================================================
# Helpers: run a coroutine without asyncio.get_event_loop (3.13 safe)
# =============================================================================


def run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _DesktopSession:
    session_id = "desk-1"

    def is_desktop_session(self):
        return True


class _WebSession:
    session_id = "web-1"

    def is_desktop_session(self):
        return False


# =============================================================================
# Fake platynui_native
# =============================================================================


class _FakeNode:
    def __init__(self, role, name, namespace="app", children=None, attrs=None):
        self.role = role
        self.name = name
        self.namespace = namespace
        self._children = children or []
        self._attrs = attrs or {}

    def children(self):
        return list(self._children)

    def attribute(self, key):
        return self._attrs.get(key)


class _FakeRuntime:
    def __init__(self, apps, shutdown_log, call_order):
        self._apps = apps
        self._shutdown_log = shutdown_log
        call_order.append("runtime_constructed")

    def evaluate(self, query):
        assert query == "/app:*"
        return list(self._apps)

    def shutdown(self):
        self._shutdown_log.append(True)


def _install_fake_native(monkeypatch, apps, shutdown_log, call_order, raise_import=False):
    """Patch sys.modules['platynui_native'] and ensure_x11_session_env."""
    if raise_import:
        # Remove the module and force ImportError on import.
        monkeypatch.setitem(sys.modules, "platynui_native", None)
    else:
        module = types.ModuleType("platynui_native")
        module.Runtime = lambda: _FakeRuntime(apps, shutdown_log, call_order)
        monkeypatch.setitem(sys.modules, "platynui_native", module)

    def _fake_ensure(*args, **kwargs):
        call_order.append("ensure_x11")
        return None

    # ui_tree_service now obtains the runtime from the plugin's runtime broker
    # (change: platynui-desktop-safety-isolation). Patch ensure_x11 at the
    # source and reset the broker so each test binds the fresh fake Runtime.
    import robotmcp.plugins.builtin.platynui_plugin as plugin_mod

    monkeypatch.setattr(plugin_mod, "ensure_x11_session_env", _fake_ensure)
    plugin_mod._reset_runtime_broker_for_tests()


# =============================================================================
# Non-desktop session
# =============================================================================


class TestNonDesktop:
    def test_web_session_returns_failure_with_hint(self):
        result = run(get_ui_tree(_WebSession()))
        assert result["success"] is False
        assert "hint" in result
        assert "page_source" in result["hint"]

    def test_session_without_method_returns_failure(self):
        class NoMethod:
            session_id = "x"

        result = run(get_ui_tree(NoMethod()))
        assert result["success"] is False


# =============================================================================
# Desktop session: happy path
# =============================================================================


class TestDesktopHappyPath:
    def _apps(self):
        button = _FakeNode("Button", "OK", "control", attrs={"Bounds": "1,2,3,4"})
        frame = _FakeNode(
            "Frame", "MainWindow", "control", children=[button],
            attrs={"IsVisible": True},
        )
        calc = _FakeNode("Application", "gnome-calculator", "app", children=[frame])
        other = _FakeNode("Application", "firefox", "app")
        return [calc, other]

    def test_application_list_no_filter(self, monkeypatch):
        shutdown_log, call_order = [], []
        _install_fake_native(monkeypatch, self._apps(), shutdown_log, call_order)
        result = run(get_ui_tree(_DesktopSession()))
        assert result["success"] is True
        assert result["application_count"] == 2
        names = [a["name"] for a in result["applications"]]
        assert "gnome-calculator" in names and "firefox" in names
        # No expansion -> no 'expanded' flag on any entry
        assert all(not a.get("expanded") for a in result["applications"])

    def test_filter_expands_matching_app(self, monkeypatch):
        shutdown_log, call_order = [], []
        _install_fake_native(monkeypatch, self._apps(), shutdown_log, call_order)
        result = run(
            get_ui_tree(_DesktopSession(), app_filters=["gnome-calculator"])
        )
        assert result["success"] is True
        assert result["expanded_applications"] == 1
        calc = next(a for a in result["applications"] if a["name"] == "gnome-calculator")
        assert calc.get("expanded") is True
        assert "children" in calc  # Frame expanded under the app

    def test_filter_no_match_sets_hint(self, monkeypatch):
        shutdown_log, call_order = [], []
        _install_fake_native(monkeypatch, self._apps(), shutdown_log, call_order)
        result = run(get_ui_tree(_DesktopSession(), app_filters=["nonexistent"]))
        assert result["success"] is True
        assert result["expanded_applications"] == 0
        assert "hint" in result

    def test_runtime_not_shut_down_per_call(self, monkeypatch):
        # Change platynui-desktop-safety-isolation: ui_tree now reuses the
        # shared runtime broker and must NOT shut the runtime down per call
        # (the per-call create+shutdown was the root cause of "not available
        # after shutdown"). Two calls reuse the same broker runtime.
        shutdown_log, call_order = [], []
        _install_fake_native(monkeypatch, self._apps(), shutdown_log, call_order)
        run(get_ui_tree(_DesktopSession()))
        run(get_ui_tree(_DesktopSession()))
        assert shutdown_log == []  # never shut down per-call
        # ensure_x11 + a single Runtime bind across both calls
        assert call_order.count("ensure_x11") == 1

    def test_ensure_x11_invoked_before_runtime(self, monkeypatch):
        shutdown_log, call_order = [], []
        _install_fake_native(monkeypatch, self._apps(), shutdown_log, call_order)
        run(get_ui_tree(_DesktopSession()))
        assert "ensure_x11" in call_order
        assert "runtime_constructed" in call_order
        assert call_order.index("ensure_x11") < call_order.index(
            "runtime_constructed"
        )


# =============================================================================
# Budget limits
# =============================================================================


class TestBudgetLimits:
    def _deep_app(self, name="bigapp", n_children=10):
        leaves = [
            _FakeNode("Item", f"item{i}", "control") for i in range(n_children)
        ]
        frame = _FakeNode("Frame", "Win", "control", children=leaves)
        return _FakeNode("Application", name, "app", children=[frame])

    def test_max_children_truncates(self, monkeypatch):
        shutdown_log, call_order = [], []
        app = self._deep_app(n_children=10)
        _install_fake_native(monkeypatch, [app], shutdown_log, call_order)
        result = run(
            get_ui_tree(
                _DesktopSession(),
                app_filters=["bigapp"],
                max_depth=3,
                max_children=2,
                max_nodes=200,
            )
        )
        expanded = result["applications"][0]
        frame = expanded["children"][0]
        assert len(frame.get("children", [])) <= 2
        assert frame.get("children_truncated") is True

    def test_max_nodes_budget_respected(self, monkeypatch):
        shutdown_log, call_order = [], []
        app = self._deep_app(n_children=10)
        _install_fake_native(monkeypatch, [app], shutdown_log, call_order)
        result = run(
            get_ui_tree(
                _DesktopSession(),
                app_filters=["bigapp"],
                max_depth=5,
                max_children=50,
                max_nodes=1,  # tiny budget
            )
        )
        # With a 1-node budget only one descendant is expanded.
        expanded = result["applications"][0]
        frame = expanded.get("children", [])
        assert len(frame) <= 1

    def test_max_depth_limits_expansion(self, monkeypatch):
        shutdown_log, call_order = [], []
        app = self._deep_app(n_children=3)
        _install_fake_native(monkeypatch, [app], shutdown_log, call_order)
        result = run(
            get_ui_tree(
                _DesktopSession(),
                app_filters=["bigapp"],
                max_depth=1,  # only one level under app
                max_children=50,
                max_nodes=200,
            )
        )
        expanded = result["applications"][0]
        # depth=1: app's direct child (Frame) present, but Frame's children not
        frame = expanded["children"][0]
        assert "children" not in frame


# =============================================================================
# ImportError path
# =============================================================================


class TestImportError:
    def test_import_error_returns_structured_error(self, monkeypatch):
        shutdown_log, call_order = [], []
        _install_fake_native(
            monkeypatch, [], shutdown_log, call_order, raise_import=True
        )
        result = run(get_ui_tree(_DesktopSession()))
        assert result["success"] is False
        assert "platynui-native" in result["error"]
        assert "hint" in result


# =============================================================================
# page_source_service short-circuit for desktop sessions
# =============================================================================


class TestPageSourceShortCircuit:
    def test_desktop_session_returns_empty_without_rf_context(self):
        from robotmcp.components.execution.page_source_service import (
            PageSourceService,
        )

        service = PageSourceService()

        class _Desktop:
            session_id = "d1"
            variables: dict = {}
            imported_libraries = ["PlatynUI.BareMetal"]

            def is_desktop_session(self):
                # If the RF-context path were reached it would raise; this
                # returning True must short-circuit before any RF calls.
                return True

        result = service._get_page_source_via_rf_context(_Desktop())
        assert result == ""
