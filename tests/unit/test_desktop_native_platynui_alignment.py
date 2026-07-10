"""Unit tests for the desktop-native-platynui-alignment change.

Native-first where PlatynUI provides the capability; a guarded, documented EWMH
probe ONLY for the gaps the spike found (window-presence, WM-active).
"""

from __future__ import annotations

import pytest


# ── D1: guarded EWMH window-presence probe + native providers ───────


class TestWindowPresenceProbe:
    def test_unknown_without_pid_or_names(self):
        from robotmcp.components.execution.platynui_focus import x11_window_present

        assert x11_window_present() == "unknown"

    def test_returns_tristate(self):
        from robotmcp.components.execution.platynui_focus import x11_window_present

        # On any environment the result is one of the three states, never raises.
        assert x11_window_present(["no-such-window-xyz"], pid=987654) in (
            "present",
            "absent",
            "unknown",
        )

    def test_documented_as_fallback(self):
        # The probe must carry the spike-citation in its docstring (it is the
        # documented fallback for a missing native capability).
        from robotmcp.components.execution.platynui_focus import x11_window_present

        doc = (x11_window_present.__doc__ or "").lower()
        assert "no window list" in doc or "internal-only" in doc
        assert "fallback" in doc


class TestNativeProviders:
    def test_native_providers_uses_runtime_api(self, monkeypatch):
        import robotmcp.plugins.builtin.platynui_plugin as p

        class _RT:
            def providers(self):
                return [{"name": "AT-SPI2"}, {"name": "WindowManager"}]

        monkeypatch.setattr(p, "get_runtime", lambda: _RT())
        out = p.native_providers()
        assert [d["name"] for d in out] == ["AT-SPI2", "WindowManager"]

    def test_native_providers_empty_when_runtime_none(self, monkeypatch):
        import robotmcp.plugins.builtin.platynui_plugin as p

        monkeypatch.setattr(p, "get_runtime", lambda: None)
        assert p.native_providers() == []

    def test_native_providers_never_raises(self, monkeypatch):
        import robotmcp.plugins.builtin.platynui_plugin as p

        def _boom():
            raise RuntimeError("disposed")

        monkeypatch.setattr(p, "get_runtime", _boom)
        assert p.native_providers() == []


# ── D2: get_ui_tree exposure diagnostic ─────────────────────────────


class TestExposureDiagnostic:
    def _patch(self, monkeypatch, presence, providers=None):
        import robotmcp.components.execution.platynui_focus as f
        import robotmcp.plugins.builtin.platynui_plugin as p

        monkeypatch.setattr(f, "x11_window_present", lambda app_names=None, pid=None: presence)
        monkeypatch.setattr(p, "native_providers", lambda: providers or [{"name": "AT-SPI2"}])

    def test_window_present_not_exposed(self, monkeypatch):
        from robotmcp.components.execution import ui_tree_service as u

        self._patch(monkeypatch, "present")
        d = u._build_exposure_diagnostic(["gnome-calculator"])
        assert d["type"] == "accessibility_not_exposed"
        assert d["window_present"] is True
        assert d["providers"] == ["AT-SPI2"]
        assert "remediation" in d
        assert "not a locator problem" in d["message"].lower()
        # Harness finding (change: desktop-a11y-atspi-backend): remediation must
        # recommend the backend NAME (GTK_A11Y=atspi), never GTK_A11Y=1 which
        # modern GTK rejects (app then exposes NO AT-SPI tree).
        remediation = " ".join(d["remediation"])
        assert "GTK_A11Y=atspi" in remediation
        # any mention of the old value must be as a rejected anti-pattern
        assert "rejects GTK_A11Y=1" in remediation

    def test_providers_use_real_native_keys(self, monkeypatch):
        # native runtime.providers() dicts use id/display_name/technology, NOT
        # "name" — the diagnostic must surface real names, never [null].
        import robotmcp.components.execution.platynui_focus as f
        import robotmcp.plugins.builtin.platynui_plugin as p
        from robotmcp.components.execution import ui_tree_service as u

        monkeypatch.setattr(f, "x11_window_present", lambda app_names=None, pid=None: "present")
        monkeypatch.setattr(
            p,
            "native_providers",
            lambda: [
                {"id": "atspi", "display_name": "AT-SPI2", "technology": "atspi"},
                {"id": "wm", "technology": "ewmh"},  # no display_name → falls back
            ],
        )
        d = u._build_exposure_diagnostic(["gnome-calculator"])
        assert d["providers"] == ["AT-SPI2", "ewmh"]
        assert None not in d["providers"]

    def test_window_absent(self, monkeypatch):
        from robotmcp.components.execution import ui_tree_service as u

        self._patch(monkeypatch, "absent")
        d = u._build_exposure_diagnostic(["x"])
        assert d["type"] == "app_window_absent"
        assert d["window_present"] is False

    def test_unknown_undetermined(self, monkeypatch):
        from robotmcp.components.execution import ui_tree_service as u

        self._patch(monkeypatch, "unknown")
        d = u._build_exposure_diagnostic(["x"])
        assert d["type"] == "accessibility_exposure_undetermined"
        assert d["window_present"] is None

    def test_wiring_no_match_adds_diagnostic(self, monkeypatch):
        # A filter given, no app resolves -> diagnostic present.
        import robotmcp.components.execution.ui_tree_service as u
        import robotmcp.plugins.builtin.platynui_plugin as p

        class _RT:
            def clear_cache(self):
                pass

            def evaluate(self, expr):
                return []  # no apps

            def desktop_info(self):
                return {}

        monkeypatch.setattr(p, "get_runtime", lambda: _RT())
        self._patch(monkeypatch, "present")
        res = u._collect_ui_tree_sync(["gnome-calculator"], 3, 20, 200)
        assert res["success"] is True
        assert "accessibility_diagnostic" in res
        assert res["accessibility_diagnostic"]["type"] == "accessibility_not_exposed"

    def test_wiring_app_present_no_diagnostic(self, monkeypatch):
        # The requested app resolves -> NO exposure diagnostic.
        import robotmcp.components.execution.ui_tree_service as u
        import robotmcp.plugins.builtin.platynui_plugin as p

        class _Node:
            role = "Application"
            name = "gnome-calculator"
            namespace = "app"

        class _RT:
            def clear_cache(self):
                pass

            def evaluate(self, expr):
                return [_Node()]

            def desktop_info(self):
                return {}

        monkeypatch.setattr(p, "get_runtime", lambda: _RT())
        monkeypatch.setattr(u, "_expand_subtree", lambda *a, **k: {})
        monkeypatch.setattr(u, "_window_visibility", lambda *a, **k: None)
        self._patch(monkeypatch, "present")
        res = u._collect_ui_tree_sync(["gnome-calculator"], 3, 20, 200)
        assert res["success"] is True
        assert res["expanded_applications"] == 1
        assert "accessibility_diagnostic" not in res


# ── D4: native-first window focus/raise ─────────────────────────────


class TestNativeFirstFocus:
    def _mgr(self):
        from robotmcp.components.execution.platynui_focus import PlatynUIFocusManager

        m = PlatynUIFocusManager.__new__(PlatynUIFocusManager)
        m._last_focused_scope = None
        return m

    def test_native_window_surface_activate_first(self, monkeypatch):
        m = self._mgr()
        monkeypatch.setattr(m, "_is_active", lambda w: None)

        class _WS:
            def activate(self):
                pass

        monkeypatch.setattr(m, "_window_surface", staticmethod(lambda w: _WS()))
        ok, strategy, _ready = m.focus_window(object(), scope=None)
        assert ok is True
        assert strategy == "window_surface:activate"  # native path, no ctypes

    def test_native_runtime_focus_before_ctypes(self, monkeypatch):
        m = self._mgr()
        monkeypatch.setattr(m, "_is_active", lambda w: None)
        monkeypatch.setattr(m, "_window_surface", staticmethod(lambda w: None))

        class _RT:
            # no bring_to_front; has focus
            def focus(self, w):
                pass

        monkeypatch.setattr(m, "_get_runtime", lambda: _RT())
        # ctypes raise must NOT be reached
        monkeypatch.setattr(m, "_x11_raise", lambda w: pytest.fail("ctypes reached"))
        ok, strategy, _ready = m.focus_window(object(), scope=None)
        assert ok is True
        assert strategy == "focus"

    def test_ctypes_fallback_only_when_native_unavailable(self, monkeypatch):
        m = self._mgr()
        monkeypatch.setattr(m, "_is_active", lambda w: None)
        monkeypatch.setattr(m, "_window_surface", staticmethod(lambda w: None))
        monkeypatch.setattr(m, "_get_runtime", lambda: None)  # no native runtime
        called = {}
        monkeypatch.setattr(m, "_x11_raise", lambda w: called.setdefault("yes", True) or True)
        ok, strategy, _ready = m.focus_window(object(), scope=None)
        assert ok is True
        assert strategy == "x11_raise"
        assert called.get("yes") is True


# ── D5: guidance references the diagnostic + native-first rule ──────


class TestExposureGuidance:
    def _g(self):
        from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter

        return RobotFrameworkNativeConverter().get_platynui_locator_guidance()

    def test_accessibility_exposure_section(self):
        g = self._g()
        assert "accessibility_exposure" in g
        rules = " ".join(g["accessibility_exposure"]["rules"])
        assert "accessibility_not_exposed" in rules
        assert "app_window_absent" in rules
        # Harness finding (change: desktop-a11y-atspi-backend): the bridge
        # remediation must name GTK_A11Y=atspi, not the GTK-rejected "=1".
        assert "GTK_A11Y=atspi" in rules
        assert "GTK_A11Y=1" not in rules

    def test_discourages_coordinate_ocr(self):
        rules = " ".join(self._g()["accessibility_exposure"]["rules"]).lower()
        assert "do not switch to coordinate" in rules or "coordinate clicks or ocr" in rules

    def test_states_native_first_rule(self):
        rules = " ".join(self._g()["accessibility_exposure"]["rules"]).lower()
        assert "native api" in rules
        assert "documented" in rules or "fallback" in rules
