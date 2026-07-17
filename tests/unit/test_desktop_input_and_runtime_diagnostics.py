"""Unit tests for the desktop-input-and-runtime-diagnostics change.

Turns three silent/misleading desktop failures into actionable signals:
- D3 runtime bind/connect failure classification + ui_tree diagnostic
- D2 Wayland forced-X11 input warning
- D1 input-with-no-effect auto-detection (non-reentrant)
"""

from __future__ import annotations

import pytest


# ── D3: runtime failure classification + diagnostic ─────────────────


class TestRuntimeUnavailableReason:
    def _set(self, monkeypatch, *, runtime, state, last_error):
        import robotmcp.plugins.builtin.platynui_plugin as p

        monkeypatch.setattr(p, "_RUNTIME", runtime, raising=False)
        monkeypatch.setattr(p, "_RUNTIME_STATE", state, raising=False)
        monkeypatch.setattr(p, "_RUNTIME_LAST_ERROR", last_error, raising=False)
        return p

    def test_open_runtime_reason_none(self, monkeypatch):
        p = self._set(monkeypatch, runtime=object(), state="open", last_error=None)
        assert p.runtime_unavailable_reason() is None

    def test_not_installed(self, monkeypatch):
        p = self._set(
            monkeypatch, runtime=None, state="new",
            last_error="ModuleNotFoundError: No module named 'platynui_native'",
        )
        assert p.runtime_unavailable_reason() == "not_installed"

    def test_display_connect_failed(self, monkeypatch):
        p = self._set(
            monkeypatch, runtime=None, state="new",
            last_error=(
                "ProviderError: provider initialization failed for runtime: "
                "x11 connection: not available after shutdown or failed connect"
            ),
        )
        assert p.runtime_unavailable_reason() == "display_connect_failed"

    def test_disposed(self, monkeypatch):
        p = self._set(monkeypatch, runtime=None, state="disposed", last_error="x")
        assert p.runtime_unavailable_reason() == "disposed"

    def test_unclassified(self, monkeypatch):
        p = self._set(monkeypatch, runtime=None, state="new", last_error="weird boom")
        assert p.runtime_unavailable_reason() == "unavailable"

    def test_never_attempted_none(self, monkeypatch):
        p = self._set(monkeypatch, runtime=None, state="new", last_error=None)
        assert p.runtime_unavailable_reason() is None


@pytest.mark.asyncio
class TestUiTreeRuntimeDiagnostic:
    async def _ui_tree(self, monkeypatch, *, reason):
        import robotmcp.components.execution.ui_tree_service as u
        import robotmcp.plugins.builtin.platynui_plugin as p

        monkeypatch.setattr(p, "get_runtime", lambda: None)
        monkeypatch.setattr(p, "runtime_unavailable_reason", lambda: reason)

        class _DesktopSession:
            def is_desktop_session(self):
                return True

        # desktop_safety enrichment also calls evaluate_safety; keep it simple
        monkeypatch.setattr(u, "evaluate_safety", lambda s, **k: {
            "classification": "unknown", "enforcing": True, "allowed": False,
        }, raising=False)
        return await u.get_ui_tree(_DesktopSession())

    async def test_display_connect_failed_diagnostic(self, monkeypatch):
        res = await self._ui_tree(monkeypatch, reason="display_connect_failed")
        assert res["success"] is False
        assert res["error"] == "runtime_display_connect_failed"
        assert "XAUTHORITY" in res["message"]
        assert "RESTART" in res["message"]

    async def test_disposed_diagnostic(self, monkeypatch):
        res = await self._ui_tree(monkeypatch, reason="disposed")
        assert res["error"] == "runtime_disposed"
        assert "Restart" in res["message"]

    async def test_not_installed_diagnostic(self, monkeypatch):
        res = await self._ui_tree(monkeypatch, reason="not_installed")
        assert "not installed" in res["error"]
        assert "pip install" in res["hint"]


# ── D2: Wayland forced-X11 input warning ────────────────────────────


class TestWaylandInputWarning:
    def test_interaction_on_wayland_warns(self, monkeypatch):
        import robotmcp.plugins.builtin.platynui_plugin as p
        from robotmcp.components.execution.desktop_execution_signals import (
            wayland_input_warning,
        )

        monkeypatch.setattr(p, "_WAYLAND_ORIGIN", True, raising=False)
        h = wayland_input_warning("Pointer Click")
        assert h is not None
        assert h["type"] == "wayland_x11_input_blocked_risk"
        assert "blocks" in h["message"].lower()
        assert "remediation" in h
        # read/query operations are explicitly noted as unaffected
        assert "unaffected" in h["message"].lower()

    def test_keyboard_on_wayland_warns(self, monkeypatch):
        import robotmcp.plugins.builtin.platynui_plugin as p
        from robotmcp.components.execution.desktop_execution_signals import (
            wayland_input_warning,
        )

        monkeypatch.setattr(p, "_WAYLAND_ORIGIN", True, raising=False)
        assert wayland_input_warning("Keyboard Type") is not None

    def test_read_keyword_no_warn(self, monkeypatch):
        import robotmcp.plugins.builtin.platynui_plugin as p
        from robotmcp.components.execution.desktop_execution_signals import (
            wayland_input_warning,
        )

        monkeypatch.setattr(p, "_WAYLAND_ORIGIN", True, raising=False)
        assert wayland_input_warning("Query") is None
        assert wayland_input_warning("Get Attribute") is None

    def test_x11_origin_no_warn(self, monkeypatch):
        import robotmcp.plugins.builtin.platynui_plugin as p
        from robotmcp.components.execution.desktop_execution_signals import (
            wayland_input_warning,
        )

        monkeypatch.setattr(p, "_WAYLAND_ORIGIN", False, raising=False)
        assert wayland_input_warning("Pointer Click") is None

    def test_origin_detection_from_env_wayland(self, monkeypatch):
        import robotmcp.plugins.builtin.platynui_plugin as p

        monkeypatch.setattr(p, "_WAYLAND_ORIGIN", None, raising=False)
        p._record_session_origin({"XDG_SESSION_TYPE": "wayland"})
        assert p.was_wayland_session() is True

    def test_origin_detection_from_socket(self, monkeypatch, tmp_path):
        import robotmcp.plugins.builtin.platynui_plugin as p

        (tmp_path / "wayland-0").write_text("")
        monkeypatch.setattr(p, "_WAYLAND_ORIGIN", None, raising=False)
        # env says x11 + no WAYLAND_DISPLAY, but the socket exists → Wayland
        p._record_session_origin(
            {"XDG_SESSION_TYPE": "x11", "XDG_RUNTIME_DIR": str(tmp_path)}
        )
        assert p.was_wayland_session() is True

    def test_origin_x11_no_socket(self, monkeypatch, tmp_path):
        import robotmcp.plugins.builtin.platynui_plugin as p

        monkeypatch.setattr(p, "_WAYLAND_ORIGIN", None, raising=False)
        p._record_session_origin(
            {"XDG_SESSION_TYPE": "x11", "XDG_RUNTIME_DIR": str(tmp_path)}
        )
        assert p.was_wayland_session() is False


# ── D1: input-effect auto-detection ─────────────────────────────────


class TestInputEffectSnapshot:
    def _ke(self):
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        return KeywordExecutor.__new__(KeywordExecutor)

    def test_skips_non_keyboard(self):
        ke = self._ke()
        assert ke._desktop_text_count_before("Pointer Click", ["//control:Text"]) is None

    def test_skips_focused_none_target(self):
        ke = self._ke()
        assert ke._desktop_text_count_before("Keyboard Type", ["${None}", "hi"]) is None

    def test_skips_option_arg(self):
        ke = self._ke()
        assert ke._desktop_text_count_before("Keyboard Type", ["env:X=1", "hi"]) is None

    def test_reads_char_count_via_native_runtime(self, monkeypatch):
        # With an explicit target descriptor, reads CharacterCount via the
        # native runtime (non-reentrant), not via RF keyword execution.
        import robotmcp.plugins.builtin.platynui_plugin as p

        class _Attr:
            def value(self):
                return 7

        class _Node:
            def attribute(self, name):
                assert name == "native:Text.CharacterCount"
                return _Attr()

        class _RT:
            def evaluate_single(self, locator):
                return _Node()

        monkeypatch.setattr(p, "get_runtime", lambda: _RT())
        ke = self._ke()
        assert ke._desktop_text_count_before(
            "Keyboard Type", ["//control:Text", "hi"]
        ) == 7

    def test_returns_none_when_runtime_unavailable(self, monkeypatch):
        import robotmcp.plugins.builtin.platynui_plugin as p

        monkeypatch.setattr(p, "get_runtime", lambda: None)
        ke = self._ke()
        assert ke._desktop_text_count_before("Keyboard Type", ["//control:Text"]) is None

    def test_input_effect_hint_unchanged_warns(self):
        # The decision helper warns on success + unchanged count.
        from robotmcp.components.execution.desktop_execution_signals import (
            input_effect_hint,
        )

        h = input_effect_hint(
            keyword="Keyboard Type", success=True, state_before=0, state_after=0
        )
        assert h is not None and h["type"] == "desktop_input_no_effect"

    def test_input_effect_hint_changed_no_warn(self):
        from robotmcp.components.execution.desktop_execution_signals import (
            input_effect_hint,
        )

        assert (
            input_effect_hint(
                keyword="Keyboard Type", success=True, state_before=0, state_after=5
            )
            is None
        )
