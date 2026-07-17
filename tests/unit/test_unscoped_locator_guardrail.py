"""Unscoped desktop locator guardrail
(change: desktop-unscoped-locator-guardrail).

Reproduced 2026-06-16: a desktop `Query //control:Paragraph` walks the whole
session AT-SPI tree (every desktop app) — 36.9s on a busy desktop — long
enough to exceed the MCP client's request timeout and kill the transport.
`//` is absolute XPath; it ignores Set Root. The guardrail refuses such
locators pre-flight with an app-scoped rewrite, unless explicitly opted in.
"""

from __future__ import annotations

import types

import pytest

from robotmcp.components.execution.desktop_execution_signals import (
    is_query_keyword,
    is_unscoped_locator,
)
from robotmcp.components.execution.keyword_executor import KeywordExecutor


class TestDetection:
    @pytest.mark.parametrize("kw", [
        "Query", "Evaluate", "PlatynUI.BareMetal.Query",
        "platynui.baremetal.evaluate",
    ])
    def test_query_keywords_recognized(self, kw):
        assert is_query_keyword(kw) is True

    @pytest.mark.parametrize("kw", ["Pointer Click", "Keyboard Type", "Set Root"])
    def test_non_query_keywords_ignored(self, kw):
        assert is_query_keyword(kw) is False

    @pytest.mark.parametrize("xp", [
        "//control:Paragraph",
        "  //control:Button",            # leading whitespace
        "//*",
        "*",
        "descendant::control:Button",
        "descendant-or-self::control:Edit",
    ])
    def test_unscoped_detected(self, xp):
        assert is_unscoped_locator(xp) is True

    @pytest.mark.parametrize("xp", [
        "/app:*[@Name='soffice']//control:Paragraph",
        "/app:*",
        "control:Button[@Name='OK']",
        ".//control:Edit",
        "count(//control:Paragraph)",
        "count(//control:Button)",
        "string(//control:Label)",
        "child::control:Frame",
    ])
    def test_scoped_or_aggregate_allowed(self, xp):
        assert is_unscoped_locator(xp) is False

    def test_non_string_is_not_unscoped(self):
        assert is_unscoped_locator(None) is False
        assert is_unscoped_locator(123) is False
        assert is_unscoped_locator("") is False


def _desktop_session(**attrs):
    s = types.SimpleNamespace(
        is_desktop_session=lambda: True,
        desktop_aut_pid=None,
        platynui_allow_unscoped=False,
        desktop_unscoped_warned=False,
    )
    for k, v in attrs.items():
        setattr(s, k, v)
    return s


class TestGuard:
    def setup_method(self):
        self.ex = KeywordExecutor.__new__(KeywordExecutor)

    def test_unscoped_query_refused_with_rewrite(self):
        sess = _desktop_session()
        out = self.ex._unscoped_locator_guard(
            sess, "Query", ["//control:Paragraph"]
        )
        assert out is not None
        assert out["success"] is False
        hint = out["hints"][0]
        assert hint["type"] == "unscoped_desktop_locator"
        assert "/app:*[@Name='<app>']//control:Paragraph" in hint["message"]
        assert "count(" in hint["message"]  # mentions the allowed discovery form

    def test_rewrite_uses_known_app_name(self, monkeypatch):
        sess = _desktop_session(desktop_aut_pid=4242)
        monkeypatch.setattr(
            KeywordExecutor, "_infer_session_app_name",
            staticmethod(lambda s: "soffice"),
        )
        out = self.ex._unscoped_locator_guard(
            sess, "Query", ["//control:Paragraph"]
        )
        assert "/app:*[@Name='soffice']//control:Paragraph" in out["hints"][0]["message"]

    def test_scoped_query_proceeds(self):
        sess = _desktop_session()
        assert self.ex._unscoped_locator_guard(
            sess, "Query", ["/app:*[@Name='soffice']//control:Paragraph"]
        ) is None

    def test_count_discovery_proceeds(self):
        sess = _desktop_session()
        assert self.ex._unscoped_locator_guard(
            sess, "Query", ["count(//control:Paragraph)"]
        ) is None

    def test_non_query_keyword_proceeds(self):
        sess = _desktop_session()
        assert self.ex._unscoped_locator_guard(
            sess, "Pointer Click", ["//control:Button"]
        ) is None

    def test_env_optout_downgrades_to_pending_warning(self, monkeypatch):
        monkeypatch.setenv("ROBOTMCP_PLATYNUI_ALLOW_UNSCOPED", "1")
        sess = _desktop_session()
        out = self.ex._unscoped_locator_guard(
            sess, "Query", ["//control:Paragraph"]
        )
        assert out is None  # proceeds
        assert sess._pending_unscoped_hint["type"] == "unscoped_desktop_locator_allowed"

    def test_session_optout_downgrades(self):
        sess = _desktop_session(platynui_allow_unscoped=True)
        out = self.ex._unscoped_locator_guard(
            sess, "Query", ["//control:Paragraph"]
        )
        assert out is None
        assert getattr(sess, "_pending_unscoped_hint", None) is not None

    def test_optout_warning_is_one_time(self):
        sess = _desktop_session(platynui_allow_unscoped=True,
                                desktop_unscoped_warned=True)
        out = self.ex._unscoped_locator_guard(
            sess, "Query", ["//control:Paragraph"]
        )
        assert out is None
        # Already warned -> no new pending hint queued.
        assert getattr(sess, "_pending_unscoped_hint", None) is None


@pytest.mark.asyncio
class TestEndToEndGuard:
    async def test_report_shape_refused_via_engine(self):
        # The exact report shape: a desktop session running the unscoped
        # Query is refused instantly (no native walk) through execute_step.
        from robotmcp.components.execution.execution_coordinator import (
            ExecutionCoordinator,
        )

        engine = ExecutionCoordinator()
        sid = "guardrail-e2e"
        sess = engine.session_manager.get_or_create_session(sid)
        sess.configure_from_scenario(
            "Open LibreOffice Writer desktop application", context="desktop"
        )
        assert sess.is_desktop_session() is True
        result = await engine.execute_step(
            "Query", ["//control:Paragraph"], sid, use_context=True
        )
        assert result["success"] is False
        assert any(
            h.get("type") == "unscoped_desktop_locator"
            for h in (result.get("hints") or [])
        )

    async def test_web_session_not_guarded(self):
        from robotmcp.components.execution.execution_coordinator import (
            ExecutionCoordinator,
        )

        engine = ExecutionCoordinator()
        sid = "guardrail-web"
        sess = engine.session_manager.get_or_create_session(sid)
        # Default web session: the guardrail must not refuse //-selectors.
        assert sess.is_desktop_session() is False
        guard = KeywordExecutor.__new__(KeywordExecutor)._unscoped_locator_guard(
            sess, "Query", ["//control:Paragraph"]
        )
        # Non-desktop: guard only fires for desktop sessions. The executor
        # gates the whole desktop pre-flight block on is_desktop_session, so
        # the guard is never called for web; calling it directly here, the
        # desktop check inside short-circuits.
        assert guard is None
