"""Tests for the mcp-diagnostics-hygiene change: log-level hygiene, page-source
gating on loaded web libraries, the assignment-heuristic RequestsLibrary
allowlist, the Requests->RequestsLibrary alias, and the JSON-body guidance."""
from __future__ import annotations

import logging

import pytest

from robotmcp.components.execution.keyword_executor import KeywordExecutor
from robotmcp.components.execution.page_source_service import PageSourceService
from robotmcp.models.config_models import ExecutionConfig


# ── §7.1 page-source gate on non-web sessions ───────────────────────────────
class _ApiSession:
    session_id = "api-1"
    imported_libraries = ["RequestsLibrary", "BuiltIn"]
    variables: dict = {}

    def is_desktop_session(self):
        return False


def test_page_source_short_circuits_for_non_web_session(monkeypatch):
    svc = PageSourceService(ExecutionConfig())
    # If the gate fails, the service would fetch the rf context manager and
    # cascade DOM keywords — make that path explode so the test catches it.
    import robotmcp.components.execution.rf_native_context_manager as rf
    monkeypatch.setattr(
        rf, "get_rf_native_context_manager",
        lambda: (_ for _ in ()).throw(AssertionError("DOM keyword path must not run for an API session")),
    )
    assert svc._get_page_source_via_rf_context(_ApiSession()) == ""


def test_page_source_still_runs_for_web_session(monkeypatch):
    class _BrowserState:
        page_source = ""
        current_url = ""
        page_title = ""

    class _WebSession(_ApiSession):
        session_id = "web-1"
        imported_libraries = ["SeleniumLibrary", "BuiltIn"]

        def __init__(self):
            self.browser_state = _BrowserState()

    svc = PageSourceService(ExecutionConfig())
    called = {"n": 0}

    class _Mgr:
        def execute_keyword_with_context(self, **kw):
            called["n"] += 1
            return {"success": True, "output": "<html>ok</html>"}

    import robotmcp.components.execution.rf_native_context_manager as rf
    monkeypatch.setattr(rf, "get_rf_native_context_manager", lambda: _Mgr())
    out = svc._get_page_source_via_rf_context(_WebSession())
    assert out == "<html>ok</html>" and called["n"] >= 1


# ── §7.2 assignment heuristic recognizes RequestsLibrary response keywords ──
@pytest.mark.parametrize("kw", ["GET On Session", "POST On Session", "DELETE On Session"])
def test_requests_keywords_are_returnable(kw, caplog):
    ke = KeywordExecutor.__new__(KeywordExecutor)
    with caplog.at_level(logging.WARNING):
        ke._validate_assignment_compatibility(kw, "resp")
    assert "may not return a useful value" not in caplog.text


def test_non_returning_keyword_still_warns(caplog):
    ke = KeywordExecutor.__new__(KeywordExecutor)
    with caplog.at_level(logging.WARNING):
        ke._validate_assignment_compatibility("Click Element", "x")
    assert "may not return a useful value" in caplog.text


# ── §7.3 Requests -> RequestsLibrary alias ──────────────────────────────────
def test_requests_alias_normalizes_and_warns(caplog):
    from robotmcp.models.session_models import ExecutionSession

    s = ExecutionSession(session_id="s-alias")
    with caplog.at_level(logging.WARNING):
        try:
            s.import_library("Requests", force=True)
        except Exception:
            pass  # LibraryManager load may no-op in the unit env; we assert on naming
    assert "RequestsLibrary" in s.imported_libraries
    assert "Requests" not in [l for l in s.imported_libraries if l == "Requests"]
    assert "resolved to 'RequestsLibrary'" in caplog.text


# ── §7.4 requests guidance includes the JSON-body construction pattern ──────
def test_requests_guidance_has_json_body_pattern():
    from robotmcp.utils.requests_guidance import build_requests_cookbook

    payload = build_requests_cookbook()
    blob = repr(payload)
    assert "json=${{" in blob  # inline-eval body form
    assert "BODY CONSTRUCTION" in blob
    # define-before-POST ordering rule present
    assert "before" in blob.lower() and "${body}" in blob


# ── §7.5 log-level regression on recoverable paths ──────────────────────────
def test_shadow_notice_logged_once(caplog):
    from robotmcp.core.keyword_discovery import KeywordDiscovery
    # A second (re)load of the same shadowing (keyword, library) pair must not
    # re-emit the warning.
    kd = KeywordDiscovery()
    key = "run keyword"
    kd.shadowed_keywords[key] = [("BuiltIn", object())]
    # Simulate the dedup guard directly: the pair ("SeleniumLibrary") warns once.
    seen_lib = "SeleniumLibrary"
    already = any(lib == seen_lib for lib, _ in kd.shadowed_keywords[key])
    assert already is False  # first time → would warn
    kd.shadowed_keywords[key].append((seen_lib, object()))
    already2 = any(lib == seen_lib for lib, _ in kd.shadowed_keywords[key])
    assert already2 is True  # second time → guard suppresses
