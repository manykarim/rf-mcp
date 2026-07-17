"""Unit tests for the desktop-mcp-workflow-correctness change.

Covers the maintainer-report findings (docs/gnome-calculator-mcp-maintainer-
report.md) that were independently reproduced and fixed:

- D0/D2 recommender PlatynUI-first for desktop + alias
- D1 context="desktop" routing across all sites
- D4 find_keywords PlatynUI discovery + literal-catalog guidance

Batch arg compat (D5) lives in
``tests/unit/domains/batch_execution/test_aggregates.py``.
"""

from __future__ import annotations

import asyncio
import importlib.util as _ilu
import sys

import pytest

# PlatynUI keyword-catalog discovery needs the native library installed; skip
# these when it is absent (CI does not install the external PlatynUI wheels).
requires_platynui = pytest.mark.skipif(
    _ilu.find_spec("platynui_native") is None or _ilu.find_spec("PlatynUI") is None,
    reason="PlatynUI (platynui_native) not installed — desktop keyword catalog unavailable",
)

# Unix executable resolution (resolve_executable("sh") -> /..., /bin/sh) is
# POSIX; Windows has no such paths. macOS is POSIX and passes.
_posix_only = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX executable-path resolution (sh, /bin/sh) — not on Windows",
)

from robotmcp.components.library_recommender import LibraryRecommender


GNOME_SCENARIO = (
    "Open GNOME Calculator desktop application, perform several calculations "
    "and assert each entered value and the result"
)


# ── D0/D2: recommender prefers PlatynUI for desktop ──────────────────


class TestRecommenderDesktop:
    def _names(self, result):
        return [r["library_name"] for r in result["recommendations"]]

    def test_platynui_registered(self):
        r = LibraryRecommender()
        r._initialize_registry()
        assert "PlatynUI.BareMetal" in r.libraries_registry

    def test_platynui_has_desktop_category(self):
        r = LibraryRecommender()
        r._initialize_registry()
        cats = r.libraries_registry["PlatynUI.BareMetal"].categories
        assert "desktop" in cats

    def test_alias_resolves(self):
        assert LibraryRecommender.resolve_library_alias("PlatynUI") == "PlatynUI.BareMetal"
        assert LibraryRecommender.resolve_library_alias("platynui") == "PlatynUI.BareMetal"

    def test_alias_passthrough_for_unknown(self):
        assert LibraryRecommender.resolve_library_alias("Browser") == "Browser"
        assert LibraryRecommender.resolve_library_alias(None) is None

    def test_desktop_scenario_leads_with_platynui(self):
        r = LibraryRecommender()
        names = self._names(r.recommend_libraries(GNOME_SCENARIO, context="desktop"))
        assert names, "expected at least one recommendation"
        assert names[0] == "PlatynUI.BareMetal"

    def test_desktop_scenario_not_led_by_appium(self):
        r = LibraryRecommender()
        names = self._names(r.recommend_libraries(GNOME_SCENARIO, context="desktop"))
        assert "AppiumLibrary" not in names or names.index("PlatynUI.BareMetal") < names.index(
            "AppiumLibrary"
        )

    def test_web_context_unchanged(self):
        r = LibraryRecommender()
        names = self._names(
            r.recommend_libraries("Open the login page in a browser and click submit", context="web")
        )
        assert names[0] == "Browser"
        assert "PlatynUI.BareMetal" not in names

    def test_mobile_context_unchanged(self):
        r = LibraryRecommender()
        names = self._names(
            r.recommend_libraries("Tap the button in the android mobile app", context="mobile")
        )
        assert names[0] == "AppiumLibrary"
        assert "PlatynUI.BareMetal" not in names

    def test_api_context_unchanged(self):
        r = LibraryRecommender()
        names = self._names(
            r.recommend_libraries("Send a GET request to the REST endpoint", context="api")
        )
        assert "PlatynUI.BareMetal" not in names


# ── D1: context="desktop" routing at all sites ──────────────────────


class TestDesktopContextRouting:
    def _analyze(self, context):
        from robotmcp.components.nlp_processor import NaturalLanguageProcessor

        nlp = NaturalLanguageProcessor()
        return asyncio.run(nlp.analyze_scenario(GNOME_SCENARIO, context=context))

    # Site (d): nlp_processor.analyze_scenario
    def test_nlp_desktop_session_type(self):
        result = self._analyze("desktop")
        assert result["analysis"].get("detected_session_type") == "desktop_testing"

    def test_nlp_desktop_capabilities_have_platynui_and_process(self):
        caps = set(self._analyze("desktop")["scenario"]["required_capabilities"])
        assert "PlatynUI.BareMetal" in caps
        assert "Process" in caps

    def test_nlp_desktop_capabilities_exclude_appium(self):
        caps = set(self._analyze("desktop")["scenario"]["required_capabilities"])
        assert "AppiumLibrary" not in caps

    def test_nlp_web_context_unchanged(self):
        from robotmcp.components.nlp_processor import NaturalLanguageProcessor

        nlp = NaturalLanguageProcessor()
        result = asyncio.run(
            nlp.analyze_scenario("Open login page in browser and click submit", context="web")
        )
        assert result["analysis"].get("detected_session_type") == "web_automation"
        caps = set(result["scenario"]["required_capabilities"])
        assert "PlatynUI.BareMetal" not in caps

    # Site (b): detect_platform_from_scenario
    def test_platform_detector_desktop(self):
        from robotmcp.components.execution.session_manager import SessionManager
        from robotmcp.models.session_models import PlatformType

        sm = SessionManager()
        assert sm.detect_platform_from_scenario(GNOME_SCENARIO) == PlatformType.DESKTOP

    def test_platform_detector_app_token_not_mobile_for_desktop(self):
        # "calculator application" contains the "app" token but is desktop.
        from robotmcp.components.execution.session_manager import SessionManager
        from robotmcp.models.session_models import PlatformType

        sm = SessionManager()
        assert sm.detect_platform_from_scenario(
            "Launch the calculator application on the desktop"
        ) == PlatformType.DESKTOP

    def test_platform_detector_mobile_unchanged(self):
        from robotmcp.components.execution.session_manager import SessionManager
        from robotmcp.models.session_models import PlatformType

        sm = SessionManager()
        assert sm.detect_platform_from_scenario(
            "Tap the android app button and swipe with a gesture"
        ) == PlatformType.MOBILE

    def test_platform_detector_web_unchanged(self):
        from robotmcp.components.execution.session_manager import SessionManager
        from robotmcp.models.session_models import PlatformType

        sm = SessionManager()
        assert sm.detect_platform_from_scenario(
            "Open the website in chrome and click the link"
        ) == PlatformType.WEB

    # Site (c): configure_from_scenario
    def test_configure_from_scenario_desktop_context(self):
        from robotmcp.models.session_models import (
            ExecutionSession,
            PlatformType,
            SessionType,
        )

        s = ExecutionSession(session_id="cfg-desktop")
        s.configure_from_scenario(GNOME_SCENARIO, context="desktop")
        assert s.session_type == SessionType.DESKTOP_TESTING
        assert s.platform_type == PlatformType.DESKTOP

    def test_configure_desktop_search_order_leads_platynui(self):
        from robotmcp.models.session_models import ExecutionSession

        s = ExecutionSession(session_id="cfg-desktop-order")
        s.configure_from_scenario(GNOME_SCENARIO, context="desktop")
        assert s.search_order
        assert s.search_order[0] == "PlatynUI.BareMetal"
        # PlatynUI ahead of AppiumLibrary (Appium absent from desktop order)
        assert "AppiumLibrary" not in s.search_order

    def test_configure_without_context_unchanged(self):
        # No explicit context: web-worded scenario stays web.
        from robotmcp.models.session_models import ExecutionSession, SessionType

        s = ExecutionSession(session_id="cfg-web")
        s.configure_from_scenario("Open the website in a browser and click submit")
        assert s.session_type == SessionType.WEB_AUTOMATION


# ── D3: desktop state inspection routes through ui_tree ──────────────


@pytest.mark.asyncio
class TestDesktopStateInspection:
    async def _call_state(self, monkeypatch, session, sections):
        from robotmcp.models.session_models import ExecutionSession  # noqa: F401
        import robotmcp.server as server
        import robotmcp.components.execution.ui_tree_service as ui_tree_service

        monkeypatch.setattr(
            server.execution_engine.session_manager,
            "get_session",
            lambda sid: session,
        )
        monkeypatch.setattr(server, "_track_tool_result", lambda *a, **kw: None)

        async def _fake_ui_tree(sess, app_filters=None):
            return {
                "success": True,
                "application_count": 0,
                "applications": [],
                "hint": "No desktop application is open; launch one first.",
            }

        monkeypatch.setattr(ui_tree_service, "get_ui_tree", _fake_ui_tree)

        return await server.get_session_state.fn(session_id=session.session_id, sections=sections)

    def _desktop_session(self, sid="d-state"):
        from robotmcp.models.session_models import ExecutionSession

        s = ExecutionSession(session_id=sid)
        s.configure_from_scenario(GNOME_SCENARIO, context="desktop")
        return s

    async def test_desktop_page_source_is_stub_not_mobile_error(self, monkeypatch):
        session = self._desktop_session()
        result = await self._call_state(monkeypatch, session, ["page_source"])
        ps = result["sections"]["page_source"]
        assert ps["source"] == "desktop"
        assert ps["page_source"] is None
        assert "no desktop application" in ps["message"].lower()
        # Must NOT be the mobile-source error
        assert "mobile source" not in ps["message"].lower()

    async def test_desktop_auto_includes_ui_tree(self, monkeypatch):
        session = self._desktop_session("d-uitree")
        result = await self._call_state(monkeypatch, session, ["page_source"])
        # ui_tree auto-added even though only page_source was requested
        assert "ui_tree" in result["sections"]
        assert result["sections"]["ui_tree"]["success"] is True

    async def test_web_session_page_source_unchanged(self, monkeypatch):
        # A non-desktop session must still go through the normal page_source path.
        from robotmcp.models.session_models import ExecutionSession
        import robotmcp.server as server

        session = ExecutionSession(session_id="w-state")
        session.configure_from_scenario("Open the website in a browser")

        monkeypatch.setattr(
            server.execution_engine.session_manager,
            "get_session",
            lambda sid: session,
        )
        monkeypatch.setattr(server, "_track_tool_result", lambda *a, **kw: None)

        captured = {}

        async def _fake_payload(**kwargs):
            captured["called"] = True
            return {"success": True, "source": "local", "page_source": "<html></html>"}

        monkeypatch.setattr(server, "_get_page_source_payload", _fake_payload)

        result = await server.get_session_state.fn(
            session_id="w-state", sections=["page_source"]
        )
        assert captured.get("called") is True
        assert result["sections"]["page_source"]["source"] == "local"
        # web session must NOT auto-include ui_tree
        assert "ui_tree" not in result["sections"]


# ── D4: find_keywords surfaces PlatynUI desktop keywords ────────────


@requires_platynui
@pytest.mark.asyncio
class TestPlatynUIKeywordDiscovery:
    async def _find(self, **kwargs):
        from robotmcp.server import find_keywords

        kwargs.setdefault("query", "")
        kwargs.setdefault("strategy", "catalog")
        return await find_keywords.fn(**kwargs)

    async def test_library_listing_returns_keywords(self):
        r = await self._find(library_name="PlatynUI.BareMetal")
        assert r["success"] is True
        assert r["match_count"] >= 20
        names = {c["name"] for c in r["results"]}
        assert "Pointer Click" in names
        assert "Keyboard Type" in names

    async def test_alias_listing_resolves(self):
        r = await self._find(library_name="PlatynUI")
        assert r["library"] == "PlatynUI.BareMetal"
        assert r["match_count"] >= 20

    async def test_single_term_intent_query_surfaces_desktop_keyword(self):
        r = await self._find(query="click", library_name="PlatynUI.BareMetal")
        names = {c["name"] for c in r["results"]}
        assert any("Click" in n for n in names)

    async def test_nl_catalog_query_does_not_strand(self):
        r = await self._find(
            query="get window find element ui tree", library_name="PlatynUI.BareMetal"
        )
        assert r["match_count"] == 0
        assert "hint" in r
        assert "literal substring" in r["hint"].lower()


# ── D6: executable resolution hook + Evaluate guidance ──────────────


class TestDesktopExecResolution:
    @_posix_only
    def test_resolve_server_resolvable_tool(self):
        from robotmcp.components.execution.desktop_launch_env import resolve_executable

        resolved = resolve_executable("sh")
        assert resolved is not None
        assert resolved.startswith("/")

    @_posix_only
    def test_resolve_absolute_existing_path(self):
        from robotmcp.components.execution.desktop_launch_env import resolve_executable

        assert resolve_executable("/bin/sh") == "/bin/sh"

    def test_resolve_unresolvable_returns_none(self):
        from robotmcp.components.execution.desktop_launch_env import resolve_executable

        assert resolve_executable("definitely-not-a-real-tool-xyz-123") is None

    def test_resolve_uses_server_path_not_shell(self):
        # Resolution must use the provided (server) PATH, not a login shell.
        from robotmcp.components.execution.desktop_launch_env import resolve_executable

        # With an empty PATH the unqualified name is unresolvable.
        assert resolve_executable("sh", parent_env={"PATH": ""}) is None

    def test_effective_path_surfaced(self):
        from robotmcp.components.execution.desktop_launch_env import get_effective_path

        assert isinstance(get_effective_path(), str)
        assert get_effective_path({"PATH": "/x:/y"}) == "/x:/y"

    @_posix_only
    def test_executor_resolves_desktop_process_launch(self):
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        ke = KeywordExecutor.__new__(KeywordExecutor)

        class _S:
            def is_desktop_session(self):
                return True

        args = ke._maybe_resolve_desktop_executable(_S(), "Start Process", ["sh", "-c", "true"])
        assert args[0].startswith("/")
        assert args[1:] == ["-c", "true"]

    def test_executor_skips_non_desktop(self):
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        ke = KeywordExecutor.__new__(KeywordExecutor)

        class _S:
            def is_desktop_session(self):
                return False

        args = ke._maybe_resolve_desktop_executable(_S(), "Start Process", ["sh"])
        assert args == ["sh"]

    def test_executor_skips_non_process_keyword(self):
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        ke = KeywordExecutor.__new__(KeywordExecutor)

        class _S:
            def is_desktop_session(self):
                return True

        args = ke._maybe_resolve_desktop_executable(_S(), "Pointer Click", ["sh"])
        assert args == ["sh"]


class TestEvaluateGuidance:
    def test_platynui_guidance_documents_evaluate_limit(self):
        from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter

        g = RobotFrameworkNativeConverter().get_platynui_locator_guidance()
        assert "process_and_recovery" in g
        rules = " ".join(g["process_and_recovery"]["rules"]).lower()
        assert "expression" in rules
        assert "run process" in rules


# ── D7: build_test_suite isolates pre-start steps (opt-in) ──────────


@pytest.mark.asyncio
class TestStepwiseSuiteIsolation:
    def _session_with_prestart_and_test(self):
        from unittest.mock import MagicMock
        from robotmcp.models.execution_models import ExecutionStep
        from robotmcp.models.session_models import ExecutionSession

        sess = ExecutionSession(session_id="ps1")
        # Pre-start exploratory step (recorded before any start_test → flat list)
        pre = ExecutionStep(step_id="p1", keyword="Log", arguments=["exploring"])
        pre.mark_success()
        sess.steps.append(pre)

        # Now a real test with a genuine interaction step
        sess.test_registry.start_test("Calc Test", tags=["desktop"])
        s1 = ExecutionStep(step_id="s1", keyword="Pointer Click", arguments=["/app:*//control:Button[@Name='1']"])
        s1.mark_success()
        sess.test_registry.tests["Calc Test"].steps.append(s1)
        s2 = ExecutionStep(step_id="s2", keyword="Get Attribute", arguments=["/app:*//control:Text", "Name"])
        s2.mark_success()
        sess.test_registry.tests["Calc Test"].steps.append(s2)
        sess.test_registry.end_test(status="pass")

        engine = MagicMock()
        engine.sessions = {"ps1": sess}
        return sess, engine

    async def test_pre_start_excluded_by_default(self):
        from robotmcp.components.test_builder import TestBuilder

        _sess, engine = self._session_with_prestart_and_test()
        builder = TestBuilder(execution_engine=engine)
        result = await builder.build_suite(session_id="ps1", test_name="Calc Suite")

        assert result["success"] is True
        assert result["excluded_pre_start_count"] == 1
        assert result["pre_start_summary"]
        # The pre-start Log must NOT appear in the generated .robot body
        assert "exploring" not in result["rf_text"]
        # The real interaction must be present
        assert "Pointer Click" in result["rf_text"]

    async def test_opt_in_includes_pre_start_for_template(self):
        # Adoption (prior behavior) applies to a template test with data rows.
        from unittest.mock import MagicMock
        from robotmcp.components.test_builder import TestBuilder
        from robotmcp.models.execution_models import ExecutionStep
        from robotmcp.models.session_models import ExecutionSession

        sess = ExecutionSession(session_id="ps2")
        pre = ExecutionStep(step_id="p1", keyword="Log", arguments=["adopted"])
        pre.mark_success()
        sess.steps.append(pre)
        # A template test with a data row but no steps of its own
        sess.test_registry.start_test("DD Test", template="Do Thing")
        sess.test_registry.tests["DD Test"].data_rows = [["a"]]
        sess.test_registry.end_test(status="pass")

        engine = MagicMock()
        engine.sessions = {"ps2": sess}
        builder = TestBuilder(execution_engine=engine)
        result = await builder.build_suite(session_id="ps2", include_pre_start=True)
        assert result["success"] is True
        # With opt-in, the pre-start step was adopted (excluded count stays 0)
        assert result["excluded_pre_start_count"] == 0

    async def test_real_interactions_render_not_placeholder(self):
        from robotmcp.components.test_builder import TestBuilder

        _sess, engine = self._session_with_prestart_and_test()
        builder = TestBuilder(execution_engine=engine)
        result = await builder.build_suite(session_id="ps1")
        # The generated suite reflects real PlatynUI interaction + assertion,
        # not a Log-only placeholder.
        assert "Pointer Click" in result["rf_text"]
        assert "Get Attribute" in result["rf_text"]

    async def test_start_test_message_explains_exclusion(self, monkeypatch):
        # The manage_session(start_test) warning explains default exclusion.
        import robotmcp.server as server
        from robotmcp.models.execution_models import ExecutionStep
        from robotmcp.models.session_models import ExecutionSession

        sess = ExecutionSession(session_id="ps3")
        pre = ExecutionStep(step_id="p1", keyword="Log", arguments=["x"])
        pre.mark_success()
        sess.steps.append(pre)

        monkeypatch.setattr(
            server.execution_engine.session_manager,
            "get_or_create_session",
            lambda sid: sess,
        )
        monkeypatch.setattr(
            server.execution_engine.session_manager,
            "get_session",
            lambda sid: sess,
        )
        result = await server.manage_session.fn(
            action="start_test", session_id="ps3", test_name="T1"
        )
        assert "warning" in result
        assert "excluded" in result["warning"].lower()
        assert "include_pre_start=True" in result["warning"]
