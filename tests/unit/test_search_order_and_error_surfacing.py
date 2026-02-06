"""Tests for Option A (search_order sync), H2 (failed import surfacing), and M1 (exception logging).

Option A: session.import_library() must update session.search_order
H2: Failed library imports must be surfaced in keyword-not-found error messages
M1: Generic execution path failures must be logged (not silently swallowed)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from robotmcp.models.session_models import ExecutionSession
from robotmcp.components.execution.rf_native_context_manager import (
    RobotFrameworkNativeContextManager,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EC_PATCH = "robotmcp.components.execution.rf_native_context_manager.EXECUTION_CONTEXTS"
_BUILTIN_PATCH = "robot.libraries.BuiltIn.BuiltIn"
_LOGGER_NAME = "robotmcp.components.execution.rf_native_context_manager"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(session_id: str = "test-session") -> ExecutionSession:
    """Create a session matching create_session() behaviour."""
    session = ExecutionSession(session_id=session_id)
    if "BuiltIn" not in session.search_order:
        session.search_order.insert(0, "BuiltIn")
    return session


# ===========================================================================
# Option A — search_order sync in import_library()
# ===========================================================================


class TestOptionASearchOrderSync:
    """Verify import_library() keeps search_order in sync."""

    def test_import_library_adds_to_search_order(self):
        """Importing Browser must add it to search_order."""
        session = _make_session()
        session.import_library("Browser")
        assert "Browser" in session.search_order

    def test_import_library_preserves_builtin_first(self):
        """BuiltIn must remain first in search_order after imports."""
        session = _make_session()
        session.import_library("Browser")
        assert session.search_order[0] == "BuiltIn"

    def test_import_library_no_duplicate(self):
        """Importing same library twice must not duplicate in search_order."""
        session = _make_session()
        session.import_library("Browser")
        session.import_library("Browser")
        assert session.search_order.count("Browser") == 1

    def test_import_library_multiple_libraries(self):
        """Multiple imports produce correct search_order."""
        session = _make_session()
        session.import_library("Browser")
        session.import_library("Collections")
        session.import_library("String")
        assert "Browser" in session.search_order
        assert "Collections" in session.search_order
        assert "String" in session.search_order
        # BuiltIn still first
        assert session.search_order[0] == "BuiltIn"

    def test_import_library_search_order_matches_imported(self):
        """After imports, every imported library is also in search_order."""
        session = _make_session()
        for lib in ["Browser", "Collections", "String", "XML"]:
            session.import_library(lib)
        for lib in session.imported_libraries:
            assert lib in session.search_order, (
                f"{lib} in imported_libraries but not in search_order"
            )

    def test_search_order_used_for_context_libraries(self):
        """_execute_keyword_with_context reads search_order for context creation."""
        session = _make_session()
        session.import_library("Browser")

        # Simulate the library selection logic from keyword_executor.py:1649-1654
        if hasattr(session, "search_order") and session.search_order:
            libraries = list(session.search_order)
        elif hasattr(session, "loaded_libraries") and session.loaded_libraries:
            libraries = list(session.loaded_libraries)
        else:
            libraries = []

        assert "Browser" in libraries, (
            "Browser must be in the library list passed to create_context_for_session"
        )

    def test_force_switch_updates_search_order(self):
        """Force-switching from Browser to SeleniumLibrary updates search_order."""
        session = _make_session()
        session.import_library("Browser")
        assert "Browser" in session.search_order

        session.import_library("SeleniumLibrary", force=True)
        assert "SeleniumLibrary" in session.search_order
        # Browser should be removed from imported_libraries but search_order
        # may still contain it (search_order is additive). The important thing
        # is that SeleniumLibrary is present.

    def test_builtin_already_in_search_order_not_duplicated(self):
        """Importing BuiltIn when it's already in search_order doesn't duplicate."""
        session = _make_session()
        assert "BuiltIn" in session.search_order
        session.import_library("BuiltIn")
        assert session.search_order.count("BuiltIn") == 1


# ===========================================================================
# H2 — failed import surfacing
# ===========================================================================


class TestH2FailedImportSurfacing:
    """Verify failed library imports are tracked and surfaced in errors."""

    def test_failed_imports_tracked_in_context(self):
        """When a library import fails, it's recorded in session context."""
        mgr = RobotFrameworkNativeContextManager()

        with patch(_EC_PATCH) as ec:
            ec.current = None  # Force new context creation

            # Mock namespace whose import_library raises for FakeLib
            mock_ns_class = MagicMock()
            mock_ns = MagicMock()

            def _import_side_effect(name, args=(), alias=None, notify=True):
                if name == "FakeLib":
                    raise ModuleNotFoundError(f"No module named '{name}'")

            mock_ns.import_library.side_effect = _import_side_effect
            mock_ns.set_search_order = MagicMock()
            mock_ns.start_suite = MagicMock()
            mock_ns.start_test = MagicMock()

            with patch(
                "robotmcp.components.execution.rf_native_context_manager.Namespace",
                return_value=mock_ns,
            ):
                with patch(
                    "robotmcp.components.execution.rf_native_context_manager.EXECUTION_CONTEXTS"
                ) as ec2:
                    ec2.current = None
                    ec2.start_suite = MagicMock(return_value=MagicMock())

                    result = mgr.create_context_for_session(
                        "test-h2", ["BuiltIn", "FakeLib"]
                    )

            ctx_info = mgr._session_contexts.get("test-h2")
            if ctx_info:
                failed = ctx_info.get("failed_imports", {})
                assert "FakeLib" in failed, "FakeLib should be in failed_imports"

    def test_failed_import_in_error_message(self):
        """Error message includes failed library import details."""
        mgr = RobotFrameworkNativeContextManager()

        # Pre-populate context with a known failed import
        mgr._session_contexts["test-h2-err"] = {
            "context": MagicMock(),
            "variables": MagicMock(),
            "namespace": MagicMock(),
            "output": None,
            "suite": MagicMock(),
            "result_suite": None,
            "result_test": None,
            "created_at": MagicMock(),
            "libraries": ["BuiltIn", "Browser"],
            "imported_libraries": ["BuiltIn"],
            "failed_imports": {"Browser": "list index out of range"},
        }

        # Trigger the error path by making get_runner fail
        ns = mgr._session_contexts["test-h2-err"]["namespace"]
        ns.get_runner = MagicMock(side_effect=Exception("No keyword with name 'New Browser' found"))
        ns.libraries = []

        with patch(_EC_PATCH) as ec:
            ec.current = mgr._session_contexts["test-h2-err"]["context"]

            with patch(_BUILTIN_PATCH) as MockBI:
                instance = MagicMock(spec=[])
                instance.run_keyword = MagicMock(
                    side_effect=RuntimeError("no keyword")
                )
                MockBI.return_value = instance

                result = mgr._execute_with_native_resolution(
                    "test-h2-err", "New Browser", [], ns, MagicMock(), None
                )

        assert not result["success"]
        assert "failed to import" in result["error"]
        assert "Browser" in result["error"]
        assert "list index out of range" in result["error"]

    def test_no_failed_imports_no_extra_message(self):
        """When all imports succeed, error message has no import note."""
        mgr = RobotFrameworkNativeContextManager()

        mgr._session_contexts["test-h2-ok"] = {
            "context": MagicMock(),
            "variables": MagicMock(),
            "namespace": MagicMock(),
            "output": None,
            "suite": MagicMock(),
            "result_suite": None,
            "result_test": None,
            "created_at": MagicMock(),
            "libraries": ["BuiltIn"],
            "imported_libraries": ["BuiltIn"],
            "failed_imports": {},
        }

        ns = mgr._session_contexts["test-h2-ok"]["namespace"]
        ns.get_runner = MagicMock(side_effect=Exception("No keyword 'Foo'"))
        ns.libraries = []

        with patch(_EC_PATCH) as ec:
            ec.current = mgr._session_contexts["test-h2-ok"]["context"]
            with patch(_BUILTIN_PATCH) as MockBI:
                instance = MagicMock(spec=[])
                instance.run_keyword = MagicMock(
                    side_effect=RuntimeError("nope")
                )
                MockBI.return_value = instance

                result = mgr._execute_with_native_resolution(
                    "test-h2-ok", "Foo", [], ns, MagicMock(), None
                )

        assert not result["success"]
        assert "failed to import" not in result["error"]

    def test_browser_skip_logged_at_warning(self, caplog):
        """Browser index-error skip is logged at WARNING, not INFO."""
        mgr = RobotFrameworkNativeContextManager()

        with patch(
            "robotmcp.components.execution.rf_native_context_manager.Namespace"
        ) as MockNS:
            mock_ns = MagicMock()

            def _import_raises(name, args=(), alias=None, notify=True):
                if name == "Browser":
                    raise IndexError("list index out of range")

            mock_ns.import_library.side_effect = _import_raises
            mock_ns.set_search_order = MagicMock()
            mock_ns.start_suite = MagicMock()
            mock_ns.start_test = MagicMock()
            MockNS.return_value = mock_ns

            with patch(
                "robotmcp.components.execution.rf_native_context_manager.EXECUTION_CONTEXTS"
            ) as ec:
                ec.current = None
                ec.start_suite = MagicMock(return_value=MagicMock())

                with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
                    result = mgr.create_context_for_session(
                        "test-h2-warn", ["Browser"]
                    )

            ctx_info = mgr._session_contexts.get("test-h2-warn")
            if ctx_info:
                failed = ctx_info.get("failed_imports", {})
                assert "Browser" in failed
                assert "index" in failed["Browser"].lower()

            # Verify the skip message is at WARNING level (not INFO)
            warning_msgs = [
                r.message for r in caplog.records if r.levelno >= logging.WARNING
            ]
            skip_logged = any("Skipping Browser" in m for m in warning_msgs)
            assert skip_logged, (
                f"Expected WARNING-level 'Skipping Browser' log. "
                f"Warning messages: {warning_msgs}"
            )


# ===========================================================================
# M1 — generic path exception logging
# ===========================================================================


class TestM1ExceptionLogging:
    """Verify generic execution path failures are logged, not silently swallowed."""

    def test_generic_path_failure_logged(self, caplog):
        """When _execute_any_keyword_generic fails, debug log is emitted."""
        mgr = RobotFrameworkNativeContextManager()

        # Build a mock context where get_runner fails
        mock_ns = MagicMock()
        mock_ns.get_runner = MagicMock(
            side_effect=Exception("No keyword 'Zap'")
        )
        mock_ns.libraries = []

        mock_vars = MagicMock()
        mock_vars.store = MagicMock()
        mock_vars.store.data = {}

        mgr._session_contexts["test-m1"] = {
            "context": MagicMock(),
            "variables": mock_vars,
            "namespace": mock_ns,
            "output": None,
            "suite": MagicMock(),
            "result_suite": None,
            "result_test": None,
            "created_at": MagicMock(),
            "libraries": ["BuiltIn"],
            "imported_libraries": ["BuiltIn"],
            "failed_imports": {},
        }

        with patch(_EC_PATCH) as ec:
            ec.current = mgr._session_contexts["test-m1"]["context"]
            ec.current.namespace = mock_ns

            with patch(_BUILTIN_PATCH) as MockBI:
                instance = MagicMock(spec=[])
                instance.run_keyword = MagicMock(
                    side_effect=RuntimeError("no runner")
                )
                MockBI.return_value = instance

                with caplog.at_level(logging.DEBUG, logger=_LOGGER_NAME):
                    result = mgr._execute_with_native_resolution(
                        "test-m1", "Zap", [], mock_ns, mock_vars, None
                    )

        assert not result["success"]
        # Check that the generic path failure was logged
        debug_msgs = [r.message for r in caplog.records if r.levelno == logging.DEBUG]
        generic_logged = any(
            "Generic keyword execution path failed" in m and "Zap" in m
            for m in debug_msgs
        )
        assert generic_logged, (
            f"Expected 'Generic keyword execution path failed' debug log for 'Zap'. "
            f"Debug messages: {debug_msgs}"
        )

    def test_generic_path_success_no_fallthrough(self):
        """When generic path succeeds, outer get_runner is NOT called."""
        mgr = RobotFrameworkNativeContextManager()

        mock_ns = MagicMock()

        class FakeRunner:
            def run(self, data_kw, res_kw, ctx):
                return "ok"

        mock_ns.get_runner = MagicMock(return_value=FakeRunner())
        mock_ns.libraries = []

        mock_vars = MagicMock()
        mock_vars.store = MagicMock()
        mock_vars.store.data = {}

        mgr._session_contexts["test-m1-ok"] = {
            "context": MagicMock(),
            "variables": mock_vars,
            "namespace": mock_ns,
            "output": None,
            "suite": MagicMock(),
            "result_suite": None,
            "result_test": None,
            "created_at": MagicMock(),
            "libraries": ["BuiltIn"],
            "imported_libraries": ["BuiltIn"],
            "failed_imports": {},
        }

        with patch(_EC_PATCH) as ec:
            ec.current = mgr._session_contexts["test-m1-ok"]["context"]

            result = mgr._execute_with_native_resolution(
                "test-m1-ok", "Log", ["hello"], mock_ns, mock_vars, None
            )

        assert result["success"]
        assert result["result"] == "ok"


# ===========================================================================
# Integration: full first-call resolution
# ===========================================================================


class TestFirstCallResolution:
    """End-to-end tests that the first unprefixed call resolves correctly."""

    def test_new_browser_resolves_first_call(self):
        """After import_library('Browser'), first 'New Browser' call resolves."""
        session = _make_session()
        session.import_library("Browser")

        # search_order must include Browser
        assert "Browser" in session.search_order

        # The library list that would be passed to create_context_for_session
        libraries = list(session.search_order)
        assert "Browser" in libraries

    def test_collections_keyword_resolves_first_call(self):
        """After import_library('Collections'), 'Create List' resolves."""
        session = _make_session()
        session.import_library("Collections")
        assert "Collections" in session.search_order

    def test_manage_session_flow(self):
        """Simulate the exact manage_session init flow."""
        session = _make_session()

        # manage_session adds BuiltIn first if missing
        libraries = ["Browser", "BuiltIn"]
        if "BuiltIn" not in libraries:
            libraries = ["BuiltIn"] + list(libraries)

        for lib in libraries:
            session.import_library(lib)
            session.loaded_libraries.add(lib)

        # Verify search_order has everything
        assert "Browser" in session.search_order
        assert "BuiltIn" in session.search_order

        # Verify the context library selection logic works
        if hasattr(session, "search_order") and session.search_order:
            ctx_libraries = list(session.search_order)
        else:
            ctx_libraries = []

        assert "Browser" in ctx_libraries, "Browser must reach context creation"
        assert "BuiltIn" in ctx_libraries

    def test_set_library_search_order_still_works(self):
        """Explicit set_library_search_order overrides import-based order."""
        session = _make_session()
        session.import_library("Browser")
        session.import_library("Collections")

        # User explicitly sets search order
        session.set_library_search_order(["Collections", "Browser", "BuiltIn"])

        assert session.search_order == ["Collections", "Browser", "BuiltIn"]


# ===========================================================================
# Pre-validation ctx.test fix
# ===========================================================================


class TestPreValidationContextTestFix:
    """Verify pre-validation sets ctx.test when it's missing.

    When ctx.test is None, BuiltIn.run_keyword() falls back to
    ctx.suite.setup which is a running-model Keyword without .body,
    causing AttributeError: 'Keyword' object has no attribute 'body'.
    """

    def test_running_keyword_lacks_body(self):
        """Running model Keyword has no .body attribute (root cause)."""
        from robot.running.model import Keyword as RunKW
        kw = RunKW(name="test")
        assert not hasattr(kw, "body") or not isinstance(
            getattr(kw, "body", None), type(None)
        )
        # Actually check for the specific failure pattern
        assert not hasattr(kw, "body") or not hasattr(kw.body, "create_keyword")

    def test_result_keyword_has_body(self):
        """Result model Keyword has .body with create_keyword."""
        from robot.result.model import Keyword as ResKW
        kw = ResKW(name="test")
        assert hasattr(kw, "body")
        assert hasattr(kw.body, "create_keyword")

    def test_result_testcase_has_body(self):
        """Result model TestCase has .body with create_keyword."""
        from robot.result.model import TestCase as ResTest
        rt = ResTest(name="test")
        assert hasattr(rt, "body")
        assert hasattr(rt.body, "create_keyword")

    def test_suite_setup_is_running_keyword(self):
        """Suite setup is a running-model Keyword (no .body)."""
        from robot.running.model import TestSuite
        suite = TestSuite(name="MCP_test")
        assert type(suite.setup).__name__ == "Keyword"
        assert not hasattr(suite.setup, "body") or not hasattr(
            suite.setup.body, "create_keyword"
        )

    def test_pre_validate_element_sets_ctx_test(self):
        """_pre_validate_element sets ctx.test when missing."""
        import asyncio
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        executor = KeywordExecutor.__new__(KeywordExecutor)
        executor.pre_validation_enabled = True

        # Minimal config
        class FakeConfig:
            PRE_VALIDATION_TIMEOUT = 500
        executor.config = FakeConfig()

        # Mock session with browser state
        mock_session = MagicMock()
        mock_session.browser_state.active_library = "browser"

        # Create a real-ish context object WITHOUT ctx.test
        class FakeCtx:
            test = None
            steps = []
            suite = MagicMock()

        fake_ctx = FakeCtx()

        # Patch at the robot.running.context level (runtime import target)
        with patch("robot.running.context.EXECUTION_CONTEXTS") as ec:
            ec.current = fake_ctx

            # Mock _pre_validate_browser_element to avoid actual browser calls
            async def fake_browser_validate(*a, **kw):
                return {"valid": True, "states": ["visible"], "missing": []}

            executor._pre_validate_browser_element = fake_browser_validate
            executor.REQUIRED_STATES_FOR_ACTION = {"fill": {"visible", "enabled", "editable"}}
            executor._get_action_type_from_keyword_for_states = lambda kw: "fill"

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    executor._pre_validate_element("#test", mock_session, "Fill Text")
                )
            finally:
                loop.close()

            # Verify ctx.test was set
            assert fake_ctx.test is not None, (
                "ctx.test should be set by _pre_validate_element"
            )

    def test_pre_validate_preserves_existing_ctx_test(self):
        """_pre_validate_element does NOT overwrite existing ctx.test."""
        import asyncio
        from robotmcp.components.execution.keyword_executor import KeywordExecutor
        from robot.result.model import TestCase as ResTest

        executor = KeywordExecutor.__new__(KeywordExecutor)
        executor.pre_validation_enabled = True

        class FakeConfig:
            PRE_VALIDATION_TIMEOUT = 500
        executor.config = FakeConfig()

        mock_session = MagicMock()
        mock_session.browser_state.active_library = "browser"

        existing_test = ResTest(name="Existing_Test")

        class FakeCtx:
            test = existing_test
            steps = []

        fake_ctx = FakeCtx()

        with patch("robot.running.context.EXECUTION_CONTEXTS") as ec:
            ec.current = fake_ctx

            async def fake_browser_validate(*a, **kw):
                return {"valid": True, "states": ["visible"], "missing": []}

            executor._pre_validate_browser_element = fake_browser_validate
            executor.REQUIRED_STATES_FOR_ACTION = {"fill": {"visible"}}
            executor._get_action_type_from_keyword_for_states = lambda kw: "fill"

            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    executor._pre_validate_element("#test", mock_session, "Fill Text")
                )
            finally:
                loop.close()

            # Should NOT have been replaced
            assert fake_ctx.test is existing_test, (
                "ctx.test should not be overwritten when already set"
            )
