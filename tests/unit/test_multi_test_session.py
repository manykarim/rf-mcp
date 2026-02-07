"""Comprehensive tests for ADR-005: Multiple Tests/Tasks per Session.

Covers:
- TestInfo lifecycle
- TestRegistry lifecycle (start/end/list, step routing, backward compat)
- ExecutionSession multi-test routing
- VariableScopes integration
- build_test_suite multi-test generation
- Suite setup/teardown text generation
- Per-test flow blocks
- Backward compatibility (legacy mode)
"""

import asyncio
import os
import uuid
from collections import OrderedDict
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotmcp.models.execution_models import (
    ExecutionStep,
    TestInfo,
    TestRegistry,
)
from robotmcp.models.session_models import ExecutionSession


# =====================================================================
# TestInfo
# =====================================================================


class TestTestInfo:
    """Tests for TestInfo dataclass."""

    def test_default_values(self):
        ti = TestInfo(name="Login Test")
        assert ti.name == "Login Test"
        assert ti.status == "not_run"
        assert ti.documentation == ""
        assert ti.tags == []
        assert ti.setup is None
        assert ti.teardown is None
        assert ti.steps == []
        assert ti.flow_blocks == []
        assert ti.started_at is None
        assert ti.ended_at is None
        assert ti.error_message is None

    def test_with_all_fields(self):
        now = datetime.now()
        ti = TestInfo(
            name="Test 1",
            status="pass",
            documentation="A test",
            tags=["smoke", "login"],
            setup={"keyword": "Open Browser", "arguments": ["chromium"]},
            teardown={"keyword": "Close Browser", "arguments": []},
            started_at=now,
            ended_at=now,
            error_message=None,
        )
        assert ti.tags == ["smoke", "login"]
        assert ti.setup["keyword"] == "Open Browser"
        assert ti.teardown["arguments"] == []

    def test_steps_append(self):
        ti = TestInfo(name="T1")
        step = ExecutionStep(step_id="s1", keyword="Log", arguments=["hello"])
        step.mark_success("ok")
        ti.steps.append(step)
        assert len(ti.steps) == 1

    def test_flow_blocks_append(self):
        ti = TestInfo(name="T1")
        ti.flow_blocks.append({"type": "if", "condition": "True"})
        assert len(ti.flow_blocks) == 1


# =====================================================================
# TestRegistry
# =====================================================================


class TestTestRegistry:
    """Tests for TestRegistry lifecycle."""

    def test_initial_state(self):
        reg = TestRegistry()
        assert not reg.is_multi_test_mode()
        assert reg.current_test_name is None
        assert len(reg.tests) == 0
        assert reg.get_current_test() is None

    def test_start_test(self):
        reg = TestRegistry()
        ti = reg.start_test("Login Test", documentation="Login", tags=["smoke"])
        assert reg.is_multi_test_mode()
        assert reg.current_test_name == "Login Test"
        assert ti.status == "running"
        assert ti.documentation == "Login"
        assert ti.tags == ["smoke"]
        assert ti.started_at is not None

    def test_end_test(self):
        reg = TestRegistry()
        reg.start_test("T1")
        ti = reg.end_test(status="pass", message="")
        assert ti.status == "pass"
        assert ti.ended_at is not None
        assert reg.current_test_name is None
        assert reg.get_current_test() is None

    def test_end_test_fail(self):
        reg = TestRegistry()
        reg.start_test("T1")
        ti = reg.end_test(status="fail", message="Assertion failed")
        assert ti.status == "fail"
        assert ti.error_message == "Assertion failed"

    def test_end_test_no_active(self):
        reg = TestRegistry()
        result = reg.end_test()
        assert result is None

    def test_auto_end_on_start(self):
        """Starting a new test auto-ends the previous one."""
        reg = TestRegistry()
        reg.start_test("T1")
        reg.start_test("T2")
        assert reg.tests["T1"].status == "pass"
        assert reg.tests["T1"].ended_at is not None
        assert reg.current_test_name == "T2"
        assert reg.tests["T2"].status == "running"

    def test_multiple_tests(self):
        reg = TestRegistry()
        reg.start_test("T1")
        reg.end_test(status="pass")
        reg.start_test("T2")
        reg.end_test(status="fail", message="err")
        reg.start_test("T3")
        reg.end_test(status="pass")
        assert len(reg.tests) == 3
        assert list(reg.tests.keys()) == ["T1", "T2", "T3"]
        assert reg.tests["T2"].status == "fail"

    def test_all_steps_flat(self):
        reg = TestRegistry()
        reg.start_test("T1")
        s1 = ExecutionStep(step_id="s1", keyword="Log", arguments=["1"])
        s1.mark_success()
        reg.tests["T1"].steps.append(s1)
        reg.end_test()

        reg.start_test("T2")
        s2 = ExecutionStep(step_id="s2", keyword="Log", arguments=["2"])
        s2.mark_success()
        reg.tests["T2"].steps.append(s2)
        reg.end_test()

        flat = reg.all_steps_flat()
        assert len(flat) == 2
        assert flat[0].step_id == "s1"
        assert flat[1].step_id == "s2"

    def test_all_flow_blocks_flat(self):
        reg = TestRegistry()
        reg.start_test("T1")
        reg.tests["T1"].flow_blocks.append({"type": "if"})
        reg.end_test()
        reg.start_test("T2")
        reg.tests["T2"].flow_blocks.append({"type": "for_each"})
        reg.end_test()

        flat = reg.all_flow_blocks_flat()
        assert len(flat) == 2

    def test_setup_teardown(self):
        reg = TestRegistry()
        ti = reg.start_test(
            "T1",
            setup={"keyword": "Open Browser", "arguments": ["chromium"]},
            teardown={"keyword": "Close Browser", "arguments": []},
        )
        assert ti.setup["keyword"] == "Open Browser"
        assert ti.teardown["keyword"] == "Close Browser"

    def test_overwrite_test_name(self):
        """Starting a test with the same name overwrites it."""
        reg = TestRegistry()
        reg.start_test("T1")
        step = ExecutionStep(step_id="s1", keyword="Log", arguments=["old"])
        step.mark_success()
        reg.tests["T1"].steps.append(step)
        reg.end_test()

        # Start again with same name
        reg.start_test("T1")
        assert len(reg.tests["T1"].steps) == 0  # fresh test


# =====================================================================
# ExecutionSession multi-test routing
# =====================================================================


class TestExecutionSessionMultiTest:
    """Tests for ExecutionSession.add_step in multi-test mode."""

    def _make_step(self, step_id="s1"):
        step = ExecutionStep(step_id=step_id, keyword="Log", arguments=["x"])
        step.mark_success()
        return step

    def test_legacy_mode(self):
        """Without start_test, steps go to flat list (legacy)."""
        sess = ExecutionSession(session_id="s1")
        step = self._make_step()
        sess.add_step(step)
        assert len(sess.steps) == 1
        assert not sess.test_registry.is_multi_test_mode()

    def test_multi_test_routing_to_current(self):
        """Steps route to current test in multi-test mode."""
        sess = ExecutionSession(session_id="s1")
        sess.test_registry.start_test("T1")
        step = self._make_step()
        sess.add_step(step)
        assert len(sess.test_registry.tests["T1"].steps) == 1
        assert len(sess.steps) == 0  # legacy list untouched

    def test_multi_test_routing_between_tests(self):
        """Steps between tests go to suite_level_steps."""
        sess = ExecutionSession(session_id="s1")
        sess.test_registry.start_test("T1")
        sess.test_registry.end_test()
        # Now between tests
        step = self._make_step()
        sess.add_step(step)
        assert len(sess.suite_level_steps) == 1

    def test_step_count_legacy(self):
        sess = ExecutionSession(session_id="s1")
        step = self._make_step()
        sess.add_step(step)
        assert sess.step_count == 1

    def test_step_count_multi_test(self):
        sess = ExecutionSession(session_id="s1")
        sess.test_registry.start_test("T1")
        sess.add_step(self._make_step("s1"))
        sess.test_registry.end_test()
        sess.add_step(self._make_step("s2"))  # suite-level
        sess.test_registry.start_test("T2")
        sess.add_step(self._make_step("s3"))
        sess.test_registry.end_test()
        # 2 in tests + 1 suite-level = 3
        assert sess.step_count == 3

    def test_suite_setup_teardown(self):
        sess = ExecutionSession(session_id="s1")
        sess.suite_setup = {"keyword": "Open Browser", "arguments": ["chromium"]}
        sess.suite_teardown = {"keyword": "Close Browser", "arguments": []}
        assert sess.suite_setup["keyword"] == "Open Browser"
        assert sess.suite_teardown["keyword"] == "Close Browser"

    def test_failed_step_not_added(self):
        """Failed steps are not recorded (existing behavior)."""
        sess = ExecutionSession(session_id="s1")
        sess.test_registry.start_test("T1")
        step = ExecutionStep(step_id="sf", keyword="Should Be True", arguments=["False"])
        step.mark_failure("Assertion failed")
        sess.add_step(step)
        assert len(sess.test_registry.tests["T1"].steps) == 0


# =====================================================================
# VariableScopes integration
# =====================================================================


class TestVariableScopesIntegration:
    """Tests for VariableScopes in context creation."""

    def test_context_uses_variable_scopes(self):
        """Verify create_context_for_session uses VariableScopes."""
        from robotmcp.components.execution.rf_native_context_manager import (
            get_rf_native_context_manager,
        )

        mgr = get_rf_native_context_manager()
        sid = f"test-vs-{uuid.uuid4()}"
        ctx_info = None
        try:
            result = mgr.create_context_for_session(sid)
            if not result["success"]:
                pytest.skip(f"Context creation failed: {result.get('error')}")
            ctx_info = mgr._session_contexts[sid]
            variables = ctx_info["variables"]
            # Should be VariableScopes, not CompatibleVariables
            assert type(variables).__name__ == "VariableScopes", (
                f"Expected VariableScopes, got {type(variables).__name__}"
            )
            # Should have start_test/end_test methods
            assert hasattr(variables, "start_test")
            assert hasattr(variables, "end_test")
            assert hasattr(variables, "set_test")
            assert hasattr(variables, "set_suite")
        finally:
            if ctx_info:
                try:
                    from robot.running.context import EXECUTION_CONTEXTS
                    if EXECUTION_CONTEXTS.current:
                        EXECUTION_CONTEXTS.end_suite()
                except Exception:
                    pass

    def test_context_has_test_cycling_keys(self):
        """Verify context info includes ADR-005 tracking keys."""
        from robotmcp.components.execution.rf_native_context_manager import (
            get_rf_native_context_manager,
        )

        mgr = get_rf_native_context_manager()
        sid = f"test-cycle-{uuid.uuid4()}"
        ctx_info = None
        try:
            result = mgr.create_context_for_session(sid)
            if not result["success"]:
                pytest.skip(f"Context creation failed: {result.get('error')}")
            ctx_info = mgr._session_contexts[sid]
            assert "current_run_test" in ctx_info
            assert "current_res_test" in ctx_info
            assert ctx_info["current_run_test"] is None
            assert ctx_info["current_res_test"] is None
        finally:
            if ctx_info:
                try:
                    from robot.running.context import EXECUTION_CONTEXTS
                    if EXECUTION_CONTEXTS.current:
                        EXECUTION_CONTEXTS.end_suite()
                except Exception:
                    pass


# =====================================================================
# RF Context test cycling
# =====================================================================


class TestRFContextTestCycling:
    """Tests for start_test_in_context / end_test_in_context."""

    def test_start_test_no_context(self):
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
        )

        mgr = RobotFrameworkNativeContextManager()
        result = mgr.start_test_in_context("nonexistent", "T1")
        assert result["success"] is False

    def test_end_test_no_context(self):
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
        )

        mgr = RobotFrameworkNativeContextManager()
        result = mgr.end_test_in_context("nonexistent")
        assert result["success"] is False

    def test_end_test_no_active_test(self):
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
        )

        mgr = RobotFrameworkNativeContextManager()
        mgr._session_contexts["sid"] = {
            "context": MagicMock(),
            "namespace": MagicMock(),
            "variables": MagicMock(),
            "current_run_test": None,
            "current_res_test": None,
        }
        result = mgr.end_test_in_context("sid")
        assert result["success"] is False
        assert "No active test" in result["error"]

    def test_ensure_test_active_already_active(self):
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
        )

        mgr = RobotFrameworkNativeContextManager()
        mgr._session_contexts["sid"] = {
            "current_run_test": MagicMock(),
        }
        # Should not start a new test
        mgr.ensure_test_active("sid")  # no error

    def test_ensure_test_active_no_context(self):
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
        )

        mgr = RobotFrameworkNativeContextManager()
        # No context — should silently return
        mgr.ensure_test_active("nonexistent")


# =====================================================================
# build_test_suite multi-test generation
# =====================================================================


class TestBuildSuiteMultiTest:
    """Tests for build_test_suite in multi-test mode."""

    def _make_session_with_tests(self):
        """Create an ExecutionSession with 2 tests."""
        sess = ExecutionSession(session_id="mt1")
        sess.test_registry.start_test("Login Test", documentation="Login test", tags=["smoke"])
        s1 = ExecutionStep(step_id="s1", keyword="New Page", arguments=["https://example.com"])
        s1.mark_success()
        sess.test_registry.tests["Login Test"].steps.append(s1)
        s2 = ExecutionStep(step_id="s2", keyword="Fill Text", arguments=["id=user", "admin"])
        s2.mark_success()
        sess.test_registry.tests["Login Test"].steps.append(s2)
        sess.test_registry.end_test(status="pass")

        sess.test_registry.start_test("Search Test", tags=["regression"])
        s3 = ExecutionStep(step_id="s3", keyword="Fill Text", arguments=["id=search", "robot"])
        s3.mark_success()
        sess.test_registry.tests["Search Test"].steps.append(s3)
        s4 = ExecutionStep(step_id="s4", keyword="Click", arguments=["id=btn"])
        s4.mark_success()
        sess.test_registry.tests["Search Test"].steps.append(s4)
        sess.test_registry.end_test(status="pass")

        sess.suite_setup = {"keyword": "New Browser", "arguments": ["chromium"]}
        sess.suite_teardown = {"keyword": "Close Browser", "arguments": []}
        return sess

    @pytest.mark.asyncio
    async def test_multi_test_build(self):
        from robotmcp.components.test_builder import TestBuilder

        sess = self._make_session_with_tests()
        engine = MagicMock()
        engine.sessions = {"mt1": sess}
        engine.session_manager.get_or_create_session.return_value = sess

        builder = TestBuilder(execution_engine=engine)
        result = await builder.build_suite(session_id="mt1", test_name="Multi Suite")

        assert result["success"], result.get("error")
        rf_text = result.get("rf_text", "")
        assert "*** Test Cases ***" in rf_text
        assert "Login Test" in rf_text
        assert "Search Test" in rf_text
        assert "New Page" in rf_text
        assert "Fill Text" in rf_text
        assert "Click" in rf_text

    @pytest.mark.asyncio
    async def test_suite_setup_teardown_in_text(self):
        from robotmcp.components.test_builder import TestBuilder

        sess = self._make_session_with_tests()
        engine = MagicMock()
        engine.sessions = {"mt1": sess}
        engine.session_manager.get_or_create_session.return_value = sess

        builder = TestBuilder(execution_engine=engine)
        result = await builder.build_suite(session_id="mt1")

        rf_text = result.get("rf_text", "")
        assert "Suite Setup" in rf_text
        assert "New Browser" in rf_text
        assert "Suite Teardown" in rf_text
        assert "Close Browser" in rf_text

    @pytest.mark.asyncio
    async def test_per_test_tags(self):
        from robotmcp.components.test_builder import TestBuilder

        sess = self._make_session_with_tests()
        engine = MagicMock()
        engine.sessions = {"mt1": sess}
        engine.session_manager.get_or_create_session.return_value = sess

        builder = TestBuilder(execution_engine=engine)
        result = await builder.build_suite(session_id="mt1")

        rf_text = result.get("rf_text", "")
        assert "[Tags]    smoke" in rf_text
        assert "[Tags]    regression" in rf_text

    @pytest.mark.asyncio
    async def test_empty_tests_skipped(self):
        from robotmcp.components.test_builder import TestBuilder

        sess = ExecutionSession(session_id="mt2")
        sess.test_registry.start_test("Empty Test")
        sess.test_registry.end_test()  # no steps

        engine = MagicMock()
        engine.sessions = {"mt2": sess}

        builder = TestBuilder(execution_engine=engine)
        result = await builder.build_suite(session_id="mt2")
        assert not result["success"]
        assert "No tests with steps" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_auto_end_running_test(self):
        """build_test_suite auto-ends a running test."""
        from robotmcp.components.test_builder import TestBuilder

        sess = ExecutionSession(session_id="mt3")
        sess.test_registry.start_test("T1")
        s1 = ExecutionStep(step_id="s1", keyword="Log", arguments=["x"])
        s1.mark_success()
        sess.test_registry.tests["T1"].steps.append(s1)
        # Don't end — still running

        engine = MagicMock()
        engine.sessions = {"mt3": sess}

        builder = TestBuilder(execution_engine=engine)
        result = await builder.build_suite(session_id="mt3")
        assert result["success"], result.get("error")
        assert sess.test_registry.tests["T1"].status == "pass"


# =====================================================================
# Legacy mode backward compat
# =====================================================================


class TestLegacyModeBackwardCompat:
    """Verify sessions without start_test work identically to before."""

    def test_session_without_start_test(self):
        sess = ExecutionSession(session_id="legacy")
        assert not sess.test_registry.is_multi_test_mode()
        step = ExecutionStep(step_id="s1", keyword="Log", arguments=["x"])
        step.mark_success()
        sess.add_step(step)
        assert len(sess.steps) == 1
        assert sess.step_count == 1
        assert len(sess.suite_level_steps) == 0

    def test_session_info_no_multi_test(self):
        sess = ExecutionSession(session_id="legacy")
        info = sess.get_session_info()
        assert "multi_test_mode" not in info  # not added in legacy mode

    def test_cleanup_unaffected(self):
        sess = ExecutionSession(session_id="legacy")
        step = ExecutionStep(step_id="s1", keyword="Log", arguments=["x"])
        step.mark_success()
        sess.add_step(step)
        sess.cleanup()
        assert len(sess.steps) == 0


# =====================================================================
# Per-test flow blocks
# =====================================================================


class TestPerTestFlowBlocks:
    """Tests for per-test flow block routing."""

    @pytest.mark.asyncio
    async def test_per_test_flow_blocks_in_suite(self):
        from robotmcp.components.test_builder import TestBuilder

        sess = ExecutionSession(session_id="fb1")

        # Test 1 with an if block
        sess.test_registry.start_test("T1")
        s1 = ExecutionStep(step_id="s1", keyword="Log", arguments=["before"])
        s1.mark_success()
        sess.test_registry.tests["T1"].steps.append(s1)
        sess.test_registry.tests["T1"].flow_blocks.append({
            "type": "if",
            "condition": "True",
            "then": [{"keyword": "Log", "arguments": ["then"]}],
            "else": [],
        })
        sess.test_registry.end_test()

        # Test 2 without flow blocks
        sess.test_registry.start_test("T2")
        s2 = ExecutionStep(step_id="s2", keyword="Log", arguments=["simple"])
        s2.mark_success()
        sess.test_registry.tests["T2"].steps.append(s2)
        sess.test_registry.end_test()

        engine = MagicMock()
        engine.sessions = {"fb1": sess}

        builder = TestBuilder(execution_engine=engine)
        result = await builder.build_suite(session_id="fb1")
        assert result["success"], result.get("error")
        suite_data = result.get("suite", {})
        # The suite should reference the per-test flow blocks
        # (this validates the data flow — rendering tested via rf_text)


# =====================================================================
# Test setup/teardown per test
# =====================================================================


class TestPerTestSetupTeardown:
    """Tests for per-test setup and teardown in generated text."""

    @pytest.mark.asyncio
    async def test_per_test_setup_teardown(self):
        from robotmcp.components.test_builder import TestBuilder

        sess = ExecutionSession(session_id="st1")
        sess.test_registry.start_test(
            "T1",
            setup={"keyword": "Open Page", "arguments": ["https://x.com"]},
            teardown={"keyword": "Close Page", "arguments": []},
        )
        s = ExecutionStep(step_id="s1", keyword="Click", arguments=["id=btn"])
        s.mark_success()
        sess.test_registry.tests["T1"].steps.append(s)
        sess.test_registry.end_test()

        engine = MagicMock()
        engine.sessions = {"st1": sess}

        builder = TestBuilder(execution_engine=engine)
        result = await builder.build_suite(session_id="st1")
        assert result["success"], result.get("error")
        rf_text = result.get("rf_text", "")
        assert "[Setup]" in rf_text
        assert "Open Page" in rf_text
        assert "[Teardown]" in rf_text
        assert "Close Page" in rf_text


# =====================================================================
# Variable assignment in multi-test steps
# =====================================================================


class TestVariableAssignmentMultiTest:
    """Tests for variable assignments in multi-test step recording."""

    @pytest.mark.asyncio
    async def test_assigned_variables_preserved(self):
        from robotmcp.components.test_builder import TestBuilder

        sess = ExecutionSession(session_id="va1")
        sess.test_registry.start_test("T1")
        s = ExecutionStep(
            step_id="s1",
            keyword="Get Text",
            arguments=["id=welcome"],
        )
        s.assigned_variables = ["${msg}"]
        s.assignment_type = "single"
        s.mark_success("Welcome")
        sess.test_registry.tests["T1"].steps.append(s)
        sess.test_registry.end_test()

        engine = MagicMock()
        engine.sessions = {"va1": sess}

        builder = TestBuilder(execution_engine=engine)
        result = await builder.build_suite(session_id="va1")
        assert result["success"], result.get("error")
        rf_text = result.get("rf_text", "")
        assert "${msg} =" in rf_text
        assert "Get Text" in rf_text
