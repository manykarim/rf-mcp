"""ADR-020 Phase 0: Namespace lifecycle tests.

Tests verifying RF context lifecycle correctness after ADR-020 fixes:
- Scope depth (3 not 4 after start_test)
- Variable isolation across test boundaries
- TEST-scoped library safety
- Multi-test cycling stability
- Suite variable persistence
"""
import tempfile
import pytest

from robot.running.context import EXECUTION_CONTEXTS
from robot.running.namespace import Namespace
from robot.variables.scopes import VariableScopes
from robot.output import Output
from robot.running.model import TestSuite, TestCase as RunTest
from robot.result.model import TestSuite as ResultSuite, TestCase as ResTest
from robot.conf import Languages, RobotSettings

__test__ = True


# ── Helpers ───────────────────────────────────────────────────────────


def _make_rf_context():
    """Create a proper RF execution context following RF-native ordering."""
    variables = VariableScopes(RobotSettings())
    suite = TestSuite(name="TestSuite")
    temp_dir = tempfile.mkdtemp(prefix="rf_adr020_")
    settings = RobotSettings(outputdir=temp_dir, output=None, console="none")
    output = Output(settings)
    try:
        output.library_listeners.new_suite_scope()
    except Exception:
        pass
    result_suite = ResultSuite(name=suite.name)
    namespace = Namespace(variables, result_suite, suite.resource, Languages())
    # RF-native ordering: start_suite BEFORE context push
    namespace.start_suite()
    ctx = EXECUTION_CONTEXTS.start_suite(result_suite, namespace, output, dry_run=False)
    ctx.set_suite_variables(result_suite)
    namespace.handle_imports()
    namespace.variables.resolve_delayed()
    return ctx, namespace, variables


def _cleanup():
    """Clean up EXECUTION_CONTEXTS."""
    try:
        EXECUTION_CONTEXTS.end_suite()
    except Exception:
        pass


# ── T0.1: Scope depth after start_test ──────────────────────────────


class TestScopeDepth:
    """Verify variable scope stack has correct depth."""

    def test_scope_depth_is_3_after_start_test(self):
        """After start_test, scopes should be global + suite + test = 3."""
        ctx, namespace, variables = _make_rf_context()
        try:
            # Should have 2 scopes: global + suite
            assert len(variables._scopes) == 2, (
                f"Expected 2 scopes (global+suite) before start_test, got {len(variables._scopes)}"
            )
            # Start test via ctx (which calls namespace.start_test internally)
            run_test = RunTest(name="Test1")
            res_test = ResTest(name="Test1")
            ctx.start_test(run_test, res_test)
            # Should have exactly 3 scopes: global + suite + test
            assert len(variables._scopes) == 3, (
                f"Expected 3 scopes (global+suite+test) after start_test, got {len(variables._scopes)}"
            )
        finally:
            _cleanup()

    def test_scope_depth_returns_to_2_after_end_test(self):
        """After end_test, scopes should return to global + suite = 2."""
        ctx, namespace, variables = _make_rf_context()
        try:
            run_test = RunTest(name="Test1")
            res_test = ResTest(name="Test1")
            ctx.start_test(run_test, res_test)
            assert len(variables._scopes) == 3
            ctx.end_test(res_test)
            assert len(variables._scopes) == 2, (
                f"Expected 2 scopes after end_test, got {len(variables._scopes)}"
            )
        finally:
            _cleanup()


# ── T0.2: Variable isolation across tests ────────────────────────────


class TestVariableIsolation:
    """Verify test-scoped variables are properly cleaned between tests."""

    def test_test_variable_cleaned_after_end_test(self):
        """Variables set with Set Test Variable should not survive end_test."""
        ctx, namespace, variables = _make_rf_context()
        try:
            run_test = RunTest(name="Test1")
            res_test = ResTest(name="Test1")
            ctx.start_test(run_test, res_test)
            # Set a test variable via RF
            from robot.libraries.BuiltIn import BuiltIn
            bi = BuiltIn()
            bi.run_keyword("Set Test Variable", "${MY_TEST_VAR}", "test1_value")
            val = bi.get_variable_value("${MY_TEST_VAR}")
            assert val == "test1_value"
            # End test
            ctx.end_test(res_test)
            # Variable should be gone
            with pytest.raises(Exception):
                variables["${MY_TEST_VAR}"]
        finally:
            _cleanup()

    def test_test_variable_does_not_leak_to_next_test(self):
        """Test variables from Test 1 must not be visible in Test 2."""
        ctx, namespace, variables = _make_rf_context()
        try:
            from robot.libraries.BuiltIn import BuiltIn
            bi = BuiltIn()
            # Test 1
            run1 = RunTest(name="Test1")
            res1 = ResTest(name="Test1")
            ctx.start_test(run1, res1)
            bi.run_keyword("Set Test Variable", "${LEAK_CHECK}", "from_test1")
            ctx.end_test(res1)
            # Test 2
            run2 = RunTest(name="Test2")
            res2 = ResTest(name="Test2")
            ctx.start_test(run2, res2)
            val = bi.get_variable_value("${LEAK_CHECK}", default="NOT_FOUND")
            assert val == "NOT_FOUND", f"Test variable leaked: {val}"
            ctx.end_test(res2)
        finally:
            _cleanup()


# ── T0.3: Suite variable persistence ─────────────────────────────────


class TestSuiteVariablePersistence:
    """Verify suite variables persist across test boundaries."""

    def test_suite_variable_survives_test_end(self):
        """Suite variables should persist after test scope is popped."""
        ctx, namespace, variables = _make_rf_context()
        try:
            from robot.libraries.BuiltIn import BuiltIn
            bi = BuiltIn()
            run1 = RunTest(name="Test1")
            res1 = ResTest(name="Test1")
            ctx.start_test(run1, res1)
            bi.run_keyword("Set Suite Variable", "${SUITE_VAR}", "suite_value")
            ctx.end_test(res1)
            # Suite variable should still be accessible
            val = variables["${SUITE_VAR}"]
            assert val == "suite_value"
        finally:
            _cleanup()

    def test_suite_variable_visible_in_next_test(self):
        """Suite variables set in Test 1 should be visible in Test 2."""
        ctx, namespace, variables = _make_rf_context()
        try:
            from robot.libraries.BuiltIn import BuiltIn
            bi = BuiltIn()
            # Test 1: set suite var
            run1 = RunTest(name="Test1")
            res1 = ResTest(name="Test1")
            ctx.start_test(run1, res1)
            bi.run_keyword("Set Suite Variable", "${CROSS_TEST}", "shared")
            ctx.end_test(res1)
            # Test 2: read suite var
            run2 = RunTest(name="Test2")
            res2 = ResTest(name="Test2")
            ctx.start_test(run2, res2)
            val = bi.get_variable_value("${CROSS_TEST}")
            assert val == "shared"
            ctx.end_test(res2)
        finally:
            _cleanup()


# ── T0.4: Multi-test cycling stability ───────────────────────────────


class TestMultiTestCycling:
    """Verify scope stability across 5+ test cycles."""

    def test_five_test_cycles_maintain_scope_depth(self):
        """Scope depth should remain consistent across 5 test cycles."""
        ctx, namespace, variables = _make_rf_context()
        try:
            base_depth = len(variables._scopes)
            assert base_depth == 2
            for i in range(5):
                run_t = RunTest(name=f"Test_{i}")
                res_t = ResTest(name=f"Test_{i}")
                ctx.start_test(run_t, res_t)
                assert len(variables._scopes) == 3, (
                    f"Cycle {i}: expected 3 scopes during test, got {len(variables._scopes)}"
                )
                ctx.end_test(res_t)
                assert len(variables._scopes) == 2, (
                    f"Cycle {i}: expected 2 scopes after end_test, got {len(variables._scopes)}"
                )
        finally:
            _cleanup()

    def test_ten_test_cycles_no_scope_drift(self):
        """After 10 test cycles, final scope depth should be exactly 2."""
        ctx, namespace, variables = _make_rf_context()
        try:
            for i in range(10):
                run_t = RunTest(name=f"Test_{i}")
                res_t = ResTest(name=f"Test_{i}")
                ctx.start_test(run_t, res_t)
                ctx.end_test(res_t)
            assert len(variables._scopes) == 2, (
                f"Scope drift after 10 cycles: expected 2, got {len(variables._scopes)}"
            )
        finally:
            _cleanup()

    def test_variable_isolation_across_five_tests(self):
        """Variables should be properly isolated across 5 consecutive tests."""
        ctx, namespace, variables = _make_rf_context()
        try:
            from robot.libraries.BuiltIn import BuiltIn
            bi = BuiltIn()
            for i in range(5):
                run_t = RunTest(name=f"Test_{i}")
                res_t = ResTest(name=f"Test_{i}")
                ctx.start_test(run_t, res_t)
                bi.run_keyword("Set Test Variable", "${ITER}", str(i))
                val = bi.get_variable_value("${ITER}")
                assert val == str(i)
                ctx.end_test(res_t)
            # After all tests, variable should be gone
            with pytest.raises(Exception):
                variables["${ITER}"]
        finally:
            _cleanup()


# ── T0.5: TEST-scoped library safety ─────────────────────────────────


class TestScopedLibrarySafety:
    """Verify that single start_test/end_test is safe for TEST-scoped libraries."""

    def test_global_scope_lib_unaffected_by_test_cycling(self):
        """GLOBAL-scoped libraries (BuiltIn) should be unaffected by test cycling."""
        ctx, namespace, variables = _make_rf_context()
        try:
            for lib in namespace.libraries:
                if lib.name == "BuiltIn":
                    # GlobalScopeManager has no instance_cache manipulation
                    assert lib.scope.name == "GLOBAL"
            # Cycle tests - should not crash
            for i in range(3):
                run_t = RunTest(name=f"Test_{i}")
                res_t = ResTest(name=f"Test_{i}")
                ctx.start_test(run_t, res_t)
                ctx.end_test(res_t)
        finally:
            _cleanup()

    def test_test_scope_manager_push_pop_balanced(self):
        """Verify TestScopeManager.start_test/end_test push/pop is balanced with single calls."""
        from robot.running.libraryscopes import TestScopeManager
        import inspect

        # Verify start_test pushes to instance_cache
        src_start = inspect.getsource(TestScopeManager.start_test)
        assert "instance_cache.append" in src_start

        # Verify end_test pops from instance_cache
        src_end = inspect.getsource(TestScopeManager.end_test)
        assert "instance_cache.pop" in src_end


# ── T0.6: rf-mcp context manager integration ─────────────────────────


class TestRfMcpContextManagerLifecycle:
    """Test rf-mcp's create_context_for_session produces correct scope depth."""

    def test_create_context_correct_scope_depth(self):
        """create_context_for_session should produce 3 scopes (global+suite+test)."""
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
        )
        mgr = RobotFrameworkNativeContextManager()
        session_id = "test_scope_depth"
        try:
            result = mgr.create_context_for_session(session_id, libraries=[])
            assert result["success"], f"Context creation failed: {result}"
            ctx_info = mgr._session_contexts.get(session_id)
            assert ctx_info is not None
            variables = ctx_info["variables"]
            # Should be 3: global + suite + test (initial MCP test)
            assert len(variables._scopes) == 3, (
                f"Expected 3 scopes after create_context_for_session, got {len(variables._scopes)}"
            )
        finally:
            mgr.cleanup_context(session_id)
            try:
                EXECUTION_CONTEXTS.end_suite()
            except Exception:
                pass

    def test_start_end_test_in_context_scope_depth(self):
        """start_test_in_context / end_test_in_context should maintain correct scope depth."""
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
        )
        mgr = RobotFrameworkNativeContextManager()
        session_id = "test_cycling"
        try:
            result = mgr.create_context_for_session(session_id, libraries=[])
            assert result["success"]
            ctx_info = mgr._session_contexts.get(session_id)
            variables = ctx_info["variables"]
            initial_depth = len(variables._scopes)
            # Start a new named test
            mgr.start_test_in_context(session_id, "NamedTest1")
            assert len(variables._scopes) == initial_depth, (
                f"Expected {initial_depth} scopes after cycling, got {len(variables._scopes)}"
            )
            # End test
            mgr.end_test_in_context(session_id)
            # Should be initial_depth - 1 (test scope popped, no new test)
            assert len(variables._scopes) == initial_depth - 1, (
                f"Expected {initial_depth - 1} scopes after end_test, got {len(variables._scopes)}"
            )
        finally:
            mgr.cleanup_context(session_id)
            try:
                EXECUTION_CONTEXTS.end_suite()
            except Exception:
                pass

    def test_multi_test_cycling_no_scope_drift(self):
        """Multiple start_test/end_test cycles should not accumulate scopes."""
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
        )
        mgr = RobotFrameworkNativeContextManager()
        session_id = "test_multi_cycle"
        try:
            result = mgr.create_context_for_session(session_id, libraries=[])
            assert result["success"]
            ctx_info = mgr._session_contexts.get(session_id)
            variables = ctx_info["variables"]
            for i in range(5):
                mgr.start_test_in_context(session_id, f"Test_{i}")
                depth_during = len(variables._scopes)
                mgr.end_test_in_context(session_id)
                depth_after = len(variables._scopes)
                assert depth_during == 3, (
                    f"Cycle {i}: expected 3 scopes during test, got {depth_during}"
                )
                assert depth_after == 2, (
                    f"Cycle {i}: expected 2 scopes after end, got {depth_after}"
                )
        finally:
            mgr.cleanup_context(session_id)
            try:
                EXECUTION_CONTEXTS.end_suite()
            except Exception:
                pass


# ── T0.7/T0.8: Keyword dispatch tests ────────────────────────────────


class TestKeywordDispatch:
    """Verify keyword execution goes through RF runner, not direct method calls."""

    def test_variable_resolution_in_arguments(self):
        """${var} in arguments must be resolved by RF runner."""
        ctx, namespace, variables = _make_rf_context()
        try:
            run_test = RunTest(name="Test1")
            res_test = ResTest(name="Test1")
            ctx.start_test(run_test, res_test)
            from robot.libraries.BuiltIn import BuiltIn
            bi = BuiltIn()
            # Set a variable
            bi.run_keyword("Set Test Variable", "${MY_NAME}", "World")
            # Use variable in argument — should resolve
            result = bi.run_keyword("Set Variable", "Hello ${MY_NAME}")
            assert result == "Hello World", f"Variable not resolved: {result}"
            ctx.end_test(res_test)
        finally:
            _cleanup()

    def test_runner_run_auto_assigns(self):
        """runner.run() with assign= should auto-assign variables."""
        ctx, namespace, variables = _make_rf_context()
        try:
            from robot.running.model import Keyword as RunKeyword
            from robot.result.model import Keyword as ResultKeyword

            run_test = RunTest(name="Test1")
            res_test = ResTest(name="Test1")
            ctx.start_test(run_test, res_test)
            namespace.import_library("Collections")

            runner = namespace.get_runner("Create List")
            data_kw = RunKeyword(
                name="Create List",
                args=("a", "b", "c"),
                assign=("${my_list}",),
            )
            res_kw = ResultKeyword(name="Create List")
            res_kw.parent = res_test
            runner.run(data_kw, res_kw, ctx)

            # RF should have auto-assigned
            val = variables["${my_list}"]
            assert val == ["a", "b", "c"], f"Auto-assign failed: {val}"
            ctx.end_test(res_test)
        finally:
            _cleanup()
