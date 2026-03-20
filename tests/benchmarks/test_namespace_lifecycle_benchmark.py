"""ADR-020 benchmarks: Namespace lifecycle and keyword execution metrics."""
import tempfile
import time
import pytest

from robot.running.context import EXECUTION_CONTEXTS
from robot.running.namespace import Namespace
from robot.variables.scopes import VariableScopes
from robot.output import Output
from robot.running.model import TestSuite, TestCase as RunTest, Keyword as RunKeyword
from robot.result.model import TestSuite as ResultSuite, TestCase as ResTest, Keyword as ResultKeyword
from robot.conf import Languages, RobotSettings


def _make_rf_context():
    variables = VariableScopes(RobotSettings())
    suite = TestSuite(name="Bench")
    temp_dir = tempfile.mkdtemp(prefix="rf_bench_")
    settings = RobotSettings(outputdir=temp_dir, output=None, console="none")
    output = Output(settings)
    try:
        output.library_listeners.new_suite_scope()
    except Exception:
        pass
    result_suite = ResultSuite(name=suite.name)
    namespace = Namespace(variables, result_suite, suite.resource, Languages())
    namespace.start_suite()
    ctx = EXECUTION_CONTEXTS.start_suite(result_suite, namespace, output, dry_run=False)
    ctx.set_suite_variables(result_suite)
    namespace.handle_imports()
    namespace.variables.resolve_delayed()
    return ctx, namespace, variables


class TestScopeDepthBenchmark:
    """Benchmark variable scope operations."""

    def test_start_end_test_cycle_time(self):
        """Measure time for 100 start_test + end_test cycles."""
        ctx, namespace, variables = _make_rf_context()
        try:
            N = 100
            start = time.perf_counter()
            for i in range(N):
                run_t = RunTest(name=f"T_{i}")
                res_t = ResTest(name=f"T_{i}")
                ctx.start_test(run_t, res_t)
                ctx.end_test(res_t)
            elapsed = time.perf_counter() - start
            per_cycle_us = (elapsed / N) * 1_000_000
            print(f"\n  ADR-020: {N} test cycles in {elapsed*1000:.1f}ms ({per_cycle_us:.0f}µs/cycle)")
            assert per_cycle_us < 5000, f"Test cycle too slow: {per_cycle_us:.0f}µs"
        finally:
            try:
                EXECUTION_CONTEXTS.end_suite()
            except Exception:
                pass


class TestKeywordExecutionBenchmark:
    """Benchmark keyword execution through RF runner."""

    def test_evaluate_keyword_time(self):
        """Measure time for 50 Evaluate keyword executions."""
        ctx, namespace, variables = _make_rf_context()
        try:
            run_test = RunTest(name="BenchTest")
            res_test = ResTest(name="BenchTest")
            ctx.start_test(run_test, res_test)

            runner = namespace.get_runner("Evaluate")
            N = 50
            start = time.perf_counter()
            for i in range(N):
                data_kw = RunKeyword(name="Evaluate", args=(f"{i} + 1",))
                res_kw = ResultKeyword(name="Evaluate")
                res_kw.parent = res_test
                runner.run(data_kw, res_kw, ctx)
            elapsed = time.perf_counter() - start
            per_kw_us = (elapsed / N) * 1_000_000
            print(f"\n  ADR-020: {N} Evaluate calls in {elapsed*1000:.1f}ms ({per_kw_us:.0f}µs/keyword)")
            assert per_kw_us < 10000, f"Keyword execution too slow: {per_kw_us:.0f}µs"

            ctx.end_test(res_test)
        finally:
            try:
                EXECUTION_CONTEXTS.end_suite()
            except Exception:
                pass


class TestContextCreationBenchmark:
    """Benchmark rf-mcp context creation."""

    def test_create_context_time(self):
        """Measure time for context creation via rf-mcp."""
        from robotmcp.components.execution.rf_native_context_manager import (
            RobotFrameworkNativeContextManager,
        )
        mgr = RobotFrameworkNativeContextManager()
        session_id = "bench_ctx"
        try:
            start = time.perf_counter()
            result = mgr.create_context_for_session(session_id, libraries=[])
            elapsed = time.perf_counter() - start
            elapsed_ms = elapsed * 1000
            assert result["success"]
            print(f"\n  ADR-020: create_context_for_session in {elapsed_ms:.1f}ms")
            assert elapsed_ms < 5000, f"Context creation too slow: {elapsed_ms:.1f}ms"
        finally:
            mgr.cleanup_context(session_id)
            try:
                EXECUTION_CONTEXTS.end_suite()
            except Exception:
                pass
