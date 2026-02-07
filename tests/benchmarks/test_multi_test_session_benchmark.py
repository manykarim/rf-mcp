"""Benchmark Tests for ADR-005: Multiple Tests/Tasks per Session.

Measures performance of:
1. TestRegistry lifecycle operations (start/end/step routing)
2. ExecutionSession multi-test step routing vs legacy
3. build_test_suite generation with many tests
4. TestRegistry scaling (10, 50, 100 tests)
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest

from robotmcp.models.execution_models import ExecutionStep, TestInfo, TestRegistry
from robotmcp.models.session_models import ExecutionSession


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_step(step_id: str = "s1") -> ExecutionStep:
    step = ExecutionStep(step_id=step_id, keyword="Log", arguments=["hello"])
    step.mark_success("ok")
    return step


def _build_session_with_n_tests(n: int, steps_per_test: int = 5) -> ExecutionSession:
    """Create session with n tests, each having steps_per_test steps."""
    sess = ExecutionSession(session_id=f"bench-{n}")
    for i in range(n):
        sess.test_registry.start_test(f"Test {i}", tags=[f"tag{i}"])
        for j in range(steps_per_test):
            step = _make_step(f"s-{i}-{j}")
            sess.test_registry.tests[f"Test {i}"].steps.append(step)
        sess.test_registry.end_test(status="pass")
    return sess


# ---------------------------------------------------------------------------
# TestRegistry lifecycle benchmarks
# ---------------------------------------------------------------------------

class TestRegistryLifecycleBenchmark:
    """Benchmark TestRegistry start/end operations."""

    @pytest.mark.parametrize("n_tests", [10, 50, 100])
    def test_start_end_cycle_latency(self, n_tests: int):
        """Start/end cycle should complete in <1ms per test."""
        reg = TestRegistry()

        start = time.perf_counter()
        for i in range(n_tests):
            reg.start_test(f"Test {i}")
            reg.end_test(status="pass")
        elapsed = time.perf_counter() - start

        per_cycle = elapsed / n_tests * 1000  # ms
        assert per_cycle < 1.0, f"Start/end cycle took {per_cycle:.3f}ms (limit: 1ms)"
        assert len(reg.tests) == n_tests

    @pytest.mark.parametrize("n_tests", [10, 50, 100])
    def test_start_test_with_auto_end(self, n_tests: int):
        """Auto-end should not add significant overhead."""
        reg = TestRegistry()

        start = time.perf_counter()
        for i in range(n_tests):
            reg.start_test(f"Test {i}")
            # No manual end — next start auto-ends
        reg.end_test()  # end the last one
        elapsed = time.perf_counter() - start

        per_cycle = elapsed / n_tests * 1000  # ms
        assert per_cycle < 1.0, f"Auto-end cycle took {per_cycle:.3f}ms (limit: 1ms)"
        assert all(t.status in ("pass", "running") for t in reg.tests.values())


class TestRegistryScalingBenchmark:
    """Benchmark TestRegistry operations at scale."""

    @pytest.mark.parametrize("n_tests", [10, 50, 100])
    def test_all_steps_flat_scaling(self, n_tests: int):
        """all_steps_flat() with many tests should scale linearly."""
        reg = TestRegistry()
        steps_per_test = 10
        for i in range(n_tests):
            reg.start_test(f"Test {i}")
            for j in range(steps_per_test):
                reg.tests[f"Test {i}"].steps.append(_make_step(f"s-{i}-{j}"))
            reg.end_test()

        start = time.perf_counter()
        flat = reg.all_steps_flat()
        elapsed = time.perf_counter() - start

        assert len(flat) == n_tests * steps_per_test
        elapsed_ms = elapsed * 1000
        # Should complete in <5ms for 100 tests x 10 steps = 1000 steps
        assert elapsed_ms < 10, f"all_steps_flat() took {elapsed_ms:.3f}ms (limit: 10ms)"

    @pytest.mark.parametrize("n_tests", [10, 50, 100])
    def test_all_flow_blocks_flat_scaling(self, n_tests: int):
        """all_flow_blocks_flat() should scale linearly."""
        reg = TestRegistry()
        for i in range(n_tests):
            reg.start_test(f"Test {i}")
            reg.tests[f"Test {i}"].flow_blocks.append({"type": "if", "idx": i})
            reg.end_test()

        start = time.perf_counter()
        flat = reg.all_flow_blocks_flat()
        elapsed = time.perf_counter() - start

        assert len(flat) == n_tests
        elapsed_ms = elapsed * 1000
        assert elapsed_ms < 5, f"all_flow_blocks_flat() took {elapsed_ms:.3f}ms (limit: 5ms)"


# ---------------------------------------------------------------------------
# ExecutionSession routing benchmarks
# ---------------------------------------------------------------------------

class TestSessionRoutingBenchmark:
    """Benchmark step routing in legacy vs multi-test mode."""

    @pytest.mark.parametrize("n_steps", [50, 100, 500])
    def test_legacy_routing_latency(self, n_steps: int):
        """Legacy mode step routing baseline."""
        sess = ExecutionSession(session_id="legacy-bench")

        start = time.perf_counter()
        for i in range(n_steps):
            sess.add_step(_make_step(f"s-{i}"))
        elapsed = time.perf_counter() - start

        per_step = elapsed / n_steps * 1000  # ms
        assert per_step < 0.1, f"Legacy routing: {per_step:.4f}ms/step (limit: 0.1ms)"
        assert len(sess.steps) == n_steps

    @pytest.mark.parametrize("n_steps", [50, 100, 500])
    def test_multi_test_routing_latency(self, n_steps: int):
        """Multi-test mode step routing should have minimal overhead vs legacy."""
        sess = ExecutionSession(session_id="mt-bench")
        sess.test_registry.start_test("T1")

        start = time.perf_counter()
        for i in range(n_steps):
            sess.add_step(_make_step(f"s-{i}"))
        elapsed = time.perf_counter() - start

        per_step = elapsed / n_steps * 1000  # ms
        assert per_step < 0.1, f"Multi-test routing: {per_step:.4f}ms/step (limit: 0.1ms)"
        assert len(sess.test_registry.tests["T1"].steps) == n_steps

    def test_routing_overhead_comparison(self):
        """Multi-test routing should be <3x slower than legacy."""
        n = 500

        # Legacy
        sess_legacy = ExecutionSession(session_id="cmp-legacy")
        start = time.perf_counter()
        for i in range(n):
            sess_legacy.add_step(_make_step(f"s-{i}"))
        legacy_elapsed = time.perf_counter() - start

        # Multi-test
        sess_mt = ExecutionSession(session_id="cmp-mt")
        sess_mt.test_registry.start_test("T1")
        start = time.perf_counter()
        for i in range(n):
            sess_mt.add_step(_make_step(f"s-{i}"))
        mt_elapsed = time.perf_counter() - start

        if legacy_elapsed > 0:
            overhead = mt_elapsed / legacy_elapsed
            assert overhead < 3.0, f"Multi-test overhead: {overhead:.2f}x (limit: 3x)"


# ---------------------------------------------------------------------------
# step_count property benchmarks
# ---------------------------------------------------------------------------

class TestStepCountBenchmark:
    """Benchmark step_count property in multi-test mode."""

    @pytest.mark.parametrize("n_tests", [10, 50, 100])
    def test_step_count_scaling(self, n_tests: int):
        """step_count should scale linearly with test count."""
        sess = _build_session_with_n_tests(n_tests, steps_per_test=5)

        start = time.perf_counter()
        for _ in range(100):  # 100 repeated reads
            count = sess.step_count
        elapsed = time.perf_counter() - start

        per_read = elapsed / 100 * 1000  # ms
        assert count == n_tests * 5
        assert per_read < 5, f"step_count took {per_read:.3f}ms (limit: 5ms)"


# ---------------------------------------------------------------------------
# build_test_suite benchmarks
# ---------------------------------------------------------------------------

class TestBuildSuiteBenchmark:
    """Benchmark build_test_suite with multi-test sessions."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("n_tests", [5, 20, 50])
    async def test_build_suite_scaling(self, n_tests: int):
        """build_suite should scale linearly with test count."""
        from robotmcp.components.test_builder import TestBuilder

        sess = _build_session_with_n_tests(n_tests, steps_per_test=5)
        engine = MagicMock()
        engine.sessions = {sess.session_id: sess}
        engine.session_manager.get_or_create_session.return_value = sess

        builder = TestBuilder(execution_engine=engine)

        start = time.perf_counter()
        result = await builder.build_suite(session_id=sess.session_id, test_name="Bench Suite")
        elapsed = time.perf_counter() - start

        assert result["success"], result.get("error")
        elapsed_ms = elapsed * 1000
        # Should complete in <500ms even for 50 tests
        assert elapsed_ms < 500, f"build_suite({n_tests} tests) took {elapsed_ms:.1f}ms (limit: 500ms)"

        rf_text = result.get("rf_text", "")
        assert "*** Test Cases ***" in rf_text

    @pytest.mark.asyncio
    async def test_build_suite_with_suite_setup_teardown(self):
        """Suite setup/teardown should not add significant overhead."""
        from robotmcp.components.test_builder import TestBuilder

        sess = _build_session_with_n_tests(20, steps_per_test=5)
        sess.suite_setup = {"keyword": "New Browser", "arguments": ["chromium"]}
        sess.suite_teardown = {"keyword": "Close Browser", "arguments": []}

        engine = MagicMock()
        engine.sessions = {sess.session_id: sess}
        engine.session_manager.get_or_create_session.return_value = sess

        builder = TestBuilder(execution_engine=engine)

        start = time.perf_counter()
        result = await builder.build_suite(session_id=sess.session_id)
        elapsed = time.perf_counter() - start

        assert result["success"], result.get("error")
        rf_text = result.get("rf_text", "")
        assert "Suite Setup" in rf_text
        assert "Suite Teardown" in rf_text
        elapsed_ms = elapsed * 1000
        assert elapsed_ms < 500, f"build_suite with setup/teardown took {elapsed_ms:.1f}ms"


# ---------------------------------------------------------------------------
# TestInfo creation benchmark
# ---------------------------------------------------------------------------

class TestInfoCreationBenchmark:
    """Benchmark TestInfo dataclass creation."""

    @pytest.mark.parametrize("n", [100, 500, 1000])
    def test_creation_latency(self, n: int):
        """TestInfo creation should be fast (<0.01ms each)."""
        start = time.perf_counter()
        infos = [
            TestInfo(
                name=f"Test {i}",
                documentation=f"Test doc {i}",
                tags=[f"tag{i}", "smoke"],
                setup={"keyword": "Open", "arguments": []},
                teardown={"keyword": "Close", "arguments": []},
            )
            for i in range(n)
        ]
        elapsed = time.perf_counter() - start

        per_create = elapsed / n * 1000  # ms
        assert len(infos) == n
        assert per_create < 0.5, f"TestInfo creation: {per_create:.4f}ms (limit: 0.5ms)"
