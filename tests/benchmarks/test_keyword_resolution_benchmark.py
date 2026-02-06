"""Benchmark Tests for Keyword Resolution Chain (C1/C2/C3 fixes).

Measures the performance of the 3-tier resolution chain:
1. namespace.get_runner() — RF 7 native
2. Direct library method lookup (with TestLibrary unwrapping)
3. BuiltIn.run_keyword() fallback

These benchmarks validate that the fix doesn't introduce performance regressions
compared to the previous (broken) code path that relied on exception-driven fallthrough.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from robotmcp.components.execution.rf_native_context_manager import RobotFrameworkNativeContextManager


# ---------------------------------------------------------------------------
# Constants — patch targets
# ---------------------------------------------------------------------------

_EC_PATCH = "robotmcp.components.execution.rf_native_context_manager.EXECUTION_CONTEXTS"
_BUILTIN_PATCH = "robot.libraries.BuiltIn.BuiltIn"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manager() -> RFNativeContextManager:
    return RobotFrameworkNativeContextManager()


class FakeRunner:
    """Minimal runner that captures call arguments and returns a fixed result."""

    def __init__(self, result: Any = "runner_result"):
        self._result = result
        self.call_count = 0

    def run(self, data_kw, res_kw, ctx):
        self.call_count += 1
        return self._result


class FakeNamespace:
    def __init__(self, runner=None, raise_on_get_runner=False):
        self._runner = runner
        self._raise = raise_on_get_runner

    def get_runner(self, name):
        if self._raise:
            raise AttributeError(f"No runner for {name}")
        return self._runner


class FakeLibraryInstance:
    """A fake raw Python library instance with configurable keyword methods."""

    def __init__(self, methods: Dict[str, Any]):
        for name, impl in methods.items():
            setattr(self, name, impl)


class FakeTestLibrary:
    """Mimics RF 7.x TestLibrary wrapper."""

    def __init__(self, name: str, instance: Any):
        self.name = name
        self.instance = instance


def _make_ctx_with_libraries(libraries: list):
    ctx = MagicMock()
    ctx.namespace.libraries = libraries
    return ctx


def _timed(fn, *args, iterations=1000, **kwargs):
    """Run fn N times and return (total_ms, per_call_us)."""
    start = time.perf_counter()
    for _ in range(iterations):
        fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    total_ms = elapsed * 1000
    per_call_us = (elapsed / iterations) * 1_000_000
    return total_ms, per_call_us


# ===========================================================================
# Tier 1: namespace.get_runner() — the fast path
# ===========================================================================


class TestTier1GetRunnerPerformance:
    """Benchmark the primary get_runner resolution path."""

    def test_get_runner_throughput(self):
        """get_runner path should resolve 1000 keywords in < 200ms."""
        runner = FakeRunner()
        ns = FakeNamespace(runner=runner)
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = MagicMock()
            total_ms, per_call_us = _timed(
                mgr._execute_any_keyword_generic, "Click", ["loc"], ns,
                iterations=1000,
            )

        assert total_ms < 200, f"Tier 1 too slow: {total_ms:.1f}ms for 1000 calls"
        assert runner.call_count == 1000

    def test_get_runner_per_call_latency(self):
        """Single get_runner call should be < 200us."""
        runner = FakeRunner()
        ns = FakeNamespace(runner=runner)
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = MagicMock()
            _, per_call_us = _timed(
                mgr._execute_any_keyword_generic, "Click", ["loc"], ns,
                iterations=5000,
            )

        assert per_call_us < 200, f"Per-call latency too high: {per_call_us:.1f}us"

    def test_get_runner_with_many_arguments(self):
        """get_runner path with 20 args should not be significantly slower."""
        runner = FakeRunner()
        ns = FakeNamespace(runner=runner)
        mgr = _make_manager()
        args = [f"arg{i}" for i in range(20)]

        with patch(_EC_PATCH) as ec:
            ec.current = MagicMock()
            total_ms, _ = _timed(
                mgr._execute_any_keyword_generic, "Create Dictionary", args, ns,
                iterations=1000,
            )

        assert total_ms < 250, f"Many-args too slow: {total_ms:.1f}ms for 1000 calls"


# ===========================================================================
# Tier 2: Library method lookup
# ===========================================================================


class TestTier2LibrarySearchPerformance:
    """Benchmark the library method search fallback."""

    def test_library_search_single_lib(self):
        """Direct method lookup on 1 library should resolve in < 300ms for 1000 calls."""
        ns = FakeNamespace(raise_on_get_runner=True)
        mgr = _make_manager()

        raw_lib = FakeLibraryInstance({"click": lambda *a: "clicked"})
        test_lib = FakeTestLibrary("Browser", raw_lib)

        with patch(_EC_PATCH) as ec:
            ec.current = _make_ctx_with_libraries([test_lib])
            total_ms, _ = _timed(
                mgr._execute_any_keyword_generic, "Click", ["loc"], ns,
                iterations=1000,
            )

        assert total_ms < 300, f"Library search (1 lib) too slow: {total_ms:.1f}ms"

    def test_library_search_five_libs(self):
        """Search across 5 libraries where match is in the last one."""
        ns = FakeNamespace(raise_on_get_runner=True)
        mgr = _make_manager()

        libs = []
        for i in range(4):
            libs.append(FakeTestLibrary(f"Lib{i}", FakeLibraryInstance({})))
        # Target method in last library
        raw_target = FakeLibraryInstance({"new_browser": lambda *a: "browser_created"})
        libs.append(FakeTestLibrary("Browser", raw_target))

        with patch(_EC_PATCH) as ec:
            ec.current = _make_ctx_with_libraries(libs)
            total_ms, _ = _timed(
                mgr._execute_any_keyword_generic, "New Browser", [], ns,
                iterations=1000,
            )

        assert total_ms < 500, f"Library search (5 libs) too slow: {total_ms:.1f}ms"

    def test_library_unwrap_overhead(self):
        """TestLibrary unwrapping adds negligible overhead."""
        mgr = _make_manager()
        raw_lib = FakeLibraryInstance({"log": lambda *a: "logged"})

        # Direct call (no unwrap needed)
        _, direct_us = _timed(
            mgr._try_execute_from_library, "Log", ["msg"], "BuiltIn", raw_lib,
            iterations=5000,
        )

        # With TestLibrary wrapper (unwrap needed in _execute_any_keyword_generic)
        ns = FakeNamespace(raise_on_get_runner=True)
        test_lib = FakeTestLibrary("BuiltIn", raw_lib)
        with patch(_EC_PATCH) as ec:
            ec.current = _make_ctx_with_libraries([test_lib])
            _, wrapped_us = _timed(
                mgr._execute_any_keyword_generic, "Log", ["msg"], ns,
                iterations=5000,
            )

        overhead = wrapped_us - direct_us
        assert overhead < 100, f"Unwrap overhead too high: {overhead:.1f}us"


# ===========================================================================
# Tier 3: BuiltIn fallback
# ===========================================================================


class TestTier3BuiltInFallbackPerformance:
    """Benchmark the BuiltIn.run_keyword() fallback path."""

    def test_builtin_fallback_throughput(self):
        """BuiltIn fallback should resolve in < 500ms for 1000 calls."""
        mgr = _make_manager()

        with patch(_BUILTIN_PATCH) as MockBI:
            instance = MockBI.return_value
            instance.run_keyword.return_value = "fallback_result"
            total_ms, _ = _timed(
                mgr._final_fallback_execution, "Log", ["message"],
                iterations=1000,
            )

        assert total_ms < 500, f"BuiltIn fallback too slow: {total_ms:.1f}ms"

    def test_builtin_direct_method_fallback(self):
        """When run_keyword fails, direct method should still be fast."""
        mgr = _make_manager()

        with patch(_BUILTIN_PATCH) as MockBI:
            instance = MockBI.return_value
            instance.run_keyword.side_effect = RuntimeError("no namespace")
            instance.log = MagicMock(return_value="direct_log")
            total_ms, _ = _timed(
                mgr._final_fallback_execution, "Log", ["message"],
                iterations=1000,
            )

        assert total_ms < 800, f"Direct method fallback too slow: {total_ms:.1f}ms"


# ===========================================================================
# Full chain benchmarks
# ===========================================================================


class TestFullChainPerformance:
    """End-to-end benchmarks for the full resolution chain."""

    def test_happy_path_latency(self):
        """Happy path (get_runner succeeds) < 100us per call."""
        runner = FakeRunner()
        ns = FakeNamespace(runner=runner)
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = MagicMock()
            _, per_call_us = _timed(
                mgr._execute_any_keyword_generic, "Log", ["msg"], ns,
                iterations=10000,
            )

        assert per_call_us < 100, f"Happy path too slow: {per_call_us:.1f}us"

    def test_fallthrough_latency(self):
        """Full fallthrough (get_runner fails → lib search fails → BuiltIn) < 500us."""
        ns = FakeNamespace(raise_on_get_runner=True)
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = _make_ctx_with_libraries([])  # No libraries
            with patch(_BUILTIN_PATCH) as MockBI:
                instance = MockBI.return_value
                instance.run_keyword.return_value = "fallback"
                _, per_call_us = _timed(
                    mgr._execute_any_keyword_generic, "Log", ["msg"], ns,
                    iterations=5000,
                )

        assert per_call_us < 500, f"Fallthrough too slow: {per_call_us:.1f}us"

    def test_method_name_normalization_overhead(self):
        """Method name normalization (space→underscore, lowering) is trivial."""
        iterations = 100_000
        start = time.perf_counter()
        for _ in range(iterations):
            name = "New Browser"
            _ = name.replace(' ', '_').lower()
            _ = name.replace(' ', '_')
            _ = name.replace(' ', '').lower()
        elapsed_us = (time.perf_counter() - start) * 1_000_000 / iterations

        assert elapsed_us < 1, f"Name normalization too slow: {elapsed_us:.3f}us"

    def test_no_double_execution(self):
        """Keyword is executed exactly once when get_runner succeeds."""
        call_count = 0

        class CountingRunner:
            def run(self, data_kw, res_kw, ctx):
                nonlocal call_count
                call_count += 1
                return "once"

        ns = FakeNamespace(runner=CountingRunner())
        mgr = _make_manager()

        with patch(_EC_PATCH) as ec:
            ec.current = MagicMock()
            for _ in range(100):
                mgr._execute_any_keyword_generic("Click", ["loc"], ns)

        assert call_count == 100, f"Expected 100 calls, got {call_count}"
