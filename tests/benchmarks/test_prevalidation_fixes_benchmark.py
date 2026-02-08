"""Benchmarks for pre-validation timeout fixes.

Verifies that:
1. Keyword lookup in expanded ELEMENT_INTERACTION_KEYWORDS is still fast
2. Action type classification is efficient
3. Lock acquisition/release adds negligible overhead
4. Case normalization doesn't add measurable latency

Run with: uv run pytest tests/benchmarks/test_prevalidation_fixes_benchmark.py -v
"""

__test__ = True

import threading
import time

import pytest

from robotmcp.components.execution.keyword_executor import KeywordExecutor
from robotmcp.models.config_models import ExecutionConfig


@pytest.fixture
def executor():
    return KeywordExecutor(config=ExecutionConfig())


class TestKeywordLookupBenchmarks:
    """Benchmark keyword set membership checks after expansion."""

    def test_positive_lookup_latency(self, executor, benchmark_reporter):
        """Positive lookup (keyword IS in set) should be < 0.01ms."""
        iterations = 10000
        t0 = time.time()
        for _ in range(iterations):
            executor._requires_pre_validation("click")
        elapsed = (time.time() - t0) * 1000
        avg = elapsed / iterations

        benchmark_reporter.record_latency("positive_keyword_lookup", avg, 0.01, iterations)
        assert avg < 0.01

    def test_negative_lookup_latency(self, executor, benchmark_reporter):
        """Negative lookup (keyword NOT in set) should be < 0.01ms."""
        iterations = 10000
        t0 = time.time()
        for _ in range(iterations):
            executor._requires_pre_validation("log")
        elapsed = (time.time() - t0) * 1000
        avg = elapsed / iterations

        benchmark_reporter.record_latency("negative_keyword_lookup", avg, 0.01, iterations)
        assert avg < 0.01

    def test_new_keyword_lookup_latency(self, executor, benchmark_reporter):
        """New keyword lookup should be as fast as original."""
        iterations = 10000
        t0 = time.time()
        for _ in range(iterations):
            executor._requires_pre_validation("click button")
        elapsed = (time.time() - t0) * 1000
        avg = elapsed / iterations

        benchmark_reporter.record_latency("new_keyword_lookup", avg, 0.01, iterations)
        assert avg < 0.01

    def test_keyword_set_size_reasonable(self, executor):
        """Expanded set should be manageable in size."""
        size = len(executor.ELEMENT_INTERACTION_KEYWORDS)
        assert size < 100, f"ELEMENT_INTERACTION_KEYWORDS has {size} entries — keep it under 100"
        assert size > 40, f"Expected > 40 keywords after expansion, got {size}"


class TestActionTypeBenchmarks:
    """Benchmark action type classification."""

    @pytest.mark.parametrize("keyword", [
        "click", "fill text", "drag and drop", "tap",
        "submit form", "choose file", "scroll by",
    ])
    def test_action_type_classification_latency(self, executor, benchmark_reporter, keyword):
        """Action type classification should be < 0.01ms."""
        iterations = 10000
        t0 = time.time()
        for _ in range(iterations):
            executor._get_action_type_from_keyword_for_states(keyword)
        elapsed = (time.time() - t0) * 1000
        avg = elapsed / iterations

        benchmark_reporter.record_latency(
            f"action_type_{keyword.replace(' ', '_')}", avg, 0.01, iterations
        )
        assert avg < 0.01


class TestLockBenchmarks:
    """Benchmark threading lock overhead."""

    def test_lock_acquire_release_overhead(self, executor, benchmark_reporter):
        """Lock acquire+release should be < 0.01ms."""
        lock = executor._browser_timeout_lock
        iterations = 10000
        t0 = time.time()
        for _ in range(iterations):
            lock.acquire()
            lock.release()
        elapsed = (time.time() - t0) * 1000
        avg = elapsed / iterations

        benchmark_reporter.record_latency("lock_acquire_release", avg, 0.01, iterations)
        assert avg < 0.01

    def test_concurrent_lock_contention(self, executor, benchmark_reporter):
        """Multiple threads contending for the lock should complete in < 500ms."""
        lock = executor._browser_timeout_lock

        def worker():
            for _ in range(100):
                lock.acquire()
                time.sleep(0.0001)
                lock.release()

        threads = [threading.Thread(target=worker) for _ in range(4)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total = (time.time() - t0) * 1000

        benchmark_reporter.record_latency("lock_contention_4_threads", total, 500.0, 1)
        assert total < 500, f"Lock contention took {total:.1f}ms — expected < 500ms"


class TestCaseNormalizationBenchmarks:
    """Benchmark case normalization overhead."""

    def test_lowercase_normalization(self, benchmark_reporter):
        """str.lower() should add < 0.001ms overhead."""
        iterations = 100000
        s = "Browser"
        t0 = time.time()
        for _ in range(iterations):
            s.lower()
        elapsed = (time.time() - t0) * 1000
        avg = elapsed / iterations

        benchmark_reporter.record_latency("lowercase_normalization", avg, 0.001, iterations)
        assert avg < 0.001

    def test_set_membership_after_lowercase(self, executor, benchmark_reporter):
        """Lowercase + set lookup should be < 0.01ms."""
        iterations = 10000
        keyword_set = executor.ELEMENT_INTERACTION_KEYWORDS

        t0 = time.time()
        for _ in range(iterations):
            kw = "Click Button"
            kw.lower().strip() in keyword_set
        elapsed = (time.time() - t0) * 1000
        avg = elapsed / iterations

        benchmark_reporter.record_latency("lowercase_and_lookup", avg, 0.01, iterations)
        assert avg < 0.01


class TestPreValidationOverhead:
    """Measure total overhead of pre-validation decision path."""

    def test_full_preval_decision_path(self, executor, benchmark_reporter):
        """Full pre-validation decision (requires_preval + action_type + states) < 0.01ms."""
        iterations = 10000

        t0 = time.time()
        for _ in range(iterations):
            keyword = "click button"
            if executor._requires_pre_validation(keyword):
                action_type = executor._get_action_type_from_keyword_for_states(keyword)
                executor.REQUIRED_STATES_FOR_ACTION.get(action_type, {"visible"})
        elapsed = (time.time() - t0) * 1000
        avg = elapsed / iterations

        benchmark_reporter.record_latency("full_preval_decision", avg, 0.01, iterations)
        assert avg < 0.01

    def test_timeout_min_calculation(self, benchmark_reporter):
        """min(default, user_timeout) calculation should be < 0.001ms."""
        default = 500
        iterations = 100000

        t0 = time.time()
        for _ in range(iterations):
            min(default, 30000)
        elapsed = (time.time() - t0) * 1000
        avg = elapsed / iterations

        benchmark_reporter.record_latency("timeout_min_calculation", avg, 0.001, iterations)
        assert avg < 0.001
