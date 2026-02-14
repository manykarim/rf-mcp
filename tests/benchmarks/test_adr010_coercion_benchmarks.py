"""Benchmarks for ADR-010 Small LLM Resilience — Array Coercion & Guided Recovery.

Measures the performance of:
1. _coerce_string_to_list — list passthrough, JSON parse, comma split, single wrap, None
2. CoercedStringList / OptionalCoercedStringList Pydantic TypeAdapter validation
3. extract_deprecation_suggestion / resolve_deprecated_alias regex & dict lookup
4. JSON Schema generation for CoercedStringList vs plain List[str]

These are pure domain-layer benchmarks (no MCP server, no browser).

Run with: uv run pytest tests/benchmarks/test_adr010_coercion_benchmarks.py -v
"""

from __future__ import annotations

__test__ = True

import time
from typing import List, Optional

import pytest
from pydantic import TypeAdapter

from robotmcp.domains.shared.kernel import (
    CoercedStringList,
    OptionalCoercedStringList,
    _coerce_string_to_list,
    extract_deprecation_suggestion,
    resolve_deprecated_alias,
)


# ============================================================
# 1. Coercion function performance benchmarks
# ============================================================


class TestCoercionBenchmarks:
    """Benchmark _coerce_string_to_list across all input paths."""

    def test_bench_list_passthrough(self, benchmark_reporter):
        """List passthrough (no coercion needed) should be < 0.001ms."""
        data = ["Browser", "BuiltIn"]
        iterations = 100_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            _coerce_string_to_list(data)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "_coerce_string_to_list(list) passthrough",
            elapsed_ms, target_ms=0.001, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.001, f"List passthrough too slow: {avg_ms:.6f}ms"

    def test_bench_json_string_parse(self, benchmark_reporter):
        """JSON string parsing should be < 0.005ms."""
        data = '["Browser", "BuiltIn"]'
        iterations = 50_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            _coerce_string_to_list(data)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            '_coerce_string_to_list(\'["Browser","BuiltIn"]\')',
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        result = _coerce_string_to_list(data)
        assert result == ["Browser", "BuiltIn"]
        assert avg_ms < 0.005, f"JSON parse too slow: {avg_ms:.6f}ms"

    def test_bench_comma_separated(self, benchmark_reporter):
        """Comma-separated string splitting should be < 0.003ms."""
        data = "Browser, BuiltIn"
        iterations = 100_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            _coerce_string_to_list(data)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "_coerce_string_to_list('Browser, BuiltIn') comma split",
            elapsed_ms, target_ms=0.003, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        result = _coerce_string_to_list(data)
        assert result == ["Browser", "BuiltIn"]
        assert avg_ms < 0.003, f"Comma split too slow: {avg_ms:.6f}ms"

    def test_bench_single_value(self, benchmark_reporter):
        """Single value wrapping should be < 0.001ms."""
        data = "Browser"
        iterations = 100_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            _coerce_string_to_list(data)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "_coerce_string_to_list('Browser') single wrap",
            elapsed_ms, target_ms=0.001, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        result = _coerce_string_to_list(data)
        assert result == ["Browser"]
        assert avg_ms < 0.001, f"Single value wrap too slow: {avg_ms:.6f}ms"

    def test_bench_none_passthrough(self, benchmark_reporter):
        """None passthrough should be < 0.001ms."""
        iterations = 100_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            _coerce_string_to_list(None)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "_coerce_string_to_list(None) passthrough",
            elapsed_ms, target_ms=0.001, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        result = _coerce_string_to_list(None)
        assert result is None
        assert avg_ms < 0.001, f"None passthrough too slow: {avg_ms:.6f}ms"

    def test_bench_json_vs_comma_ratio(self, benchmark_reporter):
        """JSON parse overhead should be < 5x comma split."""
        json_data = '["Browser", "BuiltIn"]'
        comma_data = "Browser, BuiltIn"
        iterations = 50_000

        t0 = time.perf_counter()
        for _ in range(iterations):
            _coerce_string_to_list(comma_data)
        comma_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for _ in range(iterations):
            _coerce_string_to_list(json_data)
        json_ms = (time.perf_counter() - t0) * 1000

        comma_avg = comma_ms / iterations
        json_avg = json_ms / iterations
        ratio = json_avg / comma_avg if comma_avg > 0 else float("inf")

        benchmark_reporter.record_latency(
            "comma split path", comma_ms, target_ms=0.003, iterations=iterations,
        )
        benchmark_reporter.record_latency(
            "JSON parse path", json_ms, target_ms=0.005, iterations=iterations,
        )

        assert ratio < 5.0, (
            f"JSON overhead too high: {ratio:.2f}x "
            f"(comma={comma_avg:.6f}ms, json={json_avg:.6f}ms)"
        )

    def test_bench_many_items_comma(self, benchmark_reporter):
        """Comma-separated with 10 items should be < 0.005ms."""
        data = ", ".join(f"Library{i}" for i in range(10))
        iterations = 50_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            _coerce_string_to_list(data)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "_coerce_string_to_list(10 comma-separated items)",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        result = _coerce_string_to_list(data)
        assert len(result) == 10
        assert avg_ms < 0.005, f"10-item comma split too slow: {avg_ms:.6f}ms"

    def test_bench_many_items_json(self, benchmark_reporter):
        """JSON array with 10 items should be < 0.01ms."""
        import json as json_mod
        items = [f"Library{i}" for i in range(10)]
        data = json_mod.dumps(items)
        iterations = 50_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            _coerce_string_to_list(data)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "_coerce_string_to_list(10-item JSON array)",
            elapsed_ms, target_ms=0.01, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        result = _coerce_string_to_list(data)
        assert len(result) == 10
        assert avg_ms < 0.01, f"10-item JSON parse too slow: {avg_ms:.6f}ms"


# ============================================================
# 2. Pydantic TypeAdapter validation benchmarks
# ============================================================


class TestPydanticBenchmarks:
    """Benchmark TypeAdapter validation for CoercedStringList types."""

    def test_bench_coerced_list_from_json_string(self, benchmark_reporter):
        """CoercedStringList TypeAdapter validation from JSON string; < 0.01ms."""
        ta = TypeAdapter(CoercedStringList)
        data = '["Browser", "BuiltIn"]'
        iterations = 50_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python(data)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "CoercedStringList validate_python(JSON string)",
            elapsed_ms, target_ms=0.01, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.01, f"CoercedStringList JSON validation too slow: {avg_ms:.6f}ms"

    def test_bench_coerced_list_from_native_list(self, benchmark_reporter):
        """CoercedStringList TypeAdapter validation from native list; < 0.005ms."""
        ta = TypeAdapter(CoercedStringList)
        data = ["Browser", "BuiltIn"]
        iterations = 50_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python(data)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "CoercedStringList validate_python(native list)",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.005, f"CoercedStringList native list validation too slow: {avg_ms:.6f}ms"

    def test_bench_plain_list_validation_baseline(self, benchmark_reporter):
        """Plain List[str] TypeAdapter validation (baseline); < 0.003ms."""
        ta = TypeAdapter(List[str])
        data = ["Browser", "BuiltIn"]
        iterations = 50_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python(data)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "List[str] validate_python(native list) baseline",
            elapsed_ms, target_ms=0.003, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.003, f"Plain List[str] validation too slow: {avg_ms:.6f}ms"

    def test_bench_optional_coerced_with_none(self, benchmark_reporter):
        """OptionalCoercedStringList with None; < 0.003ms."""
        ta = TypeAdapter(OptionalCoercedStringList)
        iterations = 50_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python(None)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "OptionalCoercedStringList validate_python(None)",
            elapsed_ms, target_ms=0.003, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.003, f"OptionalCoercedStringList None validation too slow: {avg_ms:.6f}ms"

    def test_bench_optional_coerced_with_value(self, benchmark_reporter):
        """OptionalCoercedStringList with JSON string; < 0.01ms."""
        ta = TypeAdapter(OptionalCoercedStringList)
        data = '["Browser", "BuiltIn"]'
        iterations = 50_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python(data)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "OptionalCoercedStringList validate_python(JSON string)",
            elapsed_ms, target_ms=0.01, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.01, f"OptionalCoercedStringList value validation too slow: {avg_ms:.6f}ms"

    def test_bench_coerced_vs_plain_overhead(self, benchmark_reporter):
        """CoercedStringList overhead vs plain List[str] should be < 3x for native list."""
        ta_coerced = TypeAdapter(CoercedStringList)
        ta_plain = TypeAdapter(List[str])
        data = ["Browser", "BuiltIn"]
        iterations = 50_000

        t0 = time.perf_counter()
        for _ in range(iterations):
            ta_plain.validate_python(data)
        plain_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for _ in range(iterations):
            ta_coerced.validate_python(data)
        coerced_ms = (time.perf_counter() - t0) * 1000

        plain_avg = plain_ms / iterations
        coerced_avg = coerced_ms / iterations
        ratio = coerced_avg / plain_avg if plain_avg > 0 else float("inf")

        benchmark_reporter.record_latency(
            "List[str] baseline validation",
            plain_ms, target_ms=0.003, iterations=iterations,
        )
        benchmark_reporter.record_latency(
            "CoercedStringList validation (native list input)",
            coerced_ms, target_ms=0.005, iterations=iterations,
        )

        assert ratio < 3.0, (
            f"Coerced overhead too high: {ratio:.2f}x "
            f"(plain={plain_avg:.6f}ms, coerced={coerced_avg:.6f}ms)"
        )

    def test_bench_adapter_creation_coerced(self, benchmark_reporter):
        """TypeAdapter(CoercedStringList) creation should be < 0.25ms."""
        iterations = 5_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta = TypeAdapter(CoercedStringList)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "TypeAdapter(CoercedStringList) creation",
            elapsed_ms, target_ms=0.25, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert ta is not None
        assert avg_ms < 0.25, f"CoercedStringList adapter creation too slow: {avg_ms:.4f}ms"

    def test_bench_adapter_creation_optional_coerced(self, benchmark_reporter):
        """TypeAdapter(OptionalCoercedStringList) creation should be < 0.25ms."""
        iterations = 5_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta = TypeAdapter(OptionalCoercedStringList)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "TypeAdapter(OptionalCoercedStringList) creation",
            elapsed_ms, target_ms=0.25, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert ta is not None
        assert avg_ms < 0.25, f"OptionalCoercedStringList adapter creation too slow: {avg_ms:.4f}ms"


# ============================================================
# 3. Deprecation detection performance benchmarks
# ============================================================


class TestDeprecationBenchmarks:
    """Benchmark deprecation regex extraction and alias lookup."""

    def test_bench_extract_deprecation_suggestion_match(self, benchmark_reporter):
        """Deprecation regex extraction (matching input) should be < 0.005ms."""
        msg = "DeprecationWarning: 'GET' is deprecated. Use 'GET On Session' instead."
        iterations = 50_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            extract_deprecation_suggestion(msg)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "extract_deprecation_suggestion(match)",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        result = extract_deprecation_suggestion(msg)
        assert result is not None
        assert "GET On Session" in result
        assert avg_ms < 0.005, f"Deprecation regex match too slow: {avg_ms:.6f}ms"

    def test_bench_extract_deprecation_suggestion_no_match(self, benchmark_reporter):
        """Deprecation regex extraction (no match) should be < 0.003ms."""
        msg = "SomeOtherError: something went wrong"
        iterations = 100_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            extract_deprecation_suggestion(msg)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "extract_deprecation_suggestion(no match)",
            elapsed_ms, target_ms=0.003, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        result = extract_deprecation_suggestion(msg)
        assert result is None
        assert avg_ms < 0.003, f"Deprecation regex no-match too slow: {avg_ms:.6f}ms"

    def test_bench_resolve_deprecated_alias_hit(self, benchmark_reporter):
        """Deprecated alias lookup (hit) should be < 0.001ms."""
        iterations = 100_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            resolve_deprecated_alias("GET")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "resolve_deprecated_alias('GET') hit",
            elapsed_ms, target_ms=0.001, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        result = resolve_deprecated_alias("GET")
        assert result == "GET On Session"
        assert avg_ms < 0.001, f"Alias lookup hit too slow: {avg_ms:.6f}ms"

    def test_bench_resolve_deprecated_alias_miss(self, benchmark_reporter):
        """Deprecated alias lookup (miss) should be < 0.001ms."""
        iterations = 100_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            resolve_deprecated_alias("Click Element")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "resolve_deprecated_alias('Click Element') miss",
            elapsed_ms, target_ms=0.001, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        result = resolve_deprecated_alias("Click Element")
        assert result is None
        assert avg_ms < 0.001, f"Alias lookup miss too slow: {avg_ms:.6f}ms"

    def test_bench_all_deprecated_aliases(self, benchmark_reporter):
        """Looking up all 7 deprecated aliases should be < 0.005ms per sweep."""
        aliases = ["get", "post", "put", "delete", "patch", "head", "options"]
        iterations = 20_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            for kw in aliases:
                resolve_deprecated_alias(kw)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "resolve_deprecated_alias all 7 aliases sweep",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.005, f"All-aliases sweep too slow: {avg_ms:.4f}ms"

    def test_bench_deprecation_regex_various_patterns(self, benchmark_reporter):
        """Various deprecation message formats; all < 0.005ms avg."""
        messages = [
            "DeprecationWarning: 'GET' is deprecated. Use 'GET On Session' instead.",
            "Warning: use 'POST On Session' instead of POST.",
            "Deprecated in favor of DELETE On Session.",
            "This keyword is no longer supported.",  # no match
            "",  # empty string
        ]
        iterations = 20_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            for msg in messages:
                extract_deprecation_suggestion(msg)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "extract_deprecation_suggestion 5 patterns sweep",
            elapsed_ms, target_ms=0.02, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.02, f"Multi-pattern regex sweep too slow: {avg_ms:.4f}ms"


# ============================================================
# 4. Schema generation performance benchmarks
# ============================================================


class TestSchemaBenchmarks:
    """Benchmark JSON Schema generation for CoercedStringList types."""

    def test_bench_coerced_schema_generation(self, benchmark_reporter):
        """CoercedStringList JSON Schema generation should be < 0.15ms."""
        ta = TypeAdapter(CoercedStringList)
        iterations = 5_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            schema = ta.json_schema()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "CoercedStringList json_schema()",
            elapsed_ms, target_ms=0.15, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        # Schema must be transparent — identical to plain List[str]
        assert schema.get("type") == "array", f"Expected array type, got: {schema}"
        assert avg_ms < 0.15, f"CoercedStringList schema generation too slow: {avg_ms:.4f}ms"

    def test_bench_plain_list_schema_generation(self, benchmark_reporter):
        """Plain List[str] schema generation (baseline) should be < 0.15ms."""
        ta = TypeAdapter(List[str])
        iterations = 5_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            schema = ta.json_schema()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "List[str] json_schema() baseline",
            elapsed_ms, target_ms=0.15, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert schema.get("type") == "array"
        assert avg_ms < 0.15, f"Plain List[str] schema generation too slow: {avg_ms:.4f}ms"

    def test_bench_optional_coerced_schema_generation(self, benchmark_reporter):
        """OptionalCoercedStringList JSON Schema generation should be < 0.15ms."""
        ta = TypeAdapter(OptionalCoercedStringList)
        iterations = 5_000
        t0 = time.perf_counter()
        for _ in range(iterations):
            schema = ta.json_schema()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "OptionalCoercedStringList json_schema()",
            elapsed_ms, target_ms=0.15, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        # Optional type produces anyOf with null
        assert avg_ms < 0.15, f"OptionalCoercedStringList schema too slow: {avg_ms:.4f}ms"

    def test_bench_schema_transparency(self, benchmark_reporter):
        """CoercedStringList schema must be identical to plain List[str] schema."""
        ta_coerced = TypeAdapter(CoercedStringList)
        ta_plain = TypeAdapter(List[str])

        schema_coerced = ta_coerced.json_schema()
        schema_plain = ta_plain.json_schema()

        # Core schema transparency invariant from ADR-010
        assert schema_coerced == schema_plain, (
            f"Schema mismatch!\n"
            f"CoercedStringList: {schema_coerced}\n"
            f"List[str]:         {schema_plain}"
        )

    def test_bench_coerced_vs_plain_schema_overhead(self, benchmark_reporter):
        """CoercedStringList schema generation overhead should be < 2x plain List[str]."""
        ta_coerced = TypeAdapter(CoercedStringList)
        ta_plain = TypeAdapter(List[str])
        iterations = 5_000

        t0 = time.perf_counter()
        for _ in range(iterations):
            ta_plain.json_schema()
        plain_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for _ in range(iterations):
            ta_coerced.json_schema()
        coerced_ms = (time.perf_counter() - t0) * 1000

        plain_avg = plain_ms / iterations
        coerced_avg = coerced_ms / iterations
        ratio = coerced_avg / plain_avg if plain_avg > 0 else float("inf")

        benchmark_reporter.record_latency(
            "List[str] schema baseline",
            plain_ms, target_ms=0.1, iterations=iterations,
        )
        benchmark_reporter.record_latency(
            "CoercedStringList schema",
            coerced_ms, target_ms=0.15, iterations=iterations,
        )

        assert ratio < 2.0, (
            f"Coerced schema overhead too high: {ratio:.2f}x "
            f"(plain={plain_avg:.6f}ms, coerced={coerced_avg:.6f}ms)"
        )


# ============================================================
# 5. Correctness sanity checks (non-benchmark, fast)
# ============================================================


class TestADR010Correctness:
    """Non-benchmark correctness tests for ADR-010 coercion and deprecation."""

    def test_coerce_list_passthrough(self):
        """Native list passes through unchanged."""
        data = ["Browser", "BuiltIn"]
        assert _coerce_string_to_list(data) is data

    def test_coerce_json_string(self):
        """JSON array string is parsed to list."""
        assert _coerce_string_to_list('["Browser", "BuiltIn"]') == ["Browser", "BuiltIn"]

    def test_coerce_comma_separated(self):
        """Comma-separated string is split."""
        assert _coerce_string_to_list("Browser, BuiltIn") == ["Browser", "BuiltIn"]

    def test_coerce_single_value(self):
        """Single string is wrapped in list."""
        assert _coerce_string_to_list("Browser") == ["Browser"]

    def test_coerce_none_passthrough(self):
        """None passes through unchanged."""
        assert _coerce_string_to_list(None) is None

    def test_coerce_empty_string(self):
        """Empty string passes through (no wrapping)."""
        result = _coerce_string_to_list("")
        assert result == ""

    def test_coerce_whitespace_string(self):
        """Whitespace-only string passes through."""
        result = _coerce_string_to_list("   ")
        assert result == "   "

    def test_coerce_json_with_whitespace(self):
        """JSON array with leading/trailing whitespace is parsed."""
        assert _coerce_string_to_list('  ["A", "B"]  ') == ["A", "B"]

    def test_deprecated_alias_all_7(self):
        """All 7 deprecated aliases resolve correctly."""
        expected = {
            "get": "GET On Session",
            "post": "POST On Session",
            "put": "PUT On Session",
            "delete": "DELETE On Session",
            "patch": "PATCH On Session",
            "head": "HEAD On Session",
            "options": "OPTIONS On Session",
        }
        for alias, replacement in expected.items():
            assert resolve_deprecated_alias(alias) == replacement

    def test_deprecated_alias_case_insensitive(self):
        """Alias lookup is case-insensitive."""
        assert resolve_deprecated_alias("GET") == "GET On Session"
        assert resolve_deprecated_alias("get") == "GET On Session"
        assert resolve_deprecated_alias("Get") == "GET On Session"

    def test_extract_deprecation_use_pattern(self):
        """Extract from 'Use X instead' pattern."""
        msg = "DeprecationWarning: 'GET' is deprecated. Use 'GET On Session' instead."
        result = extract_deprecation_suggestion(msg)
        assert result is not None
        assert "GET On Session" in result

    def test_extract_deprecation_favor_pattern(self):
        """Extract from 'in favor of X' pattern."""
        msg = "Deprecated in favor of DELETE On Session."
        result = extract_deprecation_suggestion(msg)
        assert result is not None
        assert "DELETE On Session" in result

    def test_extract_deprecation_no_match(self):
        """Non-deprecation message returns None."""
        assert extract_deprecation_suggestion("SomeOtherError") is None
        assert extract_deprecation_suggestion("") is None

    def test_pydantic_coerced_validates_json_string(self):
        """TypeAdapter(CoercedStringList) accepts JSON string."""
        ta = TypeAdapter(CoercedStringList)
        result = ta.validate_python('["Browser"]')
        assert result == ["Browser"]

    def test_pydantic_coerced_validates_comma_string(self):
        """TypeAdapter(CoercedStringList) accepts comma-separated string."""
        ta = TypeAdapter(CoercedStringList)
        result = ta.validate_python("Browser, BuiltIn")
        assert result == ["Browser", "BuiltIn"]

    def test_pydantic_optional_coerced_accepts_none(self):
        """TypeAdapter(OptionalCoercedStringList) accepts None."""
        ta = TypeAdapter(OptionalCoercedStringList)
        result = ta.validate_python(None)
        assert result is None
