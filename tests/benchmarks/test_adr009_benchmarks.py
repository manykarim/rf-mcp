"""Benchmarks for ADR-009 Type-Constrained Tool Parameters.

Measures the performance of Pydantic TypeAdapter operations on the
Literal type aliases defined in robotmcp.domains.shared.kernel:
- TypeAdapter creation latency
- Validation throughput (valid values, case normalization, rejection)
- JSON Schema generation
- Literal vs plain str baseline comparison
- Optional[Literal] wrapping overhead
- Full parameter model simulation (manage_session signature)

These are pure domain-layer benchmarks (no MCP server, no browser).

Run with: uv run pytest tests/benchmarks/test_adr009_benchmarks.py -v
"""

from __future__ import annotations

__test__ = True

import time
from typing import Optional

import pytest
from pydantic import TypeAdapter, ValidationError, create_model

from robotmcp.domains.shared.kernel import (
    AutomationContext,
    DetailLevel,
    IntentVerb,
    ModelTierLiteral,
    SessionAction,
    TestStatus,
    ToolProfileName,
)


# ============================================================
# TypeAdapter creation benchmarks
# ============================================================


class TestTypeAdapterCreation:
    """Benchmark TypeAdapter instantiation for each Literal type alias."""

    def test_bench_session_action_adapter_creation(self, benchmark_reporter):
        """TypeAdapter(SessionAction) creation should be < 0.25ms."""
        iterations = 5000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta = TypeAdapter(SessionAction)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "TypeAdapter(SessionAction) creation",
            elapsed_ms, target_ms=0.25, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert ta is not None
        assert avg_ms < 0.25, f"SessionAction adapter creation too slow: {avg_ms:.4f}ms"

    def test_bench_intent_verb_adapter_creation(self, benchmark_reporter):
        """TypeAdapter(IntentVerb) creation should be < 0.25ms."""
        iterations = 5000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta = TypeAdapter(IntentVerb)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "TypeAdapter(IntentVerb) creation",
            elapsed_ms, target_ms=0.25, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert ta is not None
        assert avg_ms < 0.25, f"IntentVerb adapter creation too slow: {avg_ms:.4f}ms"

    def test_bench_detail_level_adapter_creation(self, benchmark_reporter):
        """TypeAdapter(DetailLevel) creation should be < 0.25ms."""
        iterations = 5000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta = TypeAdapter(DetailLevel)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "TypeAdapter(DetailLevel) creation",
            elapsed_ms, target_ms=0.25, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert ta is not None
        assert avg_ms < 0.25, f"DetailLevel adapter creation too slow: {avg_ms:.4f}ms"

    def test_bench_automation_context_adapter_creation(self, benchmark_reporter):
        """TypeAdapter(AutomationContext) creation should be < 0.25ms."""
        iterations = 5000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta = TypeAdapter(AutomationContext)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "TypeAdapter(AutomationContext) creation",
            elapsed_ms, target_ms=0.25, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert ta is not None
        assert avg_ms < 0.25, f"AutomationContext adapter creation too slow: {avg_ms:.4f}ms"


# ============================================================
# Validation throughput benchmarks
# ============================================================


class TestValidationThroughput:
    """Benchmark validation speed for each Literal type alias."""

    def test_bench_session_action_validation(self, benchmark_reporter):
        """SessionAction validation should be < 0.005ms per call."""
        ta = TypeAdapter(SessionAction)
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python("init")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "SessionAction validate_python('init')",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.005, f"SessionAction validation too slow: {avg_ms:.6f}ms"

    def test_bench_session_action_validation_last_value(self, benchmark_reporter):
        """Validating the LAST enum value should be equally fast (no linear scan)."""
        ta = TypeAdapter(SessionAction)
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python("set_tool_profile")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "SessionAction validate_python('set_tool_profile')",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.005, f"Last-value validation too slow: {avg_ms:.6f}ms"

    def test_bench_intent_verb_validation(self, benchmark_reporter):
        """IntentVerb validation should be < 0.005ms per call."""
        ta = TypeAdapter(IntentVerb)
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python("click")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "IntentVerb validate_python('click')",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.005, f"IntentVerb validation too slow: {avg_ms:.6f}ms"

    def test_bench_detail_level_validation(self, benchmark_reporter):
        """DetailLevel validation should be < 0.005ms per call."""
        ta = TypeAdapter(DetailLevel)
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python("standard")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "DetailLevel validate_python('standard')",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.005, f"DetailLevel validation too slow: {avg_ms:.6f}ms"

    def test_bench_automation_context_validation(self, benchmark_reporter):
        """AutomationContext validation should be < 0.005ms per call."""
        ta = TypeAdapter(AutomationContext)
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python("web")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "AutomationContext validate_python('web')",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.005, f"AutomationContext validation too slow: {avg_ms:.6f}ms"

    def test_bench_all_session_action_values(self, benchmark_reporter):
        """Validating all 20 SessionAction values should be < 0.1ms per sweep."""
        ta = TypeAdapter(SessionAction)
        all_values = [
            "init", "initialize", "bootstrap",
            "import_library", "library",
            "import_resource", "resource",
            "set_variables", "variables",
            "import_variables", "load_variables",
            "start_test", "end_test", "start_task", "end_task",
            "list_tests",
            "set_suite_setup", "set_suite_teardown",
            "set_tool_profile", "tool_profile",
        ]
        iterations = 5000
        t0 = time.perf_counter()
        for _ in range(iterations):
            for v in all_values:
                ta.validate_python(v)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "SessionAction validate all 20 values sweep",
            elapsed_ms, target_ms=0.1, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.1, f"All-values sweep too slow: {avg_ms:.4f}ms"

    def test_bench_all_intent_verb_values(self, benchmark_reporter):
        """Validating all 8 IntentVerb values should be < 0.04ms per sweep."""
        ta = TypeAdapter(IntentVerb)
        all_values = [
            "navigate", "click", "fill", "hover",
            "select", "assert_visible", "extract_text", "wait_for",
        ]
        iterations = 5000
        t0 = time.perf_counter()
        for _ in range(iterations):
            for v in all_values:
                ta.validate_python(v)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "IntentVerb validate all 8 values sweep",
            elapsed_ms, target_ms=0.04, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.04, f"All-verbs sweep too slow: {avg_ms:.4f}ms"


# ============================================================
# Normalization overhead benchmarks
# ============================================================


class TestNormalizationOverhead:
    """Benchmark the cost of case normalization via BeforeValidator."""

    def test_bench_session_action_normalization_uppercase(self, benchmark_reporter):
        """Uppercase input triggers _normalize_str; should be < 0.005ms."""
        ta = TypeAdapter(SessionAction)
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python("Init")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "SessionAction normalize 'Init' -> 'init'",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.005, f"Normalization too slow: {avg_ms:.6f}ms"

    def test_bench_session_action_normalization_allcaps(self, benchmark_reporter):
        """ALL CAPS input; should be < 0.005ms."""
        ta = TypeAdapter(SessionAction)
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python("INIT")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "SessionAction normalize 'INIT' -> 'init'",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.005, f"Normalization too slow: {avg_ms:.6f}ms"

    def test_bench_session_action_normalization_whitespace(self, benchmark_reporter):
        """Whitespace-padded input; should be < 0.005ms."""
        ta = TypeAdapter(SessionAction)
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python("  init  ")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "SessionAction normalize '  init  ' -> 'init'",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.005, f"Normalization too slow: {avg_ms:.6f}ms"

    def test_bench_intent_verb_normalization_mixed_case(self, benchmark_reporter):
        """Mixed-case IntentVerb input; should be < 0.005ms."""
        ta = TypeAdapter(IntentVerb)
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python("Click")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "IntentVerb normalize 'Click' -> 'click'",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.005, f"Normalization too slow: {avg_ms:.6f}ms"

    def test_bench_normalization_vs_already_lowercase(self, benchmark_reporter):
        """Compare normalization overhead: already-lowercase vs mixed-case."""
        ta = TypeAdapter(SessionAction)
        iterations = 50000

        # Already lowercase (no-op path through str.lower())
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python("init")
        elapsed_lower_ms = (time.perf_counter() - t0) * 1000

        # Mixed case (actual normalization)
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python("Init")
        elapsed_mixed_ms = (time.perf_counter() - t0) * 1000

        avg_lower = elapsed_lower_ms / iterations
        avg_mixed = elapsed_mixed_ms / iterations

        benchmark_reporter.record_latency(
            "SessionAction lowercase path",
            elapsed_lower_ms, target_ms=0.005, iterations=iterations,
        )
        benchmark_reporter.record_latency(
            "SessionAction mixed-case path",
            elapsed_mixed_ms, target_ms=0.005, iterations=iterations,
        )

        # Normalization overhead should be < 2x the lowercase path
        assert avg_mixed < avg_lower * 2.0 + 0.001, (
            f"Normalization overhead too high: {avg_mixed:.6f}ms vs {avg_lower:.6f}ms"
        )


# ============================================================
# Schema generation benchmarks
# ============================================================


class TestSchemaGeneration:
    """Benchmark JSON Schema generation for ADR-009 types."""

    def test_bench_session_action_schema_generation(self, benchmark_reporter):
        """SessionAction JSON Schema generation should be < 0.15ms."""
        ta = TypeAdapter(SessionAction)
        iterations = 5000
        t0 = time.perf_counter()
        for _ in range(iterations):
            schema = ta.json_schema()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "SessionAction json_schema()",
            elapsed_ms, target_ms=0.15, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert "enum" in schema, "Schema must contain 'enum' key"
        assert len(schema["enum"]) == 20, f"Expected 20 enum values, got {len(schema['enum'])}"
        assert avg_ms < 0.15, f"Schema generation too slow: {avg_ms:.4f}ms"

    def test_bench_intent_verb_schema_generation(self, benchmark_reporter):
        """IntentVerb JSON Schema generation should be < 0.1ms."""
        ta = TypeAdapter(IntentVerb)
        iterations = 5000
        t0 = time.perf_counter()
        for _ in range(iterations):
            schema = ta.json_schema()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "IntentVerb json_schema()",
            elapsed_ms, target_ms=0.1, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert "enum" in schema, "Schema must contain 'enum' key"
        assert len(schema["enum"]) == 8, f"Expected 8 enum values, got {len(schema['enum'])}"
        assert avg_ms < 0.1, f"Schema generation too slow: {avg_ms:.4f}ms"

    def test_bench_detail_level_schema_generation(self, benchmark_reporter):
        """DetailLevel JSON Schema generation should be < 0.1ms."""
        ta = TypeAdapter(DetailLevel)
        iterations = 5000
        t0 = time.perf_counter()
        for _ in range(iterations):
            schema = ta.json_schema()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "DetailLevel json_schema()",
            elapsed_ms, target_ms=0.1, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert "enum" in schema, "Schema must contain 'enum' key"
        assert avg_ms < 0.1, f"Schema generation too slow: {avg_ms:.4f}ms"

    def test_bench_automation_context_schema_generation(self, benchmark_reporter):
        """AutomationContext JSON Schema generation should be < 0.1ms."""
        ta = TypeAdapter(AutomationContext)
        iterations = 5000
        t0 = time.perf_counter()
        for _ in range(iterations):
            schema = ta.json_schema()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "AutomationContext json_schema()",
            elapsed_ms, target_ms=0.1, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert "enum" in schema, "Schema must contain 'enum' key"
        assert len(schema["enum"]) == 6, f"Expected 6 enum values, got {len(schema['enum'])}"
        assert avg_ms < 0.1, f"Schema generation too slow: {avg_ms:.4f}ms"

    def test_bench_all_types_schema_generation(self, benchmark_reporter):
        """Generating schemas for all 4 key types should be < 0.4ms total."""
        adapters = [
            TypeAdapter(SessionAction),
            TypeAdapter(IntentVerb),
            TypeAdapter(DetailLevel),
            TypeAdapter(AutomationContext),
        ]
        iterations = 2000
        t0 = time.perf_counter()
        for _ in range(iterations):
            for ta in adapters:
                ta.json_schema()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "All 4 key types json_schema() sweep",
            elapsed_ms, target_ms=0.4, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.4, f"All-types schema sweep too slow: {avg_ms:.4f}ms"


# ============================================================
# Plain str baseline benchmarks
# ============================================================


class TestPlainStrBaseline:
    """Compare Literal validation overhead against plain str baseline."""

    def test_bench_plain_str_validation(self, benchmark_reporter):
        """Plain str validation baseline; should be < 0.003ms."""
        ta = TypeAdapter(str)
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python("init")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "TypeAdapter(str) validate_python('init') baseline",
            elapsed_ms, target_ms=0.003, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.003, f"Plain str validation too slow: {avg_ms:.6f}ms"

    def test_bench_plain_str_adapter_creation(self, benchmark_reporter):
        """Plain str TypeAdapter creation baseline; should be < 0.05ms."""
        iterations = 5000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta = TypeAdapter(str)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "TypeAdapter(str) creation baseline",
            elapsed_ms, target_ms=0.05, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert ta is not None
        assert avg_ms < 0.05, f"Plain str adapter creation too slow: {avg_ms:.4f}ms"

    def test_bench_literal_vs_str_overhead_ratio(self, benchmark_reporter):
        """Literal+BeforeValidator overhead should be < 5x plain str."""
        ta_str = TypeAdapter(str)
        ta_literal = TypeAdapter(SessionAction)
        iterations = 50000

        t0 = time.perf_counter()
        for _ in range(iterations):
            ta_str.validate_python("init")
        str_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for _ in range(iterations):
            ta_literal.validate_python("init")
        literal_ms = (time.perf_counter() - t0) * 1000

        str_avg = str_ms / iterations
        literal_avg = literal_ms / iterations
        ratio = literal_avg / str_avg if str_avg > 0 else float("inf")

        benchmark_reporter.record_latency(
            "str baseline validation",
            str_ms, target_ms=0.003, iterations=iterations,
        )
        benchmark_reporter.record_latency(
            "SessionAction Literal validation",
            literal_ms, target_ms=0.005, iterations=iterations,
        )

        # Literal overhead should be reasonable (< 5x str)
        assert ratio < 5.0, (
            f"Literal overhead too high: {ratio:.2f}x "
            f"(str={str_avg:.6f}ms, literal={literal_avg:.6f}ms)"
        )


# ============================================================
# Optional[Literal] benchmarks
# ============================================================


class TestOptionalLiteralValidation:
    """Benchmark Optional[Literal] wrapping for nullable parameters."""

    def test_bench_optional_tool_profile_validation(self, benchmark_reporter):
        """Optional[ToolProfileName] with valid value; should be < 0.005ms."""
        ta = TypeAdapter(Optional[ToolProfileName])
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python("browser_exec")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "Optional[ToolProfileName] validate 'browser_exec'",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.005, f"Optional Literal validation too slow: {avg_ms:.6f}ms"

    def test_bench_optional_tool_profile_none(self, benchmark_reporter):
        """Optional[ToolProfileName] with None; should be < 0.005ms."""
        ta = TypeAdapter(Optional[ToolProfileName])
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python(None)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "Optional[ToolProfileName] validate None",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.005, f"Optional None validation too slow: {avg_ms:.6f}ms"

    def test_bench_optional_model_tier_validation(self, benchmark_reporter):
        """Optional[ModelTierLiteral] with valid value; should be < 0.005ms."""
        ta = TypeAdapter(Optional[ModelTierLiteral])
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python("standard")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "Optional[ModelTierLiteral] validate 'standard'",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.005, f"Optional ModelTier validation too slow: {avg_ms:.6f}ms"

    def test_bench_optional_model_tier_none(self, benchmark_reporter):
        """Optional[ModelTierLiteral] with None; should be < 0.005ms."""
        ta = TypeAdapter(Optional[ModelTierLiteral])
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ta.validate_python(None)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "Optional[ModelTierLiteral] validate None",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.005, f"Optional None validation too slow: {avg_ms:.6f}ms"

    def test_bench_optional_vs_required_overhead(self, benchmark_reporter):
        """Optional wrapping overhead should be < 2x required."""
        ta_required = TypeAdapter(ToolProfileName)
        ta_optional = TypeAdapter(Optional[ToolProfileName])
        iterations = 50000

        t0 = time.perf_counter()
        for _ in range(iterations):
            ta_required.validate_python("browser_exec")
        required_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for _ in range(iterations):
            ta_optional.validate_python("browser_exec")
        optional_ms = (time.perf_counter() - t0) * 1000

        required_avg = required_ms / iterations
        optional_avg = optional_ms / iterations
        ratio = optional_avg / required_avg if required_avg > 0 else float("inf")

        benchmark_reporter.record_latency(
            "ToolProfileName required validation",
            required_ms, target_ms=0.005, iterations=iterations,
        )
        benchmark_reporter.record_latency(
            "Optional[ToolProfileName] validation",
            optional_ms, target_ms=0.005, iterations=iterations,
        )

        assert ratio < 2.0, (
            f"Optional overhead too high: {ratio:.2f}x "
            f"(required={required_avg:.6f}ms, optional={optional_avg:.6f}ms)"
        )


# ============================================================
# Full manage_session signature simulation
# ============================================================


class TestFullParamModelSimulation:
    """Benchmark a realistic manage_session parameter model."""

    @pytest.fixture
    def manage_session_model(self):
        """Create a Pydantic model simulating manage_session params."""
        return create_model(
            "ManageSessionParams",
            action=(SessionAction, ...),
            session_id=(str, ...),
            test_status=(TestStatus, "pass"),
            tool_profile=(Optional[ToolProfileName], None),
            model_tier=(Optional[ModelTierLiteral], None),
        )

    def test_bench_full_manage_session_params_minimal(
        self, benchmark_reporter, manage_session_model
    ):
        """Minimal manage_session params (action + session_id); < 0.02ms."""
        Model = manage_session_model
        iterations = 20000
        t0 = time.perf_counter()
        for _ in range(iterations):
            result = Model(action="init", session_id="test-123")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "ManageSessionParams(action, session_id) minimal",
            elapsed_ms, target_ms=0.02, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert result.action == "init"
        assert result.session_id == "test-123"
        assert result.test_status == "pass"  # default
        assert result.tool_profile is None  # default
        assert result.model_tier is None  # default
        assert avg_ms < 0.02, f"Minimal params too slow: {avg_ms:.4f}ms"

    def test_bench_full_manage_session_params_all_fields(
        self, benchmark_reporter, manage_session_model
    ):
        """All manage_session params provided; < 0.03ms."""
        Model = manage_session_model
        iterations = 20000
        t0 = time.perf_counter()
        for _ in range(iterations):
            result = Model(
                action="start_test",
                session_id="sess-456",
                test_status="pass",
                tool_profile="browser_exec",
                model_tier="small_context",
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "ManageSessionParams all 5 fields",
            elapsed_ms, target_ms=0.03, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert result.action == "start_test"
        assert result.tool_profile == "browser_exec"
        assert result.model_tier == "small_context"
        assert avg_ms < 0.03, f"All-fields params too slow: {avg_ms:.4f}ms"

    def test_bench_full_manage_session_params_with_normalization(
        self, benchmark_reporter, manage_session_model
    ):
        """All params with mixed case (normalization path); < 0.03ms."""
        Model = manage_session_model
        iterations = 20000
        t0 = time.perf_counter()
        for _ in range(iterations):
            result = Model(
                action="Start_Test",
                session_id="sess-789",
                test_status="PASS",
                tool_profile="Browser_Exec",
                model_tier="Small_Context",
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "ManageSessionParams all fields + normalization",
            elapsed_ms, target_ms=0.03, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert result.action == "start_test"
        assert result.test_status == "pass"
        assert result.tool_profile == "browser_exec"
        assert result.model_tier == "small_context"
        assert avg_ms < 0.03, f"Normalized params too slow: {avg_ms:.4f}ms"

    def test_bench_full_model_creation(self, benchmark_reporter):
        """create_model() itself should be < 5ms (one-time setup cost)."""
        iterations = 100
        t0 = time.perf_counter()
        for _ in range(iterations):
            create_model(
                "ManageSessionParams",
                action=(SessionAction, ...),
                session_id=(str, ...),
                test_status=(TestStatus, "pass"),
                tool_profile=(Optional[ToolProfileName], None),
                model_tier=(Optional[ModelTierLiteral], None),
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "create_model(ManageSessionParams) setup cost",
            elapsed_ms, target_ms=5.0, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 5.0, f"Model creation too slow: {avg_ms:.4f}ms"


# ============================================================
# Validation rejection benchmarks
# ============================================================


class TestValidationRejection:
    """Benchmark rejection of invalid values (error path)."""

    def test_bench_session_action_rejection(self, benchmark_reporter):
        """Invalid SessionAction should be rejected in < 0.01ms."""
        ta = TypeAdapter(SessionAction)
        iterations = 20000
        t0 = time.perf_counter()
        for _ in range(iterations):
            try:
                ta.validate_python("invalid_action")
            except ValidationError:
                pass
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "SessionAction rejection 'invalid_action'",
            elapsed_ms, target_ms=0.01, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.01, f"Rejection too slow: {avg_ms:.6f}ms"

    def test_bench_intent_verb_rejection(self, benchmark_reporter):
        """Invalid IntentVerb should be rejected in < 0.01ms."""
        ta = TypeAdapter(IntentVerb)
        iterations = 20000
        t0 = time.perf_counter()
        for _ in range(iterations):
            try:
                ta.validate_python("destroy")
            except ValidationError:
                pass
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "IntentVerb rejection 'destroy'",
            elapsed_ms, target_ms=0.01, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.01, f"Rejection too slow: {avg_ms:.6f}ms"

    def test_bench_wrong_type_rejection(self, benchmark_reporter):
        """Non-string input should be rejected in < 0.01ms."""
        ta = TypeAdapter(SessionAction)
        iterations = 20000
        t0 = time.perf_counter()
        for _ in range(iterations):
            try:
                ta.validate_python(42)
            except ValidationError:
                pass
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "SessionAction rejection int(42)",
            elapsed_ms, target_ms=0.01, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.01, f"Wrong-type rejection too slow: {avg_ms:.6f}ms"


# ============================================================
# Correctness assertions (non-benchmark, fast sanity checks)
# ============================================================


class TestADR009Correctness:
    """Non-benchmark correctness tests for ADR-009 type aliases."""

    def test_session_action_schema_has_flat_enum(self):
        """SessionAction schema must be flat {enum: [...]} without anyOf."""
        schema = TypeAdapter(SessionAction).json_schema()
        assert "enum" in schema, "Schema must have 'enum' key"
        assert "anyOf" not in schema, "Schema must NOT have 'anyOf' (flat enum)"
        assert schema["enum"] == [
            "init", "initialize", "bootstrap",
            "import_library", "library",
            "import_resource", "resource",
            "set_variables", "variables",
            "import_variables", "load_variables",
            "start_test", "end_test", "start_task", "end_task",
            "list_tests",
            "set_suite_setup", "set_suite_teardown",
            "set_tool_profile", "tool_profile",
        ]

    def test_intent_verb_schema_has_flat_enum(self):
        """IntentVerb schema must be flat {enum: [...]} without anyOf."""
        schema = TypeAdapter(IntentVerb).json_schema()
        assert "enum" in schema
        assert "anyOf" not in schema
        assert schema["enum"] == [
            "navigate", "click", "fill", "hover",
            "select", "assert_visible", "extract_text", "wait_for",
        ]

    def test_detail_level_schema_has_flat_enum(self):
        """DetailLevel schema must be flat {enum: [...]} without anyOf."""
        schema = TypeAdapter(DetailLevel).json_schema()
        assert "enum" in schema
        assert "anyOf" not in schema
        assert schema["enum"] == ["minimal", "standard", "full"]

    def test_automation_context_schema_has_flat_enum(self):
        """AutomationContext schema must be flat {enum: [...]} without anyOf."""
        schema = TypeAdapter(AutomationContext).json_schema()
        assert "enum" in schema
        assert "anyOf" not in schema
        assert schema["enum"] == ["web", "mobile", "api", "desktop", "generic", "database"]

    def test_normalization_preserves_valid_values(self):
        """All valid values survive normalization round-trip."""
        ta = TypeAdapter(SessionAction)
        for value in [
            "init", "initialize", "bootstrap",
            "import_library", "library",
            "import_resource", "resource",
            "set_variables", "variables",
            "import_variables", "load_variables",
            "start_test", "end_test", "start_task", "end_task",
            "list_tests",
            "set_suite_setup", "set_suite_teardown",
            "set_tool_profile", "tool_profile",
        ]:
            assert ta.validate_python(value) == value
            assert ta.validate_python(value.upper()) == value
            assert ta.validate_python(f"  {value}  ") == value

    def test_invalid_value_raises_validation_error(self):
        """Invalid values must raise ValidationError, not pass silently."""
        ta = TypeAdapter(SessionAction)
        with pytest.raises(ValidationError):
            ta.validate_python("nonexistent_action")

    def test_non_string_raises_validation_error(self):
        """Non-string types must raise ValidationError."""
        ta = TypeAdapter(SessionAction)
        with pytest.raises(ValidationError):
            ta.validate_python(123)
        with pytest.raises(ValidationError):
            ta.validate_python(["init"])
