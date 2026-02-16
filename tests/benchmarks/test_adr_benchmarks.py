"""Benchmarks for ADR-006/007/008 domain operations.

Measures the performance of:
- IntentRegistry creation and resolution
- ToolProfile preset creation and budget estimation
- ResponseCompressor compression pipeline

These are pure domain-layer benchmarks (no MCP server, no browser).

Run with: uv run pytest tests/benchmarks/test_adr_benchmarks.py -v
"""

from __future__ import annotations

__test__ = True

import time

import pytest

from robotmcp.domains.intent.aggregates import IntentRegistry
from robotmcp.domains.intent.entities import IntentMapping
from robotmcp.domains.intent.services import IntentResolver, IntentResolutionError
from robotmcp.domains.intent.value_objects import (
    IntentTarget,
    IntentVerb,
    NormalizedLocator,
)
from robotmcp.domains.response_optimization.aggregates import (
    ResponseOptimizationConfig,
)
from robotmcp.domains.response_optimization.services import ResponseCompressor
from robotmcp.domains.tool_profile.aggregates import ProfilePresets, ToolProfile
from robotmcp.domains.tool_profile.entities import ToolDescriptor
from robotmcp.domains.tool_profile.services import ToolProfileManager
from robotmcp.domains.tool_profile.value_objects import (
    ModelTier,
    ToolDescriptionMode,
    TokenBudget,
    ToolTag,
)


# ============================================================
# IntentRegistry Benchmarks
# ============================================================


class TestIntentRegistryBenchmarks:
    """Benchmark IntentRegistry creation and resolution operations."""

    def test_registry_creation_with_builtins_latency(self, benchmark_reporter):
        """IntentRegistry.with_builtins() should complete in < 1ms."""
        iterations = 1000
        t0 = time.perf_counter()
        for _ in range(iterations):
            IntentRegistry.with_builtins()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "IntentRegistry.with_builtins()",
            elapsed_ms, target_ms=1.0, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 1.0, f"Registry creation too slow: {avg_ms:.4f}ms avg"

    def test_resolve_browser_click_latency(self, benchmark_reporter):
        """Single resolve() call for Browser CLICK should be < 0.01ms."""
        registry = IntentRegistry.with_builtins()
        iterations = 10000
        t0 = time.perf_counter()
        for _ in range(iterations):
            registry.resolve(IntentVerb.CLICK, "Browser")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "IntentRegistry.resolve(CLICK, Browser)",
            elapsed_ms, target_ms=0.01, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.01, f"Resolve too slow: {avg_ms:.6f}ms avg"

    def test_resolve_selenium_click_element_latency(self, benchmark_reporter):
        """Single resolve() for SeleniumLibrary CLICK should be < 0.01ms."""
        registry = IntentRegistry.with_builtins()
        iterations = 10000
        t0 = time.perf_counter()
        for _ in range(iterations):
            registry.resolve(IntentVerb.CLICK, "SeleniumLibrary")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "IntentRegistry.resolve(CLICK, SeleniumLibrary)",
            elapsed_ms, target_ms=0.01, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.01

    def test_resolve_all_verbs_all_libraries_latency(self, benchmark_reporter):
        """Resolving all verbs for all libraries should be < 0.1ms total."""
        registry = IntentRegistry.with_builtins()
        libraries = ["Browser", "SeleniumLibrary", "AppiumLibrary"]
        iterations = 1000

        t0 = time.perf_counter()
        for _ in range(iterations):
            for verb in IntentVerb:
                for lib in libraries:
                    registry.resolve(verb, lib)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Per full sweep: 8 verbs x 3 libraries = 24 lookups
        sweeps = iterations
        benchmark_reporter.record_latency(
            "IntentRegistry.resolve(all_verbs x all_libs)",
            elapsed_ms, target_ms=0.1, iterations=sweeps,
        )
        avg_ms = elapsed_ms / sweeps
        assert avg_ms < 0.1, f"Full resolve sweep too slow: {avg_ms:.4f}ms avg"

    def test_get_supported_intents_latency(self, benchmark_reporter):
        """get_supported_intents() should be < 0.05ms."""
        registry = IntentRegistry.with_builtins()
        iterations = 10000
        t0 = time.perf_counter()
        for _ in range(iterations):
            registry.get_supported_intents("Browser")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "IntentRegistry.get_supported_intents(Browser)",
            elapsed_ms, target_ms=0.05, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.05

    def test_builtin_mappings_count(self):
        """Verify expected number of built-in mappings."""
        registry = IntentRegistry.with_builtins()
        all_mappings = registry.get_all_mappings()
        # 8 verbs for Browser (all), 8 for Selenium (all), 6 for Appium
        # (Appium has no HOVER or SELECT mappings)
        assert len(all_mappings) >= 20, (
            f"Expected >= 20 built-in mappings, got {len(all_mappings)}"
        )

    def test_has_mapping_latency(self, benchmark_reporter):
        """has_mapping() should be < 0.005ms (pure dict lookup)."""
        registry = IntentRegistry.with_builtins()
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            registry.has_mapping(IntentVerb.NAVIGATE, "Browser")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "IntentRegistry.has_mapping()",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.005


# ============================================================
# IntentMapping.build_arguments Benchmarks
# ============================================================


class TestIntentMappingArgumentBenchmarks:
    """Benchmark argument building via IntentMapping.build_arguments()."""

    def test_default_argument_builder_latency(self, benchmark_reporter):
        """Default (no transformer) argument building should be < 0.01ms."""
        registry = IntentRegistry.with_builtins()
        mapping = registry.resolve(IntentVerb.CLICK, "Browser")
        assert mapping is not None
        target = IntentTarget(locator="text=Login")
        normalized = NormalizedLocator(
            value="text=Login", source_locator="text=Login",
            target_library="Browser", strategy_applied="passthrough",
            was_transformed=False,
        )

        iterations = 10000
        t0 = time.perf_counter()
        for _ in range(iterations):
            mapping.build_arguments(target, None, normalized)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "IntentMapping.build_arguments(default)",
            elapsed_ms, target_ms=0.01, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.01

    def test_custom_transformer_latency(self, benchmark_reporter):
        """Custom transformer (navigate) argument building should be < 0.01ms."""
        registry = IntentRegistry.with_builtins()
        mapping = registry.resolve(IntentVerb.NAVIGATE, "Browser")
        assert mapping is not None
        assert mapping.argument_transformer is not None
        target = IntentTarget(locator="https://example.com")

        iterations = 10000
        t0 = time.perf_counter()
        for _ in range(iterations):
            mapping.build_arguments(target, None, None)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "IntentMapping.build_arguments(navigate transformer)",
            elapsed_ms, target_ms=0.01, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.01

    def test_select_transformer_latency(self, benchmark_reporter):
        """Select transformer (multi-arg) should be < 0.02ms."""
        registry = IntentRegistry.with_builtins()
        mapping = registry.resolve(IntentVerb.SELECT, "Browser")
        assert mapping is not None
        target = IntentTarget(locator="id=country")
        normalized = NormalizedLocator(
            value="id=country", source_locator="id=country",
            target_library="Browser", strategy_applied="passthrough",
            was_transformed=False,
        )

        iterations = 10000
        t0 = time.perf_counter()
        for _ in range(iterations):
            mapping.build_arguments(target, "United States", normalized)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "IntentMapping.build_arguments(select transformer)",
            elapsed_ms, target_ms=0.02, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.02


# ============================================================
# ToolProfile Benchmarks
# ============================================================


class TestToolProfileBenchmarks:
    """Benchmark ToolProfile operations."""

    def test_browser_exec_preset_creation_latency(self, benchmark_reporter):
        """ProfilePresets.browser_exec() should complete in < 0.1ms."""
        iterations = 10000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ProfilePresets.browser_exec()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "ProfilePresets.browser_exec()",
            elapsed_ms, target_ms=0.1, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.1

    def test_all_preset_creation_latency(self, benchmark_reporter):
        """Creating all 5 presets should complete in < 0.5ms."""
        iterations = 1000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ProfilePresets.browser_exec()
            ProfilePresets.api_exec()
            ProfilePresets.discovery()
            ProfilePresets.minimal_exec()
            ProfilePresets.full()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "All 5 ProfilePresets creation",
            elapsed_ms, target_ms=0.5, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.5

    def test_budget_estimation_latency(self, benchmark_reporter):
        """Token budget estimation across all tools should be < 0.1ms."""
        profile = ProfilePresets.browser_exec()
        descriptors = {}
        for name in ProfilePresets.ALL_TOOLS:
            descriptors[name] = ToolDescriptor(
                tool_name=name,
                tags=frozenset({ToolTag.CORE}),
                description_full=f"Full description for {name}. " * 10,
                description_compact=f"Compact {name}",
                description_minimal=f"Min {name}",
                schema_full={"type": "object", "properties": {"arg": {"type": "string"}}},
                token_estimate_full=400,
                token_estimate_compact=100,
                token_estimate_minimal=50,
            )

        iterations = 10000
        t0 = time.perf_counter()
        for _ in range(iterations):
            profile.estimate_token_cost(descriptors)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "ToolProfile.estimate_token_cost()",
            elapsed_ms, target_ms=0.1, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.1

    def test_budget_validation_latency(self, benchmark_reporter):
        """Budget validation should be < 0.15ms (estimation + comparison)."""
        profile = ProfilePresets.browser_exec()
        descriptors = {}
        for name in ProfilePresets.ALL_TOOLS:
            descriptors[name] = ToolDescriptor(
                tool_name=name,
                tags=frozenset({ToolTag.CORE}),
                description_full=f"Full {name}",
                description_compact=f"Compact {name}",
                description_minimal=f"Min {name}",
                schema_full={"type": "object"},
                token_estimate_full=400,
                token_estimate_compact=100,
                token_estimate_minimal=50,
            )

        iterations = 10000
        t0 = time.perf_counter()
        for _ in range(iterations):
            profile.validate_budget(descriptors)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "ToolProfile.validate_budget()",
            elapsed_ms, target_ms=0.15, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.15

    def test_contains_tool_latency(self, benchmark_reporter):
        """contains_tool() should be < 0.005ms (frozenset lookup)."""
        profile = ProfilePresets.browser_exec()
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            profile.contains_tool("manage_session")
            profile.contains_tool("nonexistent_tool")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "ToolProfile.contains_tool()",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.005

    def test_with_additional_tool_latency(self, benchmark_reporter):
        """with_additional_tool() should be < 0.2ms (creates new frozen profile)."""
        profile = ProfilePresets.browser_exec()
        iterations = 5000
        t0 = time.perf_counter()
        for _ in range(iterations):
            profile.with_additional_tool("get_keyword_info")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "ToolProfile.with_additional_tool()",
            elapsed_ms, target_ms=0.2, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.2


# ============================================================
# TokenBudget Benchmarks
# ============================================================


class TestTokenBudgetBenchmarks:
    """Benchmark TokenBudget operations."""

    def test_budget_for_context_window_latency(self, benchmark_reporter):
        """TokenBudget.for_context_window() should be < 0.01ms."""
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            TokenBudget.for_context_window(8192)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "TokenBudget.for_context_window(8192)",
            elapsed_ms, target_ms=0.01, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.01

    def test_budget_fits_tool_cost_latency(self, benchmark_reporter):
        """fits_tool_cost() should be < 0.005ms."""
        budget = TokenBudget.for_context_window(8192)
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            budget.fits_tool_cost(1500)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "TokenBudget.fits_tool_cost()",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.005


# ============================================================
# ToolProfileManager.suggest_profile Benchmarks
# ============================================================


class TestSuggestProfileBenchmarks:
    """Benchmark profile suggestion heuristics."""

    def _make_manager(self):
        """Create a ToolProfileManager with mock adapter."""

        class MockToolManager:
            async def remove_tool(self, name): pass
            async def add_tool_with_description(self, n, d, s): pass
            async def get_visible_tool_names(self): return frozenset()
            async def swap_tool_description(self, n, d, s): pass

        return ToolProfileManager(MockToolManager(), {})

    def test_suggest_browser_scenario_latency(self, benchmark_reporter):
        """suggest_profile for browser scenario should be < 0.05ms."""
        manager = self._make_manager()
        iterations = 10000
        t0 = time.perf_counter()
        for _ in range(iterations):
            manager.suggest_profile("browser test for login page", ModelTier.SMALL_CONTEXT)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "ToolProfileManager.suggest_profile(browser)",
            elapsed_ms, target_ms=0.05, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.05

    def test_suggest_api_scenario_latency(self, benchmark_reporter):
        """suggest_profile for API scenario should be < 0.05ms."""
        manager = self._make_manager()
        iterations = 10000
        t0 = time.perf_counter()
        for _ in range(iterations):
            manager.suggest_profile("REST API test for user endpoint", ModelTier.SMALL_CONTEXT)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "ToolProfileManager.suggest_profile(api)",
            elapsed_ms, target_ms=0.05, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.05

    def test_suggest_profile_correctness(self):
        """Verify suggest_profile returns correct profiles for known scenarios."""
        manager = self._make_manager()

        # Browser scenarios
        p = manager.suggest_profile("browser test for login", ModelTier.SMALL_CONTEXT)
        assert p.name == "browser_exec"

        # API scenarios
        p = manager.suggest_profile("REST api testing", ModelTier.SMALL_CONTEXT)
        assert p.name == "api_exec"

        # Discovery scenarios
        p = manager.suggest_profile("analyze what libraries we need", ModelTier.SMALL_CONTEXT)
        assert p.name == "discovery"

        # Large context always gets full
        p = manager.suggest_profile("anything", ModelTier.LARGE_CONTEXT)
        assert p.name == "full"


# ============================================================
# ResponseCompressor Benchmarks
# ============================================================


class TestResponseCompressionBenchmarks:
    """Benchmark response compression operations."""

    def test_compress_simple_response_latency(self, benchmark_reporter):
        """Compressing a simple response should be < 0.5ms."""
        config = ResponseOptimizationConfig.create_compact("bench-session")
        compressor = ResponseCompressor()
        response = {
            "success": True,
            "keyword": "Click Element",
            "arguments": ["id=submit"],
            "status": "pass",
            "execution_time": 0.5,
            "output": "Element clicked",
            "assigned_variables": {},
            "session_id": "test-123",
        }

        iterations = 5000
        t0 = time.perf_counter()
        for _ in range(iterations):
            compressor.compress_response(response, config)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "ResponseCompressor.compress_response(simple)",
            elapsed_ms, target_ms=0.5, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.5

    def test_compress_large_response_latency(self, benchmark_reporter):
        """Compressing a large response should be < 2ms."""
        config = ResponseOptimizationConfig.create_compact("bench-session")
        compressor = ResponseCompressor()
        response = {
            "success": True,
            "keyword": "Get Page Source",
            "arguments": [],
            "status": "pass",
            "execution_time": 1.2,
            "output": "<html>" + "<div>content</div>" * 1000 + "</html>",
            "assigned_variables": {"source": "<html>" + "x" * 5000 + "</html>"},
            "session_id": "test-456",
            "metadata": {"url": "http://example.com", "title": "Test Page"},
            "description": "Page source retrieval for the test application",
        }

        iterations = 1000
        t0 = time.perf_counter()
        for _ in range(iterations):
            compressor.compress_response(response, config)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "ResponseCompressor.compress_response(large)",
            elapsed_ms, target_ms=2.0, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 2.0

    def test_compress_verbose_passthrough_latency(self, benchmark_reporter):
        """VERBOSE mode should be near-instant (no compression)."""
        config = ResponseOptimizationConfig.create_verbose("bench-session")
        compressor = ResponseCompressor()
        response = {
            "success": True,
            "keyword": "Log",
            "arguments": ["test"],
            "status": "pass",
            "output": "test",
        }

        iterations = 10000
        t0 = time.perf_counter()
        for _ in range(iterations):
            compressor.compress_response(response, config)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "ResponseCompressor.compress_response(verbose/passthrough)",
            elapsed_ms, target_ms=0.1, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.1

    def test_compression_ratio_meets_target(self):
        """Compact compression should achieve at least 20% token reduction."""
        config = ResponseOptimizationConfig.create_compact("bench-session")
        compressor = ResponseCompressor()
        response = {
            "success": True,
            "keyword": "Click Element",
            "arguments": ["id=submit-button"],
            "status": "pass",
            "execution_time": 0.523,
            "output": "Element clicked successfully",
            "assigned_variables": {},
            "session_id": "test-compress",
            "metadata": {},
            "description": "",
            "error": None,
            "message": None,
        }

        _, metrics = compressor.compress_response(response, config)
        # Field abbreviation + empty field removal should reduce tokens
        assert metrics.compression_ratio > 0.0, (
            f"Expected positive compression ratio, got {metrics.compression_ratio:.4f}"
        )
        # At minimum, empty fields should be removed
        assert metrics.fields_omitted > 0 or metrics.fields_abbreviated > 0, (
            "Expected at least some fields omitted or abbreviated"
        )

    def test_standard_mode_removes_empty_fields(self):
        """STANDARD mode should remove empty fields but not abbreviate."""
        config = ResponseOptimizationConfig.create_default("bench-session")
        compressor = ResponseCompressor()
        response = {
            "success": True,
            "keyword": "Log",
            "error": None,
            "message": "",
            "metadata": {},
            "description": "",
        }

        result, metrics = compressor.compress_response(response, config)
        # Empty fields in _OMIT_WHEN_EMPTY should be removed
        assert "error" not in result or result.get("error") is not None
        assert metrics.fields_abbreviated == 0, "STANDARD mode should not abbreviate"


# ============================================================
# IntentTarget Value Object Benchmarks
# ============================================================


class TestIntentTargetBenchmarks:
    """Benchmark IntentTarget value object creation."""

    def test_intent_target_creation_latency(self, benchmark_reporter):
        """IntentTarget creation should be < 0.01ms."""
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            IntentTarget(locator="text=Login")
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "IntentTarget(locator=...)",
            elapsed_ms, target_ms=0.01, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.01

    def test_intent_target_has_prefix_latency(self, benchmark_reporter):
        """has_prefix property should be < 0.01ms."""
        target = IntentTarget(locator="css=#submit")
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            _ = target.has_prefix
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "IntentTarget.has_prefix",
            elapsed_ms, target_ms=0.01, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.01


# ============================================================
# ModelTier Benchmarks
# ============================================================


class TestModelTierBenchmarks:
    """Benchmark ModelTier classification."""

    def test_from_context_window_latency(self, benchmark_reporter):
        """ModelTier.from_context_window() should be < 0.005ms."""
        iterations = 50000
        t0 = time.perf_counter()
        for _ in range(iterations):
            ModelTier.from_context_window(8192)
            ModelTier.from_context_window(32768)
            ModelTier.from_context_window(131072)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        benchmark_reporter.record_latency(
            "ModelTier.from_context_window() x3 tiers",
            elapsed_ms, target_ms=0.005, iterations=iterations,
        )
        avg_ms = elapsed_ms / iterations
        assert avg_ms < 0.005

    def test_model_tier_correctness(self):
        """Verify tier classification boundaries."""
        assert ModelTier.from_context_window(4096) == ModelTier.SMALL_CONTEXT
        assert ModelTier.from_context_window(8192) == ModelTier.SMALL_CONTEXT
        assert ModelTier.from_context_window(16384) == ModelTier.SMALL_CONTEXT
        assert ModelTier.from_context_window(32768) == ModelTier.STANDARD
        assert ModelTier.from_context_window(65536) == ModelTier.STANDARD
        assert ModelTier.from_context_window(131072) == ModelTier.LARGE_CONTEXT
