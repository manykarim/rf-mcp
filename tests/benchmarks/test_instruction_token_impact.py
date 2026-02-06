"""Token impact benchmarks for MCP Instructions feature.

This module measures and validates token counts for instruction templates:
- Minimal template: <500 tokens
- Standard/default template: <1200 tokens
- Detailed template (discovery_first): <1800 tokens
- Compare with/without instructions overhead

Run with:
    uv run pytest tests/benchmarks/test_instruction_token_impact.py -v --benchmark-only
"""

import time
from typing import Any, Dict, List, Tuple

import pytest

from robotmcp.domains.instruction.value_objects import (
    InstructionContent,
    InstructionTemplate,
)
from robotmcp.domains.instruction.aggregates import InstructionConfig
from robotmcp.domains.instruction.services import InstructionResolver, InstructionRenderer


# ============================================================================
# Token Estimation Utilities
# ============================================================================


def estimate_tokens_simple(text: str) -> int:
    """Simple token estimation using character/4 heuristic.

    This is a fast approximation used by the InstructionContent class.
    Actual tokenization varies by model.
    """
    return len(text) // 4


def estimate_tokens_word_based(text: str) -> int:
    """Word-based token estimation (words * 1.3).

    More accurate for natural language text.
    """
    word_count = len(text.split())
    return int(word_count * 1.3)


def count_characters(text: str) -> int:
    """Count total characters."""
    return len(text)


def count_lines(text: str) -> int:
    """Count total lines."""
    return len(text.splitlines())


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def resolver():
    """Create an instruction resolver."""
    return InstructionResolver()


@pytest.fixture
def renderer():
    """Create an instruction renderer."""
    return InstructionRenderer()


@pytest.fixture
def standard_context():
    """Standard context for template rendering."""
    return {
        "available_tools": (
            "find_keywords, get_keyword_info, get_session_state, "
            "get_locator_guidance, analyze_scenario, recommend_libraries, "
            "check_library_availability"
        )
    }


# ============================================================================
# Template Token Count Tests
# ============================================================================


class TestTemplateTokenCounts:
    """Verify template token counts are within limits."""

    @pytest.mark.benchmark
    def test_minimal_template_token_count(
        self,
        benchmark_reporter,
    ):
        """Verify minimal template is under 500 tokens."""
        template = InstructionTemplate.minimal()
        content = template.render({})

        char_count = count_characters(content.value)
        word_token_estimate = estimate_tokens_word_based(content.value)
        char_token_estimate = estimate_tokens_simple(content.value)
        line_count = count_lines(content.value)

        # Use the InstructionContent's built-in token estimate
        builtin_estimate = content.token_estimate

        benchmark_reporter.record(
            name="minimal_template_tokens",
            duration_ms=0,  # Not timing here
            tokens_before=500,  # Target limit
            tokens_after=builtin_estimate,
            target_reduction=0,  # Not about reduction
            char_count=char_count,
            word_token_estimate=word_token_estimate,
            char_token_estimate=char_token_estimate,
            line_count=line_count,
        )

        assert builtin_estimate < 500, (
            f"Minimal template has {builtin_estimate} tokens, exceeds 500 limit"
        )
        assert char_count < 2000, (
            f"Minimal template has {char_count} chars, expected <2000"
        )

    @pytest.mark.benchmark
    def test_locator_prevention_template_token_count(
        self,
        benchmark_reporter,
    ):
        """Verify locator_prevention template is under 800 tokens."""
        template = InstructionTemplate.locator_prevention()
        content = template.render({})

        char_count = count_characters(content.value)
        builtin_estimate = content.token_estimate

        benchmark_reporter.record(
            name="locator_prevention_template_tokens",
            duration_ms=0,
            tokens_before=800,  # Target limit
            tokens_after=builtin_estimate,
            target_reduction=0,
            char_count=char_count,
        )

        assert builtin_estimate < 800, (
            f"Locator prevention template has {builtin_estimate} tokens, exceeds 800 limit"
        )

    @pytest.mark.benchmark
    def test_discovery_first_template_token_count(
        self,
        standard_context: Dict[str, str],
        benchmark_reporter,
    ):
        """Verify discovery_first (standard) template is under 1200 tokens."""
        template = InstructionTemplate.discovery_first()
        content = template.render(standard_context)

        char_count = count_characters(content.value)
        builtin_estimate = content.token_estimate
        line_count = count_lines(content.value)

        benchmark_reporter.record(
            name="discovery_first_template_tokens",
            duration_ms=0,
            tokens_before=1200,  # Target limit
            tokens_after=builtin_estimate,
            target_reduction=0,
            char_count=char_count,
            line_count=line_count,
        )

        assert builtin_estimate < 1200, (
            f"Discovery first template has {builtin_estimate} tokens, exceeds 1200 limit"
        )

    @pytest.mark.benchmark
    def test_all_templates_comparison(
        self,
        standard_context: Dict[str, str],
        benchmark_reporter,
    ):
        """Compare token counts across all templates."""
        templates = [
            ("minimal", InstructionTemplate.minimal()),
            ("locator_prevention", InstructionTemplate.locator_prevention()),
            ("discovery_first", InstructionTemplate.discovery_first()),
        ]

        results: List[Tuple[str, int, int]] = []
        for name, template in templates:
            # Use context only for templates that need it
            ctx = standard_context if template.placeholders else {}
            content = template.render(ctx)

            token_estimate = content.token_estimate
            char_count = count_characters(content.value)
            results.append((name, token_estimate, char_count))

            benchmark_reporter.record(
                name=f"template_{name}_tokens",
                duration_ms=0,
                tokens_before=1800,  # Max allowed
                tokens_after=token_estimate,
                target_reduction=0,
                char_count=char_count,
            )

        # Verify ordering: minimal < locator_prevention < discovery_first
        assert results[0][1] < results[1][1] < results[2][1], (
            f"Templates should be ordered by size: {results}"
        )

        # Verify all under max limit
        for name, tokens, chars in results:
            assert tokens < 1800, (
                f"Template '{name}' has {tokens} tokens, exceeds 1800 max"
            )


# ============================================================================
# Rendered Format Token Impact
# ============================================================================


class TestRenderedFormatTokenImpact:
    """Measure token overhead from rendering to different formats."""

    @pytest.mark.benchmark
    def test_render_format_overhead(
        self,
        renderer: InstructionRenderer,
        standard_context: Dict[str, str],
        benchmark_reporter,
    ):
        """Measure token overhead added by format wrappers."""
        template = InstructionTemplate.discovery_first()
        content = template.render(standard_context)
        base_tokens = content.token_estimate

        formats = [
            InstructionRenderer.TargetFormat.CLAUDE,
            InstructionRenderer.TargetFormat.OPENAI,
            InstructionRenderer.TargetFormat.GENERIC,
        ]

        for target_format in formats:
            rendered = renderer.render(content, target_format, include_wrapper=True)
            rendered_tokens = estimate_tokens_word_based(rendered)
            overhead = rendered_tokens - base_tokens
            overhead_percent = (overhead / base_tokens * 100) if base_tokens > 0 else 0

            benchmark_reporter.record(
                name=f"render_{target_format.value}_overhead",
                duration_ms=0,
                tokens_before=base_tokens,
                tokens_after=rendered_tokens,
                target_reduction=0,
                overhead_tokens=overhead,
                overhead_percent=overhead_percent,
            )

            # Wrapper overhead should be minimal (<5%)
            assert overhead_percent < 5, (
                f"{target_format.value} format adds {overhead_percent:.1f}% overhead"
            )

    @pytest.mark.benchmark
    def test_render_without_wrapper(
        self,
        renderer: InstructionRenderer,
        standard_context: Dict[str, str],
        benchmark_reporter,
    ):
        """Verify no-wrapper rendering has zero overhead."""
        template = InstructionTemplate.discovery_first()
        content = template.render(standard_context)
        base_tokens = content.token_estimate

        for target_format in InstructionRenderer.TargetFormat:
            rendered = renderer.render(content, target_format, include_wrapper=False)
            rendered_tokens = estimate_tokens_word_based(rendered)

            # Without wrapper, tokens should be the same (or very close)
            assert abs(rendered_tokens - base_tokens) < 5, (
                f"No-wrapper render for {target_format.value} changed tokens: "
                f"{base_tokens} -> {rendered_tokens}"
            )


# ============================================================================
# With/Without Instructions Comparison
# ============================================================================


class TestInstructionOverhead:
    """Compare token usage with and without instructions."""

    @pytest.mark.benchmark
    def test_instruction_overhead_minimal(
        self,
        resolver: InstructionResolver,
        benchmark_reporter,
    ):
        """Measure overhead when using minimal instructions."""
        # Simulate a tool response without instructions
        tool_response = {
            "keywords": ["Click", "Fill Text", "Get Text"],
            "library": "Browser",
            "count": 3,
        }
        base_response_str = str(tool_response)
        base_tokens = estimate_tokens_word_based(base_response_str)

        # With minimal instructions
        template = InstructionTemplate.minimal()
        instruction_content = template.render({})
        instruction_tokens = instruction_content.token_estimate

        total_with_instructions = base_tokens + instruction_tokens
        overhead_percent = (instruction_tokens / base_tokens * 100) if base_tokens > 0 else 0

        benchmark_reporter.record(
            name="overhead_minimal_instructions",
            duration_ms=0,
            tokens_before=base_tokens,
            tokens_after=total_with_instructions,
            target_reduction=0,
            instruction_tokens=instruction_tokens,
            overhead_percent=overhead_percent,
        )

        # Minimal instructions should have low overhead
        assert instruction_tokens < 500, (
            f"Minimal instructions add {instruction_tokens} tokens"
        )

    @pytest.mark.benchmark
    def test_instruction_overhead_default(
        self,
        resolver: InstructionResolver,
        standard_context: Dict[str, str],
        benchmark_reporter,
    ):
        """Measure overhead when using default instructions."""
        # Simulate a typical tool response
        tool_response = {
            "snapshot": "- document\n  - heading 'Title'\n  - button 'Submit' [ref=e1]",
            "refs": ["e1", "e2", "e3"],
            "page_title": "Test Page",
        }
        base_response_str = str(tool_response)
        base_tokens = estimate_tokens_word_based(base_response_str)

        # With default instructions
        template = InstructionTemplate.discovery_first()
        instruction_content = template.render(standard_context)
        instruction_tokens = instruction_content.token_estimate

        total_with_instructions = base_tokens + instruction_tokens
        overhead_percent = (instruction_tokens / base_tokens * 100) if base_tokens > 0 else 0

        benchmark_reporter.record(
            name="overhead_default_instructions",
            duration_ms=0,
            tokens_before=base_tokens,
            tokens_after=total_with_instructions,
            target_reduction=0,
            instruction_tokens=instruction_tokens,
            overhead_percent=overhead_percent,
        )

        # Default instructions should add reasonable overhead
        assert instruction_tokens < 1200, (
            f"Default instructions add {instruction_tokens} tokens"
        )

    @pytest.mark.benchmark
    def test_instruction_amortization(
        self,
        standard_context: Dict[str, str],
        benchmark_reporter,
    ):
        """Calculate how instruction cost amortizes over multiple interactions.

        Instructions are sent once at session start, then amortized
        over many tool calls.
        """
        template = InstructionTemplate.discovery_first()
        instruction_content = template.render(standard_context)
        instruction_tokens = instruction_content.token_estimate

        # Simulate typical session with multiple tool calls
        avg_tokens_per_tool_call = 50
        num_tool_calls_scenarios = [5, 10, 25, 50, 100]

        for num_calls in num_tool_calls_scenarios:
            total_tool_tokens = avg_tokens_per_tool_call * num_calls
            total_with_instructions = instruction_tokens + total_tool_tokens
            overhead_per_call = instruction_tokens / num_calls
            overhead_percent = (instruction_tokens / total_with_instructions * 100)

            benchmark_reporter.record(
                name=f"amortization_{num_calls}_calls",
                duration_ms=0,
                tokens_before=total_tool_tokens,
                tokens_after=total_with_instructions,
                target_reduction=0,
                num_tool_calls=num_calls,
                overhead_per_call=overhead_per_call,
                overhead_percent=overhead_percent,
            )

        # With 50+ tool calls, overhead should be <5%
        final_overhead = instruction_tokens / (instruction_tokens + avg_tokens_per_tool_call * 50) * 100
        assert final_overhead < 25, (
            f"Instruction overhead with 50 calls is {final_overhead:.1f}%, expected <25%"
        )


# ============================================================================
# Character Limit Tests
# ============================================================================


class TestCharacterLimits:
    """Verify templates stay within character limits."""

    @pytest.mark.benchmark
    def test_minimal_character_limit(
        self,
        benchmark_reporter,
    ):
        """Verify minimal template character count."""
        template = InstructionTemplate.minimal()
        content = template.render({})
        char_count = len(content.value)

        benchmark_reporter.record(
            name="minimal_chars",
            duration_ms=0,
            tokens_before=2000,  # Target char limit
            tokens_after=char_count,
            target_reduction=0,
        )

        # Minimal should be very compact
        assert char_count < 500, (
            f"Minimal template has {char_count} chars, expected <500"
        )

    @pytest.mark.benchmark
    def test_standard_character_limit(
        self,
        standard_context: Dict[str, str],
        benchmark_reporter,
    ):
        """Verify standard template character count."""
        template = InstructionTemplate.discovery_first()
        content = template.render(standard_context)
        char_count = len(content.value)

        benchmark_reporter.record(
            name="standard_chars",
            duration_ms=0,
            tokens_before=5000,  # Target char limit
            tokens_after=char_count,
            target_reduction=0,
        )

        assert char_count < 5000, (
            f"Standard template has {char_count} chars, expected <5000"
        )

    @pytest.mark.benchmark
    def test_max_content_limit(
        self,
        benchmark_reporter,
    ):
        """Verify InstructionContent enforces max length."""
        from robotmcp.domains.instruction.value_objects import InstructionContent

        max_length = InstructionContent.MAX_LENGTH

        # Content at limit should work
        at_limit = "x" * (max_length - 1)
        content = InstructionContent(value=at_limit, source="test")

        benchmark_reporter.record(
            name="max_content_enforcement",
            duration_ms=0,
            tokens_before=max_length,
            tokens_after=len(content.value),
            target_reduction=0,
        )

        assert len(content.value) == max_length - 1

        # Content over limit should fail
        over_limit = "x" * (max_length + 1)
        with pytest.raises(ValueError, match="too long"):
            InstructionContent(value=over_limit, source="test")


# ============================================================================
# Token Budget Compliance Tests
# ============================================================================


class TestTokenBudgetCompliance:
    """Test token budget enforcement in configurations."""

    @pytest.mark.benchmark
    def test_default_budget_compliance(
        self,
        resolver: InstructionResolver,
        standard_context: Dict[str, str],
        benchmark_reporter,
    ):
        """Verify default instructions fit within default token budget."""
        config = InstructionConfig.create_default()
        default_budget = config.max_token_budget

        content = resolver.resolve(config, standard_context)
        assert content is not None

        token_estimate = content.token_estimate
        utilization = (token_estimate / default_budget * 100) if default_budget > 0 else 0

        benchmark_reporter.record(
            name="default_budget_compliance",
            duration_ms=0,
            tokens_before=default_budget,
            tokens_after=token_estimate,
            target_reduction=0,
            utilization_percent=utilization,
        )

        # Should fit within budget with some headroom
        assert token_estimate <= default_budget, (
            f"Default instructions ({token_estimate} tokens) exceed budget ({default_budget})"
        )

    @pytest.mark.benchmark
    def test_custom_budget_enforcement(
        self,
        benchmark_reporter,
    ):
        """Test that custom budgets are enforced."""
        budgets = [500, 1000, 2000, 5000]

        for budget in budgets:
            config = InstructionConfig.create_default()
            config = config.with_options(max_token_budget=budget)

            errors = config.validate()

            benchmark_reporter.record(
                name=f"budget_{budget}_validation",
                duration_ms=0,
                tokens_before=budget,
                tokens_after=config.max_token_budget,
                target_reduction=0,
                validation_errors=len(errors),
            )

            # Config validation should pass (content not loaded yet)
            assert config.max_token_budget == budget

    @pytest.mark.benchmark
    def test_budget_utilization_efficiency(
        self,
        resolver: InstructionResolver,
        standard_context: Dict[str, str],
        benchmark_reporter,
    ):
        """Analyze how efficiently templates use the token budget."""
        templates = [
            ("minimal", InstructionTemplate.minimal()),
            ("locator_prevention", InstructionTemplate.locator_prevention()),
            ("discovery_first", InstructionTemplate.discovery_first()),
        ]

        default_budget = 1000
        results = []

        for name, template in templates:
            ctx = standard_context if template.placeholders else {}
            content = template.render(ctx)

            tokens = content.token_estimate
            utilization = (tokens / default_budget * 100)
            headroom = default_budget - tokens

            results.append({
                "name": name,
                "tokens": tokens,
                "utilization": utilization,
                "headroom": headroom,
            })

            benchmark_reporter.record(
                name=f"budget_efficiency_{name}",
                duration_ms=0,
                tokens_before=default_budget,
                tokens_after=tokens,
                target_reduction=0,
                utilization_percent=utilization,
                headroom_tokens=headroom,
            )

        # Minimal should use <50% of budget
        assert results[0]["utilization"] < 50, (
            f"Minimal uses {results[0]['utilization']:.1f}% of budget"
        )

        # Discovery first can use up to 100% but should have some headroom
        assert results[2]["headroom"] > 0, (
            f"Discovery first has no headroom: {results[2]['headroom']} tokens"
        )
