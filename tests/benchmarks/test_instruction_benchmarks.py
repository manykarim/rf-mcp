"""Performance benchmarks for MCP Instructions feature.

This module validates performance targets for instruction loading,
template rendering, and security validation:
- Instruction loading: <10ms
- Template rendering: <5ms
- Security validation: <2ms
- Path validation: <1ms

Run with:
    uv run pytest tests/benchmarks/test_instruction_benchmarks.py -v --benchmark-only
"""

import tempfile
import time
from pathlib import Path
from typing import Any, Dict

import pytest

from robotmcp.domains.instruction.value_objects import (
    InstructionContent,
    InstructionMode,
    InstructionPath,
    InstructionTemplate,
)
from robotmcp.domains.instruction.aggregates import InstructionConfig
from robotmcp.domains.instruction.services import (
    InstructionResolver,
    InstructionValidator,
    InstructionRenderer,
    ValidationResult,
)
from robotmcp.domains.instruction.adapters.fastmcp_adapter import FastMCPInstructionAdapter


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_instruction_file():
    """Create a temporary instruction file for benchmarks."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as f:
        f.write("""Custom instructions for Robot Framework MCP testing.

Use discovery tools before executing any keywords.
Verify keyword availability via find_keywords before execution.
Always check session state before running tests.
Use element references (refs) instead of raw locators.
Report failures with actionable error messages.
""")
        f.flush()
        yield Path(f.name)


@pytest.fixture
def large_instruction_file():
    """Create a large instruction file for stress testing."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as f:
        # Create ~10KB of instructions
        base_content = """Instruction block for testing performance.
Use discovery tools before executing keywords.
Verify availability before execution.
Check session state regularly.
""" * 50
        f.write(base_content)
        f.flush()
        yield Path(f.name)


@pytest.fixture
def resolver():
    """Create an instruction resolver for benchmarks."""
    return InstructionResolver()


@pytest.fixture
def validator():
    """Create an instruction validator with high token budget."""
    return InstructionValidator(max_token_budget=10000)


@pytest.fixture
def renderer():
    """Create an instruction renderer for benchmarks."""
    return InstructionRenderer()


@pytest.fixture
def adapter():
    """Create a FastMCP instruction adapter for benchmarks."""
    return FastMCPInstructionAdapter()


# ============================================================================
# Instruction Loading Benchmarks
# ============================================================================


class TestInstructionLoadingLatency:
    """Benchmark instruction loading time."""

    @pytest.mark.benchmark
    def test_default_instruction_loading(
        self,
        resolver: InstructionResolver,
        benchmark_reporter,
    ):
        """Target: <10ms for loading default instructions.

        Default instructions are loaded from the built-in template,
        which should be nearly instantaneous.
        """
        config = InstructionConfig.create_default()
        context = {"available_tools": "find_keywords, get_keyword_info"}

        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            _content = resolver.resolve(config, context)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="default_instruction_loading",
            duration_ms=total_ms,
            target_ms=10.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Default instruction loading {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 10ms target"
        )

    @pytest.mark.benchmark
    def test_custom_instruction_loading(
        self,
        resolver: InstructionResolver,
        temp_instruction_file: Path,
        benchmark_reporter,
    ):
        """Target: <10ms for loading custom instructions from file.

        First load may be slower due to file I/O, but subsequent loads
        should hit the cache.
        """
        path = InstructionPath(str(temp_instruction_file))
        config = InstructionConfig.create_custom(path)
        context = {}

        # First load (cold cache)
        start = time.perf_counter()
        _content = resolver.resolve(config, context)
        cold_load_ms = (time.perf_counter() - start) * 1000

        # Subsequent loads (warm cache)
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            _content = resolver.resolve(config, context)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="custom_instruction_loading_cached",
            duration_ms=total_ms,
            target_ms=10.0,
            iterations=iterations,
            cold_load_ms=cold_load_ms,
        )

        assert result.target_met, (
            f"Custom instruction loading {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 10ms target"
        )

    @pytest.mark.benchmark
    def test_large_file_loading(
        self,
        resolver: InstructionResolver,
        large_instruction_file: Path,
        benchmark_reporter,
    ):
        """Test loading performance with large instruction files."""
        path = InstructionPath(str(large_instruction_file))
        config = InstructionConfig.create_custom(path)

        # Cold load
        start = time.perf_counter()
        content = resolver.resolve(config, {})
        cold_load_ms = (time.perf_counter() - start) * 1000

        # Warm loads
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            _content = resolver.resolve(config, {})
        total_ms = (time.perf_counter() - start) * 1000

        file_size = large_instruction_file.stat().st_size
        result = benchmark_reporter.record_latency(
            name="large_file_loading",
            duration_ms=total_ms,
            target_ms=10.0,
            iterations=iterations,
            file_size_bytes=file_size,
            cold_load_ms=cold_load_ms,
            content_chars=len(content.value) if content else 0,
        )

        assert result.target_met, (
            f"Large file loading {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 10ms target for {file_size} byte file"
        )

    @pytest.mark.benchmark
    def test_instruction_loading_off_mode(
        self,
        resolver: InstructionResolver,
        benchmark_reporter,
    ):
        """Test that OFF mode returns immediately without loading."""
        config = InstructionConfig.create_off()

        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            _content = resolver.resolve(config, {})
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="off_mode_resolution",
            duration_ms=total_ms,
            target_ms=1.0,  # Should be nearly instant
            iterations=iterations,
        )

        assert result.target_met, (
            f"OFF mode resolution {result.avg_per_operation_ms:.6f}ms "
            f"exceeds 1ms target"
        )


# ============================================================================
# Template Rendering Benchmarks
# ============================================================================


class TestTemplateRenderingLatency:
    """Benchmark template rendering time."""

    @pytest.mark.benchmark
    def test_discovery_first_template_rendering(
        self,
        benchmark_reporter,
    ):
        """Target: <5ms for rendering discovery_first template.

        The discovery_first template is the most commonly used and
        should render quickly.
        """
        template = InstructionTemplate.discovery_first()
        context = {
            "available_tools": (
                "find_keywords, get_keyword_info, analyze_scenario, "
                "recommend_libraries, get_session_state, check_library_availability"
            )
        }

        iterations = 5000
        start = time.perf_counter()
        for _ in range(iterations):
            _content = template.render(context)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="discovery_first_template_render",
            duration_ms=total_ms,
            target_ms=5.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Template rendering {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 5ms target"
        )

    @pytest.mark.benchmark
    def test_minimal_template_rendering(
        self,
        benchmark_reporter,
    ):
        """Target: <5ms for rendering minimal template."""
        template = InstructionTemplate.minimal()

        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            _content = template.render({})
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="minimal_template_render",
            duration_ms=total_ms,
            target_ms=5.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Minimal template rendering {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 5ms target"
        )

    @pytest.mark.benchmark
    def test_locator_prevention_template_rendering(
        self,
        benchmark_reporter,
    ):
        """Target: <5ms for rendering locator_prevention template."""
        template = InstructionTemplate.locator_prevention()

        iterations = 5000
        start = time.perf_counter()
        for _ in range(iterations):
            _content = template.render({})
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="locator_prevention_template_render",
            duration_ms=total_ms,
            target_ms=5.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Locator prevention template rendering {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 5ms target"
        )

    @pytest.mark.benchmark
    def test_renderer_formatting(
        self,
        renderer: InstructionRenderer,
        benchmark_reporter,
    ):
        """Benchmark instruction rendering to different formats."""
        content = InstructionContent(
            value="Use discovery tools before action. Never guess keywords.",
            source="default",
        )

        formats = [
            InstructionRenderer.TargetFormat.CLAUDE,
            InstructionRenderer.TargetFormat.OPENAI,
            InstructionRenderer.TargetFormat.GENERIC,
        ]

        iterations = 3000
        start = time.perf_counter()
        for i in range(iterations):
            target_format = formats[i % len(formats)]
            _rendered = renderer.render(content, target_format)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="renderer_formatting",
            duration_ms=total_ms,
            target_ms=5.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Renderer formatting {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 5ms target"
        )


# ============================================================================
# Security Validation Benchmarks
# ============================================================================


class TestSecurityValidationLatency:
    """Benchmark security validation time."""

    @pytest.mark.benchmark
    def test_security_validation_clean_content(
        self,
        validator: InstructionValidator,
        benchmark_reporter,
    ):
        """Target: <2ms for validating clean instruction content.

        Security validation checks for dangerous patterns like script
        injection, eval, exec, and import statements.
        """
        content = InstructionContent(
            value="""Use discovery tools before executing keywords.
Verify keyword availability via find_keywords.
Check element snapshots for current page state.
Use locator guidance for finding elements.""",
            source="default",
        )

        iterations = 5000
        start = time.perf_counter()
        for _ in range(iterations):
            _result = validator.validate(content)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="security_validation_clean",
            duration_ms=total_ms,
            target_ms=2.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Security validation {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 2ms target"
        )

    @pytest.mark.benchmark
    def test_security_validation_with_dangerous_patterns(
        self,
        validator: InstructionValidator,
        benchmark_reporter,
    ):
        """Benchmark validation that detects dangerous patterns."""
        # Content with dangerous patterns (will fail validation)
        dangerous_contents = [
            InstructionContent(
                value="<script>alert('xss')</script> Use discovery tools.",
                source="custom:bad.txt",
            ),
            InstructionContent(
                value="Use eval() for dynamic code execution.",
                source="custom:bad.txt",
            ),
            InstructionContent(
                value="Try javascript:void(0) for actions.",
                source="custom:bad.txt",
            ),
        ]

        iterations = 3000
        start = time.perf_counter()
        for i in range(iterations):
            content = dangerous_contents[i % len(dangerous_contents)]
            _result = validator.validate(content)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="security_validation_dangerous",
            duration_ms=total_ms,
            target_ms=2.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Dangerous pattern detection {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 2ms target"
        )

    @pytest.mark.benchmark
    def test_security_validation_large_content(
        self,
        validator: InstructionValidator,
        benchmark_reporter,
    ):
        """Benchmark validation performance with large content."""
        # Create large content (~5000 chars)
        large_content = InstructionContent(
            value="Use discovery tools. " * 250,
            source="default",
        )

        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            _result = validator.validate(large_content)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="security_validation_large",
            duration_ms=total_ms,
            target_ms=2.0,
            iterations=iterations,
            content_chars=len(large_content.value),
        )

        assert result.target_met, (
            f"Large content validation {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 2ms target for {len(large_content.value)} chars"
        )


# ============================================================================
# Path Validation Benchmarks
# ============================================================================


class TestPathValidationLatency:
    """Benchmark path validation time."""

    @pytest.mark.benchmark
    def test_valid_path_validation(
        self,
        benchmark_reporter,
    ):
        """Target: <1ms for validating instruction file paths."""
        valid_paths = [
            "./instructions/custom.txt",
            "./my-instructions.md",
            "/absolute/path/instructions.instruction",
        ]

        iterations = 10000
        start = time.perf_counter()
        for i in range(iterations):
            path_str = valid_paths[i % len(valid_paths)]
            try:
                _path = InstructionPath(path_str)
            except ValueError:
                pass
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="valid_path_validation",
            duration_ms=total_ms,
            target_ms=1.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Path validation {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 1ms target"
        )

    @pytest.mark.benchmark
    def test_invalid_path_rejection(
        self,
        benchmark_reporter,
    ):
        """Benchmark rejection of invalid paths."""
        invalid_paths = [
            "../../../etc/passwd",
            "./file.exe",
            "./script.py",
            "file..txt",
        ]

        iterations = 10000
        start = time.perf_counter()
        for i in range(iterations):
            path_str = invalid_paths[i % len(invalid_paths)]
            try:
                _path = InstructionPath(path_str)
            except ValueError:
                pass  # Expected
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="invalid_path_rejection",
            duration_ms=total_ms,
            target_ms=1.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Invalid path rejection {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 1ms target"
        )

    @pytest.mark.benchmark
    def test_path_resolution(
        self,
        temp_instruction_file: Path,
        benchmark_reporter,
    ):
        """Benchmark path resolution with security checks."""
        instruction_path = InstructionPath(str(temp_instruction_file))
        base_path = temp_instruction_file.parent

        iterations = 5000
        start = time.perf_counter()
        for _ in range(iterations):
            _resolved = instruction_path.resolve(base_path)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="path_resolution",
            duration_ms=total_ms,
            target_ms=1.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Path resolution {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 1ms target"
        )


# ============================================================================
# Full Pipeline Benchmarks
# ============================================================================


class TestFullInstructionPipeline:
    """Benchmark the complete instruction pipeline."""

    @pytest.mark.benchmark
    def test_full_pipeline_default_mode(
        self,
        adapter: FastMCPInstructionAdapter,
        benchmark_reporter,
    ):
        """Benchmark complete pipeline: config -> resolve -> validate -> render."""
        config = InstructionConfig.create_default()
        context = {"available_tools": "find_keywords, get_keyword_info"}

        iterations = 500
        start = time.perf_counter()
        for _ in range(iterations):
            _instructions = adapter.get_server_instructions(config, context)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="full_pipeline_default",
            duration_ms=total_ms,
            target_ms=20.0,  # Full pipeline should complete in <20ms
            iterations=iterations,
        )

        assert result.target_met, (
            f"Full pipeline {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 20ms target"
        )

    @pytest.mark.benchmark
    def test_full_pipeline_custom_mode(
        self,
        adapter: FastMCPInstructionAdapter,
        temp_instruction_file: Path,
        benchmark_reporter,
    ):
        """Benchmark complete pipeline with custom instructions."""
        path = InstructionPath(str(temp_instruction_file))
        config = InstructionConfig.create_custom(path)

        iterations = 500
        start = time.perf_counter()
        for _ in range(iterations):
            _instructions = adapter.get_server_instructions(config, {})
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="full_pipeline_custom",
            duration_ms=total_ms,
            target_ms=20.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Full pipeline with custom file {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 20ms target"
        )

    @pytest.mark.benchmark
    def test_config_from_env_loading(
        self,
        adapter: FastMCPInstructionAdapter,
        benchmark_reporter,
    ):
        """Benchmark configuration loading from environment."""
        iterations = 5000
        start = time.perf_counter()
        for _ in range(iterations):
            _config = adapter.create_config_from_env()
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="config_from_env",
            duration_ms=total_ms,
            target_ms=5.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Config from env {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 5ms target"
        )


# ============================================================================
# Mode Switching Benchmarks
# ============================================================================


class TestModeSwitchingLatency:
    """Benchmark mode and configuration switching."""

    @pytest.mark.benchmark
    def test_mode_value_object_creation(
        self,
        benchmark_reporter,
    ):
        """Benchmark InstructionMode value object creation."""
        modes = ["off", "default", "custom"]

        iterations = 50000
        start = time.perf_counter()
        for i in range(iterations):
            mode_str = modes[i % len(modes)]
            _mode = InstructionMode.from_string(mode_str)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="mode_creation",
            duration_ms=total_ms,
            target_ms=0.1,  # Should be very fast
            iterations=iterations,
        )

        assert result.target_met, (
            f"Mode creation {result.avg_per_operation_ms:.6f}ms "
            f"exceeds 0.1ms target"
        )

    @pytest.mark.benchmark
    def test_config_mode_switching(
        self,
        benchmark_reporter,
    ):
        """Benchmark switching between instruction modes."""
        config = InstructionConfig.create_default()

        iterations = 10000
        start = time.perf_counter()
        for i in range(iterations):
            if i % 2 == 0:
                config = config.with_mode(InstructionMode.off())
            else:
                config = config.with_mode(InstructionMode.default())
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="config_mode_switching",
            duration_ms=total_ms,
            target_ms=1.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Config mode switching {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 1ms target"
        )

    @pytest.mark.benchmark
    def test_content_assignment(
        self,
        benchmark_reporter,
    ):
        """Benchmark assigning content to configuration."""
        config = InstructionConfig.create_default()
        content = InstructionContent(
            value="Use discovery tools before action.",
            source="default",
        )

        iterations = 5000
        start = time.perf_counter()
        for i in range(iterations):
            config = config.with_content(content, f"session-{i}")
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="content_assignment",
            duration_ms=total_ms,
            target_ms=1.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Content assignment {result.avg_per_operation_ms:.4f}ms "
            f"exceeds 1ms target"
        )
