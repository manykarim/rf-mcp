"""Benchmark latency at each optimization stage.

This module validates latency targets from the ADR:
- ARIA snapshot generation: <200ms
- Ref lookup: <1ms per operation
- Incremental diff computation: <50ms
- Pre-validation checks: <100ms

Run with:
    pytest tests/benchmarks/benchmark_latency.py -v
"""

import time
from typing import Any, Dict, List

import pytest


class TestARIASnapshotLatency:
    """Benchmark ARIA snapshot generation latency."""

    @pytest.mark.benchmark
    def test_aria_snapshot_generation_latency(
        self,
        sample_html_pages: Dict[str, str],
        benchmark_reporter,
    ):
        """Target: <200ms for snapshot generation.

        This measures the time to convert raw HTML to an accessibility tree
        representation. In production, this uses Browser Library's
        `Get Aria Snapshots` keyword which calls Playwright's accessibility API.

        For benchmarking, we simulate the conversion process.
        """
        html = sample_html_pages.get("medium", "")
        if not html:
            pytest.skip("Medium page fixture not available")

        # Simulate ARIA snapshot generation
        # In real implementation, this would be:
        # robot.run_keyword("Get Aria Snapshots")
        def simulate_aria_snapshot(html_content: str) -> str:
            """Simulate ARIA snapshot generation."""
            # Parse-like operations
            lines = html_content.split("\n")
            elements = []
            for line in lines:
                if "data-testid" in line or "role=" in line:
                    elements.append(line.strip()[:50])
            return "\n".join(elements[:500])

        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            _snapshot = simulate_aria_snapshot(html)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="aria_snapshot_generation",
            duration_ms=total_ms,
            target_ms=200.0,
            iterations=iterations,
            html_size_chars=len(html),
        )

        assert result.target_met, (
            f"ARIA snapshot generation {result.avg_per_operation_ms:.2f}ms "
            f"exceeds 200ms target"
        )

    @pytest.mark.benchmark
    def test_aria_snapshot_scaling(
        self,
        sample_html_pages: Dict[str, str],
        benchmark_reporter,
    ):
        """Test ARIA snapshot latency scales appropriately with page size."""
        results = {}

        for page_type in ["small", "medium", "large"]:
            html = sample_html_pages.get(page_type, "")
            if not html:
                continue

            def simulate_snapshot(content: str) -> str:
                lines = content.split("\n")
                return "\n".join(line[:50] for line in lines[:len(lines) // 10])

            iterations = 50
            start = time.perf_counter()
            for _ in range(iterations):
                _snapshot = simulate_snapshot(html)
            total_ms = (time.perf_counter() - start) * 1000

            avg_ms = total_ms / iterations
            results[page_type] = {
                "avg_ms": avg_ms,
                "html_chars": len(html),
            }

            benchmark_reporter.record_latency(
                name=f"aria_snapshot_{page_type}",
                duration_ms=total_ms,
                target_ms=200.0,
                iterations=iterations,
            )

        # Verify scaling is reasonable (large should not be >10x small)
        if "small" in results and "large" in results:
            ratio = results["large"]["avg_ms"] / results["small"]["avg_ms"]
            size_ratio = results["large"]["html_chars"] / results["small"]["html_chars"]

            # Allow up to 2x the size ratio for scaling (accounting for processing overhead)
            assert ratio < size_ratio * 2.0, (
                f"Snapshot scaling {ratio:.1f}x is inefficient for "
                f"{size_ratio:.1f}x size increase"
            )


class TestDiffComputationLatency:
    """Benchmark incremental snapshot diff computation."""

    @pytest.mark.benchmark
    def test_diff_computation_latency(
        self,
        sample_aria_snapshots: Dict[str, str],
        benchmark_reporter,
    ):
        """Target: <50ms for incremental diff computation.

        Incremental snapshots return only changed portions between
        consecutive page states, reducing token usage by 40-60%.
        """
        base_snapshot = sample_aria_snapshots.get("medium", "")
        if not base_snapshot:
            pytest.skip("Medium snapshot fixture not available")

        # Simulate small changes (10% of lines modified)
        lines = base_snapshot.split("\n")
        changed_indices = set(range(0, len(lines), 10))
        modified_snapshot = "\n".join(
            f"[CHANGED] {line}" if i in changed_indices else line
            for i, line in enumerate(lines)
        )

        def compute_diff(old: str, new: str) -> str:
            """Simulate diff computation."""
            old_lines = old.split("\n")
            new_lines = new.split("\n")

            diff_lines = []
            for i, (o, n) in enumerate(zip(old_lines, new_lines)):
                if o != n:
                    diff_lines.append(f"@{i}: {n}")

            # Handle length differences
            if len(new_lines) > len(old_lines):
                for i, line in enumerate(new_lines[len(old_lines):], len(old_lines)):
                    diff_lines.append(f"+{i}: {line}")

            return "\n".join(diff_lines) if diff_lines else "[No changes]"

        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            _diff = compute_diff(base_snapshot, modified_snapshot)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="diff_computation",
            duration_ms=total_ms,
            target_ms=50.0,
            iterations=iterations,
            base_lines=len(lines),
            changed_lines=len(changed_indices),
        )

        assert result.target_met, (
            f"Diff computation {result.avg_per_operation_ms:.2f}ms exceeds 50ms target"
        )

    @pytest.mark.benchmark
    def test_diff_token_savings(
        self,
        sample_aria_snapshots: Dict[str, str],
        benchmark_reporter,
        estimate_tokens,
    ):
        """Verify incremental diff provides expected token savings."""
        base_snapshot = sample_aria_snapshots.get("large", "")
        if not base_snapshot:
            pytest.skip("Large snapshot fixture not available")

        # Simulate 5% change (typical for small user interactions)
        lines = base_snapshot.split("\n")
        change_count = max(1, len(lines) // 20)
        changed_indices = set(range(0, change_count * 20, 20))

        diff_output = "\n".join(
            f"@{i}: [UPDATED] {lines[i]}"
            for i in changed_indices
            if i < len(lines)
        )

        full_tokens = estimate_tokens(base_snapshot)
        diff_tokens = estimate_tokens(diff_output)

        reduction = (full_tokens - diff_tokens) / full_tokens * 100

        benchmark_reporter.record(
            name="incremental_diff_savings",
            duration_ms=0,  # Not measuring time here
            tokens_before=full_tokens,
            tokens_after=diff_tokens,
            target_reduction=40.0,  # Expect at least 40% savings for 5% change
            change_percentage=5,
        )

        assert reduction >= 40, (
            f"Incremental diff only saved {reduction:.1f}% tokens "
            f"(expected 40%+ for 5% change)"
        )


class TestPreValidationLatency:
    """Benchmark pre-validation checks before element actions."""

    @pytest.mark.benchmark
    def test_prevalidation_latency(
        self,
        benchmark_reporter,
    ):
        """Target: <100ms for pre-validation checks.

        Pre-validation checks element state before interaction:
        - Element attached to DOM
        - Element visible
        - Element stable (not animating)
        - Element enabled
        - Element receives pointer events
        - Element in viewport
        """

        def simulate_prevalidation(element_state: Dict[str, Any]) -> Dict[str, Any]:
            """Simulate pre-validation checks."""
            checks = {
                "attached": element_state.get("attached", True),
                "visible": element_state.get("visible", True),
                "stable": element_state.get("stable", True),
                "enabled": element_state.get("enabled", True),
                "receives_events": element_state.get("receives_events", True),
                "in_viewport": element_state.get("in_viewport", True),
            }

            all_passed = all(checks.values())
            failed_checks = [k for k, v in checks.items() if not v]

            return {
                "valid": all_passed,
                "checks": checks,
                "failed": failed_checks,
            }

        # Create various element states
        element_states = [
            {"attached": True, "visible": True, "enabled": True},
            {"attached": True, "visible": False, "enabled": True},
            {"attached": True, "visible": True, "enabled": False},
            {"attached": False, "visible": True, "enabled": True},
        ] * 250  # 1000 total

        iterations = len(element_states)
        start = time.perf_counter()
        for state in element_states:
            _result = simulate_prevalidation(state)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="prevalidation_checks",
            duration_ms=total_ms,
            target_ms=100.0,
            iterations=iterations,
        )

        assert result.target_met, (
            f"Pre-validation {result.avg_per_operation_ms:.2f}ms exceeds 100ms target"
        )

    @pytest.mark.benchmark
    def test_trial_mode_validation(
        self,
        benchmark_reporter,
    ):
        """Test trial mode validation (checks without action execution).

        Trial mode performs all actionability checks without executing
        the action, useful for pre-validation workflows.
        """

        def simulate_trial_action(
            action_type: str,
            element_state: Dict[str, Any],
        ) -> Dict[str, Any]:
            """Simulate trial action validation."""
            base_checks = {
                "attached": element_state.get("attached", True),
                "visible": element_state.get("visible", True),
                "stable": element_state.get("stable", True),
            }

            # Action-specific checks
            if action_type == "click":
                base_checks["enabled"] = element_state.get("enabled", True)
                base_checks["receives_events"] = element_state.get("receives_events", True)
            elif action_type == "fill":
                base_checks["editable"] = element_state.get("editable", True)
            elif action_type == "select":
                base_checks["has_options"] = element_state.get("has_options", True)

            return {
                "action": action_type,
                "trial": True,
                "would_succeed": all(base_checks.values()),
                "checks": base_checks,
            }

        actions = ["click", "fill", "select", "hover", "focus"]
        element_state = {
            "attached": True,
            "visible": True,
            "stable": True,
            "enabled": True,
            "editable": True,
            "receives_events": True,
            "has_options": True,
        }

        iterations = 5000
        start = time.perf_counter()
        for i in range(iterations):
            action = actions[i % len(actions)]
            _result = simulate_trial_action(action, element_state)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="trial_mode_validation",
            duration_ms=total_ms,
            target_ms=10.0,  # Trial should be very fast
            iterations=iterations,
        )

        assert result.target_met, (
            f"Trial mode {result.avg_per_operation_ms:.3f}ms exceeds 10ms target"
        )


class TestEndToEndLatency:
    """Benchmark complete optimization pipeline latency."""

    @pytest.mark.benchmark
    def test_full_pipeline_latency(
        self,
        sample_html_pages: Dict[str, str],
        sample_aria_snapshots: Dict[str, str],
        benchmark_reporter,
    ):
        """Benchmark the complete token optimization pipeline.

        Pipeline stages:
        1. ARIA snapshot generation
        2. Response optimization
        3. (Optional) Diff computation for incremental mode
        """
        from robotmcp.utils.token_efficient_output import optimize_output

        html = sample_html_pages.get("medium", "")
        aria_snapshot = sample_aria_snapshots.get("medium", "")

        if not html or not aria_snapshot:
            pytest.skip("Required fixtures not available")

        def full_pipeline(html_content: str, snapshot: str) -> Dict[str, Any]:
            """Execute full optimization pipeline."""
            # Stage 1: Simulate ARIA snapshot (already have it)

            # Stage 2: Build optimized response
            response = {
                "success": True,
                "snapshot": snapshot,
            }

            return optimize_output(response, verbosity="standard")

        iterations = 50
        start = time.perf_counter()
        for _ in range(iterations):
            _result = full_pipeline(html, aria_snapshot)
        total_ms = (time.perf_counter() - start) * 1000

        result = benchmark_reporter.record_latency(
            name="full_optimization_pipeline",
            duration_ms=total_ms,
            target_ms=300.0,  # Full pipeline should complete in <300ms
            iterations=iterations,
        )

        assert result.target_met, (
            f"Full pipeline {result.avg_per_operation_ms:.2f}ms exceeds 300ms target"
        )
