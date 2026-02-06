#!/usr/bin/env python3
"""Run benchmarks and generate comprehensive report.

This script runs all token optimization benchmarks and generates:
1. Console output with pass/fail status
2. JSON results file for CI integration
3. Markdown report for documentation

Usage:
    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --output-dir results/
    python scripts/run_benchmarks.py --quick  # Run subset of benchmarks

Exit codes:
    0 - All benchmarks passed
    1 - Some benchmarks failed
    2 - Error running benchmarks
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run token optimization benchmarks and generate report"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/benchmarks"),
        help="Directory for output files (default: docs/benchmarks)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only quick benchmarks (skip e2e)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Only output JSON results (no markdown)",
    )
    parser.add_argument(
        "--markers",
        type=str,
        default="benchmark",
        help="Pytest markers to select (default: benchmark)",
    )
    return parser.parse_args()


def run_pytest_benchmarks(
    markers: str = "benchmark",
    quick: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run pytest benchmarks and collect results.

    Args:
        markers: Pytest markers to select
        quick: Skip e2e benchmarks if True
        verbose: Enable verbose output

    Returns:
        Dictionary with benchmark results
    """
    # Build pytest command
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/benchmarks/",
        f"-m={markers}",
        "--tb=short",
        "-q" if not verbose else "-v",
        "--json-report",
        "--json-report-file=.benchmark_results.json",
    ]

    if quick:
        cmd.extend(["--ignore=tests/benchmarks/benchmark_e2e.py"])

    print(f"Running: {' '.join(cmd)}")
    print("-" * 70)

    # Run pytest
    result = subprocess.run(cmd, capture_output=not verbose, text=True)

    # Load JSON report if available
    json_report_path = Path(".benchmark_results.json")
    if json_report_path.exists():
        with open(json_report_path) as f:
            report = json.load(f)
        json_report_path.unlink()  # Clean up
    else:
        report = {"tests": [], "summary": {"passed": 0, "failed": 0}}

    return {
        "exit_code": result.returncode,
        "report": report,
        "stdout": result.stdout if not verbose else "",
        "stderr": result.stderr if not verbose else "",
    }


def collect_benchmark_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    """Extract benchmark metrics from pytest report.

    Args:
        report: Pytest JSON report

    Returns:
        Structured benchmark metrics
    """
    metrics = {
        "token_reduction": [],
        "latency": [],
        "memory": [],
        "e2e": [],
    }

    for test in report.get("tests", []):
        test_name = test.get("nodeid", "")
        outcome = test.get("outcome", "unknown")

        # Categorize by test file
        if "token_reduction" in test_name:
            metrics["token_reduction"].append({
                "name": test_name.split("::")[-1],
                "passed": outcome == "passed",
                "duration": test.get("duration", 0),
            })
        elif "latency" in test_name:
            metrics["latency"].append({
                "name": test_name.split("::")[-1],
                "passed": outcome == "passed",
                "duration": test.get("duration", 0),
            })
        elif "memory" in test_name:
            metrics["memory"].append({
                "name": test_name.split("::")[-1],
                "passed": outcome == "passed",
                "duration": test.get("duration", 0),
            })
        elif "e2e" in test_name:
            metrics["e2e"].append({
                "name": test_name.split("::")[-1],
                "passed": outcome == "passed",
                "duration": test.get("duration", 0),
            })

    return metrics


def generate_markdown_report(
    metrics: Dict[str, Any],
    summary: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate markdown benchmark report.

    Args:
        metrics: Benchmark metrics by category
        summary: Test summary statistics
        output_path: Path to write markdown file
    """
    lines = [
        "# Token Optimization Benchmark Results",
        "",
        f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "## Summary",
        "",
        f"- **Total Tests**: {summary.get('total', 0)}",
        f"- **Passed**: {summary.get('passed', 0)}",
        f"- **Failed**: {summary.get('failed', 0)}",
        f"- **Success Rate**: {summary.get('success_rate', 0):.1f}%",
        "",
        "## Performance Targets",
        "",
        "| Target | Status | Notes |",
        "|--------|--------|-------|",
    ]

    # Token reduction targets
    token_passed = all(t["passed"] for t in metrics["token_reduction"]) if metrics["token_reduction"] else True
    lines.append(f"| Token Reduction (70-95%) | {'PASS' if token_passed else 'FAIL'} | ARIA snapshot vs raw HTML |")

    # Latency targets
    latency_passed = all(t["passed"] for t in metrics["latency"]) if metrics["latency"] else True
    lines.append(f"| Latency (<200ms snapshot) | {'PASS' if latency_passed else 'FAIL'} | Snapshot generation time |")

    # Memory targets
    memory_passed = all(t["passed"] for t in metrics["memory"]) if metrics["memory"] else True
    lines.append(f"| Memory (<200 bytes/ref) | {'PASS' if memory_passed else 'FAIL'} | Ref registry memory usage |")

    # E2E targets
    e2e_passed = all(t["passed"] for t in metrics["e2e"]) if metrics["e2e"] else True
    lines.append(f"| E2E Pipeline (<300ms) | {'PASS' if e2e_passed else 'FAIL'} | Full optimization pipeline |")

    lines.extend([
        "",
        "## Detailed Results",
        "",
        "### Token Reduction Benchmarks",
        "",
        "| Test | Status | Duration |",
        "|------|--------|----------|",
    ])

    for test in metrics["token_reduction"]:
        status = "PASS" if test["passed"] else "FAIL"
        lines.append(f"| {test['name']} | {status} | {test['duration']:.3f}s |")

    lines.extend([
        "",
        "### Latency Benchmarks",
        "",
        "| Test | Status | Duration |",
        "|------|--------|----------|",
    ])

    for test in metrics["latency"]:
        status = "PASS" if test["passed"] else "FAIL"
        lines.append(f"| {test['name']} | {status} | {test['duration']:.3f}s |")

    lines.extend([
        "",
        "### Memory Benchmarks",
        "",
        "| Test | Status | Duration |",
        "|------|--------|----------|",
    ])

    for test in metrics["memory"]:
        status = "PASS" if test["passed"] else "FAIL"
        lines.append(f"| {test['name']} | {status} | {test['duration']:.3f}s |")

    if metrics["e2e"]:
        lines.extend([
            "",
            "### End-to-End Benchmarks",
            "",
            "| Test | Status | Duration |",
            "|------|--------|----------|",
        ])

        for test in metrics["e2e"]:
            status = "PASS" if test["passed"] else "FAIL"
            lines.append(f"| {test['name']} | {status} | {test['duration']:.3f}s |")

    lines.extend([
        "",
        "## ADR Performance Targets Reference",
        "",
        "From `docs/improve_token_consumption_and_timeout.md`:",
        "",
        "| Feature | Target | Priority |",
        "|---------|--------|----------|",
        "| Aria snapshot tool | 70-90% reduction | P0 |",
        "| Ref-based targeting | 20-30% reduction | P0 |",
        "| Response filtering | 30-50% reduction | P1 |",
        "| Dual timeout config | Fast failure detection | P1 |",
        "| Pre-validation tool | Reduces wasted waits | P1 |",
        "| Incremental snapshots | 40-60% additional | P2 |",
        "| List folding | Up to 90% for list-heavy | P3 |",
        "",
        "---",
        "",
        "*Report generated by `scripts/run_benchmarks.py`*",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"\nMarkdown report written to: {output_path}")


def generate_json_report(
    metrics: Dict[str, Any],
    summary: Dict[str, Any],
    output_path: Path,
) -> None:
    """Generate JSON benchmark report for CI integration.

    Args:
        metrics: Benchmark metrics by category
        summary: Test summary statistics
        output_path: Path to write JSON file
    """
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": summary,
        "metrics": metrics,
        "targets": {
            "token_reduction": {"min": 70, "max": 95, "unit": "percent"},
            "snapshot_latency": {"max": 200, "unit": "ms"},
            "ref_lookup_latency": {"max": 1, "unit": "ms"},
            "diff_latency": {"max": 50, "unit": "ms"},
            "prevalidation_latency": {"max": 100, "unit": "ms"},
            "memory_per_ref": {"max": 200, "unit": "bytes"},
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"JSON report written to: {output_path}")


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for failures, 2 for errors)
    """
    args = parse_args()

    print("=" * 70)
    print("TOKEN OPTIMIZATION BENCHMARK SUITE")
    print("=" * 70)
    print()

    try:
        # Check if pytest-json-report is available
        try:
            import pytest_jsonreport  # noqa: F401
        except ImportError:
            print("Note: pytest-json-report not installed, using basic output")
            print("Install with: pip install pytest-json-report")
            print()

        # Run benchmarks
        results = run_pytest_benchmarks(
            markers=args.markers,
            quick=args.quick,
            verbose=args.verbose,
        )

        # Calculate summary
        report = results.get("report", {})
        summary_data = report.get("summary", {})
        total = summary_data.get("total", 0)
        passed = summary_data.get("passed", 0)
        failed = summary_data.get("failed", 0)

        if total == 0:
            # Fallback: count from tests list
            tests = report.get("tests", [])
            total = len(tests)
            passed = sum(1 for t in tests if t.get("outcome") == "passed")
            failed = total - passed

        summary = {
            "total": total,
            "passed": passed,
            "failed": failed,
            "success_rate": (passed / total * 100) if total > 0 else 100.0,
        }

        # Collect metrics
        metrics = collect_benchmark_metrics(report)

        # Print summary
        print()
        print("=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"Total:   {summary['total']}")
        print(f"Passed:  {summary['passed']}")
        print(f"Failed:  {summary['failed']}")
        print(f"Rate:    {summary['success_rate']:.1f}%")
        print()

        # Generate reports
        if not args.json_only:
            generate_markdown_report(
                metrics,
                summary,
                args.output_dir / "benchmark_results.md",
            )

        generate_json_report(
            metrics,
            summary,
            args.output_dir / "benchmark_results.json",
        )

        # Return appropriate exit code
        if results["exit_code"] != 0 or failed > 0:
            print("\nSome benchmarks FAILED. Review output above.")
            return 1

        print("\nAll benchmarks PASSED!")
        return 0

    except Exception as e:
        print(f"\nError running benchmarks: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
