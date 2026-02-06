"""Benchmark fixtures for token optimization testing.

This module provides fixtures used across all benchmark tests including:
- Sample HTML pages of various sizes
- Mock ARIA snapshot data
- Element registry fixtures
- Benchmark result collection and reporting
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


# ============================================================================
# Pytest configuration for benchmarks
# ============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers for benchmark tests."""
    config.addinivalue_line(
        "markers",
        "benchmark: mark test as a performance benchmark",
    )
    config.addinivalue_line(
        "markers",
        "e2e: mark test as an end-to-end benchmark requiring browser simulation",
    )


# ============================================================================
# Data classes for benchmark results
# ============================================================================


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    name: str
    duration_ms: float
    tokens_before: int
    tokens_after: int
    reduction_percent: float
    target_met: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyResult:
    """Result of a latency benchmark."""

    name: str
    duration_ms: float
    target_ms: float
    iterations: int
    avg_per_operation_ms: float
    target_met: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryResult:
    """Result of a memory benchmark."""

    name: str
    memory_bytes: int
    items_count: int
    bytes_per_item: float
    target_bytes_per_item: int
    target_met: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Benchmark Reporter
# ============================================================================


class BenchmarkReporter:
    """Collects and reports benchmark results."""

    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.latency_results: List[LatencyResult] = []
        self.memory_results: List[MemoryResult] = []
        self.start_time: Optional[float] = None

    def start(self):
        """Start the benchmark session."""
        self.start_time = time.perf_counter()

    def record(
        self,
        name: str,
        duration_ms: float,
        tokens_before: int,
        tokens_after: int,
        target_reduction: float = 70.0,
        **metadata,
    ) -> BenchmarkResult:
        """Record a token reduction benchmark result.

        Args:
            name: Name of the benchmark
            duration_ms: Duration in milliseconds
            tokens_before: Token count before optimization
            tokens_after: Token count after optimization
            target_reduction: Target reduction percentage (default 70%)
            **metadata: Additional metadata to record

        Returns:
            BenchmarkResult object
        """
        reduction_percent = (
            (tokens_before - tokens_after) / tokens_before * 100
            if tokens_before > 0
            else 0
        )
        target_met = reduction_percent >= target_reduction

        result = BenchmarkResult(
            name=name,
            duration_ms=duration_ms,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            reduction_percent=reduction_percent,
            target_met=target_met,
            metadata=metadata,
        )
        self.results.append(result)
        return result

    def record_latency(
        self,
        name: str,
        duration_ms: float,
        target_ms: float,
        iterations: int = 1,
        **metadata,
    ) -> LatencyResult:
        """Record a latency benchmark result.

        Args:
            name: Name of the benchmark
            duration_ms: Total duration in milliseconds
            target_ms: Target latency in milliseconds
            iterations: Number of iterations performed
            **metadata: Additional metadata

        Returns:
            LatencyResult object
        """
        avg_per_operation = duration_ms / iterations if iterations > 0 else duration_ms
        target_met = avg_per_operation <= target_ms

        result = LatencyResult(
            name=name,
            duration_ms=duration_ms,
            target_ms=target_ms,
            iterations=iterations,
            avg_per_operation_ms=avg_per_operation,
            target_met=target_met,
            metadata=metadata,
        )
        self.latency_results.append(result)
        return result

    def record_memory(
        self,
        name: str,
        memory_bytes: int,
        items_count: int,
        target_bytes_per_item: int,
        **metadata,
    ) -> MemoryResult:
        """Record a memory benchmark result.

        Args:
            name: Name of the benchmark
            memory_bytes: Total memory used in bytes
            items_count: Number of items stored
            target_bytes_per_item: Target memory per item in bytes
            **metadata: Additional metadata

        Returns:
            MemoryResult object
        """
        bytes_per_item = memory_bytes / items_count if items_count > 0 else 0
        target_met = bytes_per_item <= target_bytes_per_item

        result = MemoryResult(
            name=name,
            memory_bytes=memory_bytes,
            items_count=items_count,
            bytes_per_item=bytes_per_item,
            target_bytes_per_item=target_bytes_per_item,
            target_met=target_met,
            metadata=metadata,
        )
        self.memory_results.append(result)
        return result

    def report(self) -> Dict[str, Any]:
        """Generate a summary report of all benchmark results.

        Returns:
            Dictionary containing the benchmark report
        """
        total_duration = (
            time.perf_counter() - self.start_time if self.start_time else 0
        )

        token_results = [
            {
                "name": r.name,
                "duration_ms": round(r.duration_ms, 2),
                "tokens_before": r.tokens_before,
                "tokens_after": r.tokens_after,
                "reduction_percent": round(r.reduction_percent, 1),
                "target_met": r.target_met,
            }
            for r in self.results
        ]

        latency_results = [
            {
                "name": r.name,
                "avg_ms": round(r.avg_per_operation_ms, 4),
                "target_ms": r.target_ms,
                "iterations": r.iterations,
                "target_met": r.target_met,
            }
            for r in self.latency_results
        ]

        memory_results = [
            {
                "name": r.name,
                "bytes_per_item": round(r.bytes_per_item, 1),
                "target_bytes": r.target_bytes_per_item,
                "items_count": r.items_count,
                "target_met": r.target_met,
            }
            for r in self.memory_results
        ]

        targets_met = (
            sum(1 for r in self.results if r.target_met)
            + sum(1 for r in self.latency_results if r.target_met)
            + sum(1 for r in self.memory_results if r.target_met)
        )
        total_targets = (
            len(self.results) + len(self.latency_results) + len(self.memory_results)
        )

        return {
            "summary": {
                "total_benchmarks": total_targets,
                "targets_met": targets_met,
                "targets_missed": total_targets - targets_met,
                "success_rate": (
                    round(targets_met / total_targets * 100, 1)
                    if total_targets > 0
                    else 100.0
                ),
                "total_duration_seconds": round(total_duration, 2),
            },
            "token_reduction": token_results,
            "latency": latency_results,
            "memory": memory_results,
        }

    def print_report(self):
        """Print a formatted benchmark report to stdout."""
        report = self.report()

        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)

        # Summary
        summary = report["summary"]
        print(f"\nTotal benchmarks: {summary['total_benchmarks']}")
        print(f"Targets met: {summary['targets_met']}/{summary['total_benchmarks']}")
        print(f"Success rate: {summary['success_rate']}%")
        print(f"Total duration: {summary['total_duration_seconds']}s")

        # Token reduction results
        if report["token_reduction"]:
            print("\n--- Token Reduction Benchmarks ---")
            for r in report["token_reduction"]:
                status = "PASS" if r["target_met"] else "FAIL"
                print(
                    f"  [{status}] {r['name']}: "
                    f"{r['tokens_before']} -> {r['tokens_after']} tokens "
                    f"({r['reduction_percent']}% reduction) in {r['duration_ms']}ms"
                )

        # Latency results
        if report["latency"]:
            print("\n--- Latency Benchmarks ---")
            for r in report["latency"]:
                status = "PASS" if r["target_met"] else "FAIL"
                print(
                    f"  [{status}] {r['name']}: "
                    f"{r['avg_ms']}ms avg (target: <{r['target_ms']}ms) "
                    f"[{r['iterations']} iterations]"
                )

        # Memory results
        if report["memory"]:
            print("\n--- Memory Benchmarks ---")
            for r in report["memory"]:
                status = "PASS" if r["target_met"] else "FAIL"
                print(
                    f"  [{status}] {r['name']}: "
                    f"{r['bytes_per_item']} bytes/item (target: <{r['target_bytes']} bytes) "
                    f"[{r['items_count']} items]"
                )

        print("\n" + "=" * 70)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def benchmark_reporter() -> BenchmarkReporter:
    """Create a benchmark reporter for collecting results."""
    reporter = BenchmarkReporter()
    reporter.start()
    return reporter


@pytest.fixture
def sample_html_pages() -> Dict[str, str]:
    """Load sample HTML pages of various sizes.

    Returns:
        Dictionary mapping page type to HTML content
    """
    fixtures_dir = Path(__file__).parent / "fixtures"

    pages = {}
    for page_type in ["small_page", "medium_page", "large_page", "list_heavy_page"]:
        filepath = fixtures_dir / f"{page_type}.html"
        if filepath.exists():
            pages[page_type.replace("_page", "")] = filepath.read_text()
        else:
            # Generate synthetic pages if fixtures don't exist
            pages[page_type.replace("_page", "")] = _generate_synthetic_html(page_type)

    return pages


@pytest.fixture
def sample_aria_snapshots() -> Dict[str, str]:
    """Load sample ARIA snapshots corresponding to HTML pages.

    Returns:
        Dictionary mapping page type to ARIA snapshot content
    """
    fixtures_dir = Path(__file__).parent / "fixtures"

    snapshots = {}
    for page_type in ["small", "medium", "large", "list_heavy"]:
        filepath = fixtures_dir / f"{page_type}_snapshot.yaml"
        if filepath.exists():
            snapshots[page_type] = filepath.read_text()
        else:
            # Generate synthetic snapshots if fixtures don't exist
            snapshots[page_type] = _generate_synthetic_aria_snapshot(page_type)

    return snapshots


@pytest.fixture
def element_refs() -> List[Dict[str, Any]]:
    """Generate sample element references for ref registry testing.

    Returns:
        List of element reference dictionaries
    """
    refs = []
    for i in range(1000):
        refs.append(
            {
                "ref_id": f"e{i}",
                "locator": f'//div[@data-testid="element-{i}"]',
                "role": "button" if i % 3 == 0 else "link" if i % 3 == 1 else "text",
                "name": f"Element {i}",
                "visible": True,
                "enabled": True,
            }
        )
    return refs


# ============================================================================
# Synthetic data generators
# ============================================================================


def _generate_synthetic_html(page_type: str) -> str:
    """Generate synthetic HTML content for testing.

    Args:
        page_type: Type of page to generate

    Returns:
        HTML content as string
    """
    if page_type == "small_page":
        elements = 100
    elif page_type == "medium_page":
        elements = 500
    elif page_type == "large_page":
        elements = 2000
    elif page_type == "list_heavy_page":
        elements = 1000
        return _generate_list_heavy_html(elements)
    else:
        elements = 100

    return _generate_generic_html(elements)


def _generate_generic_html(element_count: int) -> str:
    """Generate generic HTML with specified number of elements.

    Args:
        element_count: Number of elements to include

    Returns:
        HTML content as string
    """
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        "  <title>Test Page</title>",
        "  <style>",
        "    .container { padding: 20px; }",
        "    .card { border: 1px solid #ccc; margin: 10px; padding: 15px; }",
        "    .btn { padding: 10px 20px; cursor: pointer; }",
        "    .hidden { display: none; }",
        "  </style>",
        "</head>",
        "<body>",
        "  <header>",
        "    <nav role='navigation'>",
        "      <a href='/'>Home</a>",
        "      <a href='/about'>About</a>",
        "      <a href='/contact'>Contact</a>",
        "    </nav>",
        "  </header>",
        "  <main class='container'>",
        "    <h1>Test Page Title</h1>",
    ]

    for i in range(element_count):
        if i % 10 == 0:
            html_parts.append(f"    <section data-testid='section-{i}'>")
            html_parts.append(f"      <h2>Section {i // 10}</h2>")

        html_parts.append(f"      <div class='card' data-testid='card-{i}'>")
        html_parts.append(f"        <h3>Card Title {i}</h3>")
        html_parts.append(
            f"        <p>This is the content for card number {i}. "
            f"It contains some descriptive text.</p>"
        )
        html_parts.append(
            f"        <button class='btn' data-testid='btn-{i}'>Action {i}</button>"
        )
        html_parts.append(f"        <a href='/item/{i}'>Learn more</a>")
        html_parts.append("      </div>")

        if i % 10 == 9:
            html_parts.append("    </section>")

    html_parts.extend(
        [
            "  </main>",
            "  <footer>",
            "    <p>Footer content here</p>",
            "  </footer>",
            "  <script>",
            "    // JavaScript code that should be ignored in ARIA snapshot",
            "    console.log('Page loaded');",
            "    document.addEventListener('DOMContentLoaded', function() {});",
            "  </script>",
            "</body>",
            "</html>",
        ]
    )

    return "\n".join(html_parts)


def _generate_list_heavy_html(item_count: int) -> str:
    """Generate HTML with many list items for folding tests.

    Args:
        item_count: Number of list items to include

    Returns:
        HTML content as string
    """
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "  <meta charset='UTF-8'>",
        "  <title>Product List</title>",
        "</head>",
        "<body>",
        "  <main>",
        "    <h1>Products</h1>",
        "    <ul class='product-list' role='list'>",
    ]

    for i in range(item_count):
        price = 9.99 + (i * 0.50)
        html_parts.append(f"      <li class='product-item' data-testid='product-{i}'>")
        html_parts.append(f"        <img src='/img/product-{i}.jpg' alt='Product {i}'>")
        html_parts.append(f"        <h3 class='product-name'>Product {i}</h3>")
        html_parts.append(f"        <p class='product-description'>Description for product {i}</p>")
        html_parts.append(f"        <span class='product-price'>${price:.2f}</span>")
        html_parts.append(f"        <button data-testid='add-cart-{i}'>Add to Cart</button>")
        html_parts.append("      </li>")

    html_parts.extend(
        [
            "    </ul>",
            "    <nav class='pagination'>",
            "      <button>Previous</button>",
            "      <span>Page 1 of 100</span>",
            "      <button>Next</button>",
            "    </nav>",
            "  </main>",
            "</body>",
            "</html>",
        ]
    )

    return "\n".join(html_parts)


def _generate_synthetic_aria_snapshot(page_type: str) -> str:
    """Generate synthetic ARIA snapshot content.

    Args:
        page_type: Type of page snapshot to generate

    Returns:
        ARIA snapshot as YAML-like string
    """
    if page_type == "small":
        return _generate_aria_for_element_count(100)
    elif page_type == "medium":
        return _generate_aria_for_element_count(500)
    elif page_type == "large":
        return _generate_aria_for_element_count(2000)
    elif page_type == "list_heavy":
        return _generate_list_aria_snapshot(1000)
    else:
        return _generate_aria_for_element_count(100)


def _generate_aria_for_element_count(element_count: int) -> str:
    """Generate ARIA snapshot for a generic page.

    Args:
        element_count: Number of elements to represent

    Returns:
        ARIA snapshot as YAML-like string
    """
    ref_counter = 1
    lines = [
        f"- document [ref=e{ref_counter}]",
    ]
    ref_counter += 1

    lines.append(f'  - heading "Test Page Title" [level=1] [ref=e{ref_counter}]')
    ref_counter += 1

    lines.append(f"  - navigation [ref=e{ref_counter}]")
    ref_counter += 1
    for nav_item in ["Home", "About", "Contact"]:
        lines.append(f'    - link "{nav_item}" [ref=e{ref_counter}]')
        ref_counter += 1

    # Generate simplified representations (much less than raw HTML)
    sections = element_count // 10
    for s in range(min(sections, 20)):  # Cap at 20 sections for readability
        lines.append(f"  - region [ref=e{ref_counter}]")
        ref_counter += 1
        lines.append(f'    - heading "Section {s}" [level=2] [ref=e{ref_counter}]')
        ref_counter += 1

        # Just a few representative items per section
        for i in range(3):
            lines.append(f'    - button "Action {s * 10 + i}" [ref=e{ref_counter}]')
            ref_counter += 1
            lines.append(f'    - link "Learn more" [ref=e{ref_counter}]')
            ref_counter += 1

        if s < sections - 1:
            lines.append(f"    - text: (... and {10 - 3} more items)")

    if sections > 20:
        lines.append(f"  - text: (... and {sections - 20} more sections)")

    return "\n".join(lines)


def _generate_list_aria_snapshot(item_count: int) -> str:
    """Generate ARIA snapshot for a list-heavy page with folding.

    Args:
        item_count: Number of list items

    Returns:
        ARIA snapshot as YAML-like string with folded representation
    """
    lines = [
        "- document [ref=e1]",
        '  - heading "Products" [level=1] [ref=e2]',
        "  - list [ref=e3]",
        '    - listitem [ref=e4]: "Product 0" - $9.99',
        '      - button "Add to Cart" [ref=e5]',
        '    - listitem [ref=e6]: "Product 1" - $10.49',
        '      - button "Add to Cart" [ref=e7]',
        '    - listitem [ref=e8]: "Product 2" - $10.99',
        '      - button "Add to Cart" [ref=e9]',
        f"    - (... and {item_count - 3} more similar items) [refs: e10-e{item_count + 9}]",
        "  - navigation [ref=e" + str(item_count + 10) + "]",
        '    - button "Previous" [ref=e' + str(item_count + 11) + "]",
        '    - text "Page 1 of 100"',
        '    - button "Next" [ref=e' + str(item_count + 12) + "]",
    ]
    return "\n".join(lines)


# ============================================================================
# Utility fixtures
# ============================================================================


@pytest.fixture
def estimate_tokens():
    """Fixture providing token estimation function.

    Returns:
        Function that estimates tokens from content
    """

    def _estimate(content: str) -> int:
        """Estimate tokens using character count / 4 heuristic."""
        return len(content) // 4

    return _estimate


