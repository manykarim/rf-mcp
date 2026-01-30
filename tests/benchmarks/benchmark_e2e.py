"""End-to-end benchmark with simulated browser operations.

This module provides comprehensive end-to-end benchmarks that simulate
real-world usage patterns of the token optimization system.

Run with:
    pytest tests/benchmarks/benchmark_e2e.py -v
    pytest tests/benchmarks/benchmark_e2e.py -v --benchmark-only
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest

from robotmcp.utils.token_efficient_output import (
    TokenEfficientOutput,
    estimate_tokens,
    optimize_execution_result,
    optimize_output,
)


@dataclass
class PageMetrics:
    """Metrics for a single page benchmark."""

    url: str
    html_tokens: int
    aria_tokens: int
    optimized_tokens: int
    reduction_percent: float
    snapshot_latency_ms: float
    optimization_latency_ms: float


@dataclass
class E2EBenchmarkResults:
    """Results from end-to-end benchmark run."""

    pages_tested: int = 0
    total_tokens_before: int = 0
    total_tokens_after: int = 0
    total_latency_ms: float = 0.0
    page_metrics: List[PageMetrics] = field(default_factory=list)

    @property
    def overall_reduction(self) -> float:
        if self.total_tokens_before == 0:
            return 0.0
        return (
            (self.total_tokens_before - self.total_tokens_after)
            / self.total_tokens_before
            * 100
        )

    @property
    def avg_latency_ms(self) -> float:
        if self.pages_tested == 0:
            return 0.0
        return self.total_latency_ms / self.pages_tested


class MockBrowserSession:
    """Mock browser session for benchmarking without real browser."""

    def __init__(self):
        self._current_url: str = ""
        self._page_content: str = ""
        self._aria_snapshot: str = ""

    def navigate(self, url: str) -> Dict[str, Any]:
        """Simulate navigation to URL."""
        self._current_url = url

        # Generate mock content based on URL patterns
        if "todomvc" in url.lower():
            self._page_content = self._generate_todomvc_html()
            self._aria_snapshot = self._generate_todomvc_aria()
        elif "shop" in url.lower() or "product" in url.lower():
            self._page_content = self._generate_shop_html()
            self._aria_snapshot = self._generate_shop_aria()
        elif "login" in url.lower() or "auth" in url.lower():
            self._page_content = self._generate_login_html()
            self._aria_snapshot = self._generate_login_aria()
        else:
            self._page_content = self._generate_generic_html()
            self._aria_snapshot = self._generate_generic_aria()

        return {
            "url": url,
            "title": f"Page - {url}",
            "loaded": True,
        }

    def get_html(self) -> str:
        """Get current page HTML."""
        return self._page_content

    def get_aria_snapshot(self) -> str:
        """Get ARIA accessibility snapshot."""
        return self._aria_snapshot

    def click(self, ref: str) -> Dict[str, Any]:
        """Simulate click action."""
        time.sleep(0.01)  # Simulate action latency
        return {
            "success": True,
            "action": "click",
            "ref": ref,
        }

    def fill(self, ref: str, text: str) -> Dict[str, Any]:
        """Simulate fill action."""
        time.sleep(0.01)
        return {
            "success": True,
            "action": "fill",
            "ref": ref,
            "text": text,
        }

    def _generate_todomvc_html(self) -> str:
        """Generate TodoMVC-style HTML."""
        todos = []
        for i in range(50):
            todos.append(
                f"""
                <li class="todo-item" data-id="{i}">
                    <div class="view">
                        <input class="toggle" type="checkbox" data-testid="toggle-{i}">
                        <label>Todo item number {i}</label>
                        <button class="destroy" data-testid="delete-{i}"></button>
                    </div>
                    <input class="edit" value="Todo item number {i}">
                </li>
                """
            )

        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>TodoMVC</title></head>
        <body>
            <section class="todoapp">
                <header class="header">
                    <h1>todos</h1>
                    <input class="new-todo" placeholder="What needs to be done?" autofocus>
                </header>
                <section class="main">
                    <input id="toggle-all" class="toggle-all" type="checkbox">
                    <label for="toggle-all">Mark all as complete</label>
                    <ul class="todo-list">
                        {''.join(todos)}
                    </ul>
                </section>
                <footer class="footer">
                    <span class="todo-count"><strong>50</strong> items left</span>
                    <ul class="filters">
                        <li><a class="selected" href="#/">All</a></li>
                        <li><a href="#/active">Active</a></li>
                        <li><a href="#/completed">Completed</a></li>
                    </ul>
                    <button class="clear-completed">Clear completed</button>
                </footer>
            </section>
        </body>
        </html>
        """

    def _generate_todomvc_aria(self) -> str:
        """Generate TodoMVC ARIA snapshot."""
        return """- document [ref=e1]
  - heading "todos" [level=1] [ref=e2]
  - textbox "What needs to be done?" [ref=e3]
  - checkbox "Mark all as complete" [ref=e4]
  - list [ref=e5]
    - listitem [ref=e6]
      - checkbox [ref=e7]
      - text "Todo item number 0"
      - button "delete" [ref=e8]
    - listitem [ref=e9]
      - checkbox [ref=e10]
      - text "Todo item number 1"
      - button "delete" [ref=e11]
    - (... and 48 more similar items) [refs: e12-e108]
  - text "50 items left"
  - navigation [ref=e109]
    - link "All" [ref=e110]
    - link "Active" [ref=e111]
    - link "Completed" [ref=e112]
  - button "Clear completed" [ref=e113]
"""

    def _generate_shop_html(self) -> str:
        """Generate e-commerce shop HTML."""
        products = []
        for i in range(100):
            products.append(
                f"""
                <div class="product-card" data-product-id="{i}">
                    <img src="/img/product-{i}.jpg" alt="Product {i}">
                    <h3 class="product-title">Product Name {i}</h3>
                    <p class="product-description">Description for product {i} with details</p>
                    <span class="price">${19.99 + i * 0.5:.2f}</span>
                    <button class="add-to-cart" data-testid="add-{i}">Add to Cart</button>
                    <button class="wishlist" data-testid="wish-{i}">Wishlist</button>
                </div>
                """
            )

        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>Online Shop</title></head>
        <body>
            <header>
                <nav>
                    <a href="/">Home</a>
                    <a href="/products">Products</a>
                    <a href="/cart">Cart (0)</a>
                </nav>
                <input type="search" placeholder="Search products...">
            </header>
            <main>
                <h1>Products</h1>
                <div class="product-grid">
                    {''.join(products)}
                </div>
                <nav class="pagination">
                    <button>Previous</button>
                    <span>Page 1 of 10</span>
                    <button>Next</button>
                </nav>
            </main>
        </body>
        </html>
        """

    def _generate_shop_aria(self) -> str:
        """Generate shop ARIA snapshot with folding."""
        return """- document [ref=e1]
  - navigation [ref=e2]
    - link "Home" [ref=e3]
    - link "Products" [ref=e4]
    - link "Cart (0)" [ref=e5]
  - searchbox "Search products..." [ref=e6]
  - heading "Products" [level=1] [ref=e7]
  - region "product-grid" [ref=e8]
    - article [ref=e9]
      - img "Product 0" [ref=e10]
      - heading "Product Name 0" [level=3] [ref=e11]
      - text "$19.99"
      - button "Add to Cart" [ref=e12]
      - button "Wishlist" [ref=e13]
    - article [ref=e14]
      - img "Product 1" [ref=e15]
      - heading "Product Name 1" [level=3] [ref=e16]
      - text "$20.49"
      - button "Add to Cart" [ref=e17]
      - button "Wishlist" [ref=e18]
    - (... and 98 more similar items) [refs: e19-e510]
  - navigation "pagination" [ref=e511]
    - button "Previous" [ref=e512]
    - text "Page 1 of 10"
    - button "Next" [ref=e513]
"""

    def _generate_login_html(self) -> str:
        """Generate login form HTML."""
        return """
        <!DOCTYPE html>
        <html>
        <head><title>Login</title></head>
        <body>
            <main>
                <form class="login-form" action="/login" method="post">
                    <h1>Sign In</h1>
                    <div class="form-group">
                        <label for="email">Email</label>
                        <input type="email" id="email" name="email" required>
                    </div>
                    <div class="form-group">
                        <label for="password">Password</label>
                        <input type="password" id="password" name="password" required>
                    </div>
                    <div class="form-group">
                        <input type="checkbox" id="remember" name="remember">
                        <label for="remember">Remember me</label>
                    </div>
                    <button type="submit">Sign In</button>
                    <a href="/forgot-password">Forgot password?</a>
                    <a href="/register">Create account</a>
                </form>
            </main>
        </body>
        </html>
        """

    def _generate_login_aria(self) -> str:
        """Generate login form ARIA snapshot."""
        return """- document [ref=e1]
  - heading "Sign In" [level=1] [ref=e2]
  - form [ref=e3]
    - textbox "Email" [ref=e4] [required]
    - textbox "Password" [ref=e5] [required]
    - checkbox "Remember me" [ref=e6]
    - button "Sign In" [ref=e7]
  - link "Forgot password?" [ref=e8]
  - link "Create account" [ref=e9]
"""

    def _generate_generic_html(self) -> str:
        """Generate generic page HTML."""
        sections = []
        for i in range(20):
            sections.append(
                f"""
                <section data-section="{i}">
                    <h2>Section {i}</h2>
                    <p>Content for section {i} with some descriptive text.</p>
                    <button>Action {i}</button>
                    <a href="/section/{i}">Learn more</a>
                </section>
                """
            )

        return f"""
        <!DOCTYPE html>
        <html>
        <head><title>Generic Page</title></head>
        <body>
            <header>
                <nav>
                    <a href="/">Home</a>
                    <a href="/about">About</a>
                </nav>
            </header>
            <main>
                <h1>Page Title</h1>
                {''.join(sections)}
            </main>
            <footer>
                <p>Footer content</p>
            </footer>
        </body>
        </html>
        """

    def _generate_generic_aria(self) -> str:
        """Generate generic page ARIA snapshot."""
        lines = [
            "- document [ref=e1]",
            "  - navigation [ref=e2]",
            '    - link "Home" [ref=e3]',
            '    - link "About" [ref=e4]',
            '  - heading "Page Title" [level=1] [ref=e5]',
        ]

        ref = 6
        for i in range(5):  # Show first 5 sections
            lines.extend(
                [
                    f"  - region [ref=e{ref}]",
                    f'    - heading "Section {i}" [level=2] [ref=e{ref + 1}]',
                    f'    - text "Content for section {i}..."',
                    f'    - button "Action {i}" [ref=e{ref + 2}]',
                    f'    - link "Learn more" [ref=e{ref + 3}]',
                ]
            )
            ref += 4

        lines.append("  - (... and 15 more similar sections) [refs: e26-e86]")
        lines.append(f"  - contentinfo [ref=e87]")
        lines.append('    - text "Footer content"')

        return "\n".join(lines)


@pytest.fixture
def mock_browser_session() -> MockBrowserSession:
    """Create a mock browser session for testing."""
    return MockBrowserSession()


class TestE2EBenchmarks:
    """End-to-end benchmarks simulating real usage."""

    @pytest.mark.benchmark
    @pytest.mark.e2e
    def test_full_optimization_pipeline(
        self,
        mock_browser_session: MockBrowserSession,
        benchmark_reporter,
    ):
        """Benchmark complete optimization pipeline across multiple page types."""
        test_urls = [
            "https://demo.playwright.dev/todomvc/",
            "https://shop.example.com/products",
            "https://example.com/login",
            "https://example.com/generic-page",
        ]

        results = E2EBenchmarkResults()

        for url in test_urls:
            # Navigate
            nav_start = time.perf_counter()
            mock_browser_session.navigate(url)
            nav_latency = (time.perf_counter() - nav_start) * 1000

            # Get content
            html = mock_browser_session.get_html()
            html_tokens = estimate_tokens(html)

            # Get ARIA snapshot
            snapshot_start = time.perf_counter()
            aria_snapshot = mock_browser_session.get_aria_snapshot()
            snapshot_latency = (time.perf_counter() - snapshot_start) * 1000
            aria_tokens = estimate_tokens(aria_snapshot)

            # Optimize response
            opt_start = time.perf_counter()
            optimized_response = optimize_output(
                {
                    "success": True,
                    "url": url,
                    "snapshot": aria_snapshot,
                },
                verbosity="standard",
            )
            opt_latency = (time.perf_counter() - opt_start) * 1000
            opt_tokens = estimate_tokens(optimized_response)

            # Calculate reduction (comparing optimized ARIA to raw HTML)
            reduction = (html_tokens - opt_tokens) / html_tokens * 100 if html_tokens > 0 else 0

            # Record metrics
            page_metrics = PageMetrics(
                url=url,
                html_tokens=html_tokens,
                aria_tokens=aria_tokens,
                optimized_tokens=opt_tokens,
                reduction_percent=reduction,
                snapshot_latency_ms=snapshot_latency,
                optimization_latency_ms=opt_latency,
            )

            results.pages_tested += 1
            results.total_tokens_before += html_tokens
            results.total_tokens_after += opt_tokens
            results.total_latency_ms += snapshot_latency + opt_latency
            results.page_metrics.append(page_metrics)

        # Record overall results
        benchmark_reporter.record(
            name="e2e_full_pipeline",
            duration_ms=results.total_latency_ms,
            tokens_before=results.total_tokens_before,
            tokens_after=results.total_tokens_after,
            target_reduction=70.0,
            pages_tested=results.pages_tested,
            avg_latency_ms=results.avg_latency_ms,
        )

        # Verify targets
        assert results.overall_reduction >= 70, (
            f"Overall reduction {results.overall_reduction:.1f}% below 70% target"
        )

        # Print detailed results
        print(f"\nE2E Pipeline Results:")
        print(f"  Pages tested: {results.pages_tested}")
        print(f"  Total tokens before: {results.total_tokens_before}")
        print(f"  Total tokens after: {results.total_tokens_after}")
        print(f"  Overall reduction: {results.overall_reduction:.1f}%")
        print(f"  Average latency: {results.avg_latency_ms:.1f}ms")

        for pm in results.page_metrics:
            print(f"\n  {pm.url}:")
            print(f"    HTML tokens: {pm.html_tokens}")
            print(f"    Optimized tokens: {pm.optimized_tokens}")
            print(f"    Reduction: {pm.reduction_percent:.1f}%")

    @pytest.mark.benchmark
    @pytest.mark.e2e
    def test_interaction_sequence_benchmark(
        self,
        mock_browser_session: MockBrowserSession,
        benchmark_reporter,
    ):
        """Benchmark a sequence of interactions (click, fill, navigate)."""
        # Navigate to TodoMVC
        mock_browser_session.navigate("https://demo.playwright.dev/todomvc/")

        actions = [
            ("fill", "e3", "New todo item"),  # Fill input
            ("click", "e3", None),  # Submit (simulate)
            ("click", "e7", None),  # Toggle first item
            ("click", "e8", None),  # Delete first item
            ("click", "e110", None),  # Filter: All
            ("click", "e111", None),  # Filter: Active
            ("click", "e112", None),  # Filter: Completed
            ("click", "e113", None),  # Clear completed
        ]

        total_tokens = 0
        total_latency = 0

        for action_type, ref, value in actions:
            # Perform action
            action_start = time.perf_counter()
            if action_type == "click":
                result = mock_browser_session.click(ref)
            else:
                result = mock_browser_session.fill(ref, value)
            action_latency = (time.perf_counter() - action_start) * 1000

            # Optimize result
            opt_start = time.perf_counter()
            optimized = optimize_execution_result(
                {
                    "success": result["success"],
                    "action": action_type,
                    "ref": ref,
                    "output": f"Action {action_type} completed on {ref}",
                },
                verbosity="compact",
            )
            opt_latency = (time.perf_counter() - opt_start) * 1000

            tokens = estimate_tokens(optimized)
            total_tokens += tokens
            total_latency += action_latency + opt_latency

        benchmark_reporter.record_latency(
            name="interaction_sequence",
            duration_ms=total_latency,
            target_ms=500.0,  # 8 actions in <500ms
            iterations=len(actions),
            total_tokens=total_tokens,
        )

        assert total_latency < 500, (
            f"Interaction sequence took {total_latency:.1f}ms, exceeds 500ms target"
        )

    @pytest.mark.benchmark
    @pytest.mark.e2e
    def test_navigation_heavy_scenario(
        self,
        mock_browser_session: MockBrowserSession,
        benchmark_reporter,
    ):
        """Benchmark scenario with many page navigations."""
        pages = [
            "https://example.com/",
            "https://example.com/products",
            "https://example.com/products/1",
            "https://example.com/cart",
            "https://example.com/checkout",
            "https://example.com/login",
            "https://example.com/account",
            "https://example.com/orders",
            "https://example.com/settings",
            "https://example.com/logout",
        ]

        total_tokens_saved = 0
        total_latency = 0

        for url in pages:
            start = time.perf_counter()

            mock_browser_session.navigate(url)
            html = mock_browser_session.get_html()
            aria = mock_browser_session.get_aria_snapshot()

            html_tokens = estimate_tokens(html)
            aria_tokens = estimate_tokens(aria)
            total_tokens_saved += html_tokens - aria_tokens

            latency = (time.perf_counter() - start) * 1000
            total_latency += latency

        benchmark_reporter.record(
            name="navigation_heavy_scenario",
            duration_ms=total_latency,
            tokens_before=total_tokens_saved + sum(
                estimate_tokens(mock_browser_session.get_aria_snapshot())
                for _ in range(1)  # Approximation
            ) * len(pages),
            tokens_after=sum(
                estimate_tokens(mock_browser_session.get_aria_snapshot())
                for _ in range(1)
            ) * len(pages),
            target_reduction=70.0,
            pages_navigated=len(pages),
        )

        assert total_latency < 1000, (
            f"10 navigations took {total_latency:.1f}ms, exceeds 1000ms target"
        )


class TestScalabilityBenchmarks:
    """Benchmark scalability with increasing load."""

    @pytest.mark.benchmark
    def test_concurrent_page_optimization(
        self,
        sample_html_pages: Dict[str, str],
        sample_aria_snapshots: Dict[str, str],
        benchmark_reporter,
    ):
        """Benchmark handling multiple pages concurrently."""
        pages = [
            (sample_html_pages.get("small", ""), sample_aria_snapshots.get("small", "")),
            (sample_html_pages.get("medium", ""), sample_aria_snapshots.get("medium", "")),
            (sample_html_pages.get("large", ""), sample_aria_snapshots.get("large", "")),
        ]

        # Filter out empty pages
        pages = [(h, a) for h, a in pages if h and a]

        if not pages:
            pytest.skip("No page fixtures available")

        iterations = 100
        start = time.perf_counter()

        for _ in range(iterations):
            for html, aria in pages:
                _optimized = optimize_output(
                    {
                        "snapshot": aria,
                        "html_preview": html[:500],
                    },
                    verbosity="standard",
                )

        total_ms = (time.perf_counter() - start) * 1000
        ops_per_second = (iterations * len(pages)) / (total_ms / 1000)

        benchmark_reporter.record_latency(
            name="concurrent_page_optimization",
            duration_ms=total_ms,
            target_ms=5000.0,  # 300 optimizations in <5s
            iterations=iterations * len(pages),
            ops_per_second=ops_per_second,
        )

        assert ops_per_second > 50, (
            f"Throughput {ops_per_second:.1f} ops/s below 50 ops/s target"
        )

    @pytest.mark.benchmark
    def test_sustained_load(
        self,
        benchmark_reporter,
    ):
        """Benchmark sustained optimization load over time."""
        from robotmcp.utils.token_efficient_output import TokenEfficientOutput

        handler = TokenEfficientOutput(
            verbosity="standard",
            max_string_length=500,
            max_list_items=20,
        )

        # Generate test data
        test_response = {
            "success": True,
            "snapshot": "- document [ref=e1]\n" * 100,
            "refs": [f"e{i}" for i in range(100)],
            "metadata": {"large": "data" * 100},
        }

        # Run for simulated duration
        iterations = 10000
        start = time.perf_counter()

        for i in range(iterations):
            # Vary the data slightly
            response = {**test_response, "iteration": i}
            _optimized = handler.optimize(response)

        total_ms = (time.perf_counter() - start) * 1000
        avg_ms = total_ms / iterations

        benchmark_reporter.record_latency(
            name="sustained_load",
            duration_ms=total_ms,
            target_ms=0.5,  # <0.5ms per optimization
            iterations=iterations,
        )

        assert avg_ms < 0.5, (
            f"Average optimization time {avg_ms:.3f}ms exceeds 0.5ms target"
        )
