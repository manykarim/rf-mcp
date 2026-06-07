"""Latency benchmarks for the PlatynUI new-core integration (ADR-025).

Validates that the desktop-session checks added to per-keyword hot paths
stay negligible for non-desktop (web/API) sessions:

- ensure_x11_session_env() no-op path: < 50µs median
- ExecutionSession.is_desktop_session(): < 20µs median
- Intent resolution for PlatynUI mappings: < 200µs median
- Timeout classification of PlatynUI keywords: < 20µs median

Run with:
    uv run pytest tests/benchmarks/test_platynui_newcore_benchmarks.py -v
"""

import time
from statistics import median

import pytest


def _median_us(fn, iterations: int = 2000) -> float:
    """Median wall-clock latency of fn() in microseconds."""
    fn()  # warmup
    samples = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)
    return median(samples)


class TestEnvShimLatency:
    @pytest.mark.benchmark
    def test_noop_path_on_x11_env(self):
        """Already-x11 environments must short-circuit fast (< 50µs)."""
        from robotmcp.plugins.builtin.platynui_plugin import ensure_x11_session_env

        env = {"XDG_SESSION_TYPE": "x11", "DISPLAY": ":0"}
        latency = _median_us(lambda: ensure_x11_session_env(env))
        assert latency < 50, f"env shim no-op path too slow: {latency:.1f}µs"

    @pytest.mark.benchmark
    def test_optout_path(self):
        """Opt-out check must short-circuit fast (< 50µs)."""
        from robotmcp.plugins.builtin.platynui_plugin import ensure_x11_session_env

        env = {
            "XDG_SESSION_TYPE": "wayland",
            "DISPLAY": ":0",
            "ROBOTMCP_PLATYNUI_KEEP_WAYLAND": "1",
        }
        latency = _median_us(lambda: ensure_x11_session_env(env))
        assert latency < 50, f"env shim opt-out path too slow: {latency:.1f}µs"


class TestDesktopSessionCheckLatency:
    @pytest.mark.benchmark
    def test_is_desktop_session_web_session(self):
        """Per-keyword desktop check for web sessions: < 20µs median."""
        from robotmcp.models.session_models import ExecutionSession

        session = ExecutionSession(session_id="bench-web")
        session.imported_libraries.extend(["BuiltIn", "Browser"])
        latency = _median_us(session.is_desktop_session)
        assert latency < 20, f"is_desktop_session too slow: {latency:.2f}µs"

    @pytest.mark.benchmark
    def test_is_desktop_session_desktop_session(self):
        from robotmcp.models.session_models import ExecutionSession

        session = ExecutionSession(session_id="bench-desktop")
        session.imported_libraries.extend(["BuiltIn", "PlatynUI.BareMetal"])
        latency = _median_us(session.is_desktop_session)
        assert latency < 20, f"is_desktop_session too slow: {latency:.2f}µs"


class TestIntentResolutionLatency:
    @pytest.mark.benchmark
    def test_platynui_mapping_resolution(self):
        """Registry resolve for PlatynUI mappings: < 200µs median."""
        from robotmcp.domains.intent.aggregates import IntentRegistry
        from robotmcp.domains.intent.value_objects import IntentVerb

        registry = IntentRegistry.with_builtins()
        latency = _median_us(
            lambda: registry.resolve(IntentVerb.CLICK, "PlatynUI.BareMetal")
        )
        assert latency < 200, f"intent resolve too slow: {latency:.1f}µs"

    @pytest.mark.benchmark
    def test_platynui_locator_passthrough(self):
        """Normalizer pass-through for PlatynUI descriptors: < 50µs median."""
        from robotmcp.domains.intent.adapters.locator_normalizer_adapter import (
            LocatorNormalizerAdapter,
        )
        from robotmcp.domains.intent.value_objects import IntentTarget

        adapter = LocatorNormalizerAdapter()
        target = IntentTarget(
            locator="/app:*[@Name='calc']//control:Button[@Name='OK']"
        )
        latency = _median_us(
            lambda: adapter.normalize(target, "PlatynUI.BareMetal")
        )
        assert latency < 50, f"normalizer pass-through too slow: {latency:.1f}µs"


class TestTimeoutClassificationLatency:
    @pytest.mark.benchmark
    def test_platynui_keyword_classification(self):
        """Classifying PlatynUI keywords: < 20µs median."""
        from robotmcp.domains.timeout.keyword_classifier import classify_keyword

        latency = _median_us(lambda: classify_keyword("Pointer Click"))
        assert latency < 20, f"classification too slow: {latency:.2f}µs"
