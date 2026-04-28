"""Tests for the Prometheus metrics module (src/robotmcp/utils/prometheus_metrics.py).

Run with:  uv run pytest tests/test_prometheus_metrics.py -v
"""
from __future__ import annotations

import importlib
import os
import sys
import time
import types
import unittest.mock as mock

import pytest

# ---------------------------------------------------------------------------
# Helpers to reset the module-level singleton between tests
# ---------------------------------------------------------------------------

def _reset_metrics_singleton() -> None:
    """Force-reset the global _metrics_instance so each test starts fresh."""
    import robotmcp.utils.prometheus_metrics as mod
    mod._metrics_instance = None


def _fresh_metrics_module() -> types.ModuleType:
    """Re-import the metrics module with a clean singleton state."""
    mod_name = "robotmcp.utils.prometheus_metrics"
    if mod_name in sys.modules:
        sys.modules.pop(mod_name)
    return importlib.import_module(mod_name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton before and after each test."""
    _reset_metrics_singleton()
    yield
    _reset_metrics_singleton()


# ---------------------------------------------------------------------------
# initialize_metrics / get_metrics
# ---------------------------------------------------------------------------

class TestInitializeMetrics:
    """Tests for initialize_metrics() and get_metrics()."""

    def test_initialize_returns_instance(self):
        from robotmcp.utils.prometheus_metrics import initialize_metrics, RobotMCPMetrics, _NoOpMetrics
        m = initialize_metrics()
        assert isinstance(m, (RobotMCPMetrics, _NoOpMetrics))

    def test_initialize_is_idempotent(self):
        from robotmcp.utils.prometheus_metrics import initialize_metrics
        m1 = initialize_metrics()
        m2 = initialize_metrics()
        assert m1 is m2

    def test_get_metrics_returns_none_before_init(self):
        from robotmcp.utils.prometheus_metrics import get_metrics
        assert get_metrics() is None

    def test_get_metrics_returns_instance_after_init(self):
        from robotmcp.utils.prometheus_metrics import initialize_metrics, get_metrics
        initialize_metrics()
        assert get_metrics() is not None

    def test_disabled_via_env_var(self, monkeypatch):
        from robotmcp.utils.prometheus_metrics import initialize_metrics, _NoOpMetrics
        monkeypatch.setenv("ROBOTMCP_METRICS_ENABLED", "0")
        m = initialize_metrics()
        assert isinstance(m, _NoOpMetrics)
        assert not m.is_enabled

    def test_disabled_via_false_string(self, monkeypatch):
        from robotmcp.utils.prometheus_metrics import initialize_metrics, _NoOpMetrics
        monkeypatch.setenv("ROBOTMCP_METRICS_ENABLED", "false")
        m = initialize_metrics()
        assert isinstance(m, _NoOpMetrics)

    def test_pod_name_from_hostname_env(self, monkeypatch):
        from robotmcp.utils.prometheus_metrics import initialize_metrics, RobotMCPMetrics
        monkeypatch.setenv("HOSTNAME", "test-pod-abc-123")
        m = initialize_metrics(pod_name=None)
        # _NoOpMetrics also has pod_name attribute
        assert m.pod_name == "test-pod-abc-123"

    def test_pod_name_override(self):
        from robotmcp.utils.prometheus_metrics import initialize_metrics
        m = initialize_metrics(pod_name="custom-pod-001")
        assert m.pod_name == "custom-pod-001"


# ---------------------------------------------------------------------------
# RobotMCPMetrics (with prometheus_client available)
# ---------------------------------------------------------------------------

class TestRobotMCPMetrics:
    """Tests for the RobotMCPMetrics class assuming prometheus_client is installed."""

    @pytest.fixture
    def metrics(self):
        from robotmcp.utils.prometheus_metrics import initialize_metrics, RobotMCPMetrics, _NoOpMetrics
        m = initialize_metrics(pod_name="test-pod")
        if isinstance(m, _NoOpMetrics):
            pytest.skip("prometheus_client not available")
        return m

    # ── track_tool_call ──────────────────────────────────────────────────

    def test_track_tool_call_increments_counter(self, metrics):
        before = metrics.tool_calls.labels(
            tool_name="execute_step", pod="test-pod",
            user_agent="test-agent-incr", username="testuser-incr"
        )._value.get()
        metrics.track_tool_call(
            "execute_step", username="testuser-incr", user_agent="test-agent-incr"
        )
        after = metrics.tool_calls.labels(
            tool_name="execute_step", pod="test-pod",
            user_agent="test-agent-incr", username="testuser-incr"
        )._value.get()
        assert after == before + 1.0

    def test_track_tool_call_uses_unknown_defaults(self, metrics):
        before = metrics.tool_calls.labels(
            tool_name="find_keywords_default", pod="test-pod",
            user_agent="unknown", username="unknown"
        )._value.get()
        metrics.track_tool_call("find_keywords_default")
        val = metrics.tool_calls.labels(
            tool_name="find_keywords_default", pod="test-pod",
            user_agent="unknown", username="unknown"
        )._value.get()
        assert val == before + 1.0

    # ── start_tool_tracking / end_tool_tracking ──────────────────────────

    def test_start_returns_context_dict(self, metrics):
        ctx = metrics.start_tool_tracking("run_test_suite")
        try:
            assert isinstance(ctx, dict)
            assert "start_time" in ctx
            assert ctx["tool_name"] == "run_test_suite"
        finally:
            metrics.end_tool_tracking(ctx)  # always clean up gauge

    def test_concurrent_requests_increments_on_start(self, metrics):
        # Reset the gauge to a known baseline to avoid cross-test state leakage
        metrics.concurrent_requests.labels(pod="test-pod").set(0)
        metrics._current_concurrent = 0
        before = metrics.concurrent_requests.labels(pod="test-pod")._value.get()
        ctx = metrics.start_tool_tracking("execute_step")
        after = metrics.concurrent_requests.labels(pod="test-pod")._value.get()
        assert after == before + 1.0
        metrics.end_tool_tracking(ctx)  # cleanup

    def test_concurrent_requests_decrements_on_end(self, metrics):
        ctx = metrics.start_tool_tracking("execute_step")
        before = metrics.concurrent_requests.labels(pod="test-pod")._value.get()
        metrics.end_tool_tracking(ctx)
        after = metrics.concurrent_requests.labels(pod="test-pod")._value.get()
        assert after == before - 1.0

    def test_end_tracking_records_duration_histogram(self, metrics):
        from prometheus_client import generate_latest
        ctx = metrics.start_tool_tracking("execute_step_hist")
        time.sleep(0.01)
        metrics.end_tool_tracking(ctx, success=True)
        # Verify via text exposition (avoids internal API differences across versions)
        output = generate_latest().decode("utf-8")
        assert "rf_mcp_tool_duration_seconds" in output
        assert 'tool_name="execute_step_hist"' in output

    def test_end_tracking_records_error_counter(self, metrics):
        ctx = metrics.start_tool_tracking("run_test_suite_err")
        before = metrics.errors.labels(
            tool_name="run_test_suite_err", pod="test-pod", error_type="timeout"
        )._value.get()
        metrics.end_tool_tracking(ctx, success=False, error_type="timeout")
        val = metrics.errors.labels(
            tool_name="run_test_suite_err", pod="test-pod", error_type="timeout"
        )._value.get()
        assert val == before + 1.0

    def test_end_tracking_noop_on_empty_context(self, metrics):
        # Should not raise
        metrics.end_tool_tracking({})

    # ── set_active_sessions ──────────────────────────────────────────────

    def test_set_active_sessions(self, metrics):
        metrics.set_active_sessions(5)
        val = metrics.active_sessions.labels(pod="test-pod")._value.get()
        assert val == 5.0

    def test_set_active_sessions_to_zero(self, metrics):
        metrics.set_active_sessions(3)
        metrics.set_active_sessions(0)
        val = metrics.active_sessions.labels(pod="test-pod")._value.get()
        assert val == 0.0

    # ── generate_metrics ─────────────────────────────────────────────────

    def test_generate_metrics_returns_tuple(self, metrics):
        content, ctype = metrics.generate_metrics()
        assert isinstance(content, str)
        assert "rf_mcp" in content

    def test_generate_metrics_content_type(self, metrics):
        _, ctype = metrics.generate_metrics()
        assert "text/plain" in ctype

    def test_is_enabled_true(self, metrics):
        assert metrics.is_enabled is True


# ---------------------------------------------------------------------------
# _NoOpMetrics
# ---------------------------------------------------------------------------

class TestNoOpMetrics:
    """Tests for the _NoOpMetrics stub used when metrics are disabled."""

    @pytest.fixture
    def noop(self):
        from robotmcp.utils.prometheus_metrics import _NoOpMetrics
        return _NoOpMetrics()

    def test_track_tool_call_is_noop(self, noop):
        noop.track_tool_call("execute_step")  # should not raise

    def test_start_tracking_returns_empty_dict(self, noop):
        ctx = noop.start_tool_tracking("execute_step")
        assert ctx == {}

    def test_end_tracking_is_noop(self, noop):
        noop.end_tool_tracking({"any": "thing"})  # should not raise

    def test_set_active_sessions_is_noop(self, noop):
        noop.set_active_sessions(10)  # should not raise

    def test_generate_metrics_returns_disabled_message(self, noop):
        content, _ = noop.generate_metrics()
        assert "disabled" in content.lower()

    def test_is_enabled_false(self, noop):
        assert noop.is_enabled is False


# ---------------------------------------------------------------------------
# Request context ContextVar
# ---------------------------------------------------------------------------

class TestRequestContext:
    """Tests for set_request_context() / get_request_context()."""

    def test_get_returns_empty_before_set(self):
        from robotmcp.utils.prometheus_metrics import get_request_context
        ctx = get_request_context()
        assert isinstance(ctx, dict)

    def test_set_and_get_round_trip(self):
        from robotmcp.utils.prometheus_metrics import set_request_context, get_request_context
        set_request_context(username="alice", user_agent="test-client/1.0")
        ctx = get_request_context()
        assert ctx["username"] == "alice"
        assert ctx["user_agent"] == "test-client/1.0"

    def test_set_uses_defaults(self):
        from robotmcp.utils.prometheus_metrics import set_request_context, get_request_context
        set_request_context()
        ctx = get_request_context()
        assert ctx["username"] == "unknown"
        assert ctx["user_agent"] == "unknown"

    @pytest.mark.asyncio
    async def test_context_isolated_between_async_tasks(self):
        """Each asyncio task has its own ContextVar state."""
        import asyncio
        from robotmcp.utils.prometheus_metrics import set_request_context, get_request_context

        results: list[dict] = []

        async def task_a():
            set_request_context(username="user-a")
            await asyncio.sleep(0)
            results.append(get_request_context())

        async def task_b():
            set_request_context(username="user-b")
            await asyncio.sleep(0)
            results.append(get_request_context())

        await asyncio.gather(task_a(), task_b())
        usernames = {r["username"] for r in results}
        assert usernames == {"user-a", "user-b"}


# ---------------------------------------------------------------------------
# start_metrics_http_server (port 0 = no-op)
# ---------------------------------------------------------------------------

class TestStartMetricsHttpServer:
    """Tests for start_metrics_http_server()."""

    def test_port_zero_returns_none(self):
        from robotmcp.utils.prometheus_metrics import start_metrics_http_server
        result = start_metrics_http_server(port=0)
        assert result is None

    def test_env_port_zero_returns_none(self, monkeypatch):
        from robotmcp.utils.prometheus_metrics import start_metrics_http_server
        monkeypatch.setenv("ROBOTMCP_METRICS_PORT", "0")
        result = start_metrics_http_server()
        assert result is None

    def test_start_server_binds_port(self, monkeypatch):
        """Actually start a metrics server on a random high port."""
        try:
            from prometheus_client import start_http_server  # noqa: F401
        except ImportError:
            pytest.skip("prometheus_client not available")

        from robotmcp.utils.prometheus_metrics import start_metrics_http_server
        # Use a high port that is likely free; if it's taken the test is skipped.
        port = 19_099
        try:
            result = start_metrics_http_server(port=port)
        except OSError:
            pytest.skip(f"Port {port} already in use")
        assert result == port
