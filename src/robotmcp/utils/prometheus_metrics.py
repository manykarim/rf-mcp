"""Multi-pod aware Prometheus metrics collection for MCP Robot Framework server.

This module provides Prometheus-compatible metrics that work across multiple pods:
- Track all MCP tool calls as counters with per-user, per-pod labels
- Track tool execution duration as histograms
- Track active sessions and concurrent requests as gauges
- Enable aggregation across multiple pods via Prometheus queries
- Survive pod restarts through Prometheus persistence

Metric examples (matching the mcp-atlassian pattern):
  rf_mcp_tool_calls_total{tool_name="execute_step",pod="rf-mcp-abc-123",
                           user_agent="python-httpx/0.28.1",username="user@example.com"} 42.0
  rf_mcp_tool_duration_seconds_bucket{tool_name="find_keywords",pod="rf-mcp-abc-123",...}
  rf_mcp_errors_total{tool_name="run_test_suite",pod="rf-mcp-abc-123",error_type="keyword_not_found"} 1.0
  rf_mcp_active_sessions{pod="rf-mcp-abc-123"} 2.0
  rf_mcp_concurrent_requests{pod="rf-mcp-abc-123"} 3.0

Environment variables:
  HOSTNAME                    - Pod name (set automatically by Kubernetes, defaults to socket hostname)
  ROBOTMCP_METRICS_PORT       - HTTP port for Prometheus scraping (default: 9090, 0 = disabled)
  ROBOTMCP_METRICS_ENABLED    - "1" / "true" to force-enable, "0" / "false" to force-disable
"""

from __future__ import annotations

import logging
import os
import socket
import time
from contextvars import ContextVar
from typing import Any

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        REGISTRY,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Context variable – carries per-request caller info for the duration of a
# tool call so inner helpers can read it without parameter threading.
# ---------------------------------------------------------------------------
_request_context: ContextVar[dict[str, str]] = ContextVar(
    "_request_context", default={}
)


def set_request_context(username: str = "unknown", user_agent: str = "unknown") -> None:
    """Store caller identity in the current async task's context variable."""
    _request_context.set({"username": username, "user_agent": user_agent})


def get_request_context() -> dict[str, str]:
    """Return caller identity for the current async task (username, user_agent)."""
    return _request_context.get({})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_create_metric(
    metric_class: type[Any], *args: Any, **kwargs: Any
) -> Any | None:
    """Safely create a Prometheus metric, handling duplicate registration.

    Args:
        metric_class: The metric class to instantiate (Counter, Gauge, Histogram …)
        *args: Positional arguments for the metric.
        **kwargs: Keyword arguments for the metric.

    Returns:
        The metric instance, or None if Prometheus is not available.
    """
    if not PROMETHEUS_AVAILABLE:
        return None

    try:
        return metric_class(*args, **kwargs)
    except ValueError as e:
        if "Duplicated timeseries" in str(e):
            # Metric already registered (e.g. during hot-reload or test reuse) –
            # return the existing collector from the registry.
            metric_name = args[0] if args else kwargs.get("name")
            # _names_to_collectors maps full metric names → collector object
            existing = REGISTRY._names_to_collectors.get(metric_name)  # type: ignore[attr-defined]
            if existing is None:
                # Counter suffix variants: try with/without _total
                if metric_name and metric_name.endswith("_total"):
                    existing = REGISTRY._names_to_collectors.get(metric_name[:-6])  # type: ignore[attr-defined]
                elif metric_name:
                    existing = REGISTRY._names_to_collectors.get(metric_name + "_total")  # type: ignore[attr-defined]
            if existing is None:
                # Final fallback: linear scan by _name attribute
                for collector in REGISTRY._collector_to_names:  # type: ignore[attr-defined]
                    if hasattr(collector, "_name") and collector._name == metric_name:
                        return collector
            return existing
        raise


def _resolve_pod_name() -> str:
    """Return the pod name.

    Kubernetes automatically injects the pod name via the HOSTNAME env var.
    Falls back to the machine's hostname so metrics are still identifiable
    when running locally or in Docker.
    """
    return os.environ.get("HOSTNAME") or socket.gethostname() or "unknown"


# ---------------------------------------------------------------------------
# Main metrics class
# ---------------------------------------------------------------------------

class RobotMCPMetrics:
    """Multi-pod aware metrics collector for the MCP Robot Framework server."""

    # Prometheus histogram bucket boundaries tailored for MCP tool latencies
    _DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)

    def __init__(self, pod_name: str | None = None) -> None:
        if not PROMETHEUS_AVAILABLE:
            self._metrics_enabled = False
            logger.warning(
                "prometheus-client not installed. "
                "Install it with: pip install prometheus-client"
            )
            return

        self._metrics_enabled = True
        self.pod_name = pod_name or _resolve_pod_name()
        self._current_concurrent = 0

        # ── Tool call activity counter (primary multi-user / multi-pod metric) ──
        # Mirrors mcp_atlassian_user_activity_total pattern:
        #   rf_mcp_tool_calls_total{tool_name, pod, user_agent, username}
        self.tool_calls = _safe_create_metric(
            Counter,
            "rf_mcp_tool_calls_total",
            "Total MCP tool calls per user and pod",
            ["tool_name", "pod", "user_agent", "username"],
        )

        # ── Tool execution duration histogram ──
        self.tool_duration = _safe_create_metric(
            Histogram,
            "rf_mcp_tool_duration_seconds",
            "MCP tool execution duration in seconds",
            ["tool_name", "pod"],
            buckets=self._DURATION_BUCKETS,
        )

        # ── Error counter ──
        self.errors = _safe_create_metric(
            Counter,
            "rf_mcp_errors_total",
            "Total errors per tool and pod",
            ["tool_name", "pod", "error_type"],
        )

        # ── Concurrent request gauge (per pod) ──
        self.concurrent_requests = _safe_create_metric(
            Gauge,
            "rf_mcp_concurrent_requests",
            "Current number of in-flight MCP tool calls per pod",
            ["pod"],
        )

        # ── Active sessions gauge (per pod) ──
        self.active_sessions = _safe_create_metric(
            Gauge,
            "rf_mcp_active_sessions",
            "Current number of active Robot Framework sessions per pod",
            ["pod"],
        )

        # ── Server info gauge (static labels carry build metadata) ──
        import importlib.metadata as _meta
        try:
            _version = _meta.version("rf-mcp")
        except Exception:
            _version = "unknown"

        self.server_info = _safe_create_metric(
            Gauge,
            "rf_mcp_server_info",
            "Static information about this MCP server instance",
            ["pod", "version"],
        )
        if self.server_info is not None:
            self.server_info.labels(pod=self.pod_name, version=_version).set(1)

        logger.info(
            "RobotMCPMetrics initialised for pod=%s (prometheus-client available)",
            self.pod_name,
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def track_tool_call(
        self,
        tool_name: str,
        username: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        """Increment the tool-call counter for `tool_name`.

        Args:
            tool_name:  Name of the MCP tool (e.g. ``"execute_step"``).
            username:   Caller identity (extracted from MCP request context).
            user_agent: Caller's user-agent string.
        """
        if not self._metrics_enabled or self.tool_calls is None:
            return
        self.tool_calls.labels(
            tool_name=tool_name,
            pod=self.pod_name,
            user_agent=user_agent or "unknown",
            username=username or "unknown",
        ).inc()

    def start_tool_tracking(
        self,
        tool_name: str,
        username: str | None = None,
        user_agent: str | None = None,
    ) -> dict[str, Any]:
        """Begin tracking an in-flight tool call.

        Call :meth:`end_tool_tracking` with the returned context dict when
        the call completes.

        Args:
            tool_name:  Name of the MCP tool.
            username:   Caller identity.
            user_agent: Caller's user-agent string.

        Returns:
            Opaque context dict to pass to :meth:`end_tool_tracking`.
        """
        if not self._metrics_enabled:
            return {}

        start_time = time.monotonic()
        self._current_concurrent += 1
        if self.concurrent_requests is not None:
            self.concurrent_requests.labels(pod=self.pod_name).set(
                self._current_concurrent
            )

        # Increment call counter eagerly so it's visible immediately
        self.track_tool_call(
            tool_name=tool_name,
            username=username,
            user_agent=user_agent,
        )

        return {
            "start_time": start_time,
            "tool_name": tool_name,
            "username": username,
            "user_agent": user_agent,
        }

    def end_tool_tracking(
        self, context: dict[str, Any], success: bool = True, error_type: str = ""
    ) -> None:
        """Finish tracking and record duration + error metrics.

        Args:
            context:    Context dict returned by :meth:`start_tool_tracking`.
            success:    Whether the call completed without error.
            error_type: Short error category (e.g. ``"keyword_not_found"``).
        """
        if not self._metrics_enabled or not context:
            return

        duration = time.monotonic() - context.get("start_time", time.monotonic())
        tool_name = context.get("tool_name", "unknown")

        if self.tool_duration is not None:
            self.tool_duration.labels(
                tool_name=tool_name, pod=self.pod_name
            ).observe(duration)

        if not success and self.errors is not None:
            self.errors.labels(
                tool_name=tool_name,
                pod=self.pod_name,
                error_type=error_type or "unknown",
            ).inc()

        self._current_concurrent = max(0, self._current_concurrent - 1)
        if self.concurrent_requests is not None:
            self.concurrent_requests.labels(pod=self.pod_name).set(
                self._current_concurrent
            )

    def set_active_sessions(self, count: int) -> None:
        """Update the active-sessions gauge.

        Args:
            count: Current number of active Robot Framework sessions.
        """
        if not self._metrics_enabled or self.active_sessions is None:
            return
        self.active_sessions.labels(pod=self.pod_name).set(count)

    def generate_metrics(self) -> tuple[str, str]:
        """Generate Prometheus exposition text for scraping.

        Returns:
            Tuple of ``(metrics_content, content_type)``.
        """
        if not self._metrics_enabled:
            return (
                "# rf-mcp Prometheus metrics not available – "
                "install prometheus-client: pip install prometheus-client\n",
                "text/plain; charset=utf-8",
            )
        content = generate_latest().decode("utf-8")
        return content, CONTENT_TYPE_LATEST

    @property
    def is_enabled(self) -> bool:
        """True when prometheus-client is installed and metrics are active."""
        return self._metrics_enabled


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_metrics_instance: RobotMCPMetrics | None = None


def initialize_metrics(pod_name: str | None = None) -> RobotMCPMetrics:
    """Initialise (or return) the global :class:`RobotMCPMetrics` singleton.

    Safe to call multiple times; subsequent calls return the existing instance.

    Args:
        pod_name: Pod name override (defaults to ``HOSTNAME`` env var).

    Returns:
        The global :class:`RobotMCPMetrics` instance.
    """
    global _metrics_instance
    if _metrics_instance is not None:
        return _metrics_instance

    # Honour explicit opt-out via env var
    enabled_env = os.environ.get("ROBOTMCP_METRICS_ENABLED", "").strip().lower()
    if enabled_env in {"0", "false", "no", "off"}:
        logger.info("Prometheus metrics explicitly disabled via ROBOTMCP_METRICS_ENABLED")
        # Return a no-op instance (PROMETHEUS_AVAILABLE forced off)
        _metrics_instance = _NoOpMetrics()  # type: ignore[assignment]
        return _metrics_instance  # type: ignore[return-value]

    _metrics_instance = RobotMCPMetrics(pod_name=pod_name)
    return _metrics_instance


def get_metrics() -> RobotMCPMetrics | None:
    """Return the global :class:`RobotMCPMetrics` instance, or None.

    Returns:
        The metrics instance if initialized, otherwise None.
    """
    return _metrics_instance


# ---------------------------------------------------------------------------
# No-op stub (used when metrics are explicitly disabled)
# ---------------------------------------------------------------------------

class _NoOpMetrics:
    """Drop-in replacement used when metrics are explicitly disabled."""

    pod_name: str = _resolve_pod_name()
    is_enabled: bool = False

    def track_tool_call(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        pass

    def start_tool_tracking(self, *args: Any, **kwargs: Any) -> dict:
        return {}

    def end_tool_tracking(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_active_sessions(self, count: int) -> None:
        pass

    def generate_metrics(self) -> tuple[str, str]:
        return (
            "# rf-mcp Prometheus metrics disabled (ROBOTMCP_METRICS_ENABLED=0)\n",
            "text/plain; charset=utf-8",
        )


# ---------------------------------------------------------------------------
# Optional standalone HTTP server for Prometheus scraping
# ---------------------------------------------------------------------------

def start_metrics_http_server(port: int | None = None) -> int | None:
    """Start a lightweight HTTP server that exposes ``/metrics``.

    Used when the MCP server runs in stdio mode (no built-in HTTP) so
    Prometheus can still scrape the process.

    The port is resolved as follows (first wins):
    1. ``port`` argument
    2. ``ROBOTMCP_METRICS_PORT`` environment variable
    3. Default ``9090``

    Pass ``port=0`` or set ``ROBOTMCP_METRICS_PORT=0`` to disable.

    Args:
        port: TCP port for the metrics HTTP server, or 0 to disable.

    Returns:
        The port actually bound, or None if not started.
    """
    resolved_port = port
    if resolved_port is None:
        env_port = os.environ.get("ROBOTMCP_METRICS_PORT", "9090")
        try:
            resolved_port = int(env_port)
        except ValueError:
            logger.warning(
                "Invalid ROBOTMCP_METRICS_PORT value '%s', using 9090", env_port
            )
            resolved_port = 9090

    if resolved_port == 0:
        logger.info("Prometheus metrics HTTP server disabled (port=0)")
        return None

    if not PROMETHEUS_AVAILABLE:
        logger.warning(
            "Cannot start metrics HTTP server – prometheus-client not installed. "
            "Run: pip install prometheus-client"
        )
        return None

    try:
        from prometheus_client import start_http_server

        start_http_server(resolved_port)
        logger.info(
            "Prometheus metrics HTTP server started on :%d/metrics", resolved_port
        )
        return resolved_port
    except OSError as exc:
        logger.warning(
            "Could not start Prometheus metrics HTTP server on port %d: %s",
            resolved_port,
            exc,
        )
        return None
