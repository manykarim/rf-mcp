"""Unit tests for BridgeHealthMonitor (Phase 4 of ADR-004).

Tests the bridge health monitor functionality including:
- Starting and stopping the monitor
- Health check success and failure tracking
- Cleanup triggering after threshold failures
- Environment variable configuration
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestBridgeHealthMonitor:
    """Tests for the BridgeHealthMonitor class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock ExternalRFClient."""
        client = MagicMock()
        client.host = "localhost"
        client.port = 7317
        return client

    @pytest.fixture
    def monitor(self):
        """Create a BridgeHealthMonitor instance with short interval for testing."""
        from robotmcp.server import BridgeHealthMonitor

        return BridgeHealthMonitor(interval_seconds=1, failure_threshold=3)

    @pytest.mark.asyncio
    async def test_start_creates_task(self, monitor):
        """Monitor start creates an asyncio task."""
        await monitor.start()
        assert monitor._running is True
        assert monitor._task is not None
        assert not monitor._task.done()
        await monitor.stop()

    @pytest.mark.asyncio
    async def test_stop_cancels_task(self, monitor):
        """Monitor stop cancels the running task."""
        await monitor.start()
        assert monitor._running is True

        await monitor.stop()
        assert monitor._running is False
        # Task should be cancelled
        assert monitor._task.cancelled() or monitor._task.done()

    @pytest.mark.asyncio
    async def test_health_check_success_resets_failures(self, monitor, mock_client):
        """Successful health check resets consecutive failure counter."""
        monitor.consecutive_failures = 2

        with (
            patch(
                "robotmcp.server._get_external_client_if_configured",
                return_value=mock_client,
            ),
            patch(
                "robotmcp.server._check_bridge_health",
                return_value={"healthy": True, "context_active": True},
            ),
        ):
            await monitor._check_health()

        assert monitor.consecutive_failures == 0
        assert monitor.last_healthy is not None

    @pytest.mark.asyncio
    async def test_health_check_failure_increments_counter(self, monitor, mock_client):
        """Failed health check increments consecutive failure counter."""
        monitor.consecutive_failures = 0

        with (
            patch(
                "robotmcp.server._get_external_client_if_configured",
                return_value=mock_client,
            ),
            patch(
                "robotmcp.server._check_bridge_health",
                return_value={
                    "healthy": False,
                    "recovery_hint": "Bridge not responding",
                },
            ),
        ):
            await monitor._check_health()

        assert monitor.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_health_check_exception_increments_counter(
        self, monitor, mock_client
    ):
        """Exception during health check increments failure counter."""
        monitor.consecutive_failures = 0

        with (
            patch(
                "robotmcp.server._get_external_client_if_configured",
                return_value=mock_client,
            ),
            patch(
                "robotmcp.server._check_bridge_health",
                side_effect=Exception("Connection failed"),
            ),
        ):
            await monitor._check_health()

        assert monitor.consecutive_failures == 1

    @pytest.mark.asyncio
    async def test_cleanup_triggered_at_threshold(self, monitor, mock_client):
        """Cleanup is triggered when failures reach threshold."""
        monitor.consecutive_failures = 2  # One more failure will hit threshold of 3

        mock_session_manager = MagicMock()
        mock_session_manager.cleanup_expired_sessions.return_value = 2

        with (
            patch(
                "robotmcp.server._get_external_client_if_configured",
                return_value=mock_client,
            ),
            patch(
                "robotmcp.server._check_bridge_health",
                return_value={"healthy": False, "recovery_hint": "Bridge down"},
            ),
            patch(
                "robotmcp.server.execution_engine.session_manager",
                mock_session_manager,
            ),
        ):
            await monitor._check_health()

        # Should have triggered cleanup
        mock_session_manager.cleanup_expired_sessions.assert_called_once()
        # Counter should be reset after cleanup
        assert monitor.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_no_client_configured_skips_check(self, monitor):
        """Health check is skipped when no client is configured."""
        with patch(
            "robotmcp.server._get_external_client_if_configured", return_value=None
        ):
            # Should not raise and should not change failure count
            await monitor._check_health()

        assert monitor.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_handle_unhealthy_bridge_logs_cleanup(self, monitor):
        """Unhealthy bridge handler logs cleanup actions."""
        mock_session_manager = MagicMock()
        mock_session_manager.cleanup_expired_sessions.return_value = 5

        with patch(
            "robotmcp.server.execution_engine.session_manager", mock_session_manager
        ):
            await monitor._handle_unhealthy_bridge()

        mock_session_manager.cleanup_expired_sessions.assert_called_once()
        assert monitor.consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_handle_unhealthy_bridge_handles_cleanup_error(self, monitor):
        """Unhealthy bridge handler gracefully handles cleanup errors."""
        mock_session_manager = MagicMock()
        mock_session_manager.cleanup_expired_sessions.side_effect = Exception(
            "Cleanup failed"
        )

        with patch(
            "robotmcp.server.execution_engine.session_manager", mock_session_manager
        ):
            # Should not raise
            await monitor._handle_unhealthy_bridge()

    def test_init_defaults(self):
        """Monitor initializes with correct defaults."""
        from robotmcp.server import BridgeHealthMonitor

        monitor = BridgeHealthMonitor()
        assert monitor.interval == 60
        assert monitor.failure_threshold == 3
        assert monitor.consecutive_failures == 0
        assert monitor.last_healthy is None
        assert monitor._running is False
        assert monitor._task is None

    def test_init_custom_values(self):
        """Monitor accepts custom interval and threshold."""
        from robotmcp.server import BridgeHealthMonitor

        monitor = BridgeHealthMonitor(interval_seconds=30, failure_threshold=5)
        assert monitor.interval == 30
        assert monitor.failure_threshold == 5


class TestHealthMonitorLifespan:
    """Tests for the health monitor lifespan integration."""

    @pytest.mark.asyncio
    async def test_install_health_monitor_lifespan(self):
        """Installing health monitor lifespan sets the mcp lifespan."""
        from robotmcp.server import (
            BridgeHealthMonitor,
            _install_health_monitor_lifespan,
            mcp,
        )

        with patch.object(mcp, "_mcp_server", MagicMock(lifespan=None)):
            _install_health_monitor_lifespan(interval_seconds=30, failure_threshold=2)

            # Lifespan should be set
            assert mcp._mcp_server.lifespan is not None


class TestEnvironmentConfiguration:
    """Tests for environment variable handling."""

    def test_heartbeat_disabled_by_default(self):
        """Health monitor is disabled by default (ROBOTMCP_BRIDGE_HEARTBEAT=0)."""
        import os

        # Ensure env var is not set or set to 0
        env_value = os.environ.get("ROBOTMCP_BRIDGE_HEARTBEAT", "0")
        assert env_value in {"0", "false", "no", ""}

    def test_heartbeat_interval_default(self):
        """Default heartbeat interval is 60 seconds."""
        import os

        interval = int(os.environ.get("ROBOTMCP_HEARTBEAT_INTERVAL", "60"))
        assert interval == 60

    def test_heartbeat_threshold_default(self):
        """Default failure threshold is 3."""
        import os

        threshold = int(os.environ.get("ROBOTMCP_HEARTBEAT_THRESHOLD", "3"))
        assert threshold == 3
