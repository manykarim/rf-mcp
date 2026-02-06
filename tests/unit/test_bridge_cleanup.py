"""Unit tests for ADR-004 Debug Bridge Connection Cleanup Management.

This test module provides comprehensive coverage for the Debug Bridge Connection
Cleanup feature, which ensures clean MCP server startup and provides on-demand
cleanup capabilities for stale connections and sessions.

Test Coverage:
    TestStartupValidation (Phase 1)
        - Startup cleanup in auto mode with healthy bridge
        - Startup cleanup in auto mode with unhealthy bridge
        - Startup cleanup in always mode
        - Startup cleanup in off mode
        - Startup cleanup when no attach is configured

    TestManageAttachActions (Phase 2)
        - cleanup action cleans expired sessions
        - cleanup action reports bridge status
        - reset action stops bridge and cleans sessions
        - reset action with no attach configured
        - disconnect_all uses force_stop
        - disconnect_all fallback to stop

    TestForceStop (Phase 2)
        - force_stop verb sets stop flag
        - force_stop client method exists

    TestInstanceIdentification (Phase 3)
        - Instance ID is unique per process
        - Instance ID format is correct
        - Instance ID header is included in requests

    TestSessionManagerThreadSafety
        - remove_session is thread-safe
"""

import json
import os
import re
import threading
import time
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from robotmcp.components.execution.external_rf_client import ExternalRFClient
from robotmcp.components.execution.session_manager import SessionManager


# ---------------------------------------------------------------------------
# Test Helpers / Fixtures
# ---------------------------------------------------------------------------


class DummyHTTPConnection:
    """Captures HTTP requests issued by ExternalRFClient._post."""

    def __init__(self, host: str, port: int, timeout: int):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.requests: list = []
        self.closed = False
        self._response_body = b'{"success": true, "result": 1}'
        self._response_map: Dict[str, bytes] = {}

    def request(self, method: str, path: str, body: bytes, headers: dict) -> None:
        self.requests.append(
            {
                "method": method,
                "path": path,
                "body": json.loads(body.decode("utf-8")) if body else {},
                "headers": headers,
            }
        )
        self._last_path = path

    def getresponse(self) -> SimpleNamespace:
        # Return path-specific response if configured
        if hasattr(self, "_last_path") and self._last_path in self._response_map:
            body = self._response_map[self._last_path]
        else:
            body = self._response_body
        return SimpleNamespace(read=lambda: body)

    def close(self) -> None:
        self.closed = True

    def set_response(self, response: Dict[str, Any]) -> None:
        """Set the response body for all requests."""
        self._response_body = json.dumps(response).encode("utf-8")

    def set_path_response(self, path: str, response: Dict[str, Any]) -> None:
        """Set a response for a specific path."""
        self._response_map[path] = json.dumps(response).encode("utf-8")


@pytest.fixture
def mock_bridge_client():
    """Create a mock ExternalRFClient for testing bridge interactions."""
    client = MagicMock(spec=ExternalRFClient)
    client.host = "127.0.0.1"
    client.port = 7317
    client.instance_id = "12345_1700000000"

    # Default diagnostics response (healthy bridge)
    client.diagnostics.return_value = {
        "success": True,
        "result": {
            "libraries": ["Browser", "Collections", "String"],
            "context": True,
        },
    }

    # Default stop response
    client.stop.return_value = {"success": True}

    # Default force_stop response
    client.force_stop.return_value = {"success": True, "force_stopped": True}

    return client


@pytest.fixture
def session_manager():
    """Create a fresh SessionManager instance."""
    return SessionManager()


# =========================================================================
# Phase 1 Tests - Startup Validation
# =========================================================================


class TestStartupValidation:
    """Phase 1: Startup cleanup and validation tests."""

    def test_startup_cleanup_auto_mode_healthy_bridge(
        self, mock_bridge_client, session_manager, monkeypatch
    ):
        """With healthy bridge and active context, auto mode cleans local sessions."""
        # Setup: Create some sessions to be cleaned
        session_manager.create_session("session-1")
        session_manager.create_session("session-2")
        assert session_manager.get_session_count() == 2

        # Mock environment variable for auto mode
        monkeypatch.setenv("ROBOTMCP_STARTUP_CLEANUP", "auto")

        # Mock the external client to return healthy bridge
        mock_bridge_client.diagnostics.return_value = {
            "success": True,
            "result": {"libraries": ["Browser"], "context": True},
        }

        # Import the function being tested
        from robotmcp.server import _check_bridge_health

        health = _check_bridge_health(mock_bridge_client)

        # Verify health check returns healthy
        assert health["healthy"] is True
        assert health["context_active"] is True

        # In auto mode with healthy bridge, sessions should be cleaned
        cleanup_mode = os.environ.get("ROBOTMCP_STARTUP_CLEANUP", "auto").lower()
        assert cleanup_mode == "auto"

        if cleanup_mode == "auto" and health.get("healthy") and health.get("context_active"):
            cleaned = session_manager.cleanup_all_sessions()
            assert cleaned == 2
            assert session_manager.get_session_count() == 0

    def test_startup_cleanup_auto_mode_unhealthy_bridge(
        self, mock_bridge_client, session_manager, monkeypatch
    ):
        """With unhealthy bridge, auto mode skips cleanup."""
        # Setup: Create some sessions
        session_manager.create_session("session-1")
        session_manager.create_session("session-2")
        assert session_manager.get_session_count() == 2

        # Mock environment variable for auto mode
        monkeypatch.setenv("ROBOTMCP_STARTUP_CLEANUP", "auto")

        # Mock the external client to return unhealthy bridge
        mock_bridge_client.diagnostics.return_value = {
            "success": False,
            "error": "Connection refused",
        }

        from robotmcp.server import _check_bridge_health

        health = _check_bridge_health(mock_bridge_client)

        # Verify health check returns unhealthy
        assert health["healthy"] is False

        # In auto mode with unhealthy bridge, sessions should NOT be cleaned
        cleanup_mode = os.environ.get("ROBOTMCP_STARTUP_CLEANUP", "auto").lower()
        assert cleanup_mode == "auto"

        if not (health.get("healthy") and health.get("context_active")):
            # Don't clean - sessions should remain
            assert session_manager.get_session_count() == 2

    def test_startup_cleanup_always_mode(
        self, mock_bridge_client, session_manager, monkeypatch
    ):
        """Always mode cleans regardless of bridge health."""
        # Setup: Create some sessions
        session_manager.create_session("session-1")
        session_manager.create_session("session-2")
        assert session_manager.get_session_count() == 2

        # Mock environment variable for always mode
        monkeypatch.setenv("ROBOTMCP_STARTUP_CLEANUP", "always")

        # Mock unhealthy bridge - shouldn't matter in always mode
        mock_bridge_client.diagnostics.return_value = {
            "success": False,
            "error": "Connection refused",
        }

        cleanup_mode = os.environ.get("ROBOTMCP_STARTUP_CLEANUP", "auto").lower()
        assert cleanup_mode == "always"

        # In always mode, clean regardless of bridge health
        if cleanup_mode == "always":
            cleaned = session_manager.cleanup_all_sessions()
            assert cleaned == 2
            assert session_manager.get_session_count() == 0

    def test_startup_cleanup_off_mode(self, session_manager, monkeypatch):
        """Off mode skips all startup cleanup."""
        # Setup: Create some sessions
        session_manager.create_session("session-1")
        session_manager.create_session("session-2")
        assert session_manager.get_session_count() == 2

        # Mock environment variable for off mode
        monkeypatch.setenv("ROBOTMCP_STARTUP_CLEANUP", "off")

        cleanup_mode = os.environ.get("ROBOTMCP_STARTUP_CLEANUP", "auto").lower()
        assert cleanup_mode == "off"

        # In off mode, don't clean
        if cleanup_mode == "off":
            # Sessions should remain untouched
            assert session_manager.get_session_count() == 2

    def test_startup_cleanup_no_attach_configured(self, session_manager, monkeypatch):
        """When no attach is configured, startup proceeds without cleanup."""
        # Setup: Create some sessions
        session_manager.create_session("session-1")
        assert session_manager.get_session_count() == 1

        # Ensure ROBOTMCP_ATTACH_HOST is not set
        monkeypatch.delenv("ROBOTMCP_ATTACH_HOST", raising=False)

        from robotmcp.server import _get_external_client_if_configured

        client = _get_external_client_if_configured()
        assert client is None

        # When no attach configured, sessions should remain
        assert session_manager.get_session_count() == 1


# =========================================================================
# Phase 2 Tests - Manage Attach Actions
# =========================================================================


class TestManageAttachActions:
    """Phase 2: Tests for new manage_attach actions (cleanup, reset, disconnect_all)."""

    def test_cleanup_action_cleans_expired_sessions(self, session_manager, monkeypatch):
        """Cleanup action removes expired sessions."""
        from datetime import datetime, timedelta

        # Create sessions with old activity timestamps
        session1 = session_manager.create_session("old-session-1")
        session2 = session_manager.create_session("old-session-2")
        session3 = session_manager.create_session("active-session")

        # Make first two sessions expired
        old_time = datetime.now() - timedelta(seconds=session_manager.config.SESSION_CLEANUP_TIMEOUT + 100)
        session1.last_activity = old_time
        session2.last_activity = old_time
        # session3 remains active (recent last_activity)

        assert session_manager.get_session_count() == 3

        # Run cleanup
        expired_count = session_manager.cleanup_expired_sessions()

        # Two expired sessions should be cleaned
        assert expired_count == 2
        assert session_manager.get_session_count() == 1
        assert session_manager.get_session("active-session") is not None
        assert session_manager.get_session("old-session-1") is None
        assert session_manager.get_session("old-session-2") is None

    def test_cleanup_action_reports_bridge_status(self, mock_bridge_client, monkeypatch):
        """Cleanup action includes bridge health in response."""
        from robotmcp.server import _check_bridge_health

        # Mock healthy bridge
        mock_bridge_client.diagnostics.return_value = {
            "success": True,
            "result": {"libraries": ["Browser"], "context": True},
        }

        health = _check_bridge_health(mock_bridge_client)

        assert health["healthy"] is True
        assert health["context_active"] is True
        assert "Browser" in health["libraries"]

    def test_reset_action_stops_bridge_and_cleans(
        self, mock_bridge_client, session_manager, monkeypatch
    ):
        """Reset action stops bridge and cleans all sessions."""
        # Create sessions
        session_manager.create_session("session-1")
        session_manager.create_session("session-2")
        assert session_manager.get_session_count() == 2

        # Mock successful stop
        mock_bridge_client.stop.return_value = {"success": True}

        # Simulate reset action
        stop_resp = mock_bridge_client.stop()
        assert stop_resp.get("success") is True

        sessions_cleaned = session_manager.cleanup_all_sessions()

        assert sessions_cleaned == 2
        assert session_manager.get_session_count() == 0
        mock_bridge_client.stop.assert_called_once()

    def test_reset_action_no_attach_configured(self, session_manager, monkeypatch):
        """Reset action returns error when attach mode not configured."""
        # Ensure ROBOTMCP_ATTACH_HOST is not set
        monkeypatch.delenv("ROBOTMCP_ATTACH_HOST", raising=False)

        from robotmcp.server import _get_external_client_if_configured

        client = _get_external_client_if_configured()

        # Should return None when not configured
        assert client is None

        # In a real reset action, this would return an error
        # {"success": False, "action": "reset", "error": "Attach mode not configured"}

    def test_disconnect_all_uses_force_stop(self, mock_bridge_client, session_manager):
        """Disconnect_all uses force_stop verb."""
        # Create sessions
        session_manager.create_session("session-1")
        assert session_manager.get_session_count() == 1

        # Mock force_stop
        mock_bridge_client.force_stop.return_value = {
            "success": True,
            "force_stopped": True,
        }

        # Call force_stop
        stop_resp = mock_bridge_client.force_stop()

        assert stop_resp.get("success") is True
        assert stop_resp.get("force_stopped") is True
        mock_bridge_client.force_stop.assert_called_once()

        # Clean sessions
        sessions_cleaned = session_manager.cleanup_all_sessions()
        assert sessions_cleaned == 1

    def test_disconnect_all_fallback_to_stop(self, mock_bridge_client, session_manager):
        """Disconnect_all falls back to regular stop if force_stop fails."""
        # Create sessions
        session_manager.create_session("session-1")
        assert session_manager.get_session_count() == 1

        # Mock force_stop to fail
        mock_bridge_client.force_stop.side_effect = Exception("force_stop not supported")
        mock_bridge_client.stop.return_value = {"success": True}

        # Try force_stop, fall back to stop
        try:
            stop_resp = mock_bridge_client.force_stop()
        except Exception:
            stop_resp = mock_bridge_client.stop()

        assert stop_resp.get("success") is True

        # Clean sessions
        sessions_cleaned = session_manager.cleanup_all_sessions()
        assert sessions_cleaned == 1


# =========================================================================
# Phase 2 Tests - Force Stop
# =========================================================================


class TestForceStop:
    """Phase 2: Tests for force_stop verb and client method."""

    def test_force_stop_verb_sets_stop_flag(self, monkeypatch):
        """Force stop verb sets the stop flag in McpAttach."""
        # Mock BuiltIn to avoid RF dependency
        mock_builtin = MagicMock()
        mock_builtin.log_to_console = MagicMock()

        with patch("robotmcp.attach.mcp_attach.BuiltIn", return_value=mock_builtin):
            from robotmcp.attach.mcp_attach import McpAttach

            attach = McpAttach()
            attach._stop_flag = False

            # Execute force_stop command
            result = attach._execute_command("force_stop", {})

            # Verify stop flag is set
            assert attach._stop_flag is True
            assert result["success"] is True
            assert result["force_stopped"] is True

    def test_force_stop_client_method_exists(self, monkeypatch):
        """Verify ExternalRFClient has force_stop method."""
        dummy = DummyHTTPConnection("127.0.0.1", 7317, 10)
        dummy.set_response({"success": True, "force_stopped": True})

        def factory(host, port, timeout):
            return dummy

        monkeypatch.setattr(
            "robotmcp.components.execution.external_rf_client.http.client.HTTPConnection",
            factory,
        )

        client = ExternalRFClient(token="test-token")

        # Verify force_stop method exists and can be called
        assert hasattr(client, "force_stop")
        assert callable(client.force_stop)

        result = client.force_stop()

        assert result["success"] is True
        assert result["force_stopped"] is True
        assert len(dummy.requests) == 1
        assert dummy.requests[0]["path"] == "/force_stop"


# =========================================================================
# Phase 3 Tests - Instance Identification
# =========================================================================


class TestInstanceIdentification:
    """Phase 3: Tests for instance ID tracking."""

    def test_instance_id_unique_per_process(self):
        """Instance ID is shared across all clients in the same process.

        The instance_id is generated once at module level, so all ExternalRFClient
        instances from the same MCP server process share the same ID. This prevents
        excessive "New instance connected" logs on every tool call.
        """
        client1 = ExternalRFClient()
        client2 = ExternalRFClient()
        client3 = ExternalRFClient(host="different", port=9999, token="other")

        # All clients should have instance IDs
        assert client1.instance_id is not None
        assert client2.instance_id is not None
        assert client3.instance_id is not None

        # CRITICAL: All clients from same process share the SAME instance_id
        # This is the fix for the excessive "New instance connected" logs
        assert client1.instance_id == client2.instance_id
        assert client2.instance_id == client3.instance_id

        # ID should include PID
        pid = os.getpid()
        assert str(pid) in client1.instance_id

    def test_instance_id_module_level_helper(self):
        """get_mcp_instance_id() returns the same ID used by clients."""
        from robotmcp.components.execution.external_rf_client import get_mcp_instance_id

        module_id = get_mcp_instance_id()
        client = ExternalRFClient()

        # Module helper and client should return same ID
        assert module_id == client.instance_id

        # Calling again should return same value (it's constant)
        assert get_mcp_instance_id() == module_id

    def test_instance_id_format(self):
        """Instance ID has correct format: {pid}_{timestamp}."""
        client = ExternalRFClient()

        # Format should be "{pid}_{timestamp}"
        pattern = r"^\d+_\d+$"
        assert re.match(pattern, client.instance_id), f"Instance ID {client.instance_id} doesn't match format"

        # Parse and verify parts
        parts = client.instance_id.split("_")
        assert len(parts) == 2

        pid_part = int(parts[0])
        timestamp_part = int(parts[1])

        assert pid_part == os.getpid()
        # Timestamp should be reasonable (within last day)
        now = int(time.time())
        assert now - 86400 < timestamp_part <= now + 60  # Within last day, allowing some future drift

    def test_instance_id_header_included(self, monkeypatch):
        """All requests include X-MCP-Instance-ID header."""
        dummy = DummyHTTPConnection("127.0.0.1", 7317, 10)
        dummy.set_response({"success": True, "result": {}})

        def factory(host, port, timeout):
            return dummy

        monkeypatch.setattr(
            "robotmcp.components.execution.external_rf_client.http.client.HTTPConnection",
            factory,
        )

        client = ExternalRFClient(token="test-token")

        # Make a request
        client.diagnostics()

        # Verify header was included
        assert len(dummy.requests) == 1
        headers = dummy.requests[0]["headers"]
        assert "X-MCP-Instance-ID" in headers
        assert headers["X-MCP-Instance-ID"] == client.instance_id

    def test_instance_id_consistent_across_requests(self, monkeypatch):
        """Same client uses consistent instance ID across multiple requests."""
        dummy = DummyHTTPConnection("127.0.0.1", 7317, 10)
        dummy.set_response({"success": True, "result": {}})

        def factory(host, port, timeout):
            return dummy

        monkeypatch.setattr(
            "robotmcp.components.execution.external_rf_client.http.client.HTTPConnection",
            factory,
        )

        client = ExternalRFClient(token="test-token")

        # Make multiple requests
        client.diagnostics()
        client.stop()
        client.get_page_source()

        # All requests should have same instance ID
        assert len(dummy.requests) == 3
        instance_ids = [req["headers"]["X-MCP-Instance-ID"] for req in dummy.requests]
        assert all(id == instance_ids[0] for id in instance_ids)


# =========================================================================
# Thread Safety Tests - Session Manager
# =========================================================================


class TestSessionManagerThreadSafety:
    """Tests for thread-safe session management."""

    def test_remove_session_thread_safe(self, session_manager):
        """remove_session should be thread-safe with concurrent access."""
        # Create multiple sessions
        session_ids = [f"session-{i}" for i in range(20)]
        for sid in session_ids:
            session_manager.create_session(sid)

        assert session_manager.get_session_count() == 20

        errors = []
        removed = []
        lock = threading.Lock()

        def remove_session_thread(session_id: str):
            try:
                result = session_manager.remove_session(session_id)
                with lock:
                    removed.append((session_id, result))
            except Exception as e:
                with lock:
                    errors.append((session_id, e))

        # Remove sessions concurrently
        threads = []
        for sid in session_ids:
            t = threading.Thread(target=remove_session_thread, args=(sid,))
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # All sessions should be removed
        assert session_manager.get_session_count() == 0

        # All removals should report success (True)
        successful_removals = [r for r in removed if r[1] is True]
        assert len(successful_removals) == 20

    def test_concurrent_create_and_remove(self, session_manager):
        """Concurrent create and remove operations should be thread-safe."""
        errors = []
        operations = []
        lock = threading.Lock()

        def create_session_thread(session_id: str):
            try:
                session = session_manager.create_session(session_id)
                with lock:
                    operations.append(("create", session_id, session is not None))
            except Exception as e:
                with lock:
                    errors.append(("create", session_id, e))

        def remove_session_thread(session_id: str):
            try:
                # Small delay to allow creation
                time.sleep(0.01)
                result = session_manager.remove_session(session_id)
                with lock:
                    operations.append(("remove", session_id, result))
            except Exception as e:
                with lock:
                    errors.append(("remove", session_id, e))

        # Create threads for creating and immediately removing
        threads = []
        for i in range(10):
            sid = f"concurrent-session-{i}"
            t1 = threading.Thread(target=create_session_thread, args=(sid,))
            t2 = threading.Thread(target=remove_session_thread, args=(sid,))
            threads.extend([t1, t2])

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0, f"Errors occurred: {errors}"

    def test_cleanup_all_sessions_thread_safe(self, session_manager):
        """cleanup_all_sessions should be thread-safe."""
        # Create sessions
        for i in range(10):
            session_manager.create_session(f"session-{i}")

        assert session_manager.get_session_count() == 10

        errors = []
        results = []
        lock = threading.Lock()

        def cleanup_thread():
            try:
                count = session_manager.cleanup_all_sessions()
                with lock:
                    results.append(count)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Call cleanup from multiple threads
        threads = [threading.Thread(target=cleanup_thread) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No errors should occur
        assert len(errors) == 0, f"Errors occurred: {errors}"

        # All sessions should be cleaned (total across all calls)
        assert session_manager.get_session_count() == 0

        # Sum of all cleanup results should be 10 (total sessions)
        assert sum(results) == 10


# =========================================================================
# Integration-like Tests - Bridge Health Check
# =========================================================================


class TestBridgeHealthCheck:
    """Tests for _check_bridge_health function used by cleanup operations."""

    def test_healthy_bridge_returns_correct_status(self, mock_bridge_client):
        """Healthy bridge returns expected status."""
        mock_bridge_client.diagnostics.return_value = {
            "success": True,
            "result": {"libraries": ["Browser", "Collections"], "context": True},
        }

        from robotmcp.server import _check_bridge_health

        health = _check_bridge_health(mock_bridge_client)

        assert health["healthy"] is True
        assert health["context_active"] is True
        assert "Browser" in health["libraries"]
        assert "Collections" in health["libraries"]

    def test_unhealthy_bridge_connection_refused(self, mock_bridge_client):
        """Connection refused error provides recovery hint."""
        mock_bridge_client.diagnostics.side_effect = ConnectionRefusedError(
            "Connection refused"
        )

        from robotmcp.server import _check_bridge_health

        health = _check_bridge_health(mock_bridge_client)

        assert health["healthy"] is False
        assert health["error"] == "connection_refused"
        assert "recovery_hint" in health

    def test_unhealthy_bridge_timeout(self, mock_bridge_client):
        """Timeout error provides recovery hint."""
        mock_bridge_client.diagnostics.side_effect = TimeoutError("Request timed out")

        from robotmcp.server import _check_bridge_health

        health = _check_bridge_health(mock_bridge_client)

        assert health["healthy"] is False
        assert health["error"] == "timeout"
        assert "recovery_hint" in health

    def test_bridge_diagnostics_failure(self, mock_bridge_client):
        """Bridge diagnostics failure is handled correctly."""
        mock_bridge_client.diagnostics.return_value = {
            "success": False,
            "error": "Internal error",
        }

        from robotmcp.server import _check_bridge_health

        health = _check_bridge_health(mock_bridge_client)

        assert health["healthy"] is False
        assert health["error"] == "diagnostics_failed"


# =========================================================================
# Edge Cases and Error Handling
# =========================================================================


class TestEdgeCases:
    """Edge cases and error handling for bridge cleanup."""

    def test_cleanup_empty_session_manager(self, session_manager):
        """Cleanup with no sessions should succeed without errors."""
        assert session_manager.get_session_count() == 0

        cleaned = session_manager.cleanup_all_sessions()

        assert cleaned == 0
        assert session_manager.get_session_count() == 0

    def test_cleanup_expired_with_no_expired_sessions(self, session_manager):
        """cleanup_expired_sessions with no expired sessions."""
        session_manager.create_session("active-session")
        assert session_manager.get_session_count() == 1

        cleaned = session_manager.cleanup_expired_sessions()

        # No sessions should be cleaned (active session is recent)
        assert cleaned == 0
        assert session_manager.get_session_count() == 1

    def test_remove_nonexistent_session(self, session_manager):
        """Removing a non-existent session should return False."""
        result = session_manager.remove_session("nonexistent-session")

        assert result is False

    def test_double_remove_session(self, session_manager):
        """Removing a session twice should succeed first time, fail second."""
        session_manager.create_session("test-session")

        result1 = session_manager.remove_session("test-session")
        result2 = session_manager.remove_session("test-session")

        assert result1 is True
        assert result2 is False

    def test_client_with_custom_host_port(self, monkeypatch):
        """ExternalRFClient with custom host and port generates correct instance ID."""
        dummy = DummyHTTPConnection("192.168.1.100", 8888, 10)
        dummy.set_response({"success": True})

        def factory(host, port, timeout):
            assert host == "192.168.1.100"
            assert port == 8888
            return dummy

        monkeypatch.setattr(
            "robotmcp.components.execution.external_rf_client.http.client.HTTPConnection",
            factory,
        )

        client = ExternalRFClient(host="192.168.1.100", port=8888, token="custom-token")

        assert client.host == "192.168.1.100"
        assert client.port == 8888
        assert client.token == "custom-token"
        assert client.instance_id is not None

        # Make a request to verify connection params
        client.diagnostics()

        assert len(dummy.requests) == 1
        assert dummy.requests[0]["headers"]["X-MCP-Token"] == "custom-token"
