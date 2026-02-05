"""Tests for Debug Bridge Unification improvements.

This test module provides comprehensive coverage for the Debug Bridge Unification
feature, which ensures seamless integration between the MCP server and the attach
bridge for debugging Robot Framework test sessions.

Test Coverage:
    Phase 1 - Session Auto-Initialization
        - Session auto-creation when bridge is active
        - Session info payload with bridge (no local session)
        - Session info payload fails gracefully without bridge/session
        - Session manager thread safety for concurrent operations

    Phase 2 - McpAttach Bridge Methods
        - get_page_source for Browser Library
        - get_page_source fallback to SeleniumLibrary
        - get_aria_snapshot success for Browser Library
        - get_aria_snapshot graceful failure for non-Browser
        - get_session_info retrieval
        - External RF client method coverage

    Phase 3 - Page Source Payload Bridge Integration
        - Bridge-first routing for page source
        - Local filtering applied to bridge data
        - ARIA snapshot with Browser Library
        - Graceful ARIA skip for SeleniumLibrary
        - Fallback to local execution when bridge fails
"""

import concurrent.futures
import json
import threading
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from robotmcp.attach import mcp_attach
from robotmcp.components.execution.external_rf_client import ExternalRFClient
from robotmcp.components.execution.session_manager import SessionManager


# ---------------------------------------------------------------------------
# Test Helpers / Fixtures
# ---------------------------------------------------------------------------


class StubBuiltIn:
    """Lightweight stand-in for robot.libraries.BuiltIn.BuiltIn."""

    def __init__(self, storage: Dict[str, Any] | None = None):
        self.storage = storage if storage is not None else {}
        self.calls: list = []
        self._keyword_results: Dict[str, Any] = {}

    def set_test_variable(self, name: str, value: Any) -> None:
        self.storage[name] = value
        self.calls.append(("set_test_variable", name, value))

    def set_suite_variable(self, name: str, value: Any) -> None:
        self.storage[name] = value
        self.calls.append(("set_suite_variable", name, value))

    def set_global_variable(self, name: str, value: Any) -> None:
        self.storage[name] = value
        self.calls.append(("set_global_variable", name, value))

    def get_variables(self) -> Dict[str, Any]:
        return dict(self.storage)

    def run_keyword(self, name: str, *args: Any) -> Any:
        self.calls.append(("run_keyword", name, args))
        if name in self._keyword_results:
            result = self._keyword_results[name]
            if callable(result):
                return result(name, *args)
            if isinstance(result, Exception):
                raise result
            return result
        return {"keyword": name, "args": list(args)}

    def log_to_console(self, _message: str) -> None:
        return None

    def set_keyword_result(self, keyword: str, result: Any) -> None:
        """Configure a result for a specific keyword."""
        self._keyword_results[keyword] = result


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
    """Create a mock ExternalRFClient for testing bridge routing."""
    client = MagicMock(spec=ExternalRFClient)

    # Set client attributes (used by health check and error messages)
    client.host = "127.0.0.1"
    client.port = 7317

    # Default diagnostics response
    client.diagnostics.return_value = {
        "success": True,
        "result": {
            "libraries": ["Browser", "Collections", "String"],
            "context": True,
        },
    }

    # Default get_variables response
    client.get_variables.return_value = {
        "success": True,
        "result": {
            "${BASE_URL}": "https://example.com",
            "${TIMEOUT}": "30",
        },
    }

    # Default get_page_source response
    client.get_page_source.return_value = {
        "success": True,
        "result": "<html><head><title>Test Page</title></head><body><h1>Hello</h1></body></html>",
        "library": "Browser",
    }

    # Default get_aria_snapshot response
    client.get_aria_snapshot.return_value = {
        "success": True,
        "result": "- heading: Hello\n- button: Submit",
        "format": "yaml",
    }

    # Default get_session_info response
    client.get_session_info.return_value = {
        "success": True,
        "result": {
            "context_active": True,
            "variable_count": 10,
            "suite_name": "TestSuite",
            "test_name": "TestCase",
            "libraries": ["Browser", "Collections"],
        },
    }

    # Default run_keyword responses
    def run_keyword_side_effect(keyword, args=None, **kwargs):
        if keyword == "Get Page Source":
            return {
                "success": True,
                "result": "<html><head><title>Test Page</title></head><body><h1>Hello</h1></body></html>",
            }
        elif keyword == "Get Source":
            return {
                "success": True,
                "result": "<html><body>Selenium Source</body></html>",
            }
        elif keyword == "Get Aria Snapshot":
            return {
                "success": True,
                "result": "- heading: Hello\n- button: Submit",
            }
        return {"success": False, "error": f"Unknown keyword: {keyword}"}

    client.run_keyword.side_effect = run_keyword_side_effect

    return client


@pytest.fixture
def session_manager():
    """Create a fresh SessionManager instance."""
    return SessionManager()


# =========================================================================
# Phase 1 Tests - Session Auto-Initialization
# =========================================================================


class TestSessionAutoInitFromBridge:
    """Phase 1: Session should be auto-created when bridge is active."""

    @pytest.mark.asyncio
    async def test_auto_init_session_from_bridge(self, mock_bridge_client, monkeypatch):
        """Session should be auto-created when bridge is active but no local session exists."""
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        # Mock session manager to return None (no existing session)
        # The session is None, so base_info will be empty dict
        monkeypatch.setattr(
            "robotmcp.server.execution_engine.session_manager.get_session",
            lambda sid: None,
        )
        monkeypatch.setattr(
            "robotmcp.server.execution_engine.session_manager.get_all_session_ids",
            lambda: [],
        )

        from robotmcp.server import _get_session_info_payload

        result = await _get_session_info_payload("auto-init-session")

        # Should succeed because bridge is active (provides attach_bridge info)
        assert result["success"] is True
        assert "session_info" in result
        assert "attach_bridge" in result["session_info"]
        assert result["session_info"]["attach_bridge"]["active"] is True

    @pytest.mark.asyncio
    async def test_session_info_payload_with_bridge_no_local_session(
        self, mock_bridge_client, monkeypatch
    ):
        """Session info should succeed with bridge even without local session."""
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        # No local session exists
        monkeypatch.setattr(
            "robotmcp.server.execution_engine.session_manager.get_session",
            lambda sid: None,
        )

        from robotmcp.server import _get_session_info_payload

        result = await _get_session_info_payload("nonexistent-session")

        assert result["success"] is True
        assert "attach_bridge" in result["session_info"]
        assert result["session_info"]["attach_bridge"]["active"] is True
        assert result["session_info"]["attach_bridge"]["libraries"] == [
            "Browser",
            "Collections",
            "String",
        ]
        assert result["session_info"]["attach_bridge"]["context_active"] is True

    @pytest.mark.asyncio
    async def test_session_info_payload_fails_without_bridge_and_session(
        self, monkeypatch
    ):
        """Session info should fail gracefully without bridge or local session."""
        # No bridge configured
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: None,
        )

        # No local session
        monkeypatch.setattr(
            "robotmcp.server.execution_engine.session_manager.get_session",
            lambda sid: None,
        )
        monkeypatch.setattr(
            "robotmcp.server.execution_engine.session_manager.get_all_session_ids",
            lambda: ["existing-session-1", "existing-session-2"],
        )

        from robotmcp.server import _get_session_info_payload

        result = await _get_session_info_payload("nonexistent-session")

        assert result["success"] is False
        assert "not found" in result["error"]
        assert "available_sessions" in result
        assert result["available_sessions"] == [
            "existing-session-1",
            "existing-session-2",
        ]

    def test_session_manager_thread_safety(self, session_manager):
        """SessionManager should handle concurrent session creation safely."""
        created_sessions = []
        errors = []

        def create_session_thread(session_id: str):
            try:
                session = session_manager.create_session(session_id)
                created_sessions.append((session_id, session))
            except Exception as e:
                errors.append((session_id, e))

        # Create sessions concurrently
        threads = []
        for i in range(20):
            t = threading.Thread(
                target=create_session_thread, args=(f"concurrent-session-{i}",)
            )
            threads.append(t)

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All sessions should be created without errors
        assert len(errors) == 0
        assert len(created_sessions) == 20

        # Verify all sessions are unique and properly initialized
        session_ids = set()
        for sid, session in created_sessions:
            assert session is not None
            assert session.session_id == sid
            assert "BuiltIn" in session.imported_libraries
            session_ids.add(sid)
        assert len(session_ids) == 20

    def test_session_manager_concurrent_get_or_create(self, session_manager):
        """get_or_create_session should be thread-safe for same session_id."""
        sessions = []
        errors = []

        def get_or_create_thread():
            try:
                session = session_manager.get_or_create_session("shared-session")
                sessions.append(session)
            except Exception as e:
                errors.append(e)

        # Multiple threads trying to get/create same session
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(get_or_create_thread) for _ in range(10)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0
        assert len(sessions) == 10
        # All should reference the same session
        assert all(s.session_id == "shared-session" for s in sessions)


# =========================================================================
# Phase 2 Tests - McpAttach Bridge Methods
# =========================================================================


class TestMcpAttachGetPageSource:
    """Phase 2: Test McpAttach._get_page_source for different libraries."""

    def test_mcpattach_get_page_source_browser(self, monkeypatch):
        """get_page_source should use Browser Library when available."""
        stub = StubBuiltIn()
        stub.set_keyword_result("Get Page Source", "<html><body>Browser Page</body></html>")
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()
        result = attach._get_page_source({})

        assert result["success"] is True
        assert result["result"] == "<html><body>Browser Page</body></html>"
        assert result["library"] == "Browser"

    def test_mcpattach_get_page_source_selenium(self, monkeypatch):
        """get_page_source should fallback to SeleniumLibrary when Browser fails."""
        stub = StubBuiltIn()
        # Make Browser's keyword fail
        stub.set_keyword_result("Get Page Source", RuntimeError("No browser context"))
        # Make Selenium's keyword succeed
        stub.set_keyword_result("Get Source", "<html><body>Selenium Page</body></html>")
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()
        result = attach._get_page_source({})

        assert result["success"] is True
        assert result["result"] == "<html><body>Selenium Page</body></html>"
        assert result["library"] == "SeleniumLibrary"


class TestMcpAttachGetAriaSnapshot:
    """Phase 2: Test McpAttach._get_aria_snapshot functionality."""

    def test_mcpattach_get_aria_snapshot_browser(self, monkeypatch):
        """get_aria_snapshot should return ARIA tree for Browser Library."""
        stub = StubBuiltIn()
        aria_content = "- heading: Test Page\n- button: Submit\n- textbox: Email"
        stub.set_keyword_result("Get Aria Snapshot", aria_content)
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()
        result = attach._get_aria_snapshot({"selector": "css=body", "format": "yaml"})

        assert result["success"] is True
        assert result["result"] == aria_content
        assert result["format"] == "yaml"
        assert result["selector"] == "css=body"
        assert result["library"] == "Browser"

    def test_mcpattach_get_aria_snapshot_not_available(self, monkeypatch):
        """get_aria_snapshot should return graceful error when not available."""
        stub = StubBuiltIn()
        stub.set_keyword_result(
            "Get Aria Snapshot",
            RuntimeError("No keyword with name 'Get Aria Snapshot' found"),
        )
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()
        result = attach._get_aria_snapshot({})

        assert result["success"] is False
        assert "not available" in result["error"]
        assert "note" in result
        assert "Browser Library" in result["note"]


class TestMcpAttachGetSessionInfo:
    """Phase 2: Test McpAttach._get_session_info functionality."""

    def test_mcpattach_get_session_info(self, monkeypatch):
        """get_session_info should return RF context information."""
        stub = StubBuiltIn()
        stub.storage = {
            "${VAR1}": "value1",
            "${VAR2}": "value2",
        }
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        # Mock execution context
        mock_lib = MagicMock()
        mock_lib.name = "Browser"
        mock_suite = MagicMock()
        mock_suite.name = "TestSuite"
        mock_test = MagicMock()
        mock_test.name = "TestCase"

        mock_ctx = MagicMock()
        mock_ctx.namespace.libraries = {"Browser": mock_lib, "Collections": MagicMock()}
        mock_ctx.suite = mock_suite
        mock_ctx.test = mock_test

        mock_contexts = MagicMock()
        mock_contexts.current = mock_ctx
        monkeypatch.setattr(mcp_attach, "EXECUTION_CONTEXTS", mock_contexts)

        attach = mcp_attach.McpAttach()
        result = attach._get_session_info({})

        assert result["success"] is True
        assert result["result"]["context_active"] is True
        assert result["result"]["variable_count"] == 2
        assert result["result"]["suite_name"] == "TestSuite"
        assert result["result"]["test_name"] == "TestCase"
        assert "Browser" in result["result"]["libraries"]

    def test_mcpattach_get_session_info_no_context(self, monkeypatch):
        """get_session_info should handle missing context gracefully."""
        mock_contexts = MagicMock()
        mock_contexts.current = None
        monkeypatch.setattr(mcp_attach, "EXECUTION_CONTEXTS", mock_contexts)

        attach = mcp_attach.McpAttach()
        result = attach._get_session_info({})

        assert result["success"] is False
        assert "No active execution context" in result["error"]


class TestExternalRFClientNewMethods:
    """Phase 2: Test ExternalRFClient method coverage for new bridge methods."""

    def _make_client_and_conn(self, monkeypatch) -> tuple:
        conn = DummyHTTPConnection("127.0.0.1", 7317, 10)

        def factory(host, port, timeout):
            return conn

        monkeypatch.setattr(
            "robotmcp.components.execution.external_rf_client.http.client.HTTPConnection",
            factory,
        )
        client = ExternalRFClient(token="test-token")
        return client, conn

    def test_external_rf_client_diagnostics(self, monkeypatch):
        """diagnostics should call /diagnostics endpoint."""
        client, conn = self._make_client_and_conn(monkeypatch)
        conn.set_response(
            {
                "success": True,
                "result": {"libraries": ["Browser"], "context": True},
            }
        )

        result = client.diagnostics()

        assert result["success"] is True
        assert len(conn.requests) == 1
        assert conn.requests[0]["path"] == "/diagnostics"

    def test_external_rf_client_run_keyword_with_args(self, monkeypatch):
        """run_keyword should send keyword name and arguments."""
        client, conn = self._make_client_and_conn(monkeypatch)
        conn.set_response({"success": True, "result": "page source content"})

        result = client.run_keyword("Get Page Source", [])

        assert result["success"] is True
        assert conn.requests[0]["body"]["name"] == "Get Page Source"
        assert conn.requests[0]["body"]["args"] == []

    def test_external_rf_client_run_keyword_with_timeout(self, monkeypatch):
        """run_keyword should include timeout_ms when provided."""
        client, conn = self._make_client_and_conn(monkeypatch)
        conn.set_response({"success": True, "result": "done"})

        client.run_keyword("Click", ["//button"], timeout_ms=5000)

        body = conn.requests[0]["body"]
        assert body["timeout_ms"] == 5000

    def test_external_rf_client_import_variables(self, monkeypatch):
        """import_variables should call /import_variables endpoint."""
        client, conn = self._make_client_and_conn(monkeypatch)
        conn.set_response({"success": True, "result": {"variable_file": "/vars.py"}})

        result = client.import_variables("/vars.py", args=["arg1", "arg2"])

        assert result["success"] is True
        body = conn.requests[0]["body"]
        assert body["variable_file_path"] == "/vars.py"
        assert body["args"] == ["arg1", "arg2"]

    def test_external_rf_client_set_variable_with_scope(self, monkeypatch):
        """set_variable should include scope parameter."""
        client, conn = self._make_client_and_conn(monkeypatch)
        conn.set_response({"success": True, "scope": "suite"})

        result = client.set_variable("MY_VAR", "value", scope="suite")

        assert result["success"] is True
        body = conn.requests[0]["body"]
        assert body["name"] == "MY_VAR"
        assert body["value"] == "value"
        assert body["scope"] == "suite"

    def test_external_rf_client_stop(self, monkeypatch):
        """stop should call /stop endpoint."""
        client, conn = self._make_client_and_conn(monkeypatch)
        conn.set_response({"success": True})

        result = client.stop()

        assert result["success"] is True
        assert conn.requests[0]["path"] == "/stop"


# =========================================================================
# Phase 3 Tests - Page Source Payload Bridge Integration
# =========================================================================


class TestPageSourcePayloadBridgeRouting:
    """Phase 3: Test page source payload uses bridge-first routing."""

    @pytest.mark.asyncio
    async def test_page_source_payload_uses_bridge(self, mock_bridge_client, monkeypatch):
        """Page source should be retrieved via bridge when available."""
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        from robotmcp.server import _get_page_source_payload

        result = await _get_page_source_payload(session_id="test-session")

        assert result["success"] is True
        assert result["source"] == "attach_bridge"
        assert "page_source" in result
        assert "library" in result
        mock_bridge_client.get_page_source.assert_called()

    @pytest.mark.asyncio
    async def test_page_source_payload_applies_filtering(
        self, mock_bridge_client, monkeypatch
    ):
        """Local filtering should be applied to bridge-returned page source."""
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        # Mock the filter function
        mock_filter = MagicMock(return_value="<html>Filtered Content</html>")
        monkeypatch.setattr(
            "robotmcp.components.execution.page_source_service.PageSourceService.filter_page_source",
            mock_filter,
        )

        from robotmcp.server import _get_page_source_payload

        result = await _get_page_source_payload(
            session_id="test-session",
            filtered=True,
            filtering_level="aggressive",
        )

        assert result["success"] is True
        assert result["metadata"]["filtered"] is True
        assert result["metadata"]["full"] is False
        mock_filter.assert_called_once()
        # Verify filtering_level was passed
        assert mock_filter.call_args[0][1] == "aggressive"

    @pytest.mark.asyncio
    async def test_page_source_payload_aria_snapshot_browser(
        self, mock_bridge_client, monkeypatch
    ):
        """ARIA snapshot should be retrieved for Browser Library."""
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        from robotmcp.server import _get_page_source_payload

        result = await _get_page_source_payload(
            session_id="test-session",
            include_reduced_dom=True,
        )

        assert result["success"] is True
        assert result["aria_snapshot"]["success"] is True
        assert result["aria_snapshot"]["content"] == "- heading: Hello\n- button: Submit"
        assert result["aria_snapshot"]["format"] == "yaml"
        assert result["aria_snapshot"]["source"] == "attach_bridge"

    @pytest.mark.asyncio
    async def test_page_source_payload_aria_not_available_selenium(
        self, mock_bridge_client, monkeypatch
    ):
        """ARIA snapshot should be skipped gracefully for SeleniumLibrary."""
        # Configure mock to return SeleniumLibrary response
        mock_bridge_client.get_page_source.return_value = {
            "success": True,
            "result": "<html>Selenium</html>",
            "library": "SeleniumLibrary",
        }

        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        from robotmcp.server import _get_page_source_payload

        result = await _get_page_source_payload(
            session_id="test-session",
            include_reduced_dom=True,
        )

        assert result["success"] is True
        assert result["page_source"] == "<html>Selenium</html>"
        assert result["library"] == "SeleniumLibrary"
        assert result["aria_snapshot"]["success"] is False
        assert result["aria_snapshot"]["error"] == "aria_not_available"
        assert "ARIA snapshots are only available with Browser Library" in result["aria_snapshot"]["note"]

    @pytest.mark.asyncio
    async def test_page_source_payload_aria_skipped_when_disabled(
        self, mock_bridge_client, monkeypatch
    ):
        """ARIA snapshot should be skipped when include_reduced_dom=False."""
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        from robotmcp.server import _get_page_source_payload

        result = await _get_page_source_payload(
            session_id="test-session",
            include_reduced_dom=False,
        )

        assert result["success"] is True
        assert result["aria_snapshot"].get("skipped") is True
        assert "content" not in result["aria_snapshot"]

    @pytest.mark.asyncio
    async def test_page_source_payload_fallback_to_local(self, monkeypatch):
        """Should fallback to local execution when bridge fails."""
        # Create a mock client that always fails
        mock_client = MagicMock()
        mock_client.get_page_source.return_value = {
            "success": False,
            "error": "Connection refused",
        }

        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_client,
        )

        # Mock local execution
        mock_local_result = {
            "success": True,
            "page_source": "<html>Local Page</html>",
            "source": "local",
        }

        async def mock_get_page_source(*args, **kwargs):
            return mock_local_result

        monkeypatch.setattr(
            "robotmcp.server.execution_engine.get_page_source",
            mock_get_page_source,
        )

        from robotmcp.server import _get_page_source_payload

        result = await _get_page_source_payload(session_id="test-session")

        # Should fallback to local
        assert result["success"] is True
        assert result["source"] == "local"

    @pytest.mark.asyncio
    async def test_page_source_payload_no_bridge_uses_local(self, monkeypatch):
        """Should use local execution when no bridge is configured."""
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: None,
        )

        mock_local_result = {
            "success": True,
            "page_source": "<html>Local Only</html>",
            "source": "local",
        }

        async def mock_get_page_source(*args, **kwargs):
            return mock_local_result

        monkeypatch.setattr(
            "robotmcp.server.execution_engine.get_page_source",
            mock_get_page_source,
        )

        from robotmcp.server import _get_page_source_payload

        result = await _get_page_source_payload(session_id="test-session")

        assert result["success"] is True
        assert result["source"] == "local"


# =========================================================================
# Integration Tests - Full Bridge Workflow
# =========================================================================


class TestBridgeIntegration:
    """Integration tests for complete bridge workflows."""

    @pytest.mark.asyncio
    async def test_full_session_state_via_bridge(self, mock_bridge_client, monkeypatch):
        """Complete session state retrieval should use bridge for all sections."""
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_bridge_client,
        )

        # Mock session
        mock_session = MagicMock()
        mock_session.get_session_info.return_value = {"session_id": "test"}
        monkeypatch.setattr(
            "robotmcp.server.execution_engine.session_manager.get_session",
            lambda sid: mock_session,
        )

        # Mock instruction hooks
        monkeypatch.setattr(
            "robotmcp.server._track_tool_result",
            lambda *a, **kw: None,
        )

        from robotmcp.server import get_session_state

        get_state_fn = get_session_state.fn

        result = await get_state_fn(
            session_id="test",
            sections=["summary", "page_source", "variables", "libraries"],
        )

        assert result["success"] is True

        sections = result["sections"]

        # summary should have attach_bridge info
        assert "attach_bridge" in sections["summary"]["session_info"]
        assert sections["summary"]["session_info"]["attach_bridge"]["active"] is True

        # page_source should come from bridge
        assert sections["page_source"]["source"] == "attach_bridge"

        # variables should come from bridge
        assert sections["variables"]["source"] == "attach_bridge"

        # libraries should come from bridge
        assert sections["libraries"]["source"] == "attach_bridge"

    def test_bridge_client_connection_failure_handling(self, monkeypatch):
        """Bridge client should handle connection failures gracefully."""

        def factory(host, port, timeout):
            raise ConnectionRefusedError("Connection refused")

        monkeypatch.setattr(
            "robotmcp.components.execution.external_rf_client.http.client.HTTPConnection",
            factory,
        )

        client = ExternalRFClient()
        result = client.diagnostics()

        assert result["success"] is False
        assert "connection error" in result["error"]

    def test_bridge_client_timeout_handling(self, monkeypatch):
        """Bridge client should handle timeouts correctly."""

        class SlowConnection:
            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                import time
                time.sleep(0.1)

            def getresponse(self):
                raise TimeoutError("Request timed out")

            def close(self):
                pass

        monkeypatch.setattr(
            "robotmcp.components.execution.external_rf_client.http.client.HTTPConnection",
            SlowConnection,
        )

        client = ExternalRFClient()
        result = client.run_keyword("Slow Keyword", [])

        assert result["success"] is False
        assert "error" in result


# ===========================================================================
# Phase 4 Tests - Session Resilience Features
# ===========================================================================


class TestBridgeHealthCheck:
    """Tests for _check_bridge_health() function (Phase 4)."""

    def test_healthy_bridge_with_context(self, mock_bridge_client):
        """Healthy bridge with active context returns correct status."""
        from robotmcp.server import _check_bridge_health

        result = _check_bridge_health(mock_bridge_client)

        assert result["healthy"] is True
        assert result["context_active"] is True
        assert result["libraries"] == ["Browser", "Collections", "String"]

    def test_healthy_bridge_without_context(self, mock_bridge_client):
        """Healthy bridge without active context."""
        mock_bridge_client.diagnostics.return_value = {
            "success": True,
            "result": {
                "libraries": [],
                "context": False,
            },
        }
        from robotmcp.server import _check_bridge_health

        result = _check_bridge_health(mock_bridge_client)

        assert result["healthy"] is True
        assert result["context_active"] is False

    def test_bridge_diagnostics_failure(self, mock_bridge_client):
        """Bridge returns diagnostics failure."""
        mock_bridge_client.diagnostics.return_value = {
            "success": False,
            "error": "Internal error",
        }
        from robotmcp.server import _check_bridge_health

        result = _check_bridge_health(mock_bridge_client)

        assert result["healthy"] is False
        assert result["error"] == "diagnostics_failed"
        assert "recovery_hint" in result

    def test_connection_refused_error(self, mock_bridge_client):
        """Connection refused error provides helpful recovery hint."""
        mock_bridge_client.diagnostics.side_effect = ConnectionRefusedError(
            "Connection refused"
        )
        from robotmcp.server import _check_bridge_health

        result = _check_bridge_health(mock_bridge_client)

        assert result["healthy"] is False
        assert result["error"] == "connection_refused"
        assert "McpAttach" in result["recovery_hint"]

    def test_timeout_error(self, mock_bridge_client):
        """Timeout error provides helpful recovery hint."""
        mock_bridge_client.diagnostics.side_effect = TimeoutError("Request timed out")
        from robotmcp.server import _check_bridge_health

        result = _check_bridge_health(mock_bridge_client)

        assert result["healthy"] is False
        assert result["error"] == "timeout"
        assert "blocked" in result["recovery_hint"] or "overloaded" in result["recovery_hint"]

    def test_dns_resolution_failure(self, mock_bridge_client):
        """DNS resolution failure provides helpful recovery hint."""
        mock_bridge_client.diagnostics.side_effect = Exception(
            "Name or service not known"
        )
        from robotmcp.server import _check_bridge_health

        result = _check_bridge_health(mock_bridge_client)

        assert result["healthy"] is False
        assert result["error"] == "dns_resolution_failed"
        assert "ROBOTMCP_ATTACH_HOST" in result["recovery_hint"]

    def test_network_unreachable(self, mock_bridge_client):
        """Network unreachable provides helpful recovery hint."""
        mock_bridge_client.diagnostics.side_effect = Exception(
            "Network is unreachable"
        )
        from robotmcp.server import _check_bridge_health

        result = _check_bridge_health(mock_bridge_client)

        assert result["healthy"] is False
        assert result["error"] == "network_unreachable"
        assert "connectivity" in result["recovery_hint"]

    def test_unknown_error(self, mock_bridge_client):
        """Unknown error provides generic recovery hint."""
        mock_bridge_client.diagnostics.side_effect = Exception("Something weird happened")
        from robotmcp.server import _check_bridge_health

        result = _check_bridge_health(mock_bridge_client)

        assert result["healthy"] is False
        assert "Something weird happened" in result["error"]
        assert "recovery_hint" in result


class TestBidirectionalSessionSync:
    """Tests for _sync_session_bidirectional() function (Phase 4)."""

    @pytest.mark.asyncio
    async def test_sync_from_bridge_libraries(self, mock_bridge_client, monkeypatch):
        """Sync libraries from bridge to local session."""
        # Create a mock session
        mock_session = MagicMock()
        mock_session.imported_libraries = []
        mock_session.loaded_libraries = set()
        mock_session.variables = {}

        from robotmcp.server import _sync_session_bidirectional

        result = await _sync_session_bidirectional(
            session_id="test",
            client=mock_bridge_client,
            session=mock_session,
            direction="from_bridge",
        )

        assert result["success"] is True
        assert "libraries_from_bridge" in result["synced"]
        assert "Browser" in mock_session.imported_libraries
        assert "Browser" in mock_session.loaded_libraries

    @pytest.mark.asyncio
    async def test_sync_from_bridge_variables(self, mock_bridge_client, monkeypatch):
        """Sync variables from bridge to local session."""
        mock_session = MagicMock()
        mock_session.imported_libraries = []
        mock_session.loaded_libraries = set()
        mock_session.variables = {}

        from robotmcp.server import _sync_session_bidirectional

        result = await _sync_session_bidirectional(
            session_id="test",
            client=mock_bridge_client,
            session=mock_session,
            direction="from_bridge",
        )

        assert result["success"] is True
        assert "variables_from_bridge" in result["synced"]
        # Variables should have ${} stripped
        assert "BASE_URL" in mock_session.variables
        assert mock_session.variables["BASE_URL"] == "https://example.com"

    @pytest.mark.asyncio
    async def test_sync_to_bridge_variables(self, mock_bridge_client, monkeypatch):
        """Push local variables to bridge."""
        mock_session = MagicMock()
        mock_session.imported_libraries = []
        mock_session.loaded_libraries = set()
        mock_session.variables = {
            "LOCAL_VAR": "value1",
            "ANOTHER_VAR": "value2",
        }

        mock_bridge_client.set_variable.return_value = {"success": True}

        from robotmcp.server import _sync_session_bidirectional

        result = await _sync_session_bidirectional(
            session_id="test",
            client=mock_bridge_client,
            session=mock_session,
            direction="to_bridge",
        )

        assert result["success"] is True
        assert "variables_to_bridge" in result["synced"]
        assert result["variables_pushed_count"] == 2

        # Verify set_variable was called for each variable
        assert mock_bridge_client.set_variable.call_count == 2

    @pytest.mark.asyncio
    async def test_sync_both_directions(self, mock_bridge_client, monkeypatch):
        """Sync in both directions."""
        mock_session = MagicMock()
        mock_session.imported_libraries = []
        mock_session.loaded_libraries = set()
        mock_session.variables = {"LOCAL_VAR": "local_value"}

        mock_bridge_client.set_variable.return_value = {"success": True}

        from robotmcp.server import _sync_session_bidirectional

        result = await _sync_session_bidirectional(
            session_id="test",
            client=mock_bridge_client,
            session=mock_session,
            direction="both",
        )

        assert result["success"] is True
        assert "libraries_from_bridge" in result["synced"]
        assert "variables_from_bridge" in result["synced"]
        assert "variables_to_bridge" in result["synced"]

    @pytest.mark.asyncio
    async def test_sync_handles_push_errors(self, mock_bridge_client, monkeypatch):
        """Sync reports push errors without failing."""
        mock_session = MagicMock()
        mock_session.imported_libraries = []
        mock_session.loaded_libraries = set()
        mock_session.variables = {
            "VAR1": "value1",
            "VAR2": "value2",
        }

        # First push succeeds, second fails
        mock_bridge_client.set_variable.side_effect = [
            {"success": True},
            {"success": False, "error": "Read-only variable"},
        ]

        from robotmcp.server import _sync_session_bidirectional

        result = await _sync_session_bidirectional(
            session_id="test",
            client=mock_bridge_client,
            session=mock_session,
            direction="to_bridge",
        )

        assert result["success"] is True
        assert result["variables_pushed_count"] == 1
        assert "push_errors" in result
        assert len(result["push_errors"]) == 1

    @pytest.mark.asyncio
    async def test_sync_handles_exception(self, mock_bridge_client, monkeypatch):
        """Sync handles unexpected exceptions gracefully."""
        mock_session = MagicMock()
        mock_session.imported_libraries = []
        mock_session.loaded_libraries = set()
        mock_session.variables = {}

        mock_bridge_client.diagnostics.side_effect = Exception("Network error")

        from robotmcp.server import _sync_session_bidirectional

        result = await _sync_session_bidirectional(
            session_id="test",
            client=mock_bridge_client,
            session=mock_session,
            direction="from_bridge",
        )

        assert result["success"] is False
        assert "error" in result


class TestContextVariablesPayloadWithSessionSync:
    """Tests for _get_context_variables_payload using unified session sync (Phase 4)."""

    @pytest.mark.asyncio
    async def test_variables_payload_uses_session_sync(self, mock_bridge_client, monkeypatch):
        """Variables payload should use _get_external_client_with_session_sync."""
        # Mock to track if unified function is used
        sync_calls = []

        def mock_get_external_with_sync(session_id):
            sync_calls.append(session_id)
            return mock_bridge_client, MagicMock()

        monkeypatch.setattr(
            "robotmcp.server._get_external_client_with_session_sync",
            mock_get_external_with_sync,
        )

        from robotmcp.server import _get_context_variables_payload

        result = await _get_context_variables_payload("test-session")

        # Should have called the unified sync function
        assert len(sync_calls) == 1
        assert sync_calls[0] == "test-session"

    @pytest.mark.asyncio
    async def test_variables_payload_improved_error_message(self, monkeypatch):
        """Variables payload should provide helpful error when no session/bridge."""

        def mock_get_external_with_sync(session_id):
            return None, None  # No bridge, no session

        monkeypatch.setattr(
            "robotmcp.server._get_external_client_with_session_sync",
            mock_get_external_with_sync,
        )

        from robotmcp.server import _get_context_variables_payload

        result = await _get_context_variables_payload("nonexistent")

        assert result["success"] is False
        assert "no active attach bridge" in result["error"]
        assert "hint" in result
