"""Tests for BuiltIn library handling and attach bridge improvements.

Covers the following improvements:
  R1 - BuiltIn added at session creation (session_manager.py)
  R2 - analyze_scenario attach guard (server.py)
  R3 - Variable reading bridge routing (server.py)
  R4 - import_variables verb (mcp_attach.py)
  R5 - set_variable scope (mcp_attach.py, external_rf_client.py)
  R6 - manage_session init BuiltIn (server.py)
  R7 - Truncation limit 200 (mcp_attach.py)
  R8 - Timeout forwarding (mcp_attach.py, external_rf_client.py)
"""

import json
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from robotmcp.attach import mcp_attach
from robotmcp.components.execution.external_rf_client import ExternalRFClient
from robotmcp.components.execution.session_manager import SessionManager


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------


class StubBuiltIn:
    """Lightweight stand-in for robot.libraries.BuiltIn.BuiltIn."""

    def __init__(self, storage: Dict[str, Any] | None = None):
        self.storage = storage if storage is not None else {}
        self.calls: list = []

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
        return {"keyword": name, "args": list(args)}

    def log_to_console(self, _message: str) -> None:
        return None


class DummyHTTPConnection:
    """Captures HTTP requests issued by ExternalRFClient._post."""

    def __init__(self, host: str, port: int, timeout: int):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.requests: list = []
        self.closed = False
        self._response_body = b'{"success": true, "result": 1}'

    def request(self, method: str, path: str, body: bytes, headers: dict) -> None:
        self.requests.append(
            {
                "method": method,
                "path": path,
                "body": json.loads(body.decode("utf-8")) if body else {},
                "headers": headers,
            }
        )

    def getresponse(self) -> SimpleNamespace:
        body = self._response_body
        return SimpleNamespace(read=lambda: body)

    def close(self) -> None:
        self.closed = True


# =========================================================================
# Test Group 1 -- BuiltIn Library (R1 + R6)
# =========================================================================


class TestSessionCreationBuiltIn:
    """R1: BuiltIn must be in imported_libraries, loaded_libraries, and search_order."""

    def test_session_creation_includes_builtin(self):
        """After create_session(), BuiltIn is in imported_libraries, loaded_libraries, and search_order."""
        mgr = SessionManager()
        session = mgr.create_session("test-sid-1")

        assert "BuiltIn" in session.imported_libraries
        assert "BuiltIn" in session.loaded_libraries
        assert "BuiltIn" in session.search_order

    def test_session_creation_builtin_at_front_of_search_order(self):
        """BuiltIn should be at index 0 of search_order for a freshly created session."""
        mgr = SessionManager()
        session = mgr.create_session("test-sid-2")

        assert session.search_order[0] == "BuiltIn"

    def test_get_or_create_preserves_builtin(self):
        """get_or_create_session on an existing session still has BuiltIn."""
        mgr = SessionManager()
        session = mgr.create_session("test-sid-3")

        # Manually verify before get_or_create
        assert "BuiltIn" in session.imported_libraries

        # Now call get_or_create which returns the same session
        session2 = mgr.get_or_create_session("test-sid-3")

        assert session2 is session
        assert "BuiltIn" in session2.imported_libraries
        assert "BuiltIn" in session2.loaded_libraries
        assert "BuiltIn" in session2.search_order


# =========================================================================
# Test Group 2 -- ExternalRFClient (R5 + R8)
# =========================================================================


class TestExternalRFClient:
    """R5: set_variable sends scope, R8: run_keyword sends timeout_ms."""

    def _make_client_and_conn(self, monkeypatch):
        conn = DummyHTTPConnection("127.0.0.1", 7317, 10)

        def factory(host, port, timeout):
            return conn

        monkeypatch.setattr(
            "robotmcp.components.execution.external_rf_client.http.client.HTTPConnection",
            factory,
        )
        client = ExternalRFClient(token="tok")
        return client, conn

    def test_set_variable_sends_scope(self, monkeypatch):
        """set_variable includes scope in the payload."""
        client, conn = self._make_client_and_conn(monkeypatch)

        client.set_variable("MY_VAR", "hello", scope="suite")

        assert len(conn.requests) == 1
        body = conn.requests[0]["body"]
        assert body["scope"] == "suite"
        assert body["name"] == "MY_VAR"
        assert body["value"] == "hello"

    def test_set_variable_default_scope_is_test(self, monkeypatch):
        """Default scope should be 'test' when not provided."""
        client, conn = self._make_client_and_conn(monkeypatch)

        client.set_variable("X", 1)

        body = conn.requests[0]["body"]
        assert body["scope"] == "test"

    def test_run_keyword_sends_timeout_ms(self, monkeypatch):
        """run_keyword includes timeout_ms in payload when provided."""
        client, conn = self._make_client_and_conn(monkeypatch)

        client.run_keyword("Click", ["//button"], timeout_ms=5000)

        body = conn.requests[0]["body"]
        assert body["timeout_ms"] == 5000

    def test_run_keyword_no_timeout_when_none(self, monkeypatch):
        """run_keyword omits timeout_ms from payload when not provided."""
        client, conn = self._make_client_and_conn(monkeypatch)

        client.run_keyword("Log", ["hi"])

        body = conn.requests[0]["body"]
        assert "timeout_ms" not in body


# =========================================================================
# Test Group 3 -- McpAttach Bridge (R4 + R5 + R7)
# =========================================================================


class TestMcpAttachBridge:
    """R4: import_variables, R5: set_variable scope, R7: truncation at 200."""

    # ------ R4 ------
    def test_execute_command_handles_import_variables(self, monkeypatch):
        """The import_variables verb is routed to _import_variables."""
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: StubBuiltIn())

        # Create a fake RF execution context with a namespace that has import_variables
        mock_ns = MagicMock()
        mock_ns.import_variables = MagicMock(return_value=None)
        mock_ctx = MagicMock()
        mock_ctx.namespace = mock_ns

        mock_contexts = MagicMock()
        mock_contexts.current = mock_ctx
        monkeypatch.setattr(mcp_attach, "EXECUTION_CONTEXTS", mock_contexts)

        attach = mcp_attach.McpAttach()
        result = attach._execute_command(
            "import_variables",
            {"variable_file_path": "/path/to/vars.py", "args": ["arg1"]},
        )

        assert result["success"] is True
        assert result["result"]["variable_file"] == "/path/to/vars.py"
        mock_ns.import_variables.assert_called_once_with(
            "/path/to/vars.py", args=["arg1"], overwrite=True
        )

    # ------ R5 ------
    def test_set_variable_with_scope_suite(self, monkeypatch):
        """scope='suite' calls set_suite_variable."""
        stub = StubBuiltIn()
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()
        result = attach._set_variable(
            {"name": "MY_VAR", "value": "val", "scope": "suite"}
        )

        assert result["success"] is True
        assert result["scope"] == "suite"
        # Verify set_suite_variable was called (not set_test_variable)
        suite_calls = [c for c in stub.calls if c[0] == "set_suite_variable"]
        assert len(suite_calls) == 1
        assert suite_calls[0][1] == "${MY_VAR}"
        assert suite_calls[0][2] == "val"

    def test_set_variable_with_scope_global(self, monkeypatch):
        """scope='global' calls set_global_variable."""
        stub = StubBuiltIn()
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()
        result = attach._set_variable(
            {"name": "G_VAR", "value": 42, "scope": "global"}
        )

        assert result["success"] is True
        assert result["scope"] == "global"
        global_calls = [c for c in stub.calls if c[0] == "set_global_variable"]
        assert len(global_calls) == 1
        assert global_calls[0][1] == "${G_VAR}"
        assert global_calls[0][2] == 42

    def test_set_variable_default_scope_test(self, monkeypatch):
        """Default scope calls set_test_variable."""
        stub = StubBuiltIn()
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()
        result = attach._set_variable({"name": "T_VAR", "value": "abc"})

        assert result["success"] is True
        assert result["scope"] == "test"
        test_calls = [c for c in stub.calls if c[0] == "set_test_variable"]
        assert len(test_calls) == 1
        assert test_calls[0][1] == "${T_VAR}"

    # ------ R7 ------
    def test_get_variables_truncation_at_200(self, monkeypatch):
        """Truncation happens at 200 entries, not 50."""
        # Build a BuiltIn stub that returns 250 variables
        large_vars = {f"${{VAR_{i}}}": f"value_{i}" for i in range(250)}
        stub = StubBuiltIn(storage=large_vars)
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()
        result = attach._get_variables({})

        assert result["success"] is True
        # Should have exactly 200 items (truncated)
        assert len(result["result"]) == 200
        assert result["truncated"] is True


# =========================================================================
# Test Group 4 -- Server-level improvements (R2 + R3)
# =========================================================================


class TestServerAttachImprovements:
    """R2: analyze_scenario attach guard, R3: variable bridge routing."""

    @pytest.mark.asyncio
    async def test_analyze_scenario_detects_attach_bridge(self, monkeypatch):
        """When ROBOTMCP_ATTACH_HOST is set, analyze_scenario queries bridge diagnostics."""
        # Set the environment variable so _get_external_client_if_configured returns a client
        monkeypatch.setenv("ROBOTMCP_ATTACH_HOST", "127.0.0.1")
        monkeypatch.setenv("ROBOTMCP_ATTACH_PORT", "7317")
        monkeypatch.setenv("ROBOTMCP_ATTACH_TOKEN", "test-token")

        # We need to mock the actual ExternalRFClient that gets created
        mock_client = MagicMock()
        mock_client.diagnostics.return_value = {
            "success": True,
            "result": {"libraries": ["Browser", "Collections"], "context": True},
        }
        mock_client.get_variables.return_value = {
            "success": True,
            "result": {"${BASE_URL}": "http://example.com"},
        }

        # Patch _get_external_client_if_configured in the server module
        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_client,
        )

        # We also need to mock the NLP processor to avoid heavy dependencies
        mock_nlp_result = {
            "success": True,
            "analysis": {
                "session_type": "web",
                "keywords": [],
                "libraries": [],
            },
        }

        async def mock_analyze(scenario, context):
            return mock_nlp_result

        monkeypatch.setattr(
            "robotmcp.server.nlp_processor.analyze_scenario",
            mock_analyze,
        )

        # Mock sampling to be disabled at the source module
        monkeypatch.setattr(
            "robotmcp.utils.sampling.is_sampling_enabled",
            lambda: False,
        )

        # Mock the instruction hooks
        mock_hooks = MagicMock()
        monkeypatch.setattr(
            "robotmcp.server._get_instruction_hooks",
            lambda: mock_hooks,
        )
        monkeypatch.setattr(
            "robotmcp.server._track_tool_result",
            lambda *a, **kw: None,
        )

        from robotmcp import server as server_module

        # analyze_scenario is decorated with @mcp.tool, so access .fn for the
        # underlying coroutine.
        analyze_fn = server_module.analyze_scenario.fn

        result = await analyze_fn(scenario="Open browser to example.com", context="web")

        # The result should indicate attach bridge was active
        assert result["session_info"]["attach_bridge_active"] is True
        assert "attach_note" in result["session_info"]
        # Diagnostics must have been called
        mock_client.diagnostics.assert_called()

    @pytest.mark.asyncio
    async def test_get_context_variables_routes_through_bridge(self, monkeypatch):
        """When attach bridge is configured, variables should be read from bridge."""
        mock_client = MagicMock()
        mock_client.get_variables.return_value = {
            "success": True,
            "result": {
                "${MY_VAR}": "bridge_value",
                "${TIMEOUT}": "30",
            },
        }

        monkeypatch.setattr(
            "robotmcp.server._get_external_client_if_configured",
            lambda: mock_client,
        )

        from robotmcp.server import _get_context_variables_payload

        result = await _get_context_variables_payload("some-session")

        assert result["success"] is True
        assert result["source"] == "attach_bridge"
        # Variables should be stripped of ${} wrappers
        assert "MY_VAR" in result["variables"]
        assert "TIMEOUT" in result["variables"]
        assert result["variables"]["MY_VAR"] == "bridge_value"
        mock_client.get_variables.assert_called_once()
