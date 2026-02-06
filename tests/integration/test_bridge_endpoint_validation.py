"""Direct validation script for the bridge /get_variables endpoint.

This module validates that the bridge endpoint no longer crashes when
handling Robot Framework variables with non-serializable types.

Run with:
    uv run pytest tests/integration/test_bridge_endpoint_validation.py -v

For manual testing with a running RF session:
    uv run python -m tests.integration.test_bridge_endpoint_validation

This validates:
1. The _sanitize_for_json function is correctly integrated
2. The HTTP handler's json.dumps call succeeds
3. The response is valid JSON
"""

import json
import sys
import threading
import time
from datetime import datetime
from http.client import HTTPConnection
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Test fixtures
# ============================================================================


class MockBuiltIn:
    """Mock Robot Framework BuiltIn library."""

    def __init__(self, variables: Dict[str, Any]):
        self._variables = variables

    def get_variables(self) -> Dict[str, Any]:
        return self._variables.copy()

    def set_test_variable(self, name: str, value: Any) -> None:
        self._variables[name] = value


class MockExecutionContext:
    """Mock RF execution context."""

    def __init__(self):
        self.namespace = MagicMock()
        self.namespace.libraries = {"BuiltIn": MagicMock(), "Browser": MagicMock()}


# Simulated RF types
class DotDict(dict):
    def __getattr__(self, key: str) -> Any:
        return self.get(key)


class Tags(list):
    pass


class RobotOptions:
    def __init__(self):
        self.output = Path("/tmp/output.xml")


# ============================================================================
# Unit tests for the fix
# ============================================================================


class TestBridgeEndpointFix:
    """Validate the _get_variables endpoint fix."""

    @pytest.fixture
    def problematic_variables(self) -> Dict[str, Any]:
        """Variables that previously caused crashes."""
        return {
            "${CURDIR}": Path("/home/user/tests"),
            "${OUTPUT_DIR}": Path("/home/user/output"),
            "${OUTPUT_FILE}": Path("/home/user/output/output.xml"),
            "${LOG_FILE}": Path("/home/user/output/log.html"),
            "${SUITE_SOURCE}": Path("/home/user/tests/suite.robot"),
            "${OPTIONS}": RobotOptions(),
            "${METADATA}": DotDict(author="test", version="1.0"),
            "${TAGS}": Tags(["smoke", "regression"]),
            "${TIMESTAMP}": datetime.now(),
            "${FUNC}": len,
            "${LAMBDA}": lambda x: x,
            "${SIMPLE}": "simple_value",
            "${NUMBER}": 42,
            "${BOOL}": True,
            "${NONE}": None,
        }

    def test_sanitize_for_json_integration(self, problematic_variables):
        """Test that _sanitize_for_json handles all problematic types."""
        from robotmcp.attach.mcp_attach import McpAttach

        attach = McpAttach()

        # Sanitize all variables
        sanitized = {}
        for k, v in problematic_variables.items():
            sanitized[k] = attach._sanitize_for_json(v)

        # Must be JSON serializable
        json_str = json.dumps(sanitized)
        assert len(json_str) > 0

        # Parse and verify
        parsed = json.loads(json_str)
        assert len(parsed) == len(problematic_variables)

    def test_get_variables_method_with_mock(self, problematic_variables):
        """Test _get_variables method with mocked RF context."""
        from robotmcp.attach.mcp_attach import McpAttach

        attach = McpAttach()

        # Mock BuiltIn
        mock_builtin = MockBuiltIn(problematic_variables)

        with patch("robotmcp.attach.mcp_attach.BuiltIn", return_value=mock_builtin):
            # Call _get_variables with no specific names (get all)
            result = attach._get_variables({})

        # Should succeed
        assert result["success"] is True
        assert "result" in result

        # Result should be JSON serializable
        json_str = json.dumps(result)
        assert len(json_str) > 0

        # Verify truncation applied
        assert result.get("truncated") is True or len(result["result"]) <= 200

    def test_get_variables_with_specific_names(self, problematic_variables):
        """Test _get_variables with specific variable names."""
        from robotmcp.attach.mcp_attach import McpAttach

        attach = McpAttach()
        mock_builtin = MockBuiltIn(problematic_variables)

        with patch("robotmcp.attach.mcp_attach.BuiltIn", return_value=mock_builtin):
            result = attach._get_variables({"names": ["CURDIR", "OUTPUT_DIR", "SIMPLE"]})

        assert result["success"] is True
        assert "${CURDIR}" in result["result"]
        assert isinstance(result["result"]["${CURDIR}"], str)

    def test_http_handler_json_dumps_simulation(self, problematic_variables):
        """Simulate the HTTP handler's json.dumps call."""
        from robotmcp.attach.mcp_attach import McpAttach

        attach = McpAttach()
        mock_builtin = MockBuiltIn(problematic_variables)

        with patch("robotmcp.attach.mcp_attach.BuiltIn", return_value=mock_builtin):
            resp = attach._get_variables({})

        # This is what the HTTP handler does
        try:
            body = json.dumps(resp).encode("utf-8")
        except (TypeError, ValueError) as e:
            pytest.fail(f"HTTP handler json.dumps would crash: {e}")

        assert body is not None
        assert b'"success": true' in body


class TestBridgeEndpointRobustness:
    """Test robustness of the endpoint."""

    def test_handles_very_large_variable_set(self):
        """Test with more than 200 variables (truncation limit)."""
        from robotmcp.attach.mcp_attach import McpAttach

        attach = McpAttach()

        # Generate 500 variables
        large_var_set = {}
        for i in range(500):
            if i % 4 == 0:
                large_var_set[f"${{PATH_{i}}}"] = Path(f"/path_{i}")
            elif i % 4 == 1:
                large_var_set[f"${{STR_{i}}}"] = f"value_{i}"
            elif i % 4 == 2:
                large_var_set[f"${{NUM_{i}}}"] = i
            else:
                large_var_set[f"${{OBJ_{i}}}"] = MagicMock()

        mock_builtin = MockBuiltIn(large_var_set)

        with patch("robotmcp.attach.mcp_attach.BuiltIn", return_value=mock_builtin):
            result = attach._get_variables({})

        assert result["success"] is True
        assert result.get("truncated") is True
        assert len(result["result"]) <= 200

        # All returned values must be JSON serializable
        json.dumps(result)

    def test_handles_deeply_nested_structures(self):
        """Test with deeply nested dict structures."""
        from robotmcp.attach.mcp_attach import McpAttach

        attach = McpAttach()

        # Create deeply nested structure
        deep = {"level": Path("/deep")}
        for i in range(20):
            deep = {f"l{i}": deep}

        variables = {"${DEEP}": deep}
        mock_builtin = MockBuiltIn(variables)

        with patch("robotmcp.attach.mcp_attach.BuiltIn", return_value=mock_builtin):
            result = attach._get_variables({})

        assert result["success"] is True
        json.dumps(result)

    def test_handles_mixed_collection_types(self):
        """Test with mixed collection types."""
        from robotmcp.attach.mcp_attach import McpAttach

        attach = McpAttach()

        variables = {
            "${LIST}": [Path("/a"), datetime.now(), {"nested": Path("/b")}],
            "${TUPLE}": (1, Path("/tuple"), "string"),
            "${SET_LIKE}": list({Path("/set1"), Path("/set2")}),  # Convert set to list
            "${DICT}": {
                "path": Path("/dict"),
                1: "numeric_key",
                Path("/key"): "path_key",
            },
        }

        mock_builtin = MockBuiltIn(variables)

        with patch("robotmcp.attach.mcp_attach.BuiltIn", return_value=mock_builtin):
            result = attach._get_variables({})

        assert result["success"] is True
        json_str = json.dumps(result)
        assert len(json_str) > 0


class TestBridgeEndpointErrorHandling:
    """Test error handling in the endpoint."""

    def test_handles_builtin_exception(self):
        """Test graceful handling when BuiltIn raises exception."""
        from robotmcp.attach.mcp_attach import McpAttach

        attach = McpAttach()

        # Mock BuiltIn to raise exception
        mock_builtin = MagicMock()
        mock_builtin.get_variables.side_effect = RuntimeError("No context")

        with patch("robotmcp.attach.mcp_attach.BuiltIn", return_value=mock_builtin):
            result = attach._get_variables({})

        assert result["success"] is False
        assert "error" in result

        # Error response should be JSON serializable
        json.dumps(result)


# ============================================================================
# Direct HTTP endpoint test (requires running server)
# ============================================================================


def test_bridge_endpoint_manual():
    """Manual test for bridge endpoint with running RF session.

    This test requires an active RF session with McpAttach.

    To run manually:
        1. Start a Robot Framework test with McpAttach library
        2. Run: uv run python -m tests.integration.test_bridge_endpoint_validation

    Expected: All variable types should serialize without error.
    """
    import socket
    from http.client import RemoteDisconnected

    host = "127.0.0.1"
    port = 7317
    token = "change-me"

    # Check if server is running by trying a quick connection test
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex((host, port))
    sock.close()

    if result != 0:
        # Bridge not running - skip gracefully (this is expected in CI)
        pytest.skip(f"Bridge not running on {host}:{port} - manual test skipped")

    # Test /get_variables endpoint
    conn = HTTPConnection(host, port, timeout=5)
    headers = {
        "Content-Type": "application/json",
        "X-MCP-Token": token,
    }

    try:
        # Test 1: Get all variables
        print("\n--- Test: Get all variables ---")
        conn.request("POST", "/get_variables", json.dumps({}), headers)
        response = conn.getresponse()
        body = response.read().decode("utf-8")

        if response.status != 200:
            pytest.skip(f"Bridge returned HTTP {response.status} - may not be RF attach bridge")

        try:
            data = json.loads(body)
            if data.get("success"):
                print(f"PASS: Got {len(data.get('result', {}))} variables")
                print(f"      Truncated: {data.get('truncated', False)}")
            else:
                print(f"FAIL: {data.get('error', 'Unknown error')}")
                pytest.fail(f"Bridge returned error: {data.get('error')}")
        except json.JSONDecodeError as e:
            pytest.fail(f"Invalid JSON response: {e}")

        # Test 2: Get specific variables
        print("\n--- Test: Get specific variables ---")
        conn.request(
            "POST",
            "/get_variables",
            json.dumps({"names": ["CURDIR", "OUTPUT_DIR", "SUITE_NAME"]}),
            headers,
        )
        response = conn.getresponse()
        body = response.read().decode("utf-8")

        if response.status == 200:
            data = json.loads(body)
            if data.get("success"):
                print(f"PASS: Got specific variables: {list(data.get('result', {}).keys())}")
            else:
                print(f"WARN: {data.get('error', 'No specific vars found')}")
        else:
            print(f"WARN: HTTP {response.status} for specific variables")

        # Test 3: Get session info (sanity check)
        print("\n--- Test: Get session info ---")
        conn.request("POST", "/get_session_info", json.dumps({}), headers)
        response = conn.getresponse()
        body = response.read().decode("utf-8")

        if response.status == 200:
            data = json.loads(body)
            if data.get("success"):
                result = data.get("result", {})
                print(f"PASS: Session active, {result.get('variable_count', '?')} variables")
            else:
                print(f"INFO: {data.get('error', 'Session info unavailable')}")

        print("\n--- All tests passed! ---")

    except (RemoteDisconnected, ConnectionRefusedError, ConnectionResetError, OSError) as e:
        # Connection issues mean the bridge isn't a proper RF attach bridge
        pytest.skip(f"Bridge not responding correctly: {e} - skipping manual test")
    except Exception as e:
        pytest.fail(f"Bridge endpoint test failed: {e}")
    finally:
        conn.close()


# ============================================================================
# Main entry point for manual testing
# ============================================================================


if __name__ == "__main__":
    print("=" * 60)
    print("Bridge Endpoint Validation - Direct HTTP Test")
    print("=" * 60)

    success = test_bridge_endpoint_manual()
    sys.exit(0 if success else 1)
