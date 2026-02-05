"""Unit tests for McpAttach variable serialization fixes.

Tests the _sanitize_for_json helper and JSON serialization safety in the
HTTP bridge for Robot Framework variables that may contain non-serializable types.
"""

from __future__ import annotations

import json
import queue
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Dict
from unittest.mock import MagicMock, patch

import pytest

from robotmcp.attach import mcp_attach


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


class StubBuiltIn:
    """Minimal BuiltIn stub for testing."""

    def __init__(self, variables: Dict[str, Any] | None = None):
        self.variables = variables or {}

    def get_variables(self) -> Dict[str, Any]:
        return self.variables

    def set_test_variable(self, name: str, value: Any) -> None:
        self.variables[name] = value

    def log_to_console(self, _message: str) -> None:
        pass


class CustomClass:
    """A custom class to test serialization fallback."""

    def __init__(self, value: str):
        self.value = value

    def __str__(self) -> str:
        return f"CustomClass(value={self.value})"


class UnserializableClass:
    """Class with complex internal state for testing edge cases."""

    def __init__(self):
        self.internal_func = lambda x: x * 2
        self.circular_ref = None

    def __str__(self) -> str:
        return "<UnserializableClass instance>"


def custom_function(x: int) -> int:
    """A function to test serialization of callable objects."""
    return x * 2


# =============================================================================
# Tests for _sanitize_for_json helper
# =============================================================================


class TestSanitizeForJsonScalarTypes:
    """Test that scalar types pass through _sanitize_for_json unchanged."""

    def test_string_passes_through(self):
        attach = mcp_attach.McpAttach()
        assert attach._sanitize_for_json("hello") == "hello"
        assert attach._sanitize_for_json("") == ""
        assert attach._sanitize_for_json("unicode: \u00e9\u00e0\u00fc") == "unicode: \u00e9\u00e0\u00fc"

    def test_integer_passes_through(self):
        attach = mcp_attach.McpAttach()
        assert attach._sanitize_for_json(42) == 42
        assert attach._sanitize_for_json(0) == 0
        assert attach._sanitize_for_json(-100) == -100
        assert attach._sanitize_for_json(10**20) == 10**20

    def test_float_passes_through(self):
        attach = mcp_attach.McpAttach()
        assert attach._sanitize_for_json(3.14) == 3.14
        assert attach._sanitize_for_json(0.0) == 0.0
        assert attach._sanitize_for_json(-99.99) == -99.99

    def test_boolean_passes_through(self):
        attach = mcp_attach.McpAttach()
        assert attach._sanitize_for_json(True) is True
        assert attach._sanitize_for_json(False) is False

    def test_none_passes_through(self):
        attach = mcp_attach.McpAttach()
        assert attach._sanitize_for_json(None) is None


class TestSanitizeForJsonPathObjects:
    """Test that Path objects are converted to strings."""

    def test_pathlib_path_converts_to_string(self):
        attach = mcp_attach.McpAttach()
        p = Path("/home/user/test.txt")
        result = attach._sanitize_for_json(p)
        assert isinstance(result, str)
        assert "test.txt" in result

    def test_pure_posix_path_converts_to_string(self):
        attach = mcp_attach.McpAttach()
        p = PurePosixPath("/usr/local/bin")
        result = attach._sanitize_for_json(p)
        assert isinstance(result, str)
        assert result == "/usr/local/bin"

    def test_pure_windows_path_converts_to_string(self):
        attach = mcp_attach.McpAttach()
        p = PureWindowsPath("C:/Users/test")
        result = attach._sanitize_for_json(p)
        assert isinstance(result, str)
        assert "Users" in result and "test" in result


class TestSanitizeForJsonCollections:
    """Test that lists, tuples, and dicts are recursively sanitized."""

    def test_list_is_recursively_sanitized(self):
        attach = mcp_attach.McpAttach()
        input_list = ["hello", 42, Path("/test"), True, None]
        result = attach._sanitize_for_json(input_list)
        assert isinstance(result, list)
        assert result[0] == "hello"
        assert result[1] == 42
        assert isinstance(result[2], str)
        assert result[3] is True
        assert result[4] is None

    def test_tuple_is_recursively_sanitized(self):
        attach = mcp_attach.McpAttach()
        input_tuple = ("hello", Path("/test"), 3.14)
        result = attach._sanitize_for_json(input_tuple)
        assert isinstance(result, list)  # Tuples become lists
        assert result[0] == "hello"
        assert isinstance(result[1], str)
        assert result[2] == 3.14

    def test_dict_is_recursively_sanitized(self):
        attach = mcp_attach.McpAttach()
        input_dict = {
            "name": "test",
            "path": Path("/data"),
            "count": 5,
        }
        result = attach._sanitize_for_json(input_dict)
        assert isinstance(result, dict)
        assert result["name"] == "test"
        assert isinstance(result["path"], str)
        assert result["count"] == 5

    def test_dict_keys_converted_to_string(self):
        attach = mcp_attach.McpAttach()
        # Robot Framework can sometimes have non-string keys
        input_dict = {123: "value1", "key": "value2"}
        result = attach._sanitize_for_json(input_dict)
        assert "123" in result  # Integer key becomes string
        assert "key" in result
        assert result["123"] == "value1"

    def test_empty_collections(self):
        attach = mcp_attach.McpAttach()
        assert attach._sanitize_for_json([]) == []
        assert attach._sanitize_for_json(()) == []
        assert attach._sanitize_for_json({}) == {}


class TestSanitizeForJsonComplexObjects:
    """Test that complex objects fall back to str()."""

    def test_custom_class_converted_to_string(self):
        attach = mcp_attach.McpAttach()
        obj = CustomClass("test_value")
        result = attach._sanitize_for_json(obj)
        assert isinstance(result, str)
        assert "CustomClass" in result
        assert "test_value" in result

    def test_function_converted_to_string(self):
        attach = mcp_attach.McpAttach()
        result = attach._sanitize_for_json(custom_function)
        assert isinstance(result, str)
        assert "function" in result.lower() or "custom_function" in result

    def test_lambda_converted_to_string(self):
        attach = mcp_attach.McpAttach()
        fn = lambda x: x + 1  # noqa: E731
        result = attach._sanitize_for_json(fn)
        assert isinstance(result, str)
        assert "lambda" in result.lower() or "function" in result.lower()

    def test_unserializable_class_converted_to_string(self):
        attach = mcp_attach.McpAttach()
        obj = UnserializableClass()
        result = attach._sanitize_for_json(obj)
        assert isinstance(result, str)
        assert "UnserializableClass" in result


class TestSanitizeForJsonNestedStructures:
    """Test that deeply nested structures are correctly sanitized."""

    def test_nested_dict_with_path(self):
        attach = mcp_attach.McpAttach()
        nested = {
            "level1": {
                "level2": {
                    "path": Path("/deep/nested/path"),
                    "value": 42,
                }
            }
        }
        result = attach._sanitize_for_json(nested)
        assert isinstance(result["level1"]["level2"]["path"], str)
        assert result["level1"]["level2"]["value"] == 42

    def test_list_of_dicts_with_paths(self):
        attach = mcp_attach.McpAttach()
        data = [
            {"name": "file1", "path": Path("/path/to/file1")},
            {"name": "file2", "path": Path("/path/to/file2")},
        ]
        result = attach._sanitize_for_json(data)
        assert all(isinstance(item["path"], str) for item in result)
        assert result[0]["name"] == "file1"
        assert result[1]["name"] == "file2"

    def test_dict_with_list_of_paths(self):
        attach = mcp_attach.McpAttach()
        data = {"paths": [Path("/a"), Path("/b"), Path("/c")]}
        result = attach._sanitize_for_json(data)
        assert all(isinstance(p, str) for p in result["paths"])

    def test_mixed_nested_types(self):
        attach = mcp_attach.McpAttach()
        data = {
            "config": {
                "paths": [Path("/config"), Path("/data")],
                "handlers": [custom_function, lambda x: x],
                "settings": {"debug": True, "count": 10},
            },
            "metadata": CustomClass("meta"),
        }
        result = attach._sanitize_for_json(data)

        # Verify it's JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

        # Verify structure
        assert isinstance(result["config"]["paths"][0], str)
        assert isinstance(result["config"]["handlers"][0], str)
        assert result["config"]["settings"]["debug"] is True
        assert isinstance(result["metadata"], str)


class TestSanitizeForJsonJsonSerializable:
    """Test that sanitized values are actually JSON serializable."""

    def test_all_sanitized_values_are_json_serializable(self):
        attach = mcp_attach.McpAttach()
        problematic_values = [
            Path("/test/path"),
            CustomClass("test"),
            custom_function,
            lambda x: x,
            {"nested": {"path": Path("/nested")}},
            [Path("/a"), CustomClass("b")],
            (1, 2, Path("/tuple")),
        ]
        for val in problematic_values:
            result = attach._sanitize_for_json(val)
            # This should not raise
            json_str = json.dumps(result)
            assert json_str is not None


# =============================================================================
# Tests for _get_variables with non-serializable values
# =============================================================================


class TestGetVariablesWithNonSerializableValues:
    """Test _get_variables handles non-serializable RF variable values."""

    def test_get_all_variables_with_path_values(self, monkeypatch):
        """Test that _get_variables returns JSON-serializable result for Path values."""
        stub = StubBuiltIn(
            variables={
                "${OUTPUTDIR}": Path("/robot/output"),
                "${CURDIR}": Path("/robot/tests"),
                "${TEMPDIR}": Path("/tmp"),
                "${NORMAL}": "string value",
            }
        )
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()
        response = attach._get_variables({})

        assert response["success"] is True
        result = response["result"]

        # All values should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

        # Path values should be converted to strings
        assert isinstance(result.get("${OUTPUTDIR}"), str)
        assert isinstance(result.get("${CURDIR}"), str)

    def test_get_specific_variables_with_path_values(self, monkeypatch):
        """Test that requesting specific variables handles Path objects."""
        stub = StubBuiltIn(
            variables={
                "${OUTPUTDIR}": Path("/robot/output"),
                "${NORMAL}": "value",
            }
        )
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()
        response = attach._get_variables({"names": ["OUTPUTDIR", "NORMAL"]})

        assert response["success"] is True
        result = response["result"]

        # Response should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

    def test_get_variables_with_custom_objects(self, monkeypatch):
        """Test that _get_variables handles custom object values."""
        custom_obj = CustomClass("browser_instance")
        stub = StubBuiltIn(
            variables={
                "${BROWSER}": custom_obj,
                "${PAGE}": UnserializableClass(),
                "${TEXT}": "normal string",
            }
        )
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()
        response = attach._get_variables({})

        assert response["success"] is True
        result = response["result"]

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert json_str is not None

    def test_get_variables_with_nested_non_serializable(self, monkeypatch):
        """Test that _get_variables handles nested non-serializable values."""
        stub = StubBuiltIn(
            variables={
                "${CONFIG}": {
                    "output_dir": Path("/output"),
                    "handlers": [custom_function],
                    "nested": {"path": Path("/nested")},
                },
            }
        )
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()
        response = attach._get_variables({})

        assert response["success"] is True

        # Should be JSON serializable
        json_str = json.dumps(response)
        assert json_str is not None

    def test_get_variables_truncation_with_non_serializable(self, monkeypatch):
        """Test that truncation works correctly with non-serializable values."""
        # Create 250 variables (more than the 200 limit)
        variables = {}
        for i in range(250):
            if i % 3 == 0:
                variables[f"${{VAR{i}}}"] = Path(f"/path/{i}")
            elif i % 3 == 1:
                variables[f"${{VAR{i}}}"] = CustomClass(f"value{i}")
            else:
                variables[f"${{VAR{i}}}"] = f"string_{i}"

        stub = StubBuiltIn(variables=variables)
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()
        response = attach._get_variables({})

        assert response["success"] is True
        assert response.get("truncated") is True
        assert len(response["result"]) <= 200

        # All returned values should be JSON serializable
        json_str = json.dumps(response)
        assert json_str is not None


# =============================================================================
# Tests for HTTP handler serialization defense
# =============================================================================


class TestHttpHandlerSerializationDefense:
    """Test that HTTP handler doesn't crash on serialization errors."""

    def test_execute_command_catches_exceptions(self, monkeypatch):
        """Test that _execute_command exception handling works."""
        stub = StubBuiltIn()
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()

        # Test unknown verb returns proper error
        response = attach._execute_command("unknown_verb", {})
        assert response["success"] is False
        assert "unknown verb" in response["error"]

        # Verify error response is JSON serializable
        json_str = json.dumps(response)
        assert json_str is not None

    def test_mcp_process_once_returns_error_on_exception(self, monkeypatch):
        """Test MCP_Process_Once returns error response on execution exception."""
        attach = mcp_attach.McpAttach()

        def failing_execute(verb, payload):
            raise RuntimeError("Simulated failure")

        monkeypatch.setattr(attach, "_execute_command", failing_execute)

        replyq: queue.Queue = queue.Queue()
        attach._cmdq.put(mcp_attach._Command("test_verb", {}, replyq))

        attach.MCP_Process_Once()

        response = replyq.get_nowait()
        assert response["success"] is False
        assert "Simulated failure" in response["error"]

        # Error response must be JSON serializable
        json_str = json.dumps(response)
        assert json_str is not None

    def test_get_variables_exception_returns_error(self, monkeypatch):
        """Test that _get_variables returns error on BuiltIn exception."""

        class FailingBuiltIn:
            def get_variables(self):
                raise RuntimeError("RF context not available")

        monkeypatch.setattr(mcp_attach, "BuiltIn", FailingBuiltIn)

        attach = mcp_attach.McpAttach()
        response = attach._get_variables({})

        assert response["success"] is False
        assert "error" in response

        # Error should be JSON serializable
        json_str = json.dumps(response)
        assert json_str is not None


# =============================================================================
# Integration-style tests: Full flow from HTTP request to response
# =============================================================================


class TestIntegrationFullFlow:
    """Integration-style tests simulating full HTTP request/response flow."""

    def test_full_flow_with_complex_rf_variables(self, monkeypatch):
        """Simulate full flow from HTTP request through response with complex variables."""
        # Setup: Variables with various non-serializable types
        variables = {
            "${OUTPUTDIR}": Path("/robot/output"),
            "${CURDIR}": Path("/robot/tests/suite"),
            "${BROWSER}": CustomClass("playwright_browser"),
            "${PAGE}": UnserializableClass(),
            "${NESTED}": {
                "paths": [Path("/a"), Path("/b")],
                "callback": custom_function,
            },
            "${NORMAL}": "just a string",
            "${NUMBER}": 42,
            "${BOOL}": True,
        }
        stub = StubBuiltIn(variables=variables)
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()

        # Simulate command execution
        response = attach._execute_command("get_variables", {})

        # Response must be successful
        assert response["success"] is True

        # Response must be fully JSON serializable (as would happen in HTTP handler)
        json_str = json.dumps(response)
        assert json_str is not None
        parsed = json.loads(json_str)
        assert parsed["success"] is True

    def test_full_flow_command_queue_with_non_serializable_response(self, monkeypatch):
        """Test full command queue flow with non-serializable data."""
        variables = {
            "${PATH}": Path("/test"),
            "${OBJ}": CustomClass("test"),
        }
        stub = StubBuiltIn(variables=variables)
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()

        # Create a command through the queue
        replyq: queue.Queue = queue.Queue()
        cmd = mcp_attach._Command("get_variables", {}, replyq)
        attach._cmdq.put(cmd)

        # Process the command
        attach.MCP_Process_Once()

        # Get the response
        response = replyq.get_nowait()

        # Response must be successful and JSON serializable
        assert response["success"] is True
        json_str = json.dumps(response)
        assert json_str is not None

    def test_no_exceptions_for_any_standard_rf_variable_types(self, monkeypatch):
        """Ensure no exceptions are raised for typical RF variable types."""
        # Simulate typical Robot Framework built-in variables
        typical_rf_variables = {
            "${OUTPUTDIR}": Path("/output"),
            "${OUTPUTFILE}": Path("/output/output.xml"),
            "${REPORTFILE}": Path("/output/report.html"),
            "${LOGFILE}": Path("/output/log.html"),
            "${DEBUGFILE}": None,
            "${CURDIR}": Path("/tests/suite"),
            "${TEMPDIR}": Path("/tmp"),
            "${EXECDIR}": Path("/project"),
            "${/}": "/",
            "${:}": ":",
            "${\\n}": "\n",
            "${SPACE}": " ",
            "${EMPTY}": "",
            "${True}": True,
            "${False}": False,
            "${None}": None,
            "${null}": None,
            "${TEST_NAME}": "My Test Case",
            "${TEST_DOCUMENTATION}": "Test documentation",
            "${TEST_TAGS}": ["tag1", "tag2", "tag3"],
            "${SUITE_NAME}": "My Test Suite",
            "${SUITE_SOURCE}": Path("/tests/suite.robot"),
            "${SUITE_DOCUMENTATION}": "Suite documentation",
            "${PREV_TEST_NAME}": "Previous Test",
            "${PREV_TEST_STATUS}": "PASS",
            "${PREV_TEST_MESSAGE}": "",
        }
        stub = StubBuiltIn(variables=typical_rf_variables)
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()

        # This should not raise any exception
        response = attach._get_variables({})

        assert response["success"] is True

        # Full response must be JSON serializable
        json_str = json.dumps(response)
        assert json_str is not None

        # Verify we can parse it back
        parsed = json.loads(json_str)
        assert parsed["success"] is True
        assert "${TEST_NAME}" in parsed["result"]
        assert parsed["result"]["${TEST_NAME}"] == "My Test Case"

    def test_specific_variable_names_filter_with_paths(self, monkeypatch):
        """Test requesting specific variables that contain Path objects."""
        variables = {
            "${OUTPUTDIR}": Path("/robot/output"),
            "${CURDIR}": Path("/robot/tests"),
            "${TEXT}": "some text",
        }
        stub = StubBuiltIn(variables=variables)
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()

        # Request only specific variables
        response = attach._get_variables({"names": ["OUTPUTDIR", "TEXT"]})

        assert response["success"] is True
        result = response["result"]

        # Should only contain requested variables
        assert "${OUTPUTDIR}" in result
        assert "${TEXT}" in result
        assert "${CURDIR}" not in result

        # Must be JSON serializable
        json_str = json.dumps(response)
        assert json_str is not None


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_variables_dict(self, monkeypatch):
        """Test handling of empty variables dictionary."""
        stub = StubBuiltIn(variables={})
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()
        response = attach._get_variables({})

        assert response["success"] is True
        assert response["result"] == {}

        json_str = json.dumps(response)
        assert json_str is not None

    def test_deeply_nested_structure(self, monkeypatch):
        """Test handling of deeply nested data structures."""
        # Create a deeply nested structure (10 levels deep)
        deep = {"level": 0, "path": Path("/level0")}
        current = deep
        for i in range(1, 10):
            current["nested"] = {"level": i, "path": Path(f"/level{i}")}
            current = current["nested"]

        stub = StubBuiltIn(variables={"${DEEP}": deep})
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()
        response = attach._get_variables({})

        assert response["success"] is True

        # Must be JSON serializable
        json_str = json.dumps(response)
        assert json_str is not None

    def test_large_list_of_paths(self, monkeypatch):
        """Test handling of large list containing Path objects."""
        large_list = [Path(f"/path/to/file{i}.txt") for i in range(100)]
        stub = StubBuiltIn(variables={"${FILES}": large_list})
        monkeypatch.setattr(mcp_attach, "BuiltIn", lambda: stub)

        attach = mcp_attach.McpAttach()
        response = attach._get_variables({})

        assert response["success"] is True
        assert len(response["result"]["${FILES}"]) == 100

        # All items should be strings
        assert all(isinstance(p, str) for p in response["result"]["${FILES}"])

        json_str = json.dumps(response)
        assert json_str is not None

    def test_variable_with_bytes_value(self, monkeypatch):
        """Test handling of bytes values (another non-JSON-serializable type)."""
        attach = mcp_attach.McpAttach()

        # bytes objects should be converted to string via str()
        result = attach._sanitize_for_json(b"hello bytes")
        assert isinstance(result, str)

    def test_variable_with_set_value(self, monkeypatch):
        """Test handling of set values."""
        attach = mcp_attach.McpAttach()

        # Sets should be converted to string via str()
        result = attach._sanitize_for_json({1, 2, 3})
        assert isinstance(result, str)

    def test_variable_with_complex_number(self, monkeypatch):
        """Test handling of complex number values."""
        attach = mcp_attach.McpAttach()

        # Complex numbers should be converted to string
        result = attach._sanitize_for_json(complex(1, 2))
        assert isinstance(result, str)
        assert "1" in result and "2" in result
