"""Integration tests validating the variable serialization fix.

This module tests that the _sanitize_for_json fix in mcp_attach.py works correctly
end-to-end, preventing crashes from non-serializable Robot Framework variable types.

Run with:
    uv run pytest tests/integration/test_variable_serialization_fix.py -v

These tests verify:
1. The sanitization function handles all RF variable types
2. The bridge HTTP handler doesn't crash on serialization
3. Valid JSON responses are returned
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Test fixtures simulating RF variable types
# ============================================================================


class DotDict(dict):
    """Simulates RF's DotDict."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


class Tags(list):
    """Simulates RF's Tags object."""
    pass


class RobotOptions:
    """Simulates RF's internal options (not JSON serializable)."""

    def __init__(self):
        self.output = Path("/tmp/output.xml")
        self.loglevel = "INFO"


class LibraryInstance:
    """Simulates library instance with methods."""

    def __init__(self, name: str):
        self.name = name
        self._internal = object()

    def keyword(self) -> str:
        return "result"


# ============================================================================
# Import the actual sanitization function
# ============================================================================


@pytest.fixture
def sanitize_function():
    """Get the actual sanitization function from mcp_attach."""
    from robotmcp.attach.mcp_attach import McpAttach

    attach = McpAttach()
    return attach._sanitize_for_json


# ============================================================================
# Unit tests for _sanitize_for_json
# ============================================================================


class TestSanitizeForJson:
    """Unit tests for the _sanitize_for_json method."""

    def test_none_passes_through(self, sanitize_function):
        """None should remain None."""
        assert sanitize_function(None) is None

    def test_string_passes_through(self, sanitize_function):
        """Strings should remain strings."""
        assert sanitize_function("hello") == "hello"
        assert sanitize_function("") == ""

    def test_int_passes_through(self, sanitize_function):
        """Integers should remain integers."""
        assert sanitize_function(42) == 42
        assert sanitize_function(0) == 0
        assert sanitize_function(-1) == -1

    def test_float_passes_through(self, sanitize_function):
        """Floats should remain floats."""
        assert sanitize_function(3.14) == 3.14
        assert sanitize_function(0.0) == 0.0

    def test_bool_passes_through(self, sanitize_function):
        """Booleans should remain booleans."""
        assert sanitize_function(True) is True
        assert sanitize_function(False) is False

    def test_path_converts_to_string(self, sanitize_function):
        """Path objects should become strings."""
        path = Path("/home/user/file.txt")
        result = sanitize_function(path)
        assert isinstance(result, str)
        assert result == str(path)

    def test_list_recursively_sanitized(self, sanitize_function):
        """Lists should be recursively sanitized."""
        data = [1, "two", Path("/path"), [4, Path("/nested")]]
        result = sanitize_function(data)

        assert isinstance(result, list)
        assert result[0] == 1
        assert result[1] == "two"
        assert isinstance(result[2], str)
        assert isinstance(result[3], list)
        assert isinstance(result[3][1], str)

    def test_tuple_converts_to_list(self, sanitize_function):
        """Tuples should become lists."""
        data = (1, "two", Path("/path"))
        result = sanitize_function(data)

        assert isinstance(result, list)
        assert len(result) == 3

    def test_dict_recursively_sanitized(self, sanitize_function):
        """Dicts should be recursively sanitized."""
        data = {
            "str_key": "value",
            "path_val": Path("/path"),
            "nested": {"inner": Path("/inner")},
        }
        result = sanitize_function(data)

        assert isinstance(result, dict)
        assert result["str_key"] == "value"
        assert isinstance(result["path_val"], str)
        assert isinstance(result["nested"]["inner"], str)

    def test_dict_keys_converted_to_string(self, sanitize_function):
        """Non-string dict keys should become strings."""
        data = {1: "one", Path("/key"): "path_key"}
        result = sanitize_function(data)

        assert "1" in result
        assert any("key" in k for k in result.keys())

    def test_datetime_converts_to_string(self, sanitize_function):
        """datetime objects should become strings."""
        dt = datetime.now()
        result = sanitize_function(dt)
        assert isinstance(result, str)

    def test_timedelta_converts_to_string(self, sanitize_function):
        """timedelta objects should become strings."""
        td = timedelta(hours=1, minutes=30)
        result = sanitize_function(td)
        assert isinstance(result, str)

    def test_dotdict_like_object_converts(self, sanitize_function):
        """DotDict (dict subclass) should be handled like dict."""
        data = DotDict(key="value", nested=DotDict(inner="inner_val"))
        result = sanitize_function(data)

        assert isinstance(result, dict)
        assert result["key"] == "value"
        assert result["nested"]["inner"] == "inner_val"

    def test_tags_like_object_converts(self, sanitize_function):
        """Tags (list subclass) should be handled like list."""
        tags = Tags(["smoke", "regression"])
        result = sanitize_function(tags)

        assert isinstance(result, list)

    def test_custom_object_converts_to_string(self, sanitize_function):
        """Custom objects should become string representation."""
        obj = RobotOptions()
        result = sanitize_function(obj)
        assert isinstance(result, str)

    def test_function_converts_to_string(self, sanitize_function):
        """Functions should become string representation."""
        fn = lambda x: x * 2
        result = sanitize_function(fn)
        assert isinstance(result, str)

    def test_builtin_function_converts_to_string(self, sanitize_function):
        """Built-in functions should become string representation."""
        result = sanitize_function(len)
        assert isinstance(result, str)

    def test_library_instance_converts_to_string(self, sanitize_function):
        """Library instances should become string representation."""
        lib = LibraryInstance("Browser")
        result = sanitize_function(lib)
        assert isinstance(result, str)

    def test_mock_object_converts_to_string(self, sanitize_function):
        """Mock objects should become string representation."""
        mock = MagicMock()
        result = sanitize_function(mock)
        assert isinstance(result, str)

    def test_result_is_json_serializable(self, sanitize_function):
        """All sanitized results should be JSON serializable."""
        test_values = [
            None,
            "string",
            42,
            3.14,
            True,
            Path("/path"),
            [1, 2, Path("/path")],
            {"key": Path("/value")},
            datetime.now(),
            DotDict(a=1),
            Tags(["tag1"]),
            RobotOptions(),
            lambda x: x,
            len,
        ]

        for val in test_values:
            result = sanitize_function(val)
            # Must not raise
            json_str = json.dumps({"value": result})
            assert len(json_str) > 0


class TestSanitizeForJsonEdgeCases:
    """Edge case tests for _sanitize_for_json."""

    def test_empty_structures(self, sanitize_function):
        """Empty structures should remain empty."""
        assert sanitize_function([]) == []
        assert sanitize_function({}) == {}
        assert sanitize_function(()) == []

    def test_deeply_nested_structure(self, sanitize_function):
        """Deeply nested structures should be handled."""
        deep = {"l1": {"l2": {"l3": {"l4": {"l5": Path("/deep")}}}}}
        result = sanitize_function(deep)

        assert isinstance(result["l1"]["l2"]["l3"]["l4"]["l5"], str)

    def test_mixed_list_types(self, sanitize_function):
        """Lists with mixed types should all be sanitized."""
        data = [
            Path("/path"),
            datetime.now(),
            {"nested": Path("/nested")},
            [Path("/inner")],
            None,
            True,
            42,
            "string",
        ]
        result = sanitize_function(data)

        # All should be JSON serializable
        json.dumps(result)

    def test_large_list(self, sanitize_function):
        """Large lists should be handled efficiently."""
        data = [Path(f"/path_{i}") for i in range(1000)]
        result = sanitize_function(data)

        assert len(result) == 1000
        assert all(isinstance(v, str) for v in result)

    def test_large_dict(self, sanitize_function):
        """Large dicts should be handled efficiently."""
        data = {f"key_{i}": Path(f"/path_{i}") for i in range(1000)}
        result = sanitize_function(data)

        assert len(result) == 1000
        assert all(isinstance(v, str) for v in result.values())


# ============================================================================
# Integration tests for McpAttach._get_variables
# ============================================================================


class TestGetVariablesEndpoint:
    """Test the _get_variables method that previously crashed."""

    def test_get_variables_with_paths(self, sanitize_function):
        """Simulates _get_variables with Path objects in variables."""
        # Simulate RF variables dict
        all_vars = {
            "${CURDIR}": Path("/home/user/tests"),
            "${OUTPUT_DIR}": Path("/home/user/output"),
            "${SIMPLE}": "simple_value",
            "${NUMBER}": 42,
        }

        # Simulate the _get_variables logic
        out = {}
        for i, (k, v) in enumerate(all_vars.items()):
            if i >= 200:  # Truncation limit
                break
            out[k] = sanitize_function(v)

        response = {"success": True, "result": out, "truncated": False}

        # Must not crash
        json_bytes = json.dumps(response).encode("utf-8")
        assert b'"success": true' in json_bytes

    def test_get_variables_with_complex_rf_objects(self, sanitize_function):
        """Simulates _get_variables with complex RF objects."""
        all_vars = {
            "${OPTIONS}": RobotOptions(),
            "${METADATA}": DotDict(author="test", version="1.0"),
            "${TAGS}": Tags(["smoke", "regression"]),
            "${TIMESTAMP}": datetime.now(),
            "${LIB}": LibraryInstance("Browser"),
        }

        out = {}
        for k, v in all_vars.items():
            out[k] = sanitize_function(v)

        response = {"success": True, "result": out}

        # Must be valid JSON
        json_str = json.dumps(response)
        parsed = json.loads(json_str)

        assert parsed["success"] is True
        assert len(parsed["result"]) == len(all_vars)


# ============================================================================
# End-to-end validation
# ============================================================================


class TestEndToEndValidation:
    """End-to-end tests simulating full request/response cycle."""

    def test_full_response_cycle(self, sanitize_function):
        """Test complete request -> sanitize -> JSON -> response cycle."""
        # Simulate typical RF variable set
        rf_variables = {
            "${CURDIR}": Path("/home/user/tests"),
            "${EXECDIR}": Path("/home/user/project"),
            "${OUTPUT_DIR}": Path("/home/user/output"),
            "${TEMPDIR}": Path("/tmp"),
            "${SUITE_NAME}": "Test Suite",
            "${SUITE_SOURCE}": Path("/home/user/tests/suite.robot"),
            "${TEST_NAME}": "Test Case",
            "${TEST_TAGS}": Tags(["smoke", "regression"]),
            "${SUITE_METADATA}": DotDict(author="tester", env="test"),
            "${OPTIONS}": RobotOptions(),
            "${BASE_URL}": "https://example.com",
            "${TIMEOUT}": 30,
            "${DEBUG}": True,
        }

        # Step 1: Sanitize all variables
        sanitized = {k: sanitize_function(v) for k, v in rf_variables.items()}

        # Step 2: Create response payload
        response = {
            "success": True,
            "result": sanitized,
            "truncated": False,
        }

        # Step 3: Serialize to JSON (HTTP response body)
        json_bytes = json.dumps(response).encode("utf-8")

        # Step 4: Verify parseable by client
        client_response = json.loads(json_bytes.decode("utf-8"))

        # Assertions
        assert client_response["success"] is True
        assert isinstance(client_response["result"], dict)
        assert len(client_response["result"]) == len(rf_variables)

        # Verify specific conversions
        assert isinstance(client_response["result"]["${CURDIR}"], str)
        assert isinstance(client_response["result"]["${TEST_TAGS}"], list)
        assert isinstance(client_response["result"]["${SUITE_METADATA}"], dict)
        assert isinstance(client_response["result"]["${OPTIONS}"], str)

    def test_http_handler_simulation(self, sanitize_function):
        """Simulate the HTTP handler's json.dumps call."""
        # This is the exact code path that was failing
        rf_variables = {
            "${OUTPUT_DIR}": Path("/home/user/output"),
            "${OPTIONS}": RobotOptions(),
        }

        # Sanitize
        out = {}
        for k, v in rf_variables.items():
            out[k] = sanitize_function(v)

        resp = {"success": True, "result": out}

        # This was the failing line in do_POST
        try:
            body = json.dumps(resp).encode("utf-8")
        except (TypeError, ValueError) as e:
            pytest.fail(f"JSON serialization failed: {e}")

        assert body is not None
        assert len(body) > 0


# ============================================================================
# Regression tests
# ============================================================================


class TestRegressionCases:
    """Regression tests for specific issues that were fixed."""

    def test_regression_path_object_crash(self, sanitize_function):
        """Regression: Path objects caused TypeError in json.dumps."""
        variables = {"${OUTPUT_DIR}": Path("/home/user/output")}

        sanitized = {k: sanitize_function(v) for k, v in variables.items()}
        # Must not raise TypeError
        json.dumps(sanitized)

    def test_regression_dotdict_crash(self, sanitize_function):
        """Regression: DotDict might not serialize correctly."""
        variables = {"${METADATA}": DotDict(key="value")}

        sanitized = {k: sanitize_function(v) for k, v in variables.items()}
        json_str = json.dumps(sanitized)

        # Should preserve dict structure
        parsed = json.loads(json_str)
        assert parsed["${METADATA}"]["key"] == "value"

    def test_regression_tags_crash(self, sanitize_function):
        """Regression: Tags (list subclass) should serialize as list."""
        variables = {"${TAGS}": Tags(["tag1", "tag2"])}

        sanitized = {k: sanitize_function(v) for k, v in variables.items()}
        json_str = json.dumps(sanitized)

        parsed = json.loads(json_str)
        assert isinstance(parsed["${TAGS}"], list)

    def test_regression_datetime_crash(self, sanitize_function):
        """Regression: datetime objects caused TypeError."""
        variables = {"${NOW}": datetime.now()}

        sanitized = {k: sanitize_function(v) for k, v in variables.items()}
        # Must not raise
        json.dumps(sanitized)

    def test_regression_function_crash(self, sanitize_function):
        """Regression: Function objects caused TypeError."""
        variables = {"${FUNC}": len}

        sanitized = {k: sanitize_function(v) for k, v in variables.items()}
        # Must not raise
        json.dumps(sanitized)

    def test_regression_connection_closed_error(self, sanitize_function):
        """Regression: "Remote end closed connection without response" error.

        This was caused by json.dumps raising TypeError in the HTTP handler,
        which crashed without sending a response.
        """
        # Simulate the exact variable set that caused the crash
        problematic_vars = {
            "${CURDIR}": Path("/home/user/tests"),
            "${EXECDIR}": Path("/home/user/project"),
            "${OUTPUT_DIR}": Path("/home/user/output"),
            "${OUTPUT_FILE}": Path("/home/user/output/output.xml"),
            "${LOG_FILE}": Path("/home/user/output/log.html"),
            "${REPORT_FILE}": Path("/home/user/output/report.html"),
            "${SUITE_SOURCE}": Path("/home/user/tests/suite.robot"),
            "${OPTIONS}": RobotOptions(),
            "${SUITE_METADATA}": DotDict(author="test"),
            "${TEST_TAGS}": Tags(["smoke"]),
        }

        # This sequence was failing before the fix
        out = {}
        for i, (k, v) in enumerate(problematic_vars.items()):
            if i >= 200:
                break
            out[k] = sanitize_function(v)

        response = {"success": True, "result": out, "truncated": True}

        # The line that was crashing in do_POST
        body = json.dumps(response).encode("utf-8")

        # Verify valid response
        assert b'"success": true' in body
        assert len(body) > 0

        # Verify client can parse it
        client_data = json.loads(body)
        assert client_data["success"] is True
        assert len(client_data["result"]) == len(problematic_vars)
