"""Tests for token-efficient output utilities."""

import pytest

from robotmcp.utils.token_efficient_output import (
    TokenEfficientOutput,
    compact_response,
    error_response,
    estimate_tokens,
    optimize_execution_result,
    optimize_keyword_list,
    optimize_library_list,
    optimize_output,
    optimize_session_info,
    success_response,
    token_budget_check,
)


class TestTokenEfficientOutput:
    """Test the TokenEfficientOutput class."""

    def test_omit_empty_fields(self):
        """Test that empty fields are omitted when configured."""
        handler = TokenEfficientOutput(omit_empty=True)
        response = {
            "success": True,
            "result": "data",
            "error": None,
            "message": "",
            "metadata": {},
        }
        optimized = handler.optimize(response)

        assert "success" in optimized
        assert "result" in optimized
        assert "error" not in optimized  # None should be omitted
        assert "message" not in optimized  # Empty string should be omitted
        assert "metadata" not in optimized  # Empty dict should be omitted

    def test_keep_empty_fields_when_disabled(self):
        """Test that empty fields are kept when omit_empty is False."""
        handler = TokenEfficientOutput(omit_empty=False)
        response = {
            "success": True,
            "error": None,
        }
        optimized = handler.optimize(response)

        assert "success" in optimized
        assert "error" in optimized  # None should be kept

    def test_field_abbreviation(self):
        """Test field name abbreviation in compact mode."""
        handler = TokenEfficientOutput(abbreviate_fields=True)
        response = {
            "success": True,
            "keyword": "Click",
            "arguments": ["locator"],
            "execution_time": 0.5,
        }
        optimized = handler.optimize(response)

        assert "ok" in optimized  # "success" -> "ok"
        assert "kw" in optimized  # "keyword" -> "kw"
        assert "args" in optimized  # "arguments" -> "args"
        assert "time" in optimized  # "execution_time" -> "time"

    def test_string_truncation(self):
        """Test that long strings are truncated."""
        handler = TokenEfficientOutput(max_string_length=50)
        response = {
            "output": "x" * 100,
        }
        optimized = handler.optimize(response)

        output = optimized["output"]
        assert "..." in output
        assert "[+50]" in output  # Should show how many chars were truncated

    def test_list_truncation(self):
        """Test that long lists are truncated."""
        handler = TokenEfficientOutput(max_list_items=5)
        response = {
            "items": list(range(20)),
        }
        optimized = handler.optimize(response)

        items = optimized["items"]
        assert isinstance(items, dict)
        assert items["n"] == 20
        assert items["truncated"] is True
        assert len(items["items"]) == 5

    def test_dict_truncation(self):
        """Test that large dicts are truncated."""
        handler = TokenEfficientOutput(max_dict_items=3)
        response = {
            "data": {f"key{i}": i for i in range(10)},
        }
        optimized = handler.optimize(response)

        data = optimized["data"]
        assert "_n" in data
        assert data["_truncated"] is True
        # Should have 3 items + 2 meta keys
        assert len([k for k in data.keys() if not k.startswith("_")]) == 3


class TestCompactResponse:
    """Test the compact_response function."""

    def test_basic_compaction(self):
        """Test basic response compaction."""
        response = {
            "success": True,
            "result": "data",
            "error": None,
            "warnings": [],
        }
        compacted = compact_response(response)

        assert "success" in compacted
        assert "result" in compacted
        assert "error" not in compacted
        assert "warnings" not in compacted

    def test_with_abbreviation(self):
        """Test compaction with field abbreviation."""
        response = {
            "success": True,
            "keyword": "Click",
        }
        compacted = compact_response(response, abbreviate=True)

        assert "ok" in compacted
        assert "kw" in compacted


class TestOptimizeOutput:
    """Test the optimize_output function."""

    def test_compact_verbosity(self):
        """Test compact verbosity level."""
        response = {
            "success": True,
            "result": "data",
            "metadata": {"extra": "info"},
        }
        optimized = optimize_output(response, verbosity="compact")

        assert "success" in optimized or "ok" in optimized
        assert "result" in optimized or "res" in optimized

    def test_standard_verbosity(self):
        """Test standard verbosity level."""
        response = {
            "success": True,
            "result": "data",
            "error": None,
        }
        optimized = optimize_output(response, verbosity="standard")

        assert "success" in optimized
        assert "error" not in optimized  # Empty, should be omitted

    def test_verbose_verbosity(self):
        """Test verbose verbosity level."""
        response = {
            "success": True,
            "result": "data",
        }
        optimized = optimize_output(response, verbosity="verbose")

        assert "success" in optimized
        assert "result" in optimized


class TestOptimizeKeywordList:
    """Test keyword list optimization."""

    def test_compact_keywords(self):
        """Test compact keyword list."""
        keywords = [
            {
                "name": "Click",
                "library": "Browser",
                "args": ["locator"],
                "short_doc": "Clicks on element",
            },
            {
                "name": "Type Text",
                "library": "Browser",
                "arguments": ["locator", "text"],
                "short_doc": "Types text into element",
            },
        ]
        optimized = optimize_keyword_list(keywords, verbosity="compact")

        assert len(optimized) == 2
        assert optimized[0]["name"] == "Click"
        assert optimized[0]["lib"] == "Browser"
        assert optimized[0]["args"] == 1  # Arg count only
        assert optimized[1]["args"] == 2

    def test_standard_keywords(self):
        """Test standard keyword list."""
        keywords = [
            {
                "name": "Click",
                "library": "Browser",
                "args": ["locator"],
                "short_doc": "Clicks on element. This is a longer doc that should be truncated.",
            },
        ]
        optimized = optimize_keyword_list(keywords, verbosity="standard")

        assert optimized[0]["name"] == "Click"
        assert optimized[0]["library"] == "Browser"
        assert optimized[0]["args"] == ["locator"]
        assert len(optimized[0]["doc"]) <= 100


class TestOptimizeExecutionResult:
    """Test execution result optimization."""

    def test_compact_success(self):
        """Test compact successful execution result."""
        result = {
            "success": True,
            "keyword": "Click",
            "output": "Button clicked",
            "execution_time": 0.123,
        }
        optimized = optimize_execution_result(result, verbosity="compact")

        assert optimized["ok"] is True
        assert "out" in optimized
        assert len(optimized["out"]) <= 200

    def test_compact_error(self):
        """Test compact error execution result."""
        result = {
            "success": False,
            "error": "Element not found: " + "x" * 500,
        }
        optimized = optimize_execution_result(result, verbosity="compact")

        assert optimized["ok"] is False
        assert "err" in optimized
        assert len(optimized["err"]) <= 300

    def test_standard_result(self):
        """Test standard execution result."""
        result = {
            "success": True,
            "keyword": "Click",
            "output": "Button clicked",
            "execution_time": 0.123456,
            "assigned_variables": {"result": True},
        }
        optimized = optimize_execution_result(result, verbosity="standard")

        assert optimized["ok"] is True
        assert optimized["kw"] == "Click"
        assert optimized["time"] == 0.123  # Rounded to 3 decimals
        assert optimized["vars"] == {"result": True}


class TestSuccessErrorResponse:
    """Test success/error response builders."""

    def test_success_response_minimal(self):
        """Test minimal success response."""
        response = success_response(result="data")

        assert response["success"] is True
        assert response["result"] == "data"

    def test_success_response_with_extras(self):
        """Test success response with extra fields."""
        response = success_response(
            result="data", message="Operation completed", count=10, empty_field=None
        )

        assert response["success"] is True
        assert response["result"] == "data"
        assert response["message"] == "Operation completed"
        assert response["count"] == 10
        assert "empty_field" not in response  # Should be omitted

    def test_success_response_compact(self):
        """Test compact success response."""
        response = success_response(result="data", message="Operation completed", verbosity="compact")

        assert response["success"] is True
        assert "message" not in response  # Omitted in compact mode

    def test_error_response_basic(self):
        """Test basic error response."""
        response = error_response("Something went wrong")

        assert response["success"] is False
        assert response["error"] == "Something went wrong"

    def test_error_response_truncated(self):
        """Test error response with long error message."""
        long_error = "x" * 1000
        response = error_response(long_error)

        assert response["success"] is False
        assert len(response["error"]) <= 503  # 500 + "..."


class TestTokenEstimation:
    """Test token estimation utilities."""

    def test_estimate_tokens(self):
        """Test token estimation."""
        obj = {"key": "value", "list": [1, 2, 3]}
        estimate = estimate_tokens(obj)

        assert isinstance(estimate, int)
        assert estimate > 0

    def test_token_budget_check_within_budget(self):
        """Test token budget check within budget."""
        obj = {"key": "value"}
        result = token_budget_check(obj, max_tokens=1000)

        assert result["within_budget"] is True
        assert "suggestion" not in result

    def test_token_budget_check_over_budget(self):
        """Test token budget check over budget."""
        obj = {"data": "x" * 10000}
        result = token_budget_check(obj, max_tokens=100)

        assert result["within_budget"] is False
        assert "suggestion" in result


class TestOptimizeSessionInfo:
    """Test session info optimization."""

    def test_compact_session_info(self):
        """Test compact session info."""
        session_info = {
            "session_id": "test-session",
            "libraries": [
                {"name": "Browser"},
                {"name": "BuiltIn"},
            ],
            "step_count": 5,
            "active": True,
            "extra": "data",
        }
        optimized = optimize_session_info(session_info, verbosity="compact")

        assert optimized["id"] == "test-session"
        assert optimized["libs"] == 2
        assert optimized["steps"] == 5
        assert "active" not in optimized
        assert "extra" not in optimized

    def test_standard_session_info(self):
        """Test standard session info."""
        session_info = {
            "session_id": "test-session",
            "libraries": [
                {"name": "Browser"},
                "BuiltIn",  # Can also be string
            ],
            "step_count": 5,
            "active": True,
        }
        optimized = optimize_session_info(session_info, verbosity="standard")

        assert optimized["session_id"] == "test-session"
        assert optimized["libraries"] == ["Browser", "BuiltIn"]
        assert optimized["step_count"] == 5
        assert optimized["active"] is True


class TestOptimizeLibraryList:
    """Test library list optimization."""

    def test_compact_library_list(self):
        """Test compact library list."""
        libraries = [
            {"name": "Browser", "keyword_count": 150, "status": "loaded"},
            {"name": "BuiltIn", "keyword_count": 100, "status": "loaded"},
        ]
        optimized = optimize_library_list(libraries, verbosity="compact")

        assert len(optimized) == 2
        assert optimized[0] == {"name": "Browser", "kw_count": 150}
        assert optimized[1] == {"name": "BuiltIn", "kw_count": 100}

    def test_standard_library_list(self):
        """Test standard library list."""
        libraries = [
            {"name": "Browser", "keyword_count": 150, "status": "loaded", "extra": "data"},
        ]
        optimized = optimize_library_list(libraries, verbosity="standard")

        assert optimized[0]["name"] == "Browser"
        assert optimized[0]["keywords"] == 150
        assert optimized[0]["status"] == "loaded"
        assert "extra" not in optimized[0]


class TestRedundantFieldRemoval:
    """Test redundant field removal."""

    def test_status_redundant_with_success_true(self):
        """Test that status is removed when redundant with success=True."""
        handler = TokenEfficientOutput()
        response = {
            "success": True,
            "status": "success",
            "result": "data",
        }
        optimized = handler.optimize(response)

        assert "success" in optimized
        assert "status" not in optimized

    def test_status_redundant_with_success_false(self):
        """Test that status is removed when redundant with success=False."""
        handler = TokenEfficientOutput()
        response = {
            "success": False,
            "status": "failed",
            "error": "Something went wrong",
        }
        optimized = handler.optimize(response)

        assert "success" in optimized
        assert "status" not in optimized

    def test_status_kept_when_different(self):
        """Test that status is kept when it adds information."""
        handler = TokenEfficientOutput()
        response = {
            "success": True,
            "status": "pending",  # Different from success
            "result": "data",
        }
        optimized = handler.optimize(response)

        assert "success" in optimized
        assert "status" in optimized  # Should be kept
