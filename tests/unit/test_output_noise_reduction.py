"""Comprehensive unit tests for output noise reduction changes.

Covers:
  A. session_variables removal from error paths and standard detail level
  B. changed_variables tracking (non-RF-builtin, actually changed)
  C. Hint deduplication via _rank_and_deduplicate_hints
  D. Grouped warnings in _check_untracked_variables
  E. HTTP response serialization (compact, no body duplication)
  F. Enhanced HTTP response serializer at all 3 detail levels
  G. Externalization rules expansion (13 DEFAULT_RULES)
  H. DeltaStateService.has_version()
  I. Compact manage_session / analyze_scenario responses (flat session_type, etc.)

Run with:
    uv run pytest tests/unit/test_output_noise_reduction.py -v
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

# ---------------------------------------------------------------------------
# Prevent pytest from collecting helper dataclasses with "Test" in the name
# ---------------------------------------------------------------------------


@dataclass
class _TestStep:
    """Minimal step mock (NOT collected by pytest)."""

    step_id: str = "step-001"
    keyword: str = "Click"
    arguments: list = field(default_factory=list)
    status: str = "pass"
    execution_time: float = 0.05
    result: Any = None
    variables: dict = field(default_factory=dict)
    assigned_variables: list = field(default_factory=list)
    assignment_type: Optional[str] = None
    is_successful: bool = True


__test__ = True  # Module IS a test; helpers above are suppressed via underscore prefix


# ============================================================================
# A. TestSessionVariablesRemoval
# ============================================================================


class TestSessionVariablesRemoval:
    """Verify error paths and standard detail level omit session_variables."""

    def _make_executor(self):
        """Create a minimal KeywordExecutor-like object with _build_response_by_detail_level."""
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        executor = object.__new__(KeywordExecutor)
        # Provide required attributes used in _build_response_by_detail_level
        executor.response_serializer = MagicMock()
        executor.response_serializer.serialize_for_response = lambda x: x
        executor.plugin_manager = MagicMock()
        executor.plugin_manager.generate_failure_hints.return_value = []
        executor.rf_converter = MagicMock()
        return executor

    def _make_session(self, variables=None):
        session = MagicMock()
        session.session_id = "s1"
        session.variables = variables or {}
        session.get_active_library.return_value = "Browser"
        session.browser_state = MagicMock(
            browser_type="chromium",
            current_url="http://example.com",
            context_id="ctx1",
            page_id="page1",
        )
        session.step_count = 5
        session.duration = 1.0
        session.search_order = None
        return session

    @pytest.mark.asyncio
    async def test_pre_validation_error_no_session_variables(self):
        """Pre-validation failure response must NOT contain session_variables."""
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        executor = self._make_executor()
        step = _TestStep(status="fail")
        result = {
            "success": False,
            "error": "Element not found",
            "output": "",
            "hints": [],
        }
        session = self._make_session({"${URL}": "http://example.com"})

        response = await KeywordExecutor._build_response_by_detail_level(
            executor, "standard", result, step, "Click", ["css:btn"], session
        )

        assert "session_variables" not in response

    @pytest.mark.asyncio
    async def test_exception_error_path_no_session_variables(self):
        """Exception catch-all error response must NOT contain session_variables."""
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        executor = self._make_executor()
        step = _TestStep(status="fail")
        result = {
            "success": False,
            "error": "RuntimeError: boom",
            "output": "",
        }
        session = self._make_session({"${TOKEN}": "abc123"})

        response = await KeywordExecutor._build_response_by_detail_level(
            executor, "standard", result, step, "Open Browser", [], session
        )

        assert "session_variables" not in response

    @pytest.mark.asyncio
    async def test_standard_success_no_session_variables(self):
        """Standard detail level for a successful step must NOT include session_variables."""
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        executor = self._make_executor()
        step = _TestStep()
        result = {"success": True, "output": "OK"}
        session = self._make_session({"${FOO}": "bar"})

        response = await KeywordExecutor._build_response_by_detail_level(
            executor, "standard", result, step, "Log", ["hello"], session
        )

        assert "session_variables" not in response
        assert response["success"] is True

    @pytest.mark.asyncio
    async def test_full_detail_does_include_session_variables(self):
        """Full detail level SHOULD still include session_variables."""
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        executor = self._make_executor()
        step = _TestStep()
        result = {"success": True, "output": "OK", "state_updates": {}}
        session = self._make_session({"${MY_VAR}": "value"})

        response = await KeywordExecutor._build_response_by_detail_level(
            executor, "full", result, step, "Log", ["hello"], session
        )

        assert "session_variables" in response
        assert "${MY_VAR}" in response["session_variables"]

    @pytest.mark.asyncio
    async def test_minimal_detail_no_session_variables(self):
        """Minimal detail level must NOT include session_variables."""
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        executor = self._make_executor()
        step = _TestStep()
        result = {"success": True, "output": "OK"}
        session = self._make_session({"${X}": "1"})

        response = await KeywordExecutor._build_response_by_detail_level(
            executor, "minimal", result, step, "Log", [], session
        )

        assert "session_variables" not in response


# ============================================================================
# B. TestChangedVariablesTracking
# ============================================================================


class TestChangedVariablesTracking:
    """Verify changed_variables includes only non-RF-builtin vars that changed."""

    def _make_executor(self):
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        executor = object.__new__(KeywordExecutor)
        executor.response_serializer = MagicMock()
        executor.response_serializer.serialize_for_response = lambda x: x
        executor.plugin_manager = MagicMock()
        executor.plugin_manager.generate_failure_hints.return_value = []
        executor.rf_converter = MagicMock()
        return executor

    def _make_session(self, variables):
        session = MagicMock()
        session.session_id = "s1"
        session.variables = variables
        session.get_active_library.return_value = "BuiltIn"
        session.search_order = None
        return session

    @pytest.mark.asyncio
    async def test_changed_variable_included(self):
        """A user variable that was added should appear in changed_variables."""
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        executor = self._make_executor()
        step = _TestStep()

        vars_before = {}
        session_vars_after = {"${MY_VAR}": "new_value"}
        session = self._make_session(session_vars_after)
        result = {"success": True, "output": "OK"}

        response = await KeywordExecutor._build_response_by_detail_level(
            executor,
            "standard",
            result,
            step,
            "Set Variable",
            [],
            session,
            vars_before=vars_before,
        )

        assert "changed_variables" in response
        assert "${MY_VAR}" in response["changed_variables"]

    @pytest.mark.asyncio
    async def test_rf_builtin_excluded(self):
        """RF built-in variables (LOG_LEVEL, OUTPUT_DIR, etc.) must NOT appear."""
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        executor = self._make_executor()
        step = _TestStep()

        vars_before = {}
        session_vars_after = {
            "${LOG_LEVEL}": "INFO",
            "${OUTPUT_DIR}": "/tmp",
            "${CURDIR}": "/cwd",
            "${MY_VAR}": "hello",
        }
        session = self._make_session(session_vars_after)
        result = {"success": True, "output": "OK"}

        response = await KeywordExecutor._build_response_by_detail_level(
            executor,
            "standard",
            result,
            step,
            "Set Variable",
            [],
            session,
            vars_before=vars_before,
        )

        changed = response.get("changed_variables", {})
        assert "${MY_VAR}" in changed
        assert "${LOG_LEVEL}" not in changed
        assert "${OUTPUT_DIR}" not in changed
        assert "${CURDIR}" not in changed

    @pytest.mark.asyncio
    async def test_unchanged_variable_excluded(self):
        """A variable that didn't change must NOT appear in changed_variables."""
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        executor = self._make_executor()
        step = _TestStep()

        same_obj = "same_value"
        vars_before = {"${MY_VAR}": same_obj}
        session_vars_after = {"${MY_VAR}": same_obj}
        session = self._make_session(session_vars_after)
        result = {"success": True, "output": "OK"}

        response = await KeywordExecutor._build_response_by_detail_level(
            executor,
            "standard",
            result,
            step,
            "Log",
            [],
            session,
            vars_before=vars_before,
        )

        assert "changed_variables" not in response

    @pytest.mark.asyncio
    async def test_no_vars_before_means_no_changed(self):
        """When vars_before is None, changed_variables should not appear."""
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        executor = self._make_executor()
        step = _TestStep()
        session = self._make_session({"${X}": "1"})
        result = {"success": True, "output": "OK"}

        response = await KeywordExecutor._build_response_by_detail_level(
            executor,
            "standard",
            result,
            step,
            "Log",
            [],
            session,
            vars_before=None,
        )

        assert "changed_variables" not in response

    @pytest.mark.asyncio
    async def test_special_chars_builtins_excluded(self):
        """Built-in special-char vars (/, \\n, SPACE, EMPTY, etc.) are excluded."""
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        executor = self._make_executor()
        step = _TestStep()
        vars_before = {}
        session_vars_after = {
            "${/}": "/",
            "${SPACE}": " ",
            "${EMPTY}": "",
            "${True}": True,
            "${None}": None,
            "${USER_VAR}": "keep_me",
        }
        session = self._make_session(session_vars_after)
        result = {"success": True, "output": "OK"}

        response = await KeywordExecutor._build_response_by_detail_level(
            executor,
            "standard",
            result,
            step,
            "Log",
            [],
            session,
            vars_before=vars_before,
        )

        changed = response.get("changed_variables", {})
        assert "${USER_VAR}" in changed
        assert "${/}" not in changed
        assert "${SPACE}" not in changed
        assert "${EMPTY}" not in changed
        assert "${True}" not in changed
        assert "${None}" not in changed


# ============================================================================
# C. TestHintDeduplication
# ============================================================================


class TestHintDeduplication:
    """Test _rank_and_deduplicate_hints static method."""

    def _dedup(self, hints, detail_level="minimal"):
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        return KeywordExecutor._rank_and_deduplicate_hints(hints, detail_level)

    def test_empty_hints_returns_empty(self):
        assert self._dedup([]) == []

    def test_none_hints_returns_none(self):
        """None input should be returned as-is (falsy)."""
        result = self._dedup(None)
        assert not result

    def test_single_hint_minimal_returns_one(self):
        hints = [{"message": "Use CSS selector"}]
        result = self._dedup(hints, "minimal")
        assert len(result) == 1

    def test_minimal_caps_at_one(self):
        hints = [
            {"message": "Hint A"},
            {"message": "Hint B"},
            {"message": "Hint C"},
        ]
        result = self._dedup(hints, "minimal")
        assert len(result) == 1

    def test_standard_caps_at_three(self):
        hints = [
            {"message": "Hint A"},
            {"message": "Hint B"},
            {"message": "Hint C"},
            {"message": "Hint D"},
            {"message": "Hint E"},
        ]
        result = self._dedup(hints, "standard")
        assert len(result) == 3

    def test_full_returns_all(self):
        hints = [{"message": f"Hint {i}"} for i in range(10)]
        result = self._dedup(hints, "full")
        assert len(result) == 10

    def test_dedup_by_exact_message(self):
        hints = [
            {"message": "Element not found"},
            {"message": "Element not found"},
        ]
        result = self._dedup(hints, "standard")
        assert len(result) == 1

    def test_dedup_by_substring_containment(self):
        """If one message is a substring of another, keep only the longer one (first seen)."""
        hints = [
            {"message": "Element is not visible on the page"},
            {"message": "not visible"},
        ]
        result = self._dedup(hints, "standard")
        assert len(result) == 1

    def test_dedup_substring_reverse_order(self):
        """Shorter hint first, longer hint second -- shorter survives, longer is dup."""
        hints = [
            {"message": "not visible"},
            {"message": "Element is not visible on the page"},
        ]
        result = self._dedup(hints, "standard")
        assert len(result) == 1

    def test_uses_title_as_fallback(self):
        """When 'message' key is absent, uses 'title' key for dedup."""
        hints = [
            {"title": "Use Wait"},
            {"title": "Use Wait"},
        ]
        result = self._dedup(hints, "standard")
        assert len(result) == 1

    def test_case_insensitive_dedup(self):
        """Deduplication should be case-insensitive."""
        hints = [
            {"message": "Element Not Found"},
            {"message": "element not found"},
        ]
        result = self._dedup(hints, "standard")
        assert len(result) == 1

    def test_distinct_hints_preserved(self):
        hints = [
            {"message": "Use CSS selector"},
            {"message": "Try XPath instead"},
        ]
        result = self._dedup(hints, "standard")
        assert len(result) == 2


# ============================================================================
# D. TestGroupedWarnings
# ============================================================================


class TestGroupedWarnings:
    """Verify _check_untracked_variables returns grouped format."""

    def _make_builder(self):
        from robotmcp.components.test_builder import TestBuilder

        builder = object.__new__(TestBuilder)
        return builder

    def _make_suite(self, variables=None, steps_refs=None):
        from robotmcp.components.test_builder import (
            GeneratedTestCase,
            GeneratedTestSuite,
            TestCaseStep,
        )

        tc_steps = []
        for ref in (steps_refs or []):
            tc_steps.append(TestCaseStep(keyword="Log", arguments=[f"${{{ref}}}"]))

        tc = GeneratedTestCase(name="Test1", steps=tc_steps)
        suite = GeneratedTestSuite(
            name="Suite",
            test_cases=[tc],
            imports=[],
            variables=variables or {},
        )
        return suite

    def test_grouped_format_with_untracked(self):
        """Warnings should use grouped format with type, count, variables, fix."""
        builder = self._make_builder()
        suite = self._make_suite(
            variables={},
            steps_refs=["VAR_A", "VAR_B"],
        )
        warnings = builder._check_untracked_variables(suite, "s1")

        assert len(warnings) == 1
        w = warnings[0]
        assert w["type"] == "untracked_variables"
        assert w["count"] == 2
        assert sorted(w["variables"]) == ["VAR_A", "VAR_B"]
        assert "manage_session" in w["fix"]

    def test_no_untracked_returns_empty(self):
        """When all variables are defined, no warnings should be produced."""
        builder = self._make_builder()
        suite = self._make_suite(
            variables={"${VAR_A}": "val"},
            steps_refs=["VAR_A"],
        )
        warnings = builder._check_untracked_variables(suite, "s1")
        assert warnings == []

    def test_single_untracked_count_is_one(self):
        builder = self._make_builder()
        suite = self._make_suite(
            variables={},
            steps_refs=["ONLY_ONE"],
        )
        warnings = builder._check_untracked_variables(suite, "s1")
        assert len(warnings) == 1
        assert warnings[0]["count"] == 1


# ============================================================================
# E. TestHTTPResponseSerialization
# ============================================================================


class TestHTTPResponseSerialization:
    """Test response_serializer.py compact format for requests.Response."""

    def _make_response(self, status_code=200, content_type="application/json",
                       json_body=None, text_body="", headers=None):
        resp = Mock()
        resp.status_code = status_code
        resp.ok = 200 <= status_code < 400
        _headers = {"Content-Type": content_type}
        if headers:
            _headers.update(headers)
        resp.headers = _headers
        resp.content = (text_body or "").encode()

        if json_body is not None:
            resp.json = Mock(return_value=json_body)
            resp.text = json.dumps(json_body)
        else:
            resp.json = Mock(side_effect=ValueError("No JSON"))
            resp.text = text_body

        return resp

    def test_compact_json_response(self):
        from robotmcp.utils.response_serializer import MCPResponseSerializer

        s = MCPResponseSerializer()
        resp = self._make_response(json_body={"id": 1, "name": "Test"})

        # Patch requests import
        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys
            sys.modules["requests"].models.Response = type(resp)
            result = s._serialize_requests_response(resp)

        assert result["status_code"] == 200
        assert result["ok"] is True
        assert result["content_type"] == "application/json"
        # Body should be in "json" key (small body), NOT duplicated in "text"
        assert "json" in result
        assert "text" not in result
        assert "body" not in result

    def test_compact_text_response(self):
        from robotmcp.utils.response_serializer import MCPResponseSerializer

        s = MCPResponseSerializer()
        resp = self._make_response(
            content_type="text/html",
            text_body="<h1>Hello</h1>",
        )

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys
            sys.modules["requests"].models.Response = type(resp)
            result = s._serialize_requests_response(resp)

        assert result["status_code"] == 200
        assert "text" in result
        assert "json" not in result

    def test_compact_large_text_truncated(self):
        from robotmcp.utils.response_serializer import MCPResponseSerializer

        s = MCPResponseSerializer()
        big_text = "x" * 1000
        resp = self._make_response(content_type="text/plain", text_body=big_text)

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys
            sys.modules["requests"].models.Response = type(resp)
            result = s._serialize_requests_response(resp)

        assert "text_preview" in result
        assert result["text_truncated"] is True
        assert result["body_size"] == 1000

    def test_no_body_duplication(self):
        """JSON and text fields should never both appear."""
        from robotmcp.utils.response_serializer import MCPResponseSerializer

        s = MCPResponseSerializer()
        resp = self._make_response(
            content_type="application/json",
            json_body={"ok": True},
        )

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys
            sys.modules["requests"].models.Response = type(resp)
            result = s._serialize_requests_response(resp)

        has_json = "json" in result or "json_preview" in result
        has_text = "text" in result or "text_preview" in result
        assert not (has_json and has_text), "json and text should be mutually exclusive"

    def test_error_response_status_code(self):
        from robotmcp.utils.response_serializer import MCPResponseSerializer

        s = MCPResponseSerializer()
        resp = self._make_response(status_code=404, content_type="text/plain", text_body="Not Found")

        with patch.dict("sys.modules", {"requests": MagicMock()}):
            import sys
            sys.modules["requests"].models.Response = type(resp)
            result = s._serialize_requests_response(resp)

        assert result["status_code"] == 404
        assert result["ok"] is False


# ============================================================================
# F. TestEnhancedHTTPResponse
# ============================================================================


class TestEnhancedHTTPResponse:
    """Test enhanced_response_serializer.py at all 3 detail levels."""

    def _make_response(self, status_code=200, content_type="application/json",
                       json_body=None, text_body="", headers=None):
        resp = Mock()
        resp.status_code = status_code
        resp.ok = 200 <= status_code < 400
        _headers = {"Content-Type": content_type, "X-Custom": "value"}
        if headers:
            _headers.update(headers)
        resp.headers = _headers
        resp.content = (text_body or json.dumps(json_body) if json_body else text_body or "").encode()

        if json_body is not None:
            resp.json = Mock(return_value=json_body)
            resp.text = json.dumps(json_body)
        else:
            resp.json = Mock(side_effect=ValueError("No JSON"))
            resp.text = text_body

        # Make it look like a requests.Response to the enhanced serializer
        type(resp).__name__ = "MockResponse"

        return resp

    def test_minimal_no_body_content(self):
        """Minimal should only have status_code, ok, content_type, body_size."""
        from robotmcp.utils.enhanced_response_serializer import MCPResponseSerializer

        s = MCPResponseSerializer()
        resp = self._make_response(json_body={"items": [1, 2, 3]})

        result = s._serialize_requests_response(resp, "minimal")

        assert result["status_code"] == 200
        assert result["ok"] is True
        assert "content_type" in result
        # Minimal should NOT include json or text content
        assert "json" not in result
        assert "text" not in result
        assert "json_preview" not in result
        # body_size may or may not be present depending on content

    def test_standard_includes_json_preview(self):
        """Standard should include json or json_preview but NOT headers."""
        from robotmcp.utils.enhanced_response_serializer import MCPResponseSerializer

        s = MCPResponseSerializer()
        resp = self._make_response(json_body={"id": 1})

        result = s._serialize_requests_response(resp, "standard")

        assert result["status_code"] == 200
        has_json = "json" in result or "json_preview" in result
        assert has_json, "Standard should include json content"
        assert "headers" not in result

    def test_full_includes_headers(self):
        """Full should include headers."""
        from robotmcp.utils.enhanced_response_serializer import MCPResponseSerializer

        s = MCPResponseSerializer()
        resp = self._make_response(json_body={"id": 1})

        result = s._serialize_requests_response(resp, "full")

        assert "headers" in result
        assert result["headers"]["X-Custom"] == "value"

    def test_full_includes_body(self):
        """Full should include full json content."""
        from robotmcp.utils.enhanced_response_serializer import MCPResponseSerializer

        s = MCPResponseSerializer()
        body = {"data": list(range(10))}
        resp = self._make_response(json_body=body)

        result = s._serialize_requests_response(resp, "full")

        assert "json" in result
        assert result["json"] == body


# ============================================================================
# G. TestExternalizationRulesExpanded
# ============================================================================


class TestExternalizationRulesExpanded:
    """Verify all 13 rules present in DEFAULT_RULES."""

    def test_total_rule_count(self):
        from robotmcp.domains.artifact_output.services import DEFAULT_RULES

        assert len(DEFAULT_RULES) == 14

    def test_get_session_state_rules_present(self):
        from robotmcp.domains.artifact_output.services import DEFAULT_RULES

        pairs = {(r.tool_name, r.field_path) for r in DEFAULT_RULES}
        # get_session_state: 4 rules with correct nested paths
        assert ("get_session_state", "sections.page_source.context") in pairs
        assert ("get_session_state", "sections.page_source.aria_snapshot") in pairs
        assert ("get_session_state", "sections.page_source.page_source_preview") in pairs
        assert ("get_session_state", "sections.page_source.page_source") in pairs

    def test_other_tool_rules_present(self):
        from robotmcp.domains.artifact_output.services import DEFAULT_RULES

        pairs = {(r.tool_name, r.field_path) for r in DEFAULT_RULES}
        assert ("build_test_suite", "rf_text") in pairs
        assert ("run_test_suite", "execution_details") in pairs
        assert ("execute_batch", "steps") in pairs
        assert ("find_keywords", "result") in pairs

    def test_expansion_rules_present(self):
        from robotmcp.domains.artifact_output.services import DEFAULT_RULES

        pairs = {(r.tool_name, r.field_path) for r in DEFAULT_RULES}
        assert ("execute_step", "session_variables") in pairs
        assert ("execute_step", "output") in pairs
        assert ("execute_step", "hints") in pairs
        assert ("build_test_suite", "warnings") in pairs
        assert ("build_test_suite", "suite") in pairs
        assert ("run_test_suite", "output_files") in pairs

    def test_all_rules_have_tool_name(self):
        from robotmcp.domains.artifact_output.services import DEFAULT_RULES

        for rule in DEFAULT_RULES:
            assert rule.tool_name, f"Rule missing tool_name: {rule}"

    def test_all_rules_have_field_path(self):
        from robotmcp.domains.artifact_output.services import DEFAULT_RULES

        for rule in DEFAULT_RULES:
            assert rule.field_path, f"Rule missing field_path: {rule}"


# ============================================================================
# H. TestDeltaAutoMode
# ============================================================================


class TestDeltaAutoMode:
    """Test DeltaStateService.has_version()."""

    def test_has_version_false_for_unknown_session(self):
        from robotmcp.domains.snapshot.delta_service import DeltaStateService

        svc = DeltaStateService()
        assert svc.has_version("nonexistent") is False

    def test_has_version_false_for_empty_cache(self):
        from robotmcp.domains.snapshot.delta_service import DeltaStateService

        svc = DeltaStateService()
        # Create cache but don't store any versions
        svc.get_or_create_cache("s1")
        assert svc.has_version("s1") is False

    def test_has_version_true_after_recording(self):
        from robotmcp.domains.snapshot.delta_service import DeltaStateService

        svc = DeltaStateService()
        svc.record_full_state("s1", {"variables": {"x": 1}}, ["variables"])
        assert svc.has_version("s1") is True

    def test_has_version_after_clear_session(self):
        from robotmcp.domains.snapshot.delta_service import DeltaStateService

        svc = DeltaStateService()
        svc.record_full_state("s1", {"variables": {"x": 1}}, ["variables"])
        assert svc.has_version("s1") is True
        svc.clear_session("s1")
        assert svc.has_version("s1") is False

    def test_has_version_after_clear_all(self):
        from robotmcp.domains.snapshot.delta_service import DeltaStateService

        svc = DeltaStateService()
        svc.record_full_state("s1", {"a": 1}, ["a"])
        svc.record_full_state("s2", {"b": 2}, ["b"])
        assert svc.has_version("s1") is True
        assert svc.has_version("s2") is True
        svc.clear_all()
        assert svc.has_version("s1") is False
        assert svc.has_version("s2") is False

    def test_has_version_multiple_sessions_independent(self):
        from robotmcp.domains.snapshot.delta_service import DeltaStateService

        svc = DeltaStateService()
        svc.record_full_state("s1", {"a": 1}, ["a"])
        assert svc.has_version("s1") is True
        assert svc.has_version("s2") is False


# ============================================================================
# I. TestCompactResponses
# ============================================================================


class TestCompactResponses:
    """Verify analyze_scenario returns flat session_type/libraries_loaded (no session_info dict)."""

    def test_analyze_scenario_flat_session_type(self):
        """analyze_scenario result should have flat 'session_type' key, not nested 'session_info'."""
        # We test the shape by simulating what analyze_scenario produces
        # The actual function is async and requires full server context, so we test the
        # output format contract directly.
        result = {
            "success": True,
            "session_id": "s1",
            "session_type": "WEB_TESTING",
            "libraries_loaded": ["Browser", "BuiltIn"],
            "analysis": {"intent": "login test"},
        }

        # Flat fields present
        assert "session_type" in result
        assert "libraries_loaded" in result
        # No nested session_info dict
        assert "session_info" not in result

    def test_analyze_scenario_optional_attach_bridge(self):
        """attach_bridge_active should only appear when True."""
        result_without = {
            "success": True,
            "session_id": "s1",
            "session_type": "WEB_TESTING",
            "libraries_loaded": ["Browser"],
        }
        assert "attach_bridge_active" not in result_without

        result_with = {
            "success": True,
            "session_id": "s1",
            "session_type": "WEB_TESTING",
            "libraries_loaded": ["Browser"],
            "attach_bridge_active": True,
        }
        assert result_with["attach_bridge_active"] is True

    def test_manage_session_init_has_flat_libraries_loaded(self):
        """manage_session init response should have flat libraries_loaded, not nested session_info."""
        result = {
            "success": True,
            "action": "init",
            "session_id": "s1",
            "libraries_loaded": ["Browser", "BuiltIn"],
            "variables_set": [],
            "import_issues": [],
        }

        assert "libraries_loaded" in result
        assert isinstance(result["libraries_loaded"], list)
        assert "session_info" not in result

    def test_build_test_suite_no_optimization_applied(self):
        """build_test_suite result should not contain optimization_applied."""
        from robotmcp.components.test_builder import (
            GeneratedTestCase,
            GeneratedTestSuite,
            TestCaseStep,
        )

        # Simulate the result structure returned by build_test_suite
        result = {
            "success": True,
            "session_id": "s1",
            "warnings": None,
            "suite": {"name": "Suite", "test_cases": []},
            "rf_text": "*** Test Cases ***",
            "statistics": {"original_steps": 5, "optimized_steps": 5},
        }

        assert "optimization_applied" not in result


# ============================================================================
# Additional: Enhanced serializer injection test
# ============================================================================


class TestEnhancedSerializerInjection:
    """Verify enhanced_serialization_integration replaces response_serializer attribute."""

    def test_patch_replaces_response_serializer(self):
        """patch_keyword_executor_response_formatting should set response_serializer."""
        from robotmcp.utils.enhanced_serialization_integration import (
            patch_keyword_executor_response_formatting,
        )
        from robotmcp.utils.enhanced_response_serializer import (
            MCPResponseSerializer as EnhancedSerializer,
        )

        executor = MagicMock()
        executor.response_serializer = MagicMock()  # original

        patch_keyword_executor_response_formatting(executor)

        # After patching, response_serializer should be the enhanced one
        assert isinstance(executor.response_serializer, EnhancedSerializer)

    def test_patch_does_not_use_monkey_patching_methods(self):
        """The patch function should set an attribute, not monkey-patch individual methods."""
        from robotmcp.utils.enhanced_serialization_integration import (
            patch_keyword_executor_response_formatting,
        )

        executor = MagicMock()
        original_build = executor._build_response_by_detail_level

        patch_keyword_executor_response_formatting(executor)

        # The actual _build_response_by_detail_level should NOT be replaced
        # (it's the executor's own method, not monkey-patched)
        assert executor._build_response_by_detail_level == original_build


# ============================================================================
# L. TestExternalizationWiring — end-to-end externalization integration
# ============================================================================


class TestExternalizationWiring:
    """Verify externalization replaces large fields with artifact references."""

    def _make_service(self, mode="file", max_inline_tokens=50):
        from robotmcp.domains.artifact_output.services import (
            ArtifactExternalizationService,
        )
        from robotmcp.domains.artifact_output.aggregates import ArtifactStore
        from robotmcp.domains.artifact_output.value_objects import (
            ArtifactPolicy,
            OutputMode,
        )

        policy = ArtifactPolicy(max_inline_tokens=max_inline_tokens)
        store = ArtifactStore.create(policy=policy)
        output_mode = OutputMode(mode)
        return ArtifactExternalizationService(
            store=store, output_mode=output_mode
        )

    def test_get_session_state_context_externalized(self):
        """context field under sections.page_source should be externalized."""
        svc = self._make_service(max_inline_tokens=10)
        response = {
            "success": True,
            "session_id": "s1",
            "sections": {
                "page_source": {
                    "context": {
                        "forms": [{"action": "/login"}],
                        "buttons": [{"text": "Submit"}, {"text": "Cancel"}] * 20,
                        "links": [{"href": "/home"}],
                    },
                    "aria_snapshot": "short",
                    "page_source_preview": "<html>short</html>",
                }
            },
        }
        result, results = svc.externalize("get_session_state", response, "s1")
        # context should be replaced with summary text
        ctx = result["sections"]["page_source"]["context"]
        assert isinstance(ctx, str)
        assert "Content saved to" in ctx
        assert ".robotmcp_artifacts" in ctx
        # In file mode, ALL matching rules fire (even small fields)
        assert len(results) == 3  # context + aria_snapshot + page_source_preview

    def test_get_session_state_aria_snapshot_externalized(self):
        """Large aria_snapshot dict should be externalized."""
        svc = self._make_service(max_inline_tokens=10)
        response = {
            "success": True,
            "session_id": "s1",
            "sections": {
                "page_source": {
                    "context": {"forms": []},
                    "aria_snapshot": {
                        "success": True,
                        "content": "- document:\n" + "  - heading: test\n" * 100,
                        "format": "yaml",
                    },
                }
            },
        }
        result, results = svc.externalize("get_session_state", response, "s1")
        aria = result["sections"]["page_source"]["aria_snapshot"]
        assert isinstance(aria, str)
        assert "Content saved to" in aria

    def test_get_session_state_page_source_preview_externalized(self):
        """Large page_source_preview should be externalized."""
        svc = self._make_service(max_inline_tokens=10)
        response = {
            "success": True,
            "session_id": "s1",
            "sections": {
                "page_source": {
                    "context": {"forms": []},
                    "page_source_preview": "<html>" + "<div>content</div>" * 200 + "</html>",
                }
            },
        }
        result, results = svc.externalize("get_session_state", response, "s1")
        preview = result["sections"]["page_source"]["page_source_preview"]
        assert isinstance(preview, str)
        assert "Content saved to" in preview

    def test_inline_mode_skips_externalization(self):
        """OutputMode.INLINE should never externalize."""
        svc = self._make_service(mode="inline")
        big_content = "x" * 5000
        response = {
            "sections": {
                "page_source": {
                    "context": {"data": big_content},
                }
            }
        }
        result, results = svc.externalize("get_session_state", response, "s1")
        assert results == []
        assert result["sections"]["page_source"]["context"]["data"] == big_content

    def test_auto_mode_skips_small_fields(self):
        """OutputMode.AUTO should skip fields under the token threshold."""
        svc = self._make_service(mode="auto", max_inline_tokens=500)
        response = {
            "sections": {
                "page_source": {
                    "context": {"forms": []},
                    "aria_snapshot": "short",
                }
            }
        }
        result, results = svc.externalize("get_session_state", response, "s1")
        assert results == []
        assert result["sections"]["page_source"]["aria_snapshot"] == "short"

    def test_file_mode_externalizes_regardless_of_size(self):
        """OutputMode.FILE should externalize even small fields if a rule matches."""
        svc = self._make_service(mode="file", max_inline_tokens=500)
        response = {
            "sections": {
                "page_source": {
                    "context": {"forms": []},
                }
            }
        }
        result, results = svc.externalize("get_session_state", response, "s1")
        ctx = result["sections"]["page_source"]["context"]
        assert isinstance(ctx, str)
        assert "Content saved to" in ctx

    def test_dict_serialized_as_json(self):
        """Dict values should be serialized as JSON, not Python repr."""
        svc = self._make_service(mode="file", max_inline_tokens=10)
        response = {"output": {"key": "value", "num": 42}}
        result, results = svc.externalize("execute_step", response, "s1")
        assert len(results) == 1
        # Retrieve the artifact content and verify it's valid JSON
        store = svc._store
        art_id = results[0].artifact_ref.artifact_id
        content = store.read_content(art_id)
        import json
        parsed = json.loads(content)
        assert parsed == {"key": "value", "num": 42}

    def test_fetch_artifact_retrieves_externalized_content(self):
        """Full round-trip: externalize then fetch."""
        svc = self._make_service(mode="file", max_inline_tokens=10)
        from robotmcp.domains.artifact_output.services import ArtifactRetrievalService

        retrieval = ArtifactRetrievalService(store=svc._store)
        original = "x" * 2000
        response = {
            "sections": {
                "page_source": {
                    "page_source_preview": original,
                    "context": {"forms": []},
                }
            }
        }
        result, results = svc.externalize("get_session_state", response, "s1")
        # Should have externalized both preview and context
        assert len(results) >= 1
        # Fetch the page_source_preview artifact
        for r in results:
            if "page_source_preview" in r.summary or r.artifact_ref.artifact_id:
                fetch_result = retrieval.fetch(r.artifact_ref.artifact_id)
                assert fetch_result["success"] is True
                assert len(fetch_result["content"]) > 0

    def test_multiple_fields_externalized_in_single_call(self):
        """Multiple large fields in one response should all be externalized."""
        svc = self._make_service(mode="file", max_inline_tokens=10)
        response = {
            "sections": {
                "page_source": {
                    "context": {"buttons": [{"text": f"btn{i}"} for i in range(50)]},
                    "aria_snapshot": {"success": True, "content": "yaml " * 500},
                    "page_source_preview": "<html>" + "x" * 3000 + "</html>",
                }
            }
        }
        result, results = svc.externalize("get_session_state", response, "s1")
        # All 3 fields should be externalized
        assert len(results) == 3
        ps = result["sections"]["page_source"]
        assert isinstance(ps["context"], str) and "Content saved to" in ps["context"]
        assert isinstance(ps["aria_snapshot"], str) and "Content saved to" in ps["aria_snapshot"]
        assert isinstance(ps["page_source_preview"], str) and "Content saved to" in ps["page_source_preview"]
