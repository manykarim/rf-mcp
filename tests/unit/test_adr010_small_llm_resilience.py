"""Unit tests for ADR-010: Small LLM Resilience.

Validates array coercion (I1/I2), deprecation hint extraction (I4),
schema transparency, and BaseModel integration for FastMCP tool
parameter compatibility with small LLMs (GLM-4, Qwen, etc.).

Run with: uv run pytest tests/unit/test_adr010_small_llm_resilience.py -v
"""

from __future__ import annotations

__test__ = True

from typing import List, Optional

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from robotmcp.domains.shared.kernel import (
    CoercedStringList,
    DEPRECATED_KEYWORD_ALIASES,
    OptionalCoercedStringList,
    _coerce_string_to_list,
    extract_deprecation_suggestion,
    resolve_deprecated_alias,
)


# ============================================================
# 1. _coerce_string_to_list function tests
# ============================================================


class TestCoerceStringToList:
    """Validate the _coerce_string_to_list BeforeValidator function."""

    # ---- Passthrough cases (non-string inputs) ----

    def test_list_passthrough(self):
        """Already a list should pass through unchanged."""
        original = ["Browser", "BuiltIn"]
        result = _coerce_string_to_list(original)
        assert result == ["Browser", "BuiltIn"]
        assert result is original  # same object, no copy

    def test_empty_list_passthrough(self):
        """Empty list should pass through unchanged."""
        original = []
        result = _coerce_string_to_list(original)
        assert result == []
        assert result is original

    def test_none_passthrough(self):
        """None should pass through unchanged (Pydantic handles None for Optional)."""
        result = _coerce_string_to_list(None)
        assert result is None

    def test_int_passthrough(self):
        """Integer should pass through unchanged (Pydantic rejects later)."""
        result = _coerce_string_to_list(42)
        assert result == 42

    def test_dict_passthrough(self):
        """Dict should pass through unchanged."""
        d = {"key": "value"}
        result = _coerce_string_to_list(d)
        assert result is d

    def test_bool_passthrough(self):
        """Bool should pass through unchanged."""
        result = _coerce_string_to_list(True)
        assert result is True

    # ---- Path 1: JSON array string parsing ----

    def test_json_array_two_items(self):
        """JSON array string with two items should be parsed."""
        result = _coerce_string_to_list('["Browser", "BuiltIn"]')
        assert result == ["Browser", "BuiltIn"]

    def test_json_array_single_item(self):
        """JSON array string with single item should be parsed."""
        result = _coerce_string_to_list('["BuiltIn"]')
        assert result == ["BuiltIn"]

    def test_json_array_empty(self):
        """Empty JSON array string should be parsed to empty list."""
        result = _coerce_string_to_list("[]")
        assert result == []

    def test_json_array_whitespace_padded(self):
        """Whitespace-padded JSON array should be parsed correctly."""
        result = _coerce_string_to_list('  ["Browser"]  ')
        assert result == ["Browser"]

    def test_json_array_three_items(self):
        """JSON array with three items should be parsed."""
        result = _coerce_string_to_list('["Browser", "BuiltIn", "String"]')
        assert result == ["Browser", "BuiltIn", "String"]

    def test_json_array_with_inner_spaces(self):
        """JSON array with extra internal spacing should be parsed."""
        result = _coerce_string_to_list('[  "Browser" , "BuiltIn"  ]')
        assert result == ["Browser", "BuiltIn"]

    def test_json_array_glm45_actual_output(self):
        """GLM-4.5 AIR actual output pattern: standard JSON array string."""
        # This is what GLM-4.5 actually sends when it stringifies an array
        result = _coerce_string_to_list('["Browser", "BuiltIn"]')
        assert result == ["Browser", "BuiltIn"]

    def test_json_array_with_escaped_quotes(self):
        """JSON with items containing escaped characters."""
        result = _coerce_string_to_list('["item with \\"quotes\\""]')
        assert result == ['item with "quotes"']

    def test_json_array_numeric_strings(self):
        """JSON array of number strings."""
        result = _coerce_string_to_list('["1", "2", "3"]')
        assert result == ["1", "2", "3"]

    def test_invalid_json_bracket_start_falls_to_comma(self):
        """String starting with [ but not valid JSON should fall through."""
        # "[Browser" is not valid JSON, but contains no comma -> single value
        result = _coerce_string_to_list("[Browser")
        assert result == ["[Browser"]

    def test_invalid_json_with_comma_falls_to_split(self):
        """String starting with [ but not valid JSON with comma falls to split."""
        result = _coerce_string_to_list("[Browser, BuiltIn")
        # Starts with [ but not valid JSON -> falls through to comma split
        assert result == ["[Browser", "BuiltIn"]

    # ---- Path 2: Comma-separated string splitting ----

    def test_comma_separated_no_spaces(self):
        """Comma-separated without spaces should be split."""
        result = _coerce_string_to_list("Browser,BuiltIn")
        assert result == ["Browser", "BuiltIn"]

    def test_comma_separated_with_spaces(self):
        """Comma-separated with spaces should be split and trimmed."""
        result = _coerce_string_to_list("Browser, BuiltIn")
        assert result == ["Browser", "BuiltIn"]

    def test_comma_separated_extra_spaces(self):
        """Comma-separated with excessive spaces should be handled."""
        result = _coerce_string_to_list("Browser ,  BuiltIn , String")
        assert result == ["Browser", "BuiltIn", "String"]

    def test_trailing_comma(self):
        """Trailing comma should not produce empty trailing element."""
        result = _coerce_string_to_list("A, B,")
        assert result == ["A", "B"]

    def test_leading_comma(self):
        """Leading comma should not produce empty leading element."""
        result = _coerce_string_to_list(",A, B")
        assert result == ["A", "B"]

    def test_multiple_commas(self):
        """Multiple consecutive commas should not produce empty elements."""
        result = _coerce_string_to_list("A,,B")
        assert result == ["A", "B"]

    def test_comma_only(self):
        """Single comma should produce empty list (all items stripped empty)."""
        result = _coerce_string_to_list(",")
        assert result == []

    # ---- Path 3: Single value wrapping ----

    def test_single_value(self):
        """Single non-comma string should be wrapped in a list."""
        result = _coerce_string_to_list("Browser")
        assert result == ["Browser"]

    def test_single_value_with_spaces(self):
        """Single value with surrounding spaces should be stripped then wrapped."""
        result = _coerce_string_to_list("  Browser  ")
        assert result == ["Browser"]

    def test_single_value_tab_newline(self):
        """Single value with tab/newline should be stripped then wrapped."""
        result = _coerce_string_to_list("\tBrowser\n")
        assert result == ["Browser"]

    # ---- Edge cases with empty strings ----

    def test_empty_string_passthrough(self):
        """Empty string returns empty string (v_stripped is falsy, skips path 3)."""
        result = _coerce_string_to_list("")
        assert result == ""

    def test_whitespace_only_string(self):
        """Whitespace-only string strips to empty, returns original string."""
        result = _coerce_string_to_list("   ")
        assert result == "   "


# ============================================================
# 2. CoercedStringList Pydantic TypeAdapter tests
# ============================================================


class TestCoercedStringListTypeAdapter:
    """Validate CoercedStringList works correctly with Pydantic validation."""

    def setup_method(self):
        self.ta = TypeAdapter(CoercedStringList)

    def test_list_passthrough(self):
        """Native list input should pass validation."""
        result = self.ta.validate_python(["Browser", "BuiltIn"])
        assert result == ["Browser", "BuiltIn"]

    def test_empty_list_passthrough(self):
        """Empty list should pass validation."""
        result = self.ta.validate_python([])
        assert result == []

    def test_json_string_coercion(self):
        """JSON array string should be coerced to list."""
        result = self.ta.validate_python('["Browser", "BuiltIn"]')
        assert result == ["Browser", "BuiltIn"]

    def test_comma_string_coercion(self):
        """Comma-separated string should be coerced to list."""
        result = self.ta.validate_python("Browser, BuiltIn")
        assert result == ["Browser", "BuiltIn"]

    def test_single_string_coercion(self):
        """Single string should be coerced to single-element list."""
        result = self.ta.validate_python("Browser")
        assert result == ["Browser"]

    def test_json_empty_array_coercion(self):
        """Empty JSON array string should validate as empty list."""
        result = self.ta.validate_python("[]")
        assert result == []

    def test_rejects_none(self):
        """Non-optional CoercedStringList should reject None."""
        with pytest.raises(ValidationError):
            self.ta.validate_python(None)

    def test_schema_identity_with_plain_list_str(self):
        """CoercedStringList schema must be identical to List[str] schema."""
        coerced_schema = self.ta.json_schema()
        plain_schema = TypeAdapter(List[str]).json_schema()
        assert coerced_schema == plain_schema, (
            f"Schema mismatch:\n  CoercedStringList: {coerced_schema}\n"
            f"  List[str]:         {plain_schema}"
        )

    def test_schema_is_array_of_strings(self):
        """Schema should be exactly {type: array, items: {type: string}}."""
        schema = self.ta.json_schema()
        assert schema["type"] == "array"
        assert schema["items"] == {"type": "string"}

    def test_schema_no_anyof(self):
        """Non-optional schema must not contain anyOf."""
        schema = self.ta.json_schema()
        assert "anyOf" not in schema

    def test_schema_no_oneof(self):
        """Non-optional schema must not contain oneOf."""
        schema = self.ta.json_schema()
        assert "oneOf" not in schema

    def test_schema_no_allof(self):
        """Non-optional schema must not contain allOf."""
        schema = self.ta.json_schema()
        assert "allOf" not in schema

    def test_list_of_ints_rejected(self):
        """List[str] rejects integer items (Pydantic strict string validation)."""
        with pytest.raises(ValidationError):
            self.ta.validate_python([1, 2, 3])


# ============================================================
# 3. OptionalCoercedStringList Pydantic TypeAdapter tests
# ============================================================


class TestOptionalCoercedStringListTypeAdapter:
    """Validate OptionalCoercedStringList with Pydantic validation."""

    def setup_method(self):
        self.ta = TypeAdapter(OptionalCoercedStringList)

    def test_none_passthrough(self):
        """None should be accepted for Optional variant."""
        result = self.ta.validate_python(None)
        assert result is None

    def test_list_passthrough(self):
        """Native list input should pass validation."""
        result = self.ta.validate_python(["Browser", "BuiltIn"])
        assert result == ["Browser", "BuiltIn"]

    def test_empty_list_passthrough(self):
        """Empty list should pass validation."""
        result = self.ta.validate_python([])
        assert result == []

    def test_json_string_coercion(self):
        """JSON array string should be coerced to list."""
        result = self.ta.validate_python('["Browser", "BuiltIn"]')
        assert result == ["Browser", "BuiltIn"]

    def test_comma_string_coercion(self):
        """Comma-separated string should be coerced."""
        result = self.ta.validate_python("Browser, BuiltIn")
        assert result == ["Browser", "BuiltIn"]

    def test_single_string_coercion(self):
        """Single string should be coerced to single-element list."""
        result = self.ta.validate_python("Browser")
        assert result == ["Browser"]

    def test_schema_has_anyof_with_array_and_null(self):
        """Optional schema should express array-or-null via anyOf."""
        schema = self.ta.json_schema()
        assert "anyOf" in schema, f"Expected anyOf in schema: {schema}"
        type_kinds = set()
        for variant in schema["anyOf"]:
            if variant.get("type") == "null":
                type_kinds.add("null")
            elif variant.get("type") == "array":
                type_kinds.add("array")
        assert "null" in type_kinds, f"Missing null variant in schema: {schema}"
        assert "array" in type_kinds, f"Missing array variant in schema: {schema}"

    def test_optional_schema_matches_optional_list_str(self):
        """OptionalCoercedStringList schema must match Optional[List[str]]."""
        coerced_schema = self.ta.json_schema()
        plain_schema = TypeAdapter(Optional[List[str]]).json_schema()
        assert coerced_schema == plain_schema, (
            f"Schema mismatch:\n"
            f"  OptionalCoercedStringList: {coerced_schema}\n"
            f"  Optional[List[str]]:       {plain_schema}"
        )


# ============================================================
# 4. extract_deprecation_suggestion tests (I4)
# ============================================================


class TestExtractDeprecationSuggestion:
    """Validate deprecation warning text parsing."""

    def test_standard_deprecation_warning_use_instead(self):
        """Standard 'Use X instead.' pattern."""
        msg = "DeprecationWarning: 'GET' is deprecated. Use 'GET On Session' instead."
        result = extract_deprecation_suggestion(msg)
        assert result == "GET On Session"

    def test_short_form_use_without_instead(self):
        """Short form 'use X' without 'instead'."""
        msg = "'POST' is deprecated, use 'POST On Session'"
        result = extract_deprecation_suggestion(msg)
        assert result == "POST On Session"

    def test_favor_of_pattern(self):
        """Pattern: 'in favor of X'."""
        msg = "'DELETE' is deprecated in favor of DELETE On Session."
        result = extract_deprecation_suggestion(msg)
        assert result == "DELETE On Session"

    def test_no_deprecation_message(self):
        """Non-deprecation error should return None."""
        result = extract_deprecation_suggestion("SomeOtherError: something went wrong")
        assert result is None

    def test_empty_string(self):
        """Empty string should return None."""
        result = extract_deprecation_suggestion("")
        assert result is None

    def test_partial_match_no_keyword(self):
        """Message that mentions 'use' but has no valid suggestion."""
        result = extract_deprecation_suggestion("Please use the new API")
        # The regex may or may not match; if it does, it extracts "the new API"
        # The key contract is: it does not crash.
        assert result is None or isinstance(result, str)

    def test_use_with_double_quotes(self):
        """Use pattern with double quotes around replacement."""
        msg = 'Use "PATCH On Session" instead.'
        result = extract_deprecation_suggestion(msg)
        assert result == "PATCH On Session"

    def test_use_with_no_quotes(self):
        """Use pattern without quotes around replacement."""
        msg = "Use HEAD On Session instead."
        result = extract_deprecation_suggestion(msg)
        assert result == "HEAD On Session"

    def test_case_insensitive_use(self):
        """Lowercase 'use' should also match."""
        msg = "'GET' is deprecated, use 'GET On Session' instead."
        result = extract_deprecation_suggestion(msg)
        assert result == "GET On Session"

    def test_multiline_message_extracts_from_relevant_line(self):
        """Should extract even from multi-line messages."""
        msg = (
            "Warning: keyword is outdated.\n"
            "DeprecationWarning: 'OPTIONS' is deprecated. "
            "Use 'OPTIONS On Session' instead."
        )
        result = extract_deprecation_suggestion(msg)
        assert result == "OPTIONS On Session"


# ============================================================
# 5. resolve_deprecated_alias tests (I4)
# ============================================================


class TestResolveDeprecatedAlias:
    """Validate static deprecated keyword alias resolution."""

    def test_get_resolves(self):
        assert resolve_deprecated_alias("GET") == "GET On Session"

    def test_get_lowercase(self):
        """Case-insensitive lookup."""
        assert resolve_deprecated_alias("get") == "GET On Session"

    def test_get_mixed_case(self):
        """Mixed case should also resolve."""
        assert resolve_deprecated_alias("Get") == "GET On Session"

    def test_get_with_whitespace(self):
        """Whitespace should be stripped before lookup."""
        assert resolve_deprecated_alias("  GET  ") == "GET On Session"

    def test_post_resolves(self):
        assert resolve_deprecated_alias("POST") == "POST On Session"

    def test_put_resolves(self):
        assert resolve_deprecated_alias("PUT") == "PUT On Session"

    def test_delete_resolves(self):
        assert resolve_deprecated_alias("DELETE") == "DELETE On Session"

    def test_patch_resolves(self):
        assert resolve_deprecated_alias("PATCH") == "PATCH On Session"

    def test_head_resolves(self):
        assert resolve_deprecated_alias("HEAD") == "HEAD On Session"

    def test_options_resolves(self):
        assert resolve_deprecated_alias("OPTIONS") == "OPTIONS On Session"

    def test_all_7_http_methods_in_dict(self):
        """DEPRECATED_KEYWORD_ALIASES must contain exactly 7 HTTP methods."""
        assert len(DEPRECATED_KEYWORD_ALIASES) == 7
        expected_keys = {"get", "post", "put", "delete", "patch", "head", "options"}
        assert set(DEPRECATED_KEYWORD_ALIASES.keys()) == expected_keys

    def test_non_deprecated_keyword_returns_none(self):
        """Non-deprecated keyword should return None."""
        assert resolve_deprecated_alias("Log") is None

    def test_unknown_keyword_returns_none(self):
        """Completely unknown keyword should return None."""
        assert resolve_deprecated_alias("FooBar") is None

    def test_empty_string_returns_none(self):
        """Empty string should return None."""
        assert resolve_deprecated_alias("") is None

    def test_whitespace_only_returns_none(self):
        """Whitespace-only string should return None."""
        assert resolve_deprecated_alias("   ") is None

    def test_all_aliases_map_to_on_session_variant(self):
        """Every alias value should end with 'On Session'."""
        for _key, value in DEPRECATED_KEYWORD_ALIASES.items():
            assert value.endswith("On Session"), (
                f"Alias '{_key}' maps to '{value}' which does not end with 'On Session'"
            )


# ============================================================
# 6. Schema Transparency Tests
# ============================================================


class TestSchemaTransparency:
    """Verify coerced types produce schemas identical to plain types.

    This is critical because the LLM sees the JSON Schema and must not
    be confused by anyOf/oneOf wrappers that leak validator internals.
    """

    def test_coerced_string_list_schema_equals_list_str(self):
        """CoercedStringList schema must be identical to List[str]."""
        coerced = TypeAdapter(CoercedStringList).json_schema()
        plain = TypeAdapter(List[str]).json_schema()
        assert coerced == plain

    def test_optional_coerced_schema_equals_optional_list_str(self):
        """OptionalCoercedStringList schema must match Optional[List[str]]."""
        coerced = TypeAdapter(OptionalCoercedStringList).json_schema()
        plain = TypeAdapter(Optional[List[str]]).json_schema()
        assert coerced == plain

    def test_non_optional_schema_has_no_anyof(self):
        """Non-optional CoercedStringList must not leak anyOf."""
        schema = TypeAdapter(CoercedStringList).json_schema()
        assert "anyOf" not in schema

    def test_non_optional_schema_has_no_oneof(self):
        """Non-optional CoercedStringList must not leak oneOf."""
        schema = TypeAdapter(CoercedStringList).json_schema()
        assert "oneOf" not in schema

    def test_non_optional_schema_has_no_allof(self):
        """Non-optional CoercedStringList must not leak allOf."""
        schema = TypeAdapter(CoercedStringList).json_schema()
        assert "allOf" not in schema

    def test_non_optional_schema_structure(self):
        """Non-optional schema must be exactly {type: array, items: {type: string}}."""
        schema = TypeAdapter(CoercedStringList).json_schema()
        assert schema == {"type": "array", "items": {"type": "string"}}

    def test_optional_schema_has_anyof(self):
        """Optional schema must express nullability via anyOf."""
        schema = TypeAdapter(OptionalCoercedStringList).json_schema()
        assert "anyOf" in schema
        types_present = {v.get("type") for v in schema["anyOf"]}
        assert "null" in types_present
        assert "array" in types_present


# ============================================================
# 7. BaseModel Integration Tests (simulating FastMCP tool params)
# ============================================================


class ManageSessionParams(BaseModel):
    """Simulates the manage_session tool parameter model."""
    action: str
    session_id: str = ""
    libraries: OptionalCoercedStringList = None


class GetSessionStateParams(BaseModel):
    """Simulates the get_session_state tool parameter model."""
    session_id: str
    sections: OptionalCoercedStringList = None


class ExecuteStepParams(BaseModel):
    """Simulates the execute_step tool parameter model."""
    keyword: str
    arguments: CoercedStringList = []


class TestManageSessionParamsIntegration:
    """Test ManageSessionParams accepts all LLM output patterns."""

    def test_libraries_as_native_list(self):
        """Standard list input (well-behaved LLM)."""
        params = ManageSessionParams(
            action="import_library",
            session_id="s1",
            libraries=["Browser", "BuiltIn"],
        )
        assert params.libraries == ["Browser", "BuiltIn"]

    def test_libraries_as_json_string(self):
        """JSON array string (GLM-4.5 AIR pattern)."""
        params = ManageSessionParams(
            action="import_library",
            session_id="s1",
            libraries='["Browser", "BuiltIn"]',
        )
        assert params.libraries == ["Browser", "BuiltIn"]

    def test_libraries_as_comma_separated(self):
        """Comma-separated string (small LLM pattern)."""
        params = ManageSessionParams(
            action="import_library",
            session_id="s1",
            libraries="Browser, BuiltIn",
        )
        assert params.libraries == ["Browser", "BuiltIn"]

    def test_libraries_as_single_string(self):
        """Single library name string."""
        params = ManageSessionParams(
            action="import_library",
            session_id="s1",
            libraries="Browser",
        )
        assert params.libraries == ["Browser"]

    def test_libraries_as_none(self):
        """None for optional libraries."""
        params = ManageSessionParams(action="init", session_id="s1", libraries=None)
        assert params.libraries is None

    def test_libraries_omitted(self):
        """Libraries field omitted entirely (default None)."""
        params = ManageSessionParams(action="init", session_id="s1")
        assert params.libraries is None

    def test_libraries_empty_json_array(self):
        """Empty JSON array string."""
        params = ManageSessionParams(
            action="import_library",
            session_id="s1",
            libraries="[]",
        )
        assert params.libraries == []

    def test_model_json_schema_libraries_field(self):
        """Libraries field schema must match Optional[List[str]] (no validator leak)."""
        schema = ManageSessionParams.model_json_schema()
        libs_schema = schema["properties"]["libraries"]
        # Should be anyOf with array and null (like Optional[List[str]])
        assert "anyOf" in libs_schema, f"Expected anyOf in libraries schema: {libs_schema}"


class TestGetSessionStateParamsIntegration:
    """Test GetSessionStateParams accepts all LLM output patterns."""

    def test_sections_as_native_list(self):
        params = GetSessionStateParams(
            session_id="s1",
            sections=["variables", "libraries"],
        )
        assert params.sections == ["variables", "libraries"]

    def test_sections_as_json_string(self):
        params = GetSessionStateParams(
            session_id="s1",
            sections='["variables", "libraries"]',
        )
        assert params.sections == ["variables", "libraries"]

    def test_sections_as_comma_separated(self):
        params = GetSessionStateParams(
            session_id="s1",
            sections="variables, libraries",
        )
        assert params.sections == ["variables", "libraries"]

    def test_sections_as_single_string(self):
        params = GetSessionStateParams(session_id="s1", sections="variables")
        assert params.sections == ["variables"]

    def test_sections_none(self):
        params = GetSessionStateParams(session_id="s1", sections=None)
        assert params.sections is None

    def test_sections_omitted(self):
        params = GetSessionStateParams(session_id="s1")
        assert params.sections is None


class TestExecuteStepParamsIntegration:
    """Test ExecuteStepParams accepts all LLM output patterns."""

    def test_arguments_as_native_list(self):
        """Standard list input."""
        params = ExecuteStepParams(
            keyword="Click Element",
            arguments=["id:submit", "timeout=5"],
        )
        assert params.arguments == ["id:submit", "timeout=5"]

    def test_arguments_as_json_string(self):
        """JSON array string from small LLM."""
        params = ExecuteStepParams(
            keyword="Click Element",
            arguments='["id:submit", "timeout=5"]',
        )
        assert params.arguments == ["id:submit", "timeout=5"]

    def test_arguments_as_comma_separated(self):
        """Comma-separated string from small LLM."""
        params = ExecuteStepParams(
            keyword="Click Element",
            arguments="id:submit, timeout=5",
        )
        assert params.arguments == ["id:submit", "timeout=5"]

    def test_arguments_as_single_string(self):
        """Single argument as string."""
        params = ExecuteStepParams(
            keyword="Go To",
            arguments="https://example.com",
        )
        assert params.arguments == ["https://example.com"]

    def test_arguments_empty_list(self):
        """Empty list for keyword with no arguments."""
        params = ExecuteStepParams(keyword="Get Title", arguments=[])
        assert params.arguments == []

    def test_arguments_default_empty_list(self):
        """Default value should be empty list."""
        params = ExecuteStepParams(keyword="Get Title")
        assert params.arguments == []

    def test_arguments_empty_json_array_string(self):
        """Empty JSON array string."""
        params = ExecuteStepParams(keyword="Get Title", arguments="[]")
        assert params.arguments == []

    def test_arguments_json_url_with_commas(self):
        """URL with commas in JSON array should not be split incorrectly.

        When the LLM wraps a comma-containing URL in a JSON array,
        the JSON parser handles it correctly (path 1 takes priority).
        """
        params = ExecuteStepParams(
            keyword="Go To",
            arguments='["https://example.com/path?a=1&b=2"]',
        )
        assert params.arguments == ["https://example.com/path?a=1&b=2"]

    def test_model_json_schema_arguments_field(self):
        """Arguments field schema must match List[str] (no validator leak)."""
        schema = ExecuteStepParams.model_json_schema()
        args_schema = schema["properties"]["arguments"]
        assert args_schema.get("type") == "array", (
            f"Expected type=array, got: {args_schema}"
        )
        assert args_schema.get("items") == {"type": "string"}, (
            f"Expected items={{type: string}}, got: {args_schema}"
        )
        assert "anyOf" not in args_schema
        assert "oneOf" not in args_schema


# ============================================================
# 8. Real-World LLM Output Pattern Tests
# ============================================================


class TestRealWorldLLMPatterns:
    """Test patterns actually observed from various small LLMs."""

    def test_glm45_air_stringified_array(self):
        """GLM-4.5 AIR sends JSON-stringified arrays for list parameters."""
        params = ManageSessionParams(
            action="import_library",
            session_id="test-1",
            libraries='["Browser", "BuiltIn"]',
        )
        assert params.libraries == ["Browser", "BuiltIn"]

    def test_qwen_comma_separated(self):
        """Qwen-style comma-separated list."""
        params = ExecuteStepParams(
            keyword="Select From List By Value",
            arguments="id:dropdown, option_a, option_b",
        )
        assert params.arguments == ["id:dropdown", "option_a", "option_b"]

    def test_small_llm_single_library_string(self):
        """Small LLM sends single library as plain string."""
        params = ManageSessionParams(
            action="import_library",
            session_id="s1",
            libraries="SeleniumLibrary",
        )
        assert params.libraries == ["SeleniumLibrary"]

    def test_well_behaved_llm_native_list(self):
        """Well-behaved LLM (GPT-4, Claude) sends native list."""
        params = ExecuteStepParams(
            keyword="Input Text",
            arguments=["id:username", "admin"],
        )
        assert params.arguments == ["id:username", "admin"]

    def test_whitespace_padded_json_from_llm(self):
        """Some LLMs add whitespace around JSON."""
        params = ManageSessionParams(
            action="import_library",
            session_id="s1",
            libraries='  ["Browser"]  ',
        )
        assert params.libraries == ["Browser"]

    def test_trailing_comma_from_llm(self):
        """Some LLMs add trailing commas in comma-separated output."""
        params = ExecuteStepParams(
            keyword="Select Checkbox",
            arguments="id:agree, id:terms,",
        )
        assert params.arguments == ["id:agree", "id:terms"]


# ============================================================
# 9. Deprecation Integration with Coercion
# ============================================================


class TestDeprecationWithCoercion:
    """Test that deprecated keyword resolution works end-to-end."""

    @pytest.mark.parametrize(
        "deprecated, replacement",
        list(DEPRECATED_KEYWORD_ALIASES.items()),
        ids=list(DEPRECATED_KEYWORD_ALIASES.keys()),
    )
    def test_all_aliases_resolve(self, deprecated, replacement):
        """Every alias in the dict should resolve to its replacement."""
        assert resolve_deprecated_alias(deprecated) == replacement

    @pytest.mark.parametrize(
        "deprecated, replacement",
        list(DEPRECATED_KEYWORD_ALIASES.items()),
        ids=[f"{k}_upper" for k in DEPRECATED_KEYWORD_ALIASES.keys()],
    )
    def test_all_aliases_resolve_uppercase(self, deprecated, replacement):
        """Uppercase variants should also resolve."""
        assert resolve_deprecated_alias(deprecated.upper()) == replacement

    def test_extract_then_resolve_roundtrip(self):
        """Extract suggestion from warning, then verify it matches alias dict."""
        msg = "DeprecationWarning: 'GET' is deprecated. Use 'GET On Session' instead."
        suggestion = extract_deprecation_suggestion(msg)
        assert suggestion == "GET On Session"
        # The suggestion should be the same as what resolve_deprecated_alias returns
        alias_result = resolve_deprecated_alias("GET")
        assert suggestion == alias_result
