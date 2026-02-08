"""Comprehensive unit tests for VariableResolver.

The VariableResolver (~1012 lines) had ZERO unit tests prior to this file.
Tests cover: basic resolution, builtin variables, nested variables, indexed
access, method calls, circular references, named parameters, dynamic literals,
syntax validation, and preview resolution.

Run with: uv run pytest tests/unit/test_variable_resolver.py -v
"""

__test__ = True

import os
import tempfile

import pytest

from robotmcp.components.variables.variable_resolver import (
    CircularReferenceError,
    ObjectPreservingArgument,
    VariableResolutionError,
    VariableResolver,
)


@pytest.fixture
def resolver():
    """Fresh VariableResolver instance."""
    return VariableResolver()


@pytest.fixture
def sample_vars():
    """Common test variables dict."""
    return {
        "${NAME}": "Alice",
        "${GREETING}": "Hello",
        "${COUNT}": 42,
        "${PI}": 3.14,
        "${FLAG}": True,
        "${ITEMS}": ["a", "b", "c"],
        "${CONFIG}": {"host": "localhost", "port": 8080},
        "${NESTED_KEY}": "NAME",
        "${PREFIX}": "GREET",
        "${GREET_MSG}": "Hi there",
        "${EMPTY_STR}": "",
        "${NONE_VAL}": None,
    }


# =============================================================================
# Basic scalar resolution
# =============================================================================


class TestBasicScalarResolution:
    """Test basic ${VAR} resolution."""

    def test_resolve_simple_scalar(self, resolver, sample_vars):
        result = resolver.resolve_single_argument("${NAME}", sample_vars)
        assert result == "Alice"

    def test_resolve_integer_variable(self, resolver, sample_vars):
        result = resolver.resolve_single_argument("${COUNT}", sample_vars)
        assert result == 42
        assert isinstance(result, int)

    def test_resolve_float_variable(self, resolver, sample_vars):
        result = resolver.resolve_single_argument("${PI}", sample_vars)
        assert result == 3.14

    def test_resolve_boolean_variable(self, resolver, sample_vars):
        result = resolver.resolve_single_argument("${FLAG}", sample_vars)
        assert result is True

    def test_resolve_list_variable(self, resolver, sample_vars):
        result = resolver.resolve_single_argument("${ITEMS}", sample_vars)
        assert result == ["a", "b", "c"]

    def test_resolve_dict_variable(self, resolver, sample_vars):
        result = resolver.resolve_single_argument("${CONFIG}", sample_vars)
        assert result == {"host": "localhost", "port": 8080}

    def test_resolve_none_variable(self, resolver, sample_vars):
        result = resolver.resolve_single_argument("${NONE_VAL}", sample_vars)
        assert result is None

    def test_resolve_empty_string_variable(self, resolver, sample_vars):
        result = resolver.resolve_single_argument("${EMPTY_STR}", sample_vars)
        assert result == ""

    def test_nonstring_passthrough(self, resolver, sample_vars):
        """Non-string arguments pass through unchanged."""
        result = resolver.resolve_single_argument(42, sample_vars)
        assert result == 42

    def test_plain_text_passthrough(self, resolver, sample_vars):
        """Plain text without variables passes through unchanged."""
        result = resolver.resolve_single_argument("hello world", sample_vars)
        assert result == "hello world"


# =============================================================================
# Text substitution (variables embedded in text)
# =============================================================================


class TestTextSubstitution:
    """Test variables embedded in text strings."""

    def test_variable_in_text(self, resolver, sample_vars):
        result = resolver.resolve_single_argument(
            "${GREETING} ${NAME}!", sample_vars
        )
        assert result == "Hello Alice!"

    def test_multiple_variables_in_text(self, resolver, sample_vars):
        result = resolver.resolve_single_argument(
            "${GREETING} ${NAME}, count=${COUNT}", sample_vars
        )
        assert result == "Hello Alice, count=42"

    def test_variable_at_start_of_text(self, resolver, sample_vars):
        result = resolver.resolve_single_argument("${NAME} is here", sample_vars)
        assert result == "Alice is here"

    def test_variable_at_end_of_text(self, resolver, sample_vars):
        result = resolver.resolve_single_argument("Name: ${NAME}", sample_vars)
        assert result == "Name: Alice"

    def test_repeated_variable_in_text(self, resolver, sample_vars):
        result = resolver.resolve_single_argument(
            "${NAME} and ${NAME}", sample_vars
        )
        assert result == "Alice and Alice"


# =============================================================================
# Dynamic literals
# =============================================================================


class TestDynamicLiterals:
    """Test dynamic literal resolution (booleans, numbers, None)."""

    @pytest.mark.parametrize(
        "var_name,expected",
        [
            ("${true}", True),
            ("${True}", True),
            ("${TRUE}", True),
            ("${false}", False),
            ("${False}", False),
            ("${FALSE}", False),
            ("${none}", None),
            ("${None}", None),
            ("${null}", None),
        ],
    )
    def test_boolean_and_none_literals(self, resolver, var_name, expected):
        result = resolver.resolve_single_argument(var_name, {})
        assert result is expected

    @pytest.mark.parametrize(
        "var_name,expected",
        [
            ("${42}", 42),
            ("${0}", 0),
            ("${-1}", -1),
            ("${100}", 100),
        ],
    )
    def test_integer_literals(self, resolver, var_name, expected):
        result = resolver.resolve_single_argument(var_name, {})
        assert result == expected
        assert isinstance(result, int)

    @pytest.mark.parametrize(
        "var_name,expected",
        [
            ("${3.14}", 3.14),
            ("${0.5}", 0.5),
        ],
    )
    def test_float_literals(self, resolver, var_name, expected):
        result = resolver.resolve_single_argument(var_name, {})
        assert result == pytest.approx(expected)

    @pytest.mark.parametrize(
        "var_name,expected",
        [
            ("${0xFF}", 255),
            ("${0xff}", 255),
            ("${0b1010}", 10),
            ("${0o17}", 15),
        ],
    )
    def test_numeric_base_literals(self, resolver, var_name, expected):
        result = resolver.resolve_single_argument(var_name, {})
        assert result == expected


# =============================================================================
# Built-in variables
# =============================================================================


class TestBuiltinVariables:
    """Test built-in variable resolution."""

    def test_space_builtin(self, resolver):
        result = resolver.resolve_single_argument("${SPACE}", {})
        assert result == " "

    def test_empty_builtin(self, resolver):
        result = resolver.resolve_single_argument("${EMPTY}", {})
        assert result == ""

    def test_true_builtin(self, resolver):
        result = resolver.resolve_single_argument("${TRUE}", {})
        assert result is True

    def test_false_builtin(self, resolver):
        result = resolver.resolve_single_argument("${FALSE}", {})
        assert result is False

    def test_null_builtin(self, resolver):
        result = resolver.resolve_single_argument("${NULL}", {})
        assert result is None

    def test_path_sep_builtin(self, resolver):
        result = resolver.resolve_single_argument("${/}", {})
        assert result == os.sep

    def test_pathsep_builtin(self, resolver):
        result = resolver.resolve_single_argument("${:}", {})
        assert result == os.pathsep

    def test_newline_builtin(self, resolver):
        result = resolver.resolve_single_argument("${\\n}", {})
        assert result == "\n"

    def test_tab_builtin(self, resolver):
        result = resolver.resolve_single_argument("${\\t}", {})
        assert result == "\t"

    def test_tempdir_builtin(self, resolver):
        result = resolver.resolve_single_argument("${TEMPDIR}", {})
        assert result == tempfile.gettempdir()

    def test_curdir_builtin(self, resolver):
        result = resolver.resolve_single_argument("${CURDIR}", {})
        # CURDIR is set to cwd at init time
        assert isinstance(result, str)
        assert len(result) > 0

    def test_session_vars_override_builtins(self, resolver):
        """Session variables should take precedence over builtins.

        Note: dynamic literal names (TRUE, FALSE, NONE, NULL) are resolved
        by the dynamic literal handler *before* the variable dict is checked,
        so they cannot be overridden.  Use a non-dynamic-literal builtin
        like ${SPACE} to test the override path.
        """
        custom_vars = {"${SPACE}": "custom_space"}
        result = resolver.resolve_single_argument("${SPACE}", custom_vars)
        assert result == "custom_space"

    def test_builtins_excluded_when_disabled(self, resolver):
        """When include_builtins=False, builtin vars should not be available."""
        with pytest.raises(VariableResolutionError):
            resolver.resolve_single_argument(
                "${SPACE}", {}, include_builtins=False
            )

    def test_numeric_builtin_zero(self, resolver):
        """${0} is a builtin with value 0."""
        result = resolver.resolve_single_argument("${0}", {})
        assert result == 0

    def test_numeric_builtin_one(self, resolver):
        """${1} is a builtin with value 1."""
        result = resolver.resolve_single_argument("${1}", {})
        assert result == 1

    def test_numeric_builtin_negative_one(self, resolver):
        """${-1} is a builtin with value -1."""
        result = resolver.resolve_single_argument("${-1}", {})
        assert result == -1


# =============================================================================
# Indexed access
# =============================================================================


class TestIndexedAccess:
    """Test list/dict indexed access patterns."""

    def test_list_index_zero(self, resolver, sample_vars):
        result = resolver.resolve_single_argument("${ITEMS}[0]", sample_vars)
        assert result == "a"

    def test_list_index_last(self, resolver, sample_vars):
        result = resolver.resolve_single_argument("${ITEMS}[2]", sample_vars)
        assert result == "c"

    def test_list_negative_index(self, resolver, sample_vars):
        result = resolver.resolve_single_argument("${ITEMS}[-1]", sample_vars)
        assert result == "c"

    def test_dict_key_access(self, resolver, sample_vars):
        result = resolver.resolve_single_argument("${CONFIG}[host]", sample_vars)
        assert result == "localhost"

    def test_dict_numeric_key_access(self, resolver, sample_vars):
        result = resolver.resolve_single_argument("${CONFIG}[port]", sample_vars)
        assert result == 8080

    def test_nested_list_of_dicts(self, resolver):
        variables = {"${DATA}": [{"name": "Alice"}, {"name": "Bob"}]}
        result = resolver.resolve_single_argument("${DATA}[0][name]", variables)
        assert result == "Alice"

    def test_double_index_access(self, resolver):
        variables = {"${MATRIX}": [[1, 2], [3, 4]]}
        result = resolver.resolve_single_argument("${MATRIX}[1][0]", variables)
        assert result == 3

    def test_variable_in_index(self, resolver):
        variables = {"${ITEMS}": ["a", "b", "c"], "${IDX}": "1"}
        result = resolver.resolve_single_argument("${ITEMS}[${IDX}]", variables)
        assert result == "b"

    def test_index_out_of_range(self, resolver, sample_vars):
        with pytest.raises(VariableResolutionError):
            resolver.resolve_single_argument("${ITEMS}[99]", sample_vars)

    def test_dict_missing_key(self, resolver, sample_vars):
        with pytest.raises(VariableResolutionError):
            resolver.resolve_single_argument("${CONFIG}[missing]", sample_vars)


# =============================================================================
# Nested variables
# =============================================================================


class TestNestedVariables:
    """Test nested variable resolution: ${${INNER}}."""

    def test_nested_variable_reference(self, resolver, sample_vars):
        """${${NESTED_KEY}} should resolve to ${NAME} -> Alice."""
        result = resolver.resolve_single_argument(
            "${${NESTED_KEY}}", sample_vars
        )
        assert result == "Alice"

    def test_nested_variable_with_prefix(self, resolver, sample_vars):
        """${${PREFIX}_MSG} should resolve to ${GREET_MSG} -> Hi there."""
        result = resolver.resolve_single_argument(
            "${${PREFIX}_MSG}", sample_vars
        )
        assert result == "Hi there"


# =============================================================================
# Circular reference detection
# =============================================================================


class TestCircularReferenceDetection:
    """Test circular reference detection."""

    def test_direct_circular_reference(self, resolver):
        variables = {"${A}": "${A}"}
        with pytest.raises(CircularReferenceError, match="Circular"):
            resolver.resolve_single_argument("${A}", variables)

    def test_indirect_circular_reference(self, resolver):
        variables = {"${A}": "${B}", "${B}": "${A}"}
        with pytest.raises(CircularReferenceError, match="Circular"):
            resolver.resolve_single_argument("${A}", variables)

    def test_chain_circular_reference(self, resolver):
        variables = {"${A}": "${B}", "${B}": "${C}", "${C}": "${A}"}
        with pytest.raises(CircularReferenceError, match="Circular"):
            resolver.resolve_single_argument("${A}", variables)

    def test_non_circular_chain_succeeds(self, resolver):
        """A -> B -> C (no cycle) should resolve fine."""
        variables = {"${A}": "${B}", "${B}": "${C}", "${C}": "final"}
        result = resolver.resolve_single_argument("${A}", variables)
        assert result == "final"


# =============================================================================
# Named parameter object preservation
# =============================================================================


class TestNamedParameterObjectPreservation:
    """Test ObjectPreservingArgument for dict/list params.

    Note: The ObjectPreservingArgument path in resolve_single_argument
    checks ``var_name in all_variables`` where var_name is the *bare* name
    (without ``${}``).  Standard variable dicts use ``${KEY}`` format, so
    the preservation path only activates when variables are stored with
    bare keys (as the MCP server does internally for some code paths).
    With ``${KEY}`` format, named params go through text substitution.
    """

    def test_dict_param_preserved_bare_key(self, resolver):
        """ObjectPreservingArgument is returned when variables use bare keys."""
        body = {"key": "value", "num": 42}
        # Bare key (no ${}) -- this matches the internal var_name extraction
        variables = {"BODY": body}
        result = resolver.resolve_single_argument("json=${BODY}", variables)
        assert isinstance(result, ObjectPreservingArgument)
        assert result.param_name == "json"
        assert result.value == body

    def test_list_param_preserved_bare_key(self, resolver):
        """ObjectPreservingArgument is returned for list values with bare keys."""
        items = [1, 2, 3]
        variables = {"LIST": items}
        result = resolver.resolve_single_argument("data=${LIST}", variables)
        assert isinstance(result, ObjectPreservingArgument)
        assert result.param_name == "data"
        assert result.value == items

    def test_dict_param_with_wrapped_key_stringified(self, resolver):
        """With standard ${KEY} format, object params get stringified."""
        body = {"key": "value"}
        variables = {"${BODY}": body}
        result = resolver.resolve_single_argument("json=${BODY}", variables)
        # Falls through to text substitution, gets stringified
        assert isinstance(result, str)
        assert "key" in result

    def test_scalar_param_not_preserved(self, resolver):
        variables = {"${VAL}": "hello"}
        result = resolver.resolve_single_argument("key=${VAL}", variables)
        # Scalar values never become ObjectPreservingArgument
        assert isinstance(result, str)
        assert result == "key=hello"

    def test_object_preserving_str_repr(self):
        arg = ObjectPreservingArgument("json", {"a": 1})
        text = str(arg)
        assert "json" in text
        assert "dict" in text

    def test_object_preserving_repr(self):
        arg = ObjectPreservingArgument("data", [1, 2])
        text = repr(arg)
        assert "ObjectPreservingArgument" in text


# =============================================================================
# resolve_arguments (list of args)
# =============================================================================


class TestResolveArguments:
    """Test resolve_arguments for argument lists."""

    def test_empty_args(self, resolver, sample_vars):
        result = resolver.resolve_arguments([], sample_vars)
        assert result == []

    def test_multiple_args_resolved(self, resolver, sample_vars):
        result = resolver.resolve_arguments(
            ["${NAME}", "${COUNT}", "literal"], sample_vars
        )
        assert result == ["Alice", 42, "literal"]

    def test_non_string_args_passthrough(self, resolver, sample_vars):
        result = resolver.resolve_arguments([42, True, None], sample_vars)
        assert result == [42, True, None]

    def test_mixed_args(self, resolver, sample_vars):
        result = resolver.resolve_arguments(
            ["${NAME}", 42, "text ${COUNT}"], sample_vars
        )
        assert result[0] == "Alice"
        assert result[1] == 42
        assert result[2] == "text 42"


# =============================================================================
# Method calls and attribute access
# =============================================================================


class TestMethodCallsAndAttributeAccess:
    """Test ${obj.method()} and ${obj.attr} patterns."""

    def test_method_call_on_object(self, resolver):
        class FakeResponse:
            def json(self):
                return {"id": 1, "name": "test"}

        variables = {"${RESPONSE}": FakeResponse()}
        result = resolver.resolve_single_argument(
            "${RESPONSE.json()}", variables
        )
        assert result == {"id": 1, "name": "test"}

    def test_attribute_access(self, resolver):
        class Obj:
            name = "Alice"

        variables = {"${OBJ}": Obj()}
        result = resolver.resolve_single_argument("${OBJ.name}", variables)
        assert result == "Alice"

    def test_chained_method_and_index(self, resolver):
        class FakeResponse:
            def json(self):
                return [{"id": 1}, {"id": 2}]

        variables = {"${RESPONSE}": FakeResponse()}
        result = resolver.resolve_single_argument(
            "${RESPONSE.json()[0]['id']}", variables
        )
        assert result == 1


# =============================================================================
# Variable not found errors
# =============================================================================


class TestVariableNotFound:
    """Test error handling for undefined variables."""

    def test_undefined_variable_raises(self, resolver):
        with pytest.raises(VariableResolutionError, match="not found"):
            resolver.resolve_single_argument("${NONEXISTENT}", {})

    def test_error_includes_variable_name(self, resolver):
        with pytest.raises(VariableResolutionError, match="NONEXISTENT"):
            resolver.resolve_single_argument("${NONEXISTENT}", {})

    def test_undefined_in_text_raises(self, resolver):
        with pytest.raises(VariableResolutionError):
            resolver.resolve_single_argument("Hello ${MISSING}!", {})


# =============================================================================
# Syntax validation
# =============================================================================


class TestSyntaxValidation:
    """Test validate_variable_syntax."""

    def test_valid_syntax(self, resolver):
        is_valid, errors = resolver.validate_variable_syntax(
            "${NAME} is ${AGE}"
        )
        assert is_valid is True
        assert errors == []

    def test_empty_variable_name(self, resolver):
        is_valid, errors = resolver.validate_variable_syntax("${}")
        assert is_valid is False
        assert any("Empty" in e for e in errors)

    def test_plain_text_is_valid(self, resolver):
        is_valid, errors = resolver.validate_variable_syntax("hello world")
        assert is_valid is True

    def test_unclosed_brace(self, resolver):
        is_valid, errors = resolver.validate_variable_syntax("${UNCLOSED")
        assert is_valid is False

    def test_multiple_valid_variables(self, resolver):
        is_valid, errors = resolver.validate_variable_syntax(
            "${A} ${B} ${C}"
        )
        assert is_valid is True
        assert errors == []


# =============================================================================
# Preview resolution
# =============================================================================


class TestPreviewResolution:
    """Test preview_resolution method."""

    def test_preview_finds_variables(self, resolver, sample_vars):
        preview = resolver.preview_resolution(
            "${NAME} has ${COUNT}", sample_vars
        )
        assert preview["syntax_valid"] is True
        assert "NAME" in preview["variables_found"]
        assert "COUNT" in preview["variables_found"]

    def test_preview_shows_resolved_values(self, resolver, sample_vars):
        preview = resolver.preview_resolution("${NAME}", sample_vars)
        assert preview["resolution_preview"]["${NAME}"]["resolved"] is True
        assert preview["resolution_preview"]["${NAME}"]["value"] == "Alice"

    def test_preview_shows_missing_variables(self, resolver):
        preview = resolver.preview_resolution("${MISSING}", {})
        assert "MISSING" in preview["missing_variables"]
        assert (
            preview["resolution_preview"]["${MISSING}"]["resolved"] is False
        )

    def test_preview_returns_original_text(self, resolver, sample_vars):
        text = "prefix ${NAME} suffix"
        preview = resolver.preview_resolution(text, sample_vars)
        assert preview["original"] == text


# =============================================================================
# _is_single_variable
# =============================================================================


class TestIsSingleVariable:
    """Test the _is_single_variable helper."""

    def test_simple_variable(self, resolver):
        assert resolver._is_single_variable("${VAR}") is True

    def test_variable_with_index(self, resolver):
        assert resolver._is_single_variable("${VAR}[0]") is True

    def test_variable_with_double_index(self, resolver):
        assert resolver._is_single_variable("${VAR}[0][key]") is True

    def test_variable_in_text(self, resolver):
        assert resolver._is_single_variable("hello ${VAR}") is False

    def test_variable_followed_by_text(self, resolver):
        assert resolver._is_single_variable("${VAR} hello") is False

    def test_not_a_variable(self, resolver):
        assert resolver._is_single_variable("hello") is False

    def test_empty_string(self, resolver):
        assert resolver._is_single_variable("") is False


# =============================================================================
# get_available_variables
# =============================================================================


class TestGetAvailableVariables:
    """Test get_available_variables method."""

    def test_includes_builtins(self, resolver):
        all_vars = resolver.get_available_variables()
        assert "${SPACE}" in all_vars
        assert "${TRUE}" in all_vars

    def test_includes_session_vars(self, resolver, sample_vars):
        all_vars = resolver.get_available_variables(sample_vars)
        assert "${NAME}" in all_vars
        assert "${SPACE}" in all_vars  # builtins still present

    def test_session_vars_override_builtins(self, resolver):
        custom = {"${TRUE}": "custom"}
        all_vars = resolver.get_available_variables(custom)
        assert all_vars["${TRUE}"] == "custom"


# =============================================================================
# _normalize_variable_name
# =============================================================================


class TestNormalizeVariableName:
    """Test _normalize_variable_name helper."""

    def test_bare_name_gets_wrapped(self, resolver):
        assert resolver._normalize_variable_name("FOO") == "${FOO}"

    def test_already_wrapped_unchanged(self, resolver):
        assert resolver._normalize_variable_name("${FOO}") == "${FOO}"

    def test_partial_prefix_gets_wrapped(self, resolver):
        # Does not start with ${ and end with }, so gets wrapped
        assert resolver._normalize_variable_name("$FOO") == "${$FOO}"


# =============================================================================
# VariableResolutionError features
# =============================================================================


class TestVariableResolutionError:
    """Test error class features."""

    def test_error_message_includes_variable_name(self):
        err = VariableResolutionError("MY_VAR")
        assert "MY_VAR" in str(err)

    def test_error_shows_available_vars(self):
        err = VariableResolutionError("MY_VAR", ["SOME_VAR", "OTHER_VAR"])
        msg = str(err)
        assert "MY_VAR" in msg

    def test_error_with_custom_message(self):
        err = VariableResolutionError(
            "X", message="Custom error message"
        )
        assert str(err) == "Custom error message"

    def test_circular_reference_error_chain(self):
        err = CircularReferenceError(["A", "B", "C", "A"])
        msg = str(err)
        assert "Circular" in msg
        assert "${A}" in msg
        assert "${B}" in msg
        assert "${C}" in msg

    def test_similar_variable_suggestions(self):
        err = VariableResolutionError("NAM", ["NAME", "NAMES", "COUNT"])
        msg = str(err)
        # NAM is a substring of NAME/NAMES so similarity should be high
        assert "NAM" in msg


# =============================================================================
# Edge cases and integration
# =============================================================================


class TestEdgeCases:
    """Test edge cases and integration scenarios."""

    def test_variable_value_containing_variable_syntax(self, resolver):
        """A variable whose value contains ${...} gets resolved recursively."""
        variables = {
            "${INDIRECT}": "${TARGET}",
            "${TARGET}": "final_value",
        }
        result = resolver.resolve_single_argument("${INDIRECT}", variables)
        assert result == "final_value"

    def test_escape_like_backslash_n_builtin(self, resolver):
        """${\\n} is a builtin representing newline."""
        result = resolver.resolve_single_argument("line1${\\n}line2", {})
        assert result == "line1\nline2"

    def test_none_arguments_list(self, resolver, sample_vars):
        """None in arguments list passes through."""
        result = resolver.resolve_arguments([None], sample_vars)
        assert result == [None]

    def test_boolean_in_arguments_list(self, resolver, sample_vars):
        """Boolean in arguments list passes through."""
        result = resolver.resolve_arguments([True, False], sample_vars)
        assert result == [True, False]

    def test_list_in_arguments_list(self, resolver, sample_vars):
        """Lists in arguments list pass through."""
        lst = [1, 2, 3]
        result = resolver.resolve_arguments([lst], sample_vars)
        assert result == [lst]

    def test_resolution_stack_cleared_after_success(self, resolver):
        """_resolution_stack should be empty after successful resolution."""
        variables = {"${A}": "value"}
        resolver.resolve_arguments(["${A}"], variables)
        assert resolver._resolution_stack == []

    def test_resolution_stack_cleared_after_error(self, resolver):
        """_resolution_stack should be empty after failed resolution."""
        with pytest.raises(VariableResolutionError):
            resolver.resolve_arguments(["${MISSING}"], {})
        assert resolver._resolution_stack == []

    def test_has_method_call_or_attribute_access_true(self, resolver):
        """Detect method call in ${obj.method()} pattern."""
        assert resolver._has_method_call_or_attribute_access(
            "${obj.method()}"
        ) is True

    def test_has_method_call_or_attribute_access_false_for_simple(
        self, resolver
    ):
        """Simple ${VAR} has no method/attribute access."""
        assert resolver._has_method_call_or_attribute_access(
            "${VAR}"
        ) is False

    def test_has_method_call_or_attribute_access_false_for_indexed(
        self, resolver
    ):
        """${VAR}[0] ends with ] not }, so returns False."""
        assert resolver._has_method_call_or_attribute_access(
            "${VAR}[0]"
        ) is False
