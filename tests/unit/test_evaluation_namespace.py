"""Unit tests for evaluation namespace syntax handling in execute_step and build_test_suite.

This test suite verifies the correct handling of Robot Framework's evaluation namespace
syntax, particularly the differences between:
- ${variable}: Standard RF variable substitution (string replacement BEFORE evaluation)
- $variable: Direct access in evaluation namespace (actual object, NOT quoted)

Reference: https://robotframework.org/robotframework/latest/RobotFrameworkUserGuide.html#evaluation-namespaces
"""

import pytest
import re


class TestEvaluateVariableSyntaxConversion:
    """Test the conversion of ${var} to $var for Evaluate expressions."""

    def test_simple_variable_conversion(self):
        """Test that ${var} is converted to $var."""
        expr = "${status} == 'PASS'"
        # Expected: $status == 'PASS'
        converted = re.sub(r"\$\{([A-Za-z_]\w*)\}", r"$\1", expr)
        assert converted == "$status == 'PASS'"

    def test_variable_with_suffix_conversion(self):
        """Test that ${var.suffix} is converted to $var.suffix."""
        expr = "${response.status_code} == 200"
        # Current implementation uses: r"\$\{([A-Za-z_]\w*)([^}]*)\}"
        converted = re.sub(r"\$\{([A-Za-z_]\w*)([^}]*)\}", r"$\1\2", expr)
        assert converted == "$response.status_code == 200"

    def test_variable_with_method_call(self):
        """Test that ${var.method()} is converted to $var.method()."""
        expr = "len(${items}) > 0"
        converted = re.sub(r"\$\{([A-Za-z_]\w*)([^}]*)\}", r"$\1\2", expr)
        # Expected: len($items) > 0
        assert converted == "len($items) > 0"

    def test_variable_with_index_access(self):
        """Test that ${var}[0] stays as $var[0] (but needs special handling)."""
        expr = "${list}[0] == 'first'"
        converted = re.sub(r"\$\{([A-Za-z_]\w*)([^}]*)\}", r"$\1\2", expr)
        # Expected: $list[0] == 'first'
        assert converted == "$list[0] == 'first'"

    def test_nested_variable_in_index(self):
        """Test that ${list}[${index}] is properly handled."""
        expr = "${list}[${index}]"
        # The regex handles this correctly because it's greedy
        converted = re.sub(r"\$\{([A-Za-z_]\w*)([^}]*)\}", r"$\1\2", expr)
        # Expected: $list[$index] (both converted correctly)
        assert converted == "$list[$index]"

    def test_multiple_variables(self):
        """Test that multiple ${var} instances are all converted."""
        expr = "${a} + ${b} == ${sum}"
        converted = re.sub(r"\$\{([A-Za-z_]\w*)\}", r"$\1", expr)
        assert converted == "$a + $b == $sum"

    def test_string_comparison_quoting(self):
        """Test that string comparisons need proper handling.

        When using ${variable} with strings, the value becomes a string literal,
        so it needs to be quoted. With $variable, the string is already a string object.
        """
        # WRONG: $status == PASS (PASS would be evaluated as a variable/identifier)
        # RIGHT: $status == 'PASS' (PASS is a string literal)
        expr = "$status == 'PASS'"
        assert "'" in expr or '"' in expr  # Quotes are required


class TestIfConditionSyntax:
    """Test IF condition syntax handling for Robot Framework."""

    def test_if_condition_uses_evaluation_namespace(self):
        """IF conditions use the same evaluation namespace as Evaluate."""
        # In RF, IF conditions are evaluated using Python's eval()
        # So they should use $var syntax, not ${var}
        condition = "${count} > 0"
        # Should be converted to: $count > 0
        converted = re.sub(r"\$\{([A-Za-z_]\w*)([^}]*)\}", r"$\1\2", condition)
        assert converted == "$count > 0"

    def test_if_condition_with_string_comparison(self):
        """IF condition string comparison needs quotes."""
        # WRONG for evaluation: ${status} == PASS (PASS is not quoted)
        # RIGHT: $status == 'PASS' (PASS is quoted string)
        condition = "${status} == 'PASS'"
        converted = re.sub(r"\$\{([A-Za-z_]\w*)([^}]*)\}", r"$\1\2", condition)
        assert converted == "$status == 'PASS'"

    def test_if_condition_with_boolean(self):
        """IF condition with boolean evaluation."""
        condition = "${is_enabled}"
        converted = re.sub(r"\$\{([A-Za-z_]\w*)([^}]*)\}", r"$\1\2", condition)
        assert converted == "$is_enabled"


class TestBuildTestSuiteEvaluateSyntax:
    """Test that build_test_suite correctly handles Evaluate expressions."""

    def test_evaluate_step_conversion_regex(self):
        """Verify the regex used in test_builder.py handles all cases."""
        # The current implementation in test_builder.py:
        # expr = re.sub(r"\$\{([A-Za-z_]\w*)([^}]*)\}", r"$\1\2", expr)

        test_cases = [
            # (input, expected_output)
            ("${var}", "$var"),
            ("${var.attr}", "$var.attr"),
            ("${var}[0]", "$var[0]"),
            ("len(${items})", "len($items)"),
            ("${a} + ${b}", "$a + $b"),
            # Edge cases - these may reveal bugs
            ("${var.method()}", "$var.method()"),
        ]

        for input_expr, expected in test_cases:
            converted = re.sub(r"\$\{([A-Za-z_]\w*)([^}]*)\}", r"$\1\2", input_expr)
            assert converted == expected, f"Failed: {input_expr} -> {converted}, expected {expected}"


class TestImprovedConversionRegex:
    """Test improved regex patterns for evaluation namespace conversion."""

    def _improved_convert_to_eval_syntax(self, expr: str) -> str:
        """Improved conversion that handles nested variables correctly."""
        result = expr
        # Need to convert innermost variables first, then outer ones
        # Or use a proper parser

        # Simple improvement: use a greedy match that handles nesting
        # This still won't handle all cases but is better
        import re

        # Keep converting until no more changes
        prev = ""
        while prev != result:
            prev = result
            # Convert ${var} to $var, being careful with nested braces
            result = re.sub(r"\$\{([A-Za-z_]\w*)\}", r"$\1", result)

        return result

    def test_improved_nested_variable(self):
        """Test improved handling of nested variables."""
        expr = "${list}[${index}]"
        converted = self._improved_convert_to_eval_syntax(expr)
        # Should handle inner ${index} first, then outer ${list}
        assert converted == "$list[$index]"

    def test_improved_multiple_levels(self):
        """Test multiple levels of variables."""
        expr = "${outer}[${middle}[${inner}]]"
        converted = self._improved_convert_to_eval_syntax(expr)
        assert converted == "$outer[$middle[$inner]]"


class TestEvaluationNamespacePitfalls:
    """Document and test common pitfalls with evaluation namespace."""

    def test_pitfall_unquoted_string_comparison(self):
        """Unquoted string comparison fails because identifier is not defined."""
        # WRONG: $status == PASS
        # This tries to find a variable/module named PASS
        # RIGHT: $status == 'PASS'
        wrong_expr = "$status == PASS"
        right_expr = "$status == 'PASS'"

        # The right expression uses quoted string
        assert "'" in right_expr

    def test_pitfall_curly_brace_in_evaluation(self):
        """${var} in Evaluate is replaced BEFORE evaluation, so it becomes a string."""
        # ${status} == 'PASS' with status='OK' becomes:
        # 'OK' == 'PASS' (evaluates fine but status is already a string)
        #
        # $status == 'PASS' with status='OK' becomes:
        # actual_value_of_status == 'PASS' (status is the Python object)
        #
        # For strings, both work, but for objects, $var preserves the type
        pass

    def test_pitfall_numeric_string_comparison(self):
        """Numeric comparison with ${var} may fail due to string conversion."""
        # If count = 5 (integer):
        # ${count} > 0 becomes: 5 > 0 (works for integers)
        # But if count = "5" (string):
        # ${count} > 0 becomes: "5" > 0 which fails in Python 3
        #
        # $count > 0 always uses the actual type
        pass


class TestTestBuilderEvaluationSyntax:
    """Test TestBuilder's handling of evaluation namespace syntax."""

    @pytest.fixture
    def test_builder(self):
        """Create a TestBuilder instance for testing."""
        from robotmcp.components.test_builder import TestBuilder
        return TestBuilder()

    def test_convert_to_evaluation_namespace_syntax_simple(self, test_builder):
        """Test conversion of simple ${var} to $var."""
        result = test_builder._convert_to_evaluation_namespace_syntax("${status}")
        assert result == "$status"

    def test_convert_to_evaluation_namespace_syntax_with_attr(self, test_builder):
        """Test conversion of ${var.attr} to $var.attr."""
        result = test_builder._convert_to_evaluation_namespace_syntax("${response.status_code}")
        assert result == "$response.status_code"

    def test_convert_to_evaluation_namespace_syntax_expression(self, test_builder):
        """Test conversion of complex expression."""
        result = test_builder._convert_to_evaluation_namespace_syntax("${count} > 0 and ${status} == 'PASS'")
        assert result == "$count > 0 and $status == 'PASS'"

    def test_convert_to_evaluation_namespace_syntax_empty(self, test_builder):
        """Test conversion of empty string."""
        result = test_builder._convert_to_evaluation_namespace_syntax("")
        assert result == ""

    def test_convert_to_evaluation_namespace_syntax_none(self, test_builder):
        """Test conversion of None."""
        result = test_builder._convert_to_evaluation_namespace_syntax(None)
        assert result is None

    def test_convert_to_evaluation_namespace_syntax_with_index(self, test_builder):
        """Test conversion with index access."""
        result = test_builder._convert_to_evaluation_namespace_syntax("${list}[0] == 'first'")
        assert result == "$list[0] == 'first'"

    def test_convert_to_evaluation_namespace_syntax_nested(self, test_builder):
        """Test conversion of nested variable access."""
        result = test_builder._convert_to_evaluation_namespace_syntax("${list}[${index}]")
        assert result == "$list[$index]"


class TestRenderFlowBlocksEvaluationSyntax:
    """Test that _render_flow_blocks properly converts IF conditions."""

    @pytest.fixture
    def test_builder(self):
        """Create a TestBuilder instance for testing."""
        from robotmcp.components.test_builder import TestBuilder
        return TestBuilder()

    def test_if_condition_converted(self, test_builder):
        """Test that IF condition uses $var syntax."""
        nodes = [{
            "type": "if",
            "condition": "${count} > 0",
            "then": [{"keyword": "Log", "arguments": ["Count is positive"]}],
            "else": []
        }]
        lines = test_builder._render_flow_blocks(nodes)
        # The IF line should have $count, not ${count}
        assert any("IF    $count > 0" in line for line in lines)

    def test_if_condition_with_string_comparison_converted(self, test_builder):
        """Test that IF condition with string comparison uses $var syntax."""
        nodes = [{
            "type": "if",
            "condition": "${status} == 'PASS'",
            "then": [{"keyword": "Log", "arguments": ["Status is PASS"]}],
            "else": []
        }]
        lines = test_builder._render_flow_blocks(nodes)
        assert any("IF    $status == 'PASS'" in line for line in lines)

    def test_evaluate_in_flow_body_converted(self, test_builder):
        """Test that Evaluate keyword in flow body uses $var syntax."""
        steps = [{"keyword": "Evaluate", "arguments": ["${x} + ${y}"]}]
        lines = test_builder._render_flow_body(steps, "    ")
        # Should be converted to $x + $y
        assert any("$x + $y" in line for line in lines)


class TestBuildTestSuiteVariablesSection:
    """Test the Variables section generation in build_test_suite.

    This tests the fix for the issue where session variables set via manage_session
    or execute_step were not being serialized to the *** Variables *** section
    of generated .robot files, causing standalone execution to fail.
    """

    @pytest.fixture
    def test_builder(self):
        """Create a TestBuilder instance for testing."""
        from robotmcp.components.test_builder import TestBuilder
        return TestBuilder()

    def test_format_variable_scalar_string(self, test_builder):
        """Test formatting a scalar string variable."""
        result = test_builder._format_variable_for_rf("USER_NAME", "Alice Johnson")
        assert "${USER_NAME}" in result
        assert "Alice Johnson" in result

    def test_format_variable_scalar_integer(self, test_builder):
        """Test formatting a scalar integer variable."""
        result = test_builder._format_variable_for_rf("USER_AGE", 28)
        assert "${USER_AGE}" in result
        assert "28" in result

    def test_format_variable_scalar_float(self, test_builder):
        """Test formatting a scalar float variable."""
        result = test_builder._format_variable_for_rf("USER_SCORE", 95.5)
        assert "${USER_SCORE}" in result
        assert "95.5" in result

    def test_format_variable_scalar_boolean_true(self, test_builder):
        """Test formatting a boolean True variable."""
        result = test_builder._format_variable_for_rf("IS_ENABLED", True)
        assert "${IS_ENABLED}" in result
        assert "${TRUE}" in result

    def test_format_variable_scalar_boolean_false(self, test_builder):
        """Test formatting a boolean False variable."""
        result = test_builder._format_variable_for_rf("IS_DISABLED", False)
        assert "${IS_DISABLED}" in result
        assert "${FALSE}" in result

    def test_format_variable_scalar_none(self, test_builder):
        """Test formatting a None variable."""
        result = test_builder._format_variable_for_rf("NULL_VALUE", None)
        assert "${NULL_VALUE}" in result
        assert "${NONE}" in result

    def test_format_variable_list(self, test_builder):
        """Test formatting a list variable."""
        result = test_builder._format_variable_for_rf("ITEMS", ["apple", "banana", "cherry"])
        assert "@{ITEMS}" in result
        assert "apple" in result
        assert "banana" in result
        assert "cherry" in result

    def test_format_variable_dict(self, test_builder):
        """Test formatting a dictionary variable."""
        result = test_builder._format_variable_for_rf("CONFIG", {"host": "localhost", "port": "8080"})
        assert "&{CONFIG}" in result
        assert "host=localhost" in result
        assert "port=8080" in result

    def test_format_variable_removes_existing_syntax(self, test_builder):
        """Test that existing ${} syntax is removed from variable name."""
        result = test_builder._format_variable_for_rf("${USER_NAME}", "Alice")
        assert "${USER_NAME}" in result
        assert "${${USER_NAME}}" not in result  # Should not have nested syntax

    def test_format_variable_removes_at_syntax(self, test_builder):
        """Test that existing @{} syntax is removed from variable name."""
        result = test_builder._format_variable_for_rf("@{ITEMS}", ["a", "b"])
        assert "@{ITEMS}" in result
        # Count occurrences of @{ITEMS} - should be exactly 1
        assert result.count("@{ITEMS}") == 1

    def test_format_variable_removes_ampersand_syntax(self, test_builder):
        """Test that existing &{} syntax is removed from variable name."""
        result = test_builder._format_variable_for_rf("&{CONFIG}", {"key": "value"})
        assert "&{CONFIG}" in result
        assert result.count("&{CONFIG}") == 1


class TestGeneratedTestSuiteVariablesField:
    """Test the variables field in GeneratedTestSuite dataclass."""

    def test_generated_test_suite_has_variables_field(self):
        """Test that GeneratedTestSuite dataclass has variables field."""
        from robotmcp.components.test_builder import GeneratedTestSuite
        suite = GeneratedTestSuite(
            name="TestSuite",
            test_cases=[],
            variables={"VAR1": "value1", "VAR2": 42}
        )
        assert suite.variables == {"VAR1": "value1", "VAR2": 42}

    def test_generated_test_suite_has_variable_files_field(self):
        """Test that GeneratedTestSuite dataclass has variable_files field."""
        from robotmcp.components.test_builder import GeneratedTestSuite
        suite = GeneratedTestSuite(
            name="TestSuite",
            test_cases=[],
            variable_files=["/path/to/vars.py", "/path/to/more_vars.yaml"]
        )
        assert suite.variable_files == ["/path/to/vars.py", "/path/to/more_vars.yaml"]

    def test_generated_test_suite_variables_defaults_to_none(self):
        """Test that variables field defaults to None."""
        from robotmcp.components.test_builder import GeneratedTestSuite
        suite = GeneratedTestSuite(name="TestSuite", test_cases=[])
        assert suite.variables is None
        assert suite.variable_files is None


class TestGenerateRfTextWithVariables:
    """Test _generate_rf_text includes Variables section when variables exist."""

    @pytest.fixture
    def test_builder(self):
        """Create a TestBuilder instance for testing."""
        from robotmcp.components.test_builder import TestBuilder
        return TestBuilder()

    @pytest.fixture
    def suite_with_variables(self):
        """Create a test suite with variables."""
        from robotmcp.components.test_builder import GeneratedTestSuite, GeneratedTestCase, TestCaseStep
        return GeneratedTestSuite(
            name="Test_Suite",
            test_cases=[
                GeneratedTestCase(
                    name="Test_With_Variables",
                    steps=[
                        TestCaseStep(keyword="Log", arguments=["${USER_NAME}"]),
                    ],
                ),
            ],
            imports=["Collections"],
            variables={
                "USER_NAME": "Alice Johnson",
                "USER_AGE": 28,
                "USER_SCORE": 95.5,
            }
        )

    @pytest.fixture
    def suite_without_variables(self):
        """Create a test suite without variables."""
        from robotmcp.components.test_builder import GeneratedTestSuite, GeneratedTestCase, TestCaseStep
        return GeneratedTestSuite(
            name="Test_Suite",
            test_cases=[
                GeneratedTestCase(
                    name="Simple_Test",
                    steps=[
                        TestCaseStep(keyword="Log", arguments=["Hello"]),
                    ],
                ),
            ],
            imports=["Collections"],
        )

    @pytest.mark.asyncio
    async def test_rf_text_includes_variables_section(self, test_builder, suite_with_variables):
        """Test that RF text includes *** Variables *** section when variables exist."""
        rf_text = await test_builder._generate_rf_text(suite_with_variables)
        assert "*** Variables ***" in rf_text
        assert "${USER_NAME}" in rf_text
        assert "${USER_AGE}" in rf_text
        assert "${USER_SCORE}" in rf_text
        assert "Alice Johnson" in rf_text

    @pytest.mark.asyncio
    async def test_rf_text_excludes_variables_section_when_no_variables(self, test_builder, suite_without_variables):
        """Test that RF text excludes *** Variables *** section when no variables exist."""
        rf_text = await test_builder._generate_rf_text(suite_without_variables)
        assert "*** Variables ***" not in rf_text

    @pytest.mark.asyncio
    async def test_rf_text_variables_section_comes_after_settings(self, test_builder, suite_with_variables):
        """Test that Variables section comes after Settings and before Test Cases."""
        rf_text = await test_builder._generate_rf_text(suite_with_variables)
        settings_pos = rf_text.find("*** Settings ***")
        variables_pos = rf_text.find("*** Variables ***")
        test_cases_pos = rf_text.find("*** Test Cases ***")

        # Verify order: Settings < Variables < Test Cases
        assert settings_pos < variables_pos < test_cases_pos


class TestVariableFilesInSettingsSection:
    """Test that variable files are imported in *** Settings *** section per RF documentation."""

    @pytest.fixture
    def test_builder(self):
        """Create a TestBuilder instance for testing."""
        from robotmcp.components.test_builder import TestBuilder
        return TestBuilder()

    @pytest.fixture
    def suite_with_variable_files(self):
        """Create a test suite with variable files (imported via manage_session)."""
        from robotmcp.components.test_builder import GeneratedTestSuite, GeneratedTestCase, TestCaseStep
        return GeneratedTestSuite(
            name="Test_Suite",
            test_cases=[
                GeneratedTestCase(
                    name="Test_With_Variable_Files",
                    steps=[
                        TestCaseStep(keyword="Log", arguments=["${CONFIG_VALUE}"]),
                    ],
                ),
            ],
            imports=["Collections"],
            variable_files=["/path/to/config.py", "/path/to/data.yaml"],
        )

    @pytest.fixture
    def suite_with_both_variables_and_files(self):
        """Create a test suite with both inline variables and variable files."""
        from robotmcp.components.test_builder import GeneratedTestSuite, GeneratedTestCase, TestCaseStep
        return GeneratedTestSuite(
            name="Test_Suite",
            test_cases=[
                GeneratedTestCase(
                    name="Combined_Test",
                    steps=[
                        TestCaseStep(keyword="Log", arguments=["${USER}:${CONFIG}"]),
                    ],
                ),
            ],
            imports=["Collections"],
            variables={"USER": "admin"},
            variable_files=["/path/to/config.py"],
        )

    @pytest.mark.asyncio
    async def test_variable_files_in_settings_section(self, test_builder, suite_with_variable_files):
        """Test that variable files are imported in Settings section, not Variables section."""
        rf_text = await test_builder._generate_rf_text(suite_with_variable_files)

        # Variable files should be in Settings section with "Variables" keyword
        # Path may be formatted with ${/} for cross-platform compatibility
        assert "Variables       " in rf_text
        assert "config.py" in rf_text  # File name should appear
        assert "data.yaml" in rf_text  # Second file should also appear
        # Should not have a Variables section since no inline variables
        assert "*** Variables ***" not in rf_text

    @pytest.mark.asyncio
    async def test_variable_files_come_after_libraries_in_settings(self, test_builder, suite_with_variable_files):
        """Test that variable file imports come after library imports in Settings."""
        rf_text = await test_builder._generate_rf_text(suite_with_variable_files)
        library_pos = rf_text.find("Library")
        variables_import_pos = rf_text.find("Variables       ")

        # Variables import should come after Library
        assert library_pos < variables_import_pos
        # Both should be in Settings section (before any Variables section or Test Cases)
        settings_end = rf_text.find("*** Test Cases ***")
        assert variables_import_pos < settings_end

    @pytest.mark.asyncio
    async def test_suite_with_both_has_separate_sections(self, test_builder, suite_with_both_variables_and_files):
        """Test suite with both variable files and inline variables generates correct structure."""
        rf_text = await test_builder._generate_rf_text(suite_with_both_variables_and_files)

        # Variable files should be in Settings (with "Variables" keyword)
        assert "Variables       " in rf_text  # Import in Settings

        # Inline variables should be in *** Variables *** section
        assert "*** Variables ***" in rf_text
        assert "${USER}" in rf_text

        # Settings section should contain the Variables import
        settings_section = rf_text.split("*** Variables ***")[0]
        assert "Variables       " in settings_section


class TestSuiteLevelVariablesTracking:
    """Test that only suite-level variables from manage_session are included in Variables section."""

    def test_session_has_suite_level_variables_field(self):
        """Test that ExecutionSession has suite_level_variables field."""
        from robotmcp.models.session_models import ExecutionSession
        session = ExecutionSession(session_id="test-session")
        assert hasattr(session, 'suite_level_variables')
        assert isinstance(session.suite_level_variables, set)

    def test_suite_level_variables_initially_empty(self):
        """Test that suite_level_variables is empty by default."""
        from robotmcp.models.session_models import ExecutionSession
        session = ExecutionSession(session_id="test-session")
        assert len(session.suite_level_variables) == 0

    def test_suite_level_variables_can_be_added(self):
        """Test that variables can be tracked in suite_level_variables."""
        from robotmcp.models.session_models import ExecutionSession
        session = ExecutionSession(session_id="test-session")
        session.suite_level_variables.add("BASE_URL")
        session.suite_level_variables.add("API_KEY")
        assert "BASE_URL" in session.suite_level_variables
        assert "API_KEY" in session.suite_level_variables
        assert len(session.suite_level_variables) == 2


class TestVariableReferenceScanner:
    """Test the variable reference scanner that detects untracked variables."""

    @pytest.fixture
    def test_builder(self):
        """Create a TestBuilder instance for testing."""
        from robotmcp.components.test_builder import TestBuilder
        return TestBuilder()

    @pytest.fixture
    def suite_with_variable_references(self):
        """Create a test suite with variable references in steps."""
        from robotmcp.components.test_builder import GeneratedTestSuite, GeneratedTestCase, TestCaseStep
        return GeneratedTestSuite(
            name="Test_Suite",
            test_cases=[
                GeneratedTestCase(
                    name="Test_With_Variables",
                    steps=[
                        TestCaseStep(keyword="Open Browser", arguments=["${BASE_URL}", "chrome"]),
                        TestCaseStep(keyword="Input Text", arguments=["id=username", "${USER_NAME}"]),
                        TestCaseStep(keyword="Input Password", arguments=["id=password", "${PASSWORD}"]),
                        TestCaseStep(keyword="Log", arguments=["Logging in as ${USER_NAME}"]),
                    ],
                ),
            ],
            imports=["Browser"],
            variables={"BASE_URL": "https://example.com"},  # Only BASE_URL is defined
        )

    @pytest.fixture
    def suite_with_builtin_variables(self):
        """Create a test suite using built-in RF variables."""
        from robotmcp.components.test_builder import GeneratedTestSuite, GeneratedTestCase, TestCaseStep
        return GeneratedTestSuite(
            name="Test_Suite",
            test_cases=[
                GeneratedTestCase(
                    name="Test_With_Builtins",
                    steps=[
                        TestCaseStep(keyword="Log", arguments=["Current dir: ${CURDIR}"]),
                        TestCaseStep(keyword="Log", arguments=["Test: ${TEST_NAME}"]),
                        TestCaseStep(keyword="Should Be True", arguments=["${TRUE}"]),
                    ],
                ),
            ],
            imports=["BuiltIn"],
        )

    def test_scan_finds_variable_references(self, test_builder, suite_with_variable_references):
        """Test that scanner finds all variable references in steps."""
        found_vars = test_builder._scan_variable_references(suite_with_variable_references)
        assert "BASE_URL" in found_vars
        assert "USER_NAME" in found_vars
        assert "PASSWORD" in found_vars

    def test_scan_ignores_builtin_variables(self, test_builder, suite_with_builtin_variables):
        """Test that scanner ignores built-in RF variables."""
        found_vars = test_builder._scan_variable_references(suite_with_builtin_variables)
        # Built-in variables should not be in the found set
        assert "CURDIR" not in found_vars
        assert "TEST_NAME" not in found_vars
        assert "TRUE" not in found_vars
        # The set should be empty since only builtins are used
        assert len(found_vars) == 0

    def test_check_untracked_variables_returns_warnings(self, test_builder, suite_with_variable_references):
        """Test that untracked variables generate warnings."""
        warnings = test_builder._check_untracked_variables(suite_with_variable_references, "test-session")
        # BASE_URL is defined, but USER_NAME and PASSWORD are not
        assert len(warnings) == 2
        warning_vars = {w["variable"] for w in warnings}
        assert "USER_NAME" in warning_vars
        assert "PASSWORD" in warning_vars
        assert "BASE_URL" not in warning_vars  # This one is defined

    def test_check_untracked_variables_empty_when_all_defined(self, test_builder):
        """Test that no warnings when all variables are defined."""
        from robotmcp.components.test_builder import GeneratedTestSuite, GeneratedTestCase, TestCaseStep
        suite = GeneratedTestSuite(
            name="Test_Suite",
            test_cases=[
                GeneratedTestCase(
                    name="Complete_Test",
                    steps=[
                        TestCaseStep(keyword="Log", arguments=["${MY_VAR}"]),
                    ],
                ),
            ],
            imports=["BuiltIn"],
            variables={"MY_VAR": "value"},
        )
        warnings = test_builder._check_untracked_variables(suite, "test-session")
        assert len(warnings) == 0

    def test_warning_message_contains_helpful_info(self, test_builder, suite_with_variable_references):
        """Test that warning messages are helpful."""
        warnings = test_builder._check_untracked_variables(suite_with_variable_references, "test-session")
        for warning in warnings:
            assert warning["type"] == "untracked_variable"
            assert "variable" in warning
            assert "message" in warning
            assert "manage_session" in warning["message"]
            assert "set_variables" in warning["message"]


class TestVariableHandlingEdgeCases:
    """Test edge cases in variable handling."""

    @pytest.fixture
    def test_builder(self):
        """Create a TestBuilder instance for testing."""
        from robotmcp.components.test_builder import TestBuilder
        return TestBuilder()

    def test_scan_handles_nested_variable_access(self, test_builder):
        """Test that scanner handles ${var.attr} and ${var}[0] syntax."""
        from robotmcp.components.test_builder import GeneratedTestSuite, GeneratedTestCase, TestCaseStep
        suite = GeneratedTestSuite(
            name="Test_Suite",
            test_cases=[
                GeneratedTestCase(
                    name="Nested_Access_Test",
                    steps=[
                        TestCaseStep(keyword="Log", arguments=["${response.json}"]),
                        TestCaseStep(keyword="Log", arguments=["${items}[0]"]),
                        TestCaseStep(keyword="Log", arguments=["${data.nested.value}"]),
                    ],
                ),
            ],
            imports=["BuiltIn"],
        )
        found_vars = test_builder._scan_variable_references(suite)
        # Should extract base variable names
        assert "response" in found_vars
        assert "items" in found_vars
        assert "data" in found_vars
        # Should not include the full path
        assert "response.json" not in found_vars
        assert "data.nested.value" not in found_vars

    def test_scan_handles_list_and_dict_variables(self, test_builder):
        """Test that scanner handles @{list} and &{dict} syntax."""
        from robotmcp.components.test_builder import GeneratedTestSuite, GeneratedTestCase, TestCaseStep
        suite = GeneratedTestSuite(
            name="Test_Suite",
            test_cases=[
                GeneratedTestCase(
                    name="Collection_Test",
                    steps=[
                        TestCaseStep(keyword="Log Many", arguments=["@{MY_LIST}"]),
                        TestCaseStep(keyword="Log", arguments=["&{MY_DICT}"]),
                    ],
                ),
            ],
            imports=["BuiltIn"],
        )
        found_vars = test_builder._scan_variable_references(suite)
        assert "MY_LIST" in found_vars
        assert "MY_DICT" in found_vars

    def test_scan_handles_empty_suite(self, test_builder):
        """Test that scanner handles empty test suite."""
        from robotmcp.components.test_builder import GeneratedTestSuite, GeneratedTestCase
        suite = GeneratedTestSuite(
            name="Empty_Suite",
            test_cases=[
                GeneratedTestCase(
                    name="Empty_Test",
                    steps=[],
                ),
            ],
            imports=[],
        )
        found_vars = test_builder._scan_variable_references(suite)
        assert len(found_vars) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
