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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
