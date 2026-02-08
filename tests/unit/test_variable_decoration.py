"""Unit tests for _add_variable_decoration in rf_native_context_manager.

Tests the variable naming decoration logic that converts Python variable
names (with optional LIST__/DICT__ prefixes) to Robot Framework decorated
names (${}, @{}, &{}).

Run with: uv run pytest tests/unit/test_variable_decoration.py -v
"""

__test__ = True

import pytest
from robotmcp.components.execution.rf_native_context_manager import (
    get_rf_native_context_manager,
)


@pytest.fixture
def context_manager():
    """Get the RF native context manager singleton."""
    return get_rf_native_context_manager()


class TestVariableDecoration:
    """Test _add_variable_decoration method."""

    def test_list_prefix_becomes_at_decoration(self, context_manager):
        """LIST__items -> @{items}."""
        result = context_manager._add_variable_decoration("LIST__items", ["a", "b"])
        assert result == "@{items}"

    def test_dict_prefix_becomes_ampersand_decoration(self, context_manager):
        """DICT__config -> &{config}."""
        result = context_manager._add_variable_decoration("DICT__config", {"key": "val"})
        assert result == "&{config}"

    def test_plain_list_value_inferred_as_list(self, context_manager):
        """Plain list value (no prefix) -> @{name}."""
        result = context_manager._add_variable_decoration("items", ["a", "b", "c"])
        assert result == "@{items}"

    def test_plain_tuple_value_inferred_as_list(self, context_manager):
        """Plain tuple value (no prefix) -> @{name}."""
        result = context_manager._add_variable_decoration("coords", (1, 2))
        assert result == "@{coords}"

    def test_plain_dict_value_inferred_as_dict(self, context_manager):
        """Plain dict value (no prefix) -> &{name}."""
        result = context_manager._add_variable_decoration("config", {"host": "localhost"})
        assert result == "&{config}"

    def test_scalar_value_becomes_dollar(self, context_manager):
        """Scalar value -> ${name}."""
        result = context_manager._add_variable_decoration("username", "admin")
        assert result == "${username}"

    def test_already_decorated_dollar_passthrough(self, context_manager):
        """${already_decorated} -> ${already_decorated} (no change)."""
        result = context_manager._add_variable_decoration("${already}", "value")
        assert result == "${already}"

    def test_already_decorated_at_passthrough(self, context_manager):
        """@{already_decorated} -> @{already_decorated} (no change)."""
        result = context_manager._add_variable_decoration("@{already}", [1, 2])
        assert result == "@{already}"

    def test_already_decorated_ampersand_passthrough(self, context_manager):
        """&{already_decorated} -> &{already_decorated} (no change)."""
        result = context_manager._add_variable_decoration("&{already}", {"k": "v"})
        assert result == "&{already}"

    def test_integer_scalar(self, context_manager):
        """Integer value -> ${name}."""
        result = context_manager._add_variable_decoration("count", 42)
        assert result == "${count}"

    def test_none_scalar(self, context_manager):
        """None value -> ${name} (scalar)."""
        result = context_manager._add_variable_decoration("empty_val", None)
        assert result == "${empty_val}"

    def test_boolean_scalar(self, context_manager):
        """Boolean value -> ${name} (scalar)."""
        result = context_manager._add_variable_decoration("flag", True)
        assert result == "${flag}"
