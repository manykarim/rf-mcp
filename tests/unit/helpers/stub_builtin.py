"""Shared StubBuiltIn test double for unit tests.

Consolidates 4+ duplicate implementations of StubBuiltIn used across
test files into a single reusable helper.
"""

from unittest.mock import MagicMock


class StubBuiltIn:
    """Lightweight stand-in for robot.libraries.BuiltIn.

    Provides run_keyword() that can be configured with side_effect
    or return_value, and tracks calls for assertions.
    """

    def __init__(self):
        self.calls = []
        self._side_effect = None
        self._return_value = None
        self._keyword_handlers = {}

    def run_keyword(self, name, *args, **kwargs):
        """Record the call and delegate to configured handler or side_effect."""
        self.calls.append((name, args, kwargs))

        # Check for keyword-specific handler first
        name_lower = name.lower()
        if name_lower in self._keyword_handlers:
            return self._keyword_handlers[name_lower](name, *args, **kwargs)

        if self._side_effect is not None:
            if callable(self._side_effect):
                return self._side_effect(name, *args, **kwargs)
            raise self._side_effect

        return self._return_value

    def set_side_effect(self, side_effect):
        """Configure a side effect for run_keyword calls."""
        self._side_effect = side_effect
        return self

    def set_return_value(self, value):
        """Configure a static return value for run_keyword calls."""
        self._return_value = value
        return self

    def register_keyword_handler(self, keyword_name, handler):
        """Register a handler for a specific keyword name."""
        self._keyword_handlers[keyword_name.lower()] = handler
        return self

    def get_calls_for(self, keyword_name):
        """Get all calls matching a keyword name (case-insensitive)."""
        keyword_lower = keyword_name.lower()
        return [c for c in self.calls if c[0].lower() == keyword_lower]

    def assert_called_with(self, keyword_name):
        """Assert that a specific keyword was called at least once."""
        matching = self.get_calls_for(keyword_name)
        assert matching, f"Expected call to '{keyword_name}', but it was never called. Calls: {[c[0] for c in self.calls]}"
        return matching

    def assert_not_called_with(self, keyword_name):
        """Assert that a specific keyword was never called."""
        matching = self.get_calls_for(keyword_name)
        assert not matching, f"Expected no call to '{keyword_name}', but found {len(matching)} calls"
