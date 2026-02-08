"""Helpers for mocking Robot Framework execution context in unit tests.

After S5a/S5b, pre-validation and post-execution library detection use
`namespace.get_runner(keyword).keyword.owner.name` from the RF execution
context instead of session.browser_state.active_library.  These helpers
create properly structured mocks for that chain.
"""

from contextlib import contextmanager
from unittest.mock import MagicMock, patch


@contextmanager
def rf_context_with_owner(owner_name):
    """Mock RF EXECUTION_CONTEXTS so get_runner() returns a keyword owned by *owner_name*.

    The mock chain: _EC.current.namespace.get_runner(kw).keyword.owner.name

    Args:
        owner_name: Library name string (e.g. "Browser", "SeleniumLibrary",
                    "AppiumLibrary", "BuiltIn") or None for invalid keywords.
    """
    if owner_name is not None:
        mock_owner = MagicMock()
        mock_owner.name = owner_name
    else:
        mock_owner = None

    mock_kw = MagicMock()
    mock_kw.owner = mock_owner

    mock_runner = MagicMock()
    mock_runner.keyword = mock_kw

    mock_namespace = MagicMock()
    mock_namespace.get_runner.return_value = mock_runner

    mock_ctx = MagicMock()
    mock_ctx.namespace = mock_namespace
    mock_ctx.test = MagicMock()  # non-None prevents MCP_PreValidation creation

    mock_ec = MagicMock()
    mock_ec.current = mock_ctx

    with patch("robot.running.context.EXECUTION_CONTEXTS", mock_ec):
        yield mock_ctx


@contextmanager
def no_rf_context():
    """Mock EXECUTION_CONTEXTS with current=None (no active RF context)."""
    mock_ec = MagicMock()
    mock_ec.current = None

    with patch("robot.running.context.EXECUTION_CONTEXTS", mock_ec):
        yield
