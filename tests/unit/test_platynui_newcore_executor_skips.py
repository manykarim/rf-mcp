"""Unit tests for desktop-session executor skips (ADR-025).

Covers:
- _requires_pre_validation returns False for desktop sessions even for
  ELEMENT_INTERACTION_KEYWORDS entries ('click').
- _inject_timeout_into_arguments returns arguments unchanged for desktop
  sessions.

Run with: uv run pytest tests/unit/test_platynui_newcore_executor_skips.py -q
"""

__test__ = True

import pytest

from robotmcp.components.execution.keyword_executor import KeywordExecutor
from robotmcp.models.config_models import ExecutionConfig


@pytest.fixture
def executor():
    return KeywordExecutor(config=ExecutionConfig())


class _DesktopSession:
    session_id = "desk-1"
    imported_libraries = ["PlatynUI.BareMetal", "BuiltIn"]
    variables: dict = {}

    def is_desktop_session(self):
        return True


class _WebSession:
    session_id = "web-1"
    imported_libraries = ["Browser", "BuiltIn"]
    variables: dict = {}

    def is_desktop_session(self):
        return False


# =============================================================================
# _requires_pre_validation
# =============================================================================


class TestPreValidationSkips:
    def test_desktop_session_skips_pre_validation_for_click(self, executor):
        assert (
            executor._requires_pre_validation("click", _DesktopSession()) is False
        )

    def test_desktop_session_skips_for_fill_text(self, executor):
        assert (
            executor._requires_pre_validation("fill text", _DesktopSession())
            is False
        )

    def test_web_session_still_requires_pre_validation_for_click(self, executor):
        # Web session (is_desktop_session False) keeps the curated behaviour.
        # Result is True unless the introspector confidently vetoes; for a web
        # session with no resolvable keyword the positive list wins.
        assert (
            executor._requires_pre_validation("click", _WebSession()) is True
        )

    def test_non_interaction_keyword_still_false(self, executor):
        # 'log' is not an interaction keyword -> always False regardless.
        assert executor._requires_pre_validation("log", _DesktopSession()) is False


# =============================================================================
# _inject_timeout_into_arguments
# =============================================================================


class TestTimeoutInjectionSkips:
    def test_desktop_session_returns_arguments_unchanged(self, executor):
        args = ["/app:*//control:Button[@Name='OK']"]
        result = executor._inject_timeout_into_arguments(
            "Pointer Click", args, 5000, _DesktopSession()
        )
        assert result == args
        assert result is args  # returned unchanged object

    def test_desktop_session_unchanged_even_with_timeout(self, executor):
        args = ["/app:*//control:Text", "Hello"]
        result = executor._inject_timeout_into_arguments(
            "Keyboard Type", args, 30000, _DesktopSession()
        )
        assert result == args

    def test_no_timeout_returns_arguments_unchanged(self, executor):
        args = ["text=Login"]
        result = executor._inject_timeout_into_arguments(
            "Click", args, None, _WebSession()
        )
        assert result == args
