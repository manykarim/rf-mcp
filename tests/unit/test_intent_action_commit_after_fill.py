"""intent_action(commit=True) fires a real DOM ``change`` event after FILL.

Many SPAs (Vue, React, Angular, jQuery validate, idealForms) only
commit form state in response to a real ``change`` event, which
Playwright's ``fill`` does not always emit. The old branch papered
over this with a heavy "commit_form" JS payload that mutated the
whole form. The focused replacement is a single follow-up
``Dispatch Event <selector> change`` after a successful Browser FILL,
gated behind an opt-in ``commit: bool = False`` flag.

These tests pin the gate (``_should_commit_after_fill``) and the
follow-up dispatcher (``_dispatch_change_after_fill``).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotmcp.server import (
    _dispatch_change_after_fill,
    _should_commit_after_fill,
)


# The MCP-facing IntentVerb type alias is a Literal of plain strings, so
# the gate compares against the literal string ``"fill"``. The Enum from
# domains.intent.value_objects.IntentVerb subclasses str (``str, Enum``),
# so both forms of equality work at runtime.


# ---------------------------------------------------------------------------
# Gate: _should_commit_after_fill
# ---------------------------------------------------------------------------


class TestShouldCommitAfterFill:
    """Gate fires ONLY for a successful Browser FILL with a locator arg."""

    @staticmethod
    def _kwargs(**overrides):
        defaults = dict(
            commit=True,
            intent="fill",
            result={"success": True},
            library="Browser",
            dispatched_arguments=["id=username", "alice"],
        )
        defaults.update(overrides)
        return defaults

    def test_happy_path_fires(self):
        assert _should_commit_after_fill(**self._kwargs()) is True

    def test_commit_false_skips(self):
        # The default for the MCP tool — must NOT auto-fire.
        assert _should_commit_after_fill(**self._kwargs(commit=False)) is False

    def test_failed_fill_skips(self):
        assert _should_commit_after_fill(
            **self._kwargs(result={"success": False, "error": "boom"})
        ) is False

    @pytest.mark.parametrize("verb", [
        "click", "hover", "select", "navigate",
        "assert_visible", "extract_text", "wait_for",
    ])
    def test_non_fill_intent_skips(self, verb):
        assert _should_commit_after_fill(**self._kwargs(intent=verb)) is False

    def test_enum_form_also_works(self):
        # When called from an Enum-typed call site (the value_objects
        # IntentVerb subclasses str), the gate still matches because
        # ``str.__eq__`` handles ``IntentVerb.FILL == "fill"``.
        from robotmcp.domains.intent.value_objects import IntentVerb as EnumIntent
        assert _should_commit_after_fill(
            **self._kwargs(intent=EnumIntent.FILL)
        ) is True

    @pytest.mark.parametrize("lib", [
        "SeleniumLibrary",
        "AppiumLibrary",
        "RequestsLibrary",
        None,
        "",
    ])
    def test_non_browser_library_skips(self, lib):
        assert _should_commit_after_fill(**self._kwargs(library=lib)) is False

    def test_empty_args_skips(self):
        # No locator to dispatch to → skip.
        assert _should_commit_after_fill(
            **self._kwargs(dispatched_arguments=[])
        ) is False


# ---------------------------------------------------------------------------
# Dispatcher: _dispatch_change_after_fill
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestDispatchChangeAfterFill:
    """The follow-up call's shape and best-effort semantics."""

    async def test_dispatches_change_event_with_correct_args(self):
        mock_fn = AsyncMock(return_value={"success": True})
        # Patch get_tool_fn to return our async mock for execute_step.
        with patch("robotmcp.server.get_tool_fn", return_value=mock_fn):
            result = await _dispatch_change_after_fill(
                target_locator="id=username", session_id="s1",
            )
        assert result is True
        mock_fn.assert_awaited_once()
        call_kwargs = mock_fn.await_args.kwargs
        assert call_kwargs["keyword"] == "Dispatch Event"
        assert call_kwargs["arguments"] == ["id=username", "change"]
        assert call_kwargs["session_id"] == "s1"
        # Best-effort: must not raise on failure, must not pollute history.
        assert call_kwargs["raise_on_failure"] is False
        assert call_kwargs["record"] is False

    async def test_failure_returns_false_does_not_raise(self):
        mock_fn = AsyncMock(side_effect=RuntimeError("transient"))
        with patch("robotmcp.server.get_tool_fn", return_value=mock_fn):
            # MUST NOT raise — original FILL has already succeeded.
            result = await _dispatch_change_after_fill(
                target_locator="id=foo", session_id="s1",
            )
        assert result is False

    async def test_returns_true_even_when_inner_returns_failure(self):
        # raise_on_failure=False means execute_step returns
        # ``{"success": False, ...}`` for a failed dispatch instead of
        # raising. The dispatcher considers "we tried and it didn't
        # throw" as commit_applied=True — surfacing the inner error to
        # the caller would be confusing since the FILL itself succeeded.
        mock_fn = AsyncMock(return_value={"success": False, "error": "x"})
        with patch("robotmcp.server.get_tool_fn", return_value=mock_fn):
            result = await _dispatch_change_after_fill(
                target_locator="id=foo", session_id="s1",
            )
        assert result is True
