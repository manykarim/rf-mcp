"""v0.32.5 P1 — wrapper_suggestion hint must fire on execution-time failures
that occurred AFTER the pre-validation gate passed.

The pre-validation gate at ``_pre_validate_element`` already calls
WrapperSuggester when the gate rejects "missing visible".  But the most
common idealForms pattern (decorative ``<span class="ideal-radio">``
overlay covering a visible checkbox) does NOT fail at the gate — the
gate sees the input as visible.  It fails INSIDE the keyword execution
with "Click intercepted by overlapping element".  Pre-v0.32.5 that
failure path did not consult WrapperSuggester at all, so the LLM got
generic hints and reached for ``Evaluate JavaScript`` DOM mutation.

These tests pin the post-gate injection of ``wrapper_suggestion``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from robotmcp.components.execution.keyword_executor import KeywordExecutor
from robotmcp.models.config_models import ExecutionConfig
from robotmcp.models.session_models import BrowserState, ExecutionSession


@pytest.fixture
def executor():
    return KeywordExecutor(ExecutionConfig())


@pytest.fixture
def browser_session():
    s = MagicMock(spec=ExecutionSession)
    s.session_id = "p1-test"
    s.browser_state = MagicMock(spec=BrowserState)
    s.browser_state.active_library = "browser"
    s.imported_libraries = ["Browser", "BuiltIn"]
    return s


class TestExecutionFailureMarkers:
    """Confirm the marker substrings include the documented failure modes."""

    @pytest.mark.parametrize("err_text,should_match", [
        ("Click intercepted by overlapping element", True),
        ("strict mode violation: element subtree intercepts pointer events", True),
        ("locator.click: Element is not visible", True),
        ("locator.click: Element is not enabled", True),
        ("Element is not stable, retrying", True),
        ("Element is outside of the viewport", True),
        ("Page has been closed", False),  # not actionability
        ("TimeoutError: waiting for navigation", False),  # not actionability
        ("", False),
    ])
    def test_marker_match(self, executor, err_text, should_match):
        markers = executor._ACTIONABILITY_FAILURE_MARKERS
        matched = any(m in err_text.lower() for m in markers)
        assert matched is should_match, (
            f"Marker match for '{err_text}' was {matched}, expected {should_match}"
        )


class TestWrapperSuggestionInjectedOnExecutionFailure:

    @pytest.mark.asyncio
    async def test_injects_hint_on_intercepted_click(
        self, executor, browser_session, monkeypatch
    ):
        """Given a click failure with "intercepted", the helper must call
        WrapperSuggester.suggest and append the returned hint."""
        from robotmcp.components.execution import keyword_executor as ke_mod

        async def _fake_suggest(session, locator, keyword):
            return {
                "type": "wrapper_suggestion",
                "message": f"Element '{locator}' is hidden inside a visible wrapper.",
                "suggestions": [{
                    "description": "Try the wrapper label",
                    "selector": f"*css=label >> {locator}",
                    "action_keyword": "Click",
                }],
            }

        monkeypatch.setattr(
            ke_mod.WrapperSuggester, "suggest", AsyncMock(side_effect=_fake_suggest)
        )

        hints_in: list = []
        hints_out = await executor._add_wrapper_suggestion_on_execution_failure(
            keyword="Click",
            arguments=["id=gendermale"],
            error_text="Click intercepted by overlapping element",
            session=browser_session,
            hints=hints_in,
        )
        assert len(hints_out) == 1
        h = hints_out[0]
        assert h["type"] == "wrapper_suggestion"
        assert h.get("source") == "execution_failure"
        assert any("label" in s.get("selector", "") for s in h["suggestions"])

    @pytest.mark.asyncio
    async def test_skips_when_error_is_not_actionability(
        self, executor, browser_session, monkeypatch
    ):
        from robotmcp.components.execution import keyword_executor as ke_mod
        mock_suggest = AsyncMock(return_value={"type": "wrapper_suggestion", "message": "x", "suggestions": []})
        monkeypatch.setattr(ke_mod.WrapperSuggester, "suggest", mock_suggest)

        hints_out = await executor._add_wrapper_suggestion_on_execution_failure(
            keyword="Click",
            arguments=["id=foo"],
            error_text="No keyword with name 'Foo' found",
            session=browser_session,
            hints=[],
        )
        assert hints_out == []
        mock_suggest.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_wrapper_hint_already_present(
        self, executor, browser_session, monkeypatch
    ):
        """If the gate-side hint already injected a wrapper_suggestion (e.g.
        for the same element on the prior pre-validation pass), don't add a
        duplicate."""
        from robotmcp.components.execution import keyword_executor as ke_mod
        mock_suggest = AsyncMock(return_value={
            "type": "wrapper_suggestion", "message": "x", "suggestions": [],
        })
        monkeypatch.setattr(ke_mod.WrapperSuggester, "suggest", mock_suggest)

        existing = [{"type": "wrapper_suggestion", "message": "from gate", "suggestions": []}]
        hints_out = await executor._add_wrapper_suggestion_on_execution_failure(
            keyword="Click",
            arguments=["id=foo"],
            error_text="Click intercepted by overlapping element",
            session=browser_session,
            hints=existing,
        )
        assert hints_out == existing
        mock_suggest.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_locator_argument_returns_hints_unchanged(
        self, executor, browser_session, monkeypatch
    ):
        from robotmcp.components.execution import keyword_executor as ke_mod
        mock_suggest = AsyncMock()
        monkeypatch.setattr(ke_mod.WrapperSuggester, "suggest", mock_suggest)

        hints_out = await executor._add_wrapper_suggestion_on_execution_failure(
            keyword="Sleep",
            arguments=[],
            error_text="Element is not visible",
            session=browser_session,
            hints=[],
        )
        assert hints_out == []
        mock_suggest.assert_not_called()

    @pytest.mark.asyncio
    async def test_suggester_returns_none_no_hint_added(
        self, executor, browser_session, monkeypatch
    ):
        from robotmcp.components.execution import keyword_executor as ke_mod
        monkeypatch.setattr(
            ke_mod.WrapperSuggester, "suggest", AsyncMock(return_value=None)
        )

        hints_out = await executor._add_wrapper_suggestion_on_execution_failure(
            keyword="Click",
            arguments=["id=foo"],
            error_text="Click intercepted by overlapping element",
            session=browser_session,
            hints=[],
        )
        assert hints_out == []

    @pytest.mark.asyncio
    async def test_suggester_exception_is_soft_failed(
        self, executor, browser_session, monkeypatch
    ):
        """An exception inside WrapperSuggester.suggest must not surface;
        the helper soft-fails and returns the unchanged hints list."""
        from robotmcp.components.execution import keyword_executor as ke_mod
        monkeypatch.setattr(
            ke_mod.WrapperSuggester, "suggest",
            AsyncMock(side_effect=RuntimeError("probe blew up")),
        )

        hints_out = await executor._add_wrapper_suggestion_on_execution_failure(
            keyword="Click",
            arguments=["id=foo"],
            error_text="Click intercepted by overlapping element",
            session=browser_session,
            hints=[{"type": "pre_validation_failure", "message": "x"}],
        )
        assert hints_out == [{"type": "pre_validation_failure", "message": "x"}]


class TestProvenanceMarker:
    """The injected hint must declare ``source: 'execution_failure'`` so a
    response consumer can tell it apart from a gate-side hint."""

    @pytest.mark.asyncio
    async def test_source_is_execution_failure(
        self, executor, browser_session, monkeypatch
    ):
        from robotmcp.components.execution import keyword_executor as ke_mod
        monkeypatch.setattr(
            ke_mod.WrapperSuggester, "suggest",
            AsyncMock(return_value={
                "type": "wrapper_suggestion",
                "message": "x",
                "suggestions": [{"description": "d", "selector": "s", "action_keyword": "Click"}],
            }),
        )

        hints_out = await executor._add_wrapper_suggestion_on_execution_failure(
            keyword="Click",
            arguments=["id=foo"],
            error_text="Element is not visible",
            session=browser_session,
            hints=[],
        )
        assert hints_out[0]["source"] == "execution_failure"
