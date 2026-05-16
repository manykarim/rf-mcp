"""Proposal-B integration: the record gate honours JS classification.

Pairs with test_proposal_b_evaluate_js_curation.py (classifier-level tests).
These tests exercise the full execute_keyword path with mocked context.
"""

from __future__ import annotations

__test__ = True

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from robotmcp.models.session_models import ExecutionSession


def _make_executor():
    from robotmcp.components.execution.keyword_executor import KeywordExecutor
    from robotmcp.models.config_models import ExecutionConfig

    return KeywordExecutor(ExecutionConfig(), None)


def _ok(*_args, **_kwargs):
    return {"success": True, "result": "ok", "output": "ok"}


@pytest.mark.asyncio
async def test_readonly_evaluate_js_is_not_recorded():
    """A getBoundingClientRect probe must not appear in session.steps."""
    executor = _make_executor()
    executor.pre_validation_enabled = False
    session = ExecutionSession(session_id="b-readonly")
    blm = MagicMock()

    with patch.object(
        executor,
        "_execute_keyword_with_context",
        new=AsyncMock(side_effect=_ok),
    ):
        result = await executor.execute_keyword(
            session=session,
            keyword="Evaluate JavaScript",
            arguments=["css=html", "elem => JSON.stringify(elem.getBoundingClientRect())"],
            browser_library_manager=blm,
        )

    assert len(session.steps) == 0
    assert result.get("recorded") is False


@pytest.mark.asyncio
async def test_mutation_evaluate_js_is_recorded():
    """A .value = X mutation must appear in session.steps."""
    executor = _make_executor()
    executor.pre_validation_enabled = False
    session = ExecutionSession(session_id="b-mutation")
    blm = MagicMock()

    with patch.object(
        executor,
        "_execute_keyword_with_context",
        new=AsyncMock(side_effect=_ok),
    ):
        result = await executor.execute_keyword(
            session=session,
            keyword="Evaluate JavaScript",
            arguments=["css=html", "elem => { elem.value = '25000'; }"],
            browser_library_manager=blm,
        )

    assert len(session.steps) == 1
    assert result.get("recorded") is True


@pytest.mark.asyncio
async def test_unclassifiable_evaluate_js_defaults_to_recorded():
    """An arbitrary expression that doesn't match either pattern set is kept."""
    executor = _make_executor()
    executor.pre_validation_enabled = False
    session = ExecutionSession(session_id="b-default")
    blm = MagicMock()

    with patch.object(
        executor,
        "_execute_keyword_with_context",
        new=AsyncMock(side_effect=_ok),
    ):
        result = await executor.execute_keyword(
            session=session,
            keyword="Evaluate JavaScript",
            arguments=["css=html", "() => 42"],
            browser_library_manager=blm,
        )

    assert len(session.steps) == 1
    assert result.get("recorded") is True


@pytest.mark.asyncio
async def test_explicit_record_false_overrides_classifier():
    """When the agent passes record=False, the gate respects that even if
    the JS looks like a mutation."""
    executor = _make_executor()
    executor.pre_validation_enabled = False
    session = ExecutionSession(session_id="b-override-false")
    blm = MagicMock()

    with patch.object(
        executor,
        "_execute_keyword_with_context",
        new=AsyncMock(side_effect=_ok),
    ):
        result = await executor.execute_keyword(
            session=session,
            keyword="Evaluate JavaScript",
            arguments=["css=html", "elem => { elem.value = 'x'; }"],
            browser_library_manager=blm,
            record=False,
        )

    assert len(session.steps) == 0
    assert result.get("recorded") is False


@pytest.mark.asyncio
async def test_explicit_record_true_overrides_classifier():
    """When the agent passes record=True, the gate keeps the step even if
    the JS is read-only."""
    executor = _make_executor()
    executor.pre_validation_enabled = False
    session = ExecutionSession(session_id="b-override-true")
    blm = MagicMock()

    with patch.object(
        executor,
        "_execute_keyword_with_context",
        new=AsyncMock(side_effect=_ok),
    ):
        result = await executor.execute_keyword(
            session=session,
            keyword="Evaluate JavaScript",
            arguments=["css=html", "elem => JSON.stringify(elem.getBoundingClientRect())"],
            browser_library_manager=blm,
            record=True,
        )

    assert len(session.steps) == 1
    assert result.get("recorded") is True


@pytest.mark.asyncio
async def test_execute_javascript_selenium_form_classified():
    """SeleniumLibrary's 'Execute Javascript' (JS body in arg[0]) is also gated."""
    executor = _make_executor()
    executor.pre_validation_enabled = False
    session = ExecutionSession(session_id="b-selenium")
    blm = MagicMock()

    with patch.object(
        executor,
        "_execute_keyword_with_context",
        new=AsyncMock(side_effect=_ok),
    ):
        result = await executor.execute_keyword(
            session=session,
            keyword="Execute Javascript",
            arguments=[
                "return (el => JSON.stringify(el.getBoundingClientRect()))(arguments[0]);",
                "css=#input",
            ],
            browser_library_manager=blm,
        )

    assert len(session.steps) == 0
    assert result.get("recorded") is False
