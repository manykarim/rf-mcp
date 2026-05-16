"""Tests for F-N12: execute_step record parameter.

Verifies that:
- record=True  -> step is recorded in session.steps
- record=False -> step executes but is NOT added to session.steps
- record=None, action keyword -> step IS recorded (auto-classified)
- record=None, inspection keyword -> step is NOT recorded (auto-classified)
"""

__test__ = True

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from robotmcp.models.execution_models import ExecutionStep
from robotmcp.models.session_models import ExecutionSession


def _make_executor():
    """Return a KeywordExecutor with minimal mocking."""
    from robotmcp.components.execution.keyword_executor import KeywordExecutor
    from robotmcp.models.config_models import ExecutionConfig

    config = ExecutionConfig()
    executor = KeywordExecutor(config, None)
    return executor


def _make_session():
    sess = ExecutionSession(session_id="test-record")
    return sess


def _fake_context_result(success: bool = True, output: str = "ok"):
    """Minimal result dict returned by _execute_keyword_with_context."""
    if success:
        return {"success": True, "result": output, "output": output}
    return {"success": False, "error": "simulated failure"}


@pytest.mark.asyncio
async def test_record_true_adds_step():
    """Explicit record=True: step is added to session.steps."""
    executor = _make_executor()
    executor.pre_validation_enabled = False
    session = _make_session()
    blm = MagicMock()

    with patch.object(
        executor,
        "_execute_keyword_with_context",
        new=AsyncMock(return_value=_fake_context_result()),
    ):
        await executor.execute_keyword(
            session=session,
            keyword="Click",
            arguments=["id=foo"],
            browser_library_manager=blm,
            record=True,
        )

    assert len(session.steps) == 1
    assert session.steps[0].keyword == "Click"


@pytest.mark.asyncio
async def test_record_false_does_not_add_step():
    """Explicit record=False: keyword runs but step is NOT added."""
    executor = _make_executor()
    executor.pre_validation_enabled = False
    session = _make_session()
    blm = MagicMock()

    with patch.object(
        executor,
        "_execute_keyword_with_context",
        new=AsyncMock(return_value=_fake_context_result()),
    ):
        result = await executor.execute_keyword(
            session=session,
            keyword="Click",
            arguments=["id=foo"],
            browser_library_manager=blm,
            record=False,
        )

    assert len(session.steps) == 0
    assert result["success"] is True
    assert result.get("recorded") is False


@pytest.mark.asyncio
async def test_record_none_inspection_keyword_not_added():
    """record=None + 'get title' -> auto-classified as inspection, not recorded."""
    executor = _make_executor()
    session = _make_session()
    blm = MagicMock()

    with patch.object(
        executor,
        "_execute_keyword_with_context",
        new=AsyncMock(return_value=_fake_context_result(output="My Page")),
    ):
        result = await executor.execute_keyword(
            session=session,
            keyword="Get Title",
            arguments=[],
            browser_library_manager=blm,
            record=None,
        )

    assert len(session.steps) == 0
    assert result["success"] is True
    assert result.get("recorded") is False


@pytest.mark.asyncio
async def test_record_none_action_keyword_added():
    """record=None + 'Click' -> auto-classified as action, IS recorded."""
    executor = _make_executor()
    # Disable pre-validation so it doesn't interfere with mocked execution.
    executor.pre_validation_enabled = False
    session = _make_session()
    blm = MagicMock()

    with patch.object(
        executor,
        "_execute_keyword_with_context",
        new=AsyncMock(return_value=_fake_context_result()),
    ):
        result = await executor.execute_keyword(
            session=session,
            keyword="Click",
            arguments=["id=btn"],
            browser_library_manager=blm,
            record=None,
        )

    assert len(session.steps) == 1
    assert result.get("recorded") is True


@pytest.mark.asyncio
async def test_record_none_get_url_not_added():
    """record=None + 'Get Url' -> not recorded."""
    executor = _make_executor()
    session = _make_session()
    blm = MagicMock()

    with patch.object(
        executor,
        "_execute_keyword_with_context",
        new=AsyncMock(return_value=_fake_context_result(output="https://example.com")),
    ):
        await executor.execute_keyword(
            session=session,
            keyword="Get Url",
            arguments=[],
            browser_library_manager=blm,
        )

    assert len(session.steps) == 0


@pytest.mark.asyncio
async def test_failed_step_never_recorded_regardless_of_record_flag():
    """Failed steps are never added to session.steps even with record=True."""
    executor = _make_executor()
    executor.pre_validation_enabled = False
    session = _make_session()
    blm = MagicMock()

    with patch.object(
        executor,
        "_execute_keyword_with_context",
        new=AsyncMock(return_value=_fake_context_result(success=False)),
    ):
        result = await executor.execute_keyword(
            session=session,
            keyword="Click",
            arguments=["id=missing"],
            browser_library_manager=blm,
            record=True,
        )

    assert len(session.steps) == 0
    assert result["success"] is False


@pytest.mark.asyncio
async def test_assign_to_overrides_inspection_auto_classification():
    """When assign_to is set, even an inspection keyword (Get Text) is recorded.

    The canonical RF pattern is:
        ${actual}=    Get Text    css=h1
        Should Be Equal    ${actual}    Welcome
    Without this override, the Get Text step would be silently dropped and
    the generated suite would reference an undefined ${actual}.
    """
    executor = _make_executor()
    executor.pre_validation_enabled = False
    session = _make_session()
    blm = MagicMock()

    with patch.object(
        executor,
        "_execute_keyword_with_context",
        new=AsyncMock(return_value=_fake_context_result(output="Welcome")),
    ):
        result = await executor.execute_keyword(
            session=session,
            keyword="Get Text",
            arguments=["css=h1"],
            browser_library_manager=blm,
            assign_to="actual",
        )

    assert len(session.steps) == 1
    assert result.get("recorded") is True
    assert session.steps[0].keyword == "Get Text"


@pytest.mark.asyncio
async def test_assign_to_does_not_override_explicit_record_false():
    """Explicit record=False still wins over assign_to.

    Use case: agent captures a debug snapshot into a variable but knows it
    should not appear in the generated suite.
    """
    executor = _make_executor()
    executor.pre_validation_enabled = False
    session = _make_session()
    blm = MagicMock()

    with patch.object(
        executor,
        "_execute_keyword_with_context",
        new=AsyncMock(return_value=_fake_context_result(output="page snapshot")),
    ):
        result = await executor.execute_keyword(
            session=session,
            keyword="Get Text",
            arguments=["css=body"],
            browser_library_manager=blm,
            assign_to="debug_snapshot",
            record=False,
        )

    assert len(session.steps) == 0
    assert result.get("recorded") is False
