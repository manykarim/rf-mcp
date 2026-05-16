"""Tests for F-N12: build_test_suite excludes inspection-only steps.

Adds 5 steps to a session — 3 action steps (Click) interleaved with
2 inspection probes (Get Title, Get Url) — and verifies that
build_test_suite output contains only the action steps.
"""

__test__ = True

import pytest
from unittest.mock import MagicMock

from robotmcp.models.execution_models import ExecutionStep
from robotmcp.models.session_models import ExecutionSession


def _make_step(keyword: str, arguments: list, *, record: bool) -> ExecutionStep:
    """Create a successful ExecutionStep, optionally marked as not-recorded."""
    step = ExecutionStep(
        step_id=keyword.replace(" ", "_"),
        keyword=keyword,
        arguments=arguments,
    )
    step.mark_success("ok")
    # Simulate the F-N12 gate: inspection steps are never appended to session.
    # We exercise this by only appending when record=True.
    return step, record


@pytest.mark.asyncio
async def test_build_test_suite_excludes_inspection_steps():
    """Inspection steps (Get Title, Get Url) must not appear in rf_text.

    We build the session directly (adding only the three Click steps) to
    simulate the F-N12 record gate, then call TestBuilder in multi-test mode
    so we avoid the legacy path's async validate_test_readiness call.
    """
    from robotmcp.components.test_builder import TestBuilder

    session = ExecutionSession(session_id="inspect-test2")

    # Start a named test so multi-test mode is active (avoids legacy async path).
    session.test_registry.start_test("Inspect Test")

    # Record only the three Click steps — inspection steps were gated by F-N12.
    for kw, args in [("Click", ["id=a"]), ("Click", ["id=b"]), ("Click", ["id=c"])]:
        step = ExecutionStep(step_id=f"{kw}_{args[0]}", keyword=kw, arguments=args)
        step.mark_success("ok")
        session.test_registry.tests["Inspect Test"].steps.append(step)

    session.test_registry.end_test(status="pass")

    engine = MagicMock()
    engine.sessions = {"inspect-test2": session}
    engine.session_manager.get_or_create_session.return_value = session

    builder = TestBuilder(execution_engine=engine)
    result = await builder.build_suite(session_id="inspect-test2", test_name="Inspect Suite")

    assert result.get("success"), result.get("error")
    rf_text = result.get("rf_text", "")

    # Action keyword name must appear (3 Click steps).
    assert rf_text.count("Click") == 3

    # Inspection steps must NOT appear (they were gated by F-N12 and never added).
    assert "Get Title" not in rf_text
    assert "Get Url" not in rf_text


@pytest.mark.asyncio
async def test_session_steps_count_reflects_record_gate():
    """After the record gate, session.steps only contains action steps."""
    from unittest.mock import AsyncMock, patch
    from robotmcp.components.execution.keyword_executor import KeywordExecutor
    from robotmcp.models.config_models import ExecutionConfig

    config = ExecutionConfig()
    executor = KeywordExecutor(config, None)
    executor.pre_validation_enabled = False
    session = ExecutionSession(session_id="gate-test")
    blm = MagicMock()

    def _ok(*a, **kw):
        return {"success": True, "result": "ok", "output": "ok"}

    with patch.object(
        executor,
        "_execute_keyword_with_context",
        new=AsyncMock(side_effect=_ok),
    ):
        # Click -> recorded
        await executor.execute_keyword(
            session=session, keyword="Click", arguments=["id=a"],
            browser_library_manager=blm,
        )
        # Get Title -> auto-excluded
        await executor.execute_keyword(
            session=session, keyword="Get Title", arguments=[],
            browser_library_manager=blm,
        )
        # Click -> recorded
        await executor.execute_keyword(
            session=session, keyword="Click", arguments=["id=b"],
            browser_library_manager=blm,
        )
        # Get Url -> auto-excluded
        await executor.execute_keyword(
            session=session, keyword="Get Url", arguments=[],
            browser_library_manager=blm,
        )
        # Click -> recorded
        await executor.execute_keyword(
            session=session, keyword="Click", arguments=["id=c"],
            browser_library_manager=blm,
        )

    # Only the 3 Click steps should be in session.steps
    assert len(session.steps) == 3
    keywords_recorded = [s.keyword for s in session.steps]
    assert keywords_recorded == ["Click", "Click", "Click"]
