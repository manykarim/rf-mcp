"""resume_batch argument fidelity
(change: desktop-evidence-and-display-scoping, D5).

The 2026-06-11 rerun observed a paused ``BuiltIn.Sleep`` retried with ZERO
arguments ("expected 1 to 2 arguments, got 0"): resume_batch's fix_steps
construction read only the ``args`` alias, never the canonical
``arguments`` key that execute_batch documents. These tests pin both the
fix-step resolution and the failed-step retry through the real tool.
"""

from __future__ import annotations
from robotmcp.compat.fastmcp_compat import get_tool_fn

from datetime import datetime

import pytest

from robotmcp import server
from robotmcp.container import get_container
from robotmcp.domains.batch_execution.aggregates import BatchState
from robotmcp.domains.batch_execution.entities import BatchStep
from robotmcp.domains.batch_execution.value_objects import (
    BatchId,
    OnFailurePolicy,
    RecoveryAttemptLimit,
)


def _seed_state(batch_id: str, steps, failed_at: int, session_id: str):
    state = BatchState(
        batch_id=BatchId(value=batch_id),
        session_id=session_id,
        original_steps=steps,
        results=[],
        results_map={},
        failed_at_index=failed_at,
        on_failure=OnFailurePolicy.STOP,
        max_recovery_attempts=RecoveryAttemptLimit(1),
        timeout_ms=30000,
        elapsed_ms=10.0,
        created_at=datetime.now(),
    )
    get_container().batch_state_manager.store(state)
    return state


@pytest.mark.asyncio
class TestResumeArgumentFidelity:
    async def test_failed_sleep_retries_with_original_args(self):
        sid = "resume-args-1"
        server.execution_engine.session_manager.get_or_create_session(sid)
        steps = [
            BatchStep(index=0, keyword="BuiltIn.Sleep", args=["0.01s"]),
            BatchStep(index=1, keyword="BuiltIn.Log", args=["after"]),
        ]
        _seed_state("batch_testargs0001", steps, failed_at=0, session_id=sid)

        result = await get_tool_fn(server.resume_batch)(batch_id="batch_testargs0001")
        statuses = [(s.get("keyword"), s.get("status")) for s in result.get("steps", [])]
        # The retried Sleep must NOT fail with "got 0" — it passes with its arg.
        assert result.get("status") == "PASS", result
        assert ("BuiltIn.Sleep", "PASS") in statuses
        assert "got 0" not in str(result)

    async def test_fix_steps_canonical_arguments_key(self):
        sid = "resume-args-2"
        server.execution_engine.session_manager.get_or_create_session(sid)
        steps = [BatchStep(index=0, keyword="BuiltIn.Sleep", args=["0.01s"])]
        _seed_state("batch_testargs0002", steps, failed_at=0, session_id=sid)

        result = await get_tool_fn(server.resume_batch)(
            batch_id="batch_testargs0002",
            fix_steps=[{"keyword": "BuiltIn.Log", "arguments": ["fix-hello"]}],
        )
        assert result.get("status") == "PASS", result
        # The fix Log step ran WITH its argument (Log with no args fails).
        kw_status = {s.get("keyword"): s.get("status") for s in result.get("steps", [])}
        assert kw_status.get("BuiltIn.Log") == "PASS"

    async def test_fix_steps_args_alias_still_works(self):
        sid = "resume-args-3"
        server.execution_engine.session_manager.get_or_create_session(sid)
        steps = [BatchStep(index=0, keyword="BuiltIn.Sleep", args=["0.01s"])]
        _seed_state("batch_testargs0003", steps, failed_at=0, session_id=sid)

        result = await get_tool_fn(server.resume_batch)(
            batch_id="batch_testargs0003",
            fix_steps=[{"keyword": "BuiltIn.Log", "args": ["fix-alias"]}],
        )
        assert result.get("status") == "PASS", result
        kw_status = {s.get("keyword"): s.get("status") for s in result.get("steps", [])}
        assert kw_status.get("BuiltIn.Log") == "PASS"
