"""assign_to persistence across execution paths
(change: platynui-visible-safe-targeting, I-3).

The LibreOffice validation run reported desktop read-back ``Query``
assignments "not persisting unless use_context=True". Locally, the
executor runs in context-only mode (`use_context` has no local branching
effect) and assignment persists on BOTH flag values — these tests pin
that contract so any future re-introduction of a non-context path cannot
silently drop assigned variables again.
"""

from __future__ import annotations

import pytest

from robotmcp.components.execution.execution_coordinator import ExecutionCoordinator


@pytest.fixture()
def engine():
    return ExecutionCoordinator()


@pytest.mark.asyncio
class TestAssignmentPersistence:
    async def test_non_context_assignment_persists_to_session(self, engine):
        sid = "assign-nc"
        sess = engine.session_manager.get_or_create_session(sid)
        r1 = await engine.execute_step(
            "Evaluate", ["[1, 2, 3]"], sid, assign_to="nodes", use_context=False
        )
        assert r1["success"] is True
        assert "${nodes}" in sess.variables
        assert sess.variables["${nodes}"] == [1, 2, 3]
        # The assigned variable resolves in a subsequent step.
        r2 = await engine.execute_step(
            "Evaluate", ["len($nodes)"], sid, assign_to="count", use_context=False
        )
        assert r2["success"] is True
        assert sess.variables["${count}"] == 3

    async def test_context_path_identical(self, engine):
        sid = "assign-ctx"
        sess = engine.session_manager.get_or_create_session(sid)
        r1 = await engine.execute_step(
            "Evaluate", ["[1, 2, 3]"], sid, assign_to="nodes", use_context=True
        )
        assert r1["success"] is True
        assert sess.variables.get("${nodes}") == [1, 2, 3]

    async def test_cross_flag_resolution(self, engine):
        # Assigned without context, consumed with context — one variable
        # store, no divergence between the paths.
        sid = "assign-x"
        sess = engine.session_manager.get_or_create_session(sid)
        await engine.execute_step(
            "Evaluate", ["'hello'"], sid, assign_to="greeting", use_context=False
        )
        r2 = await engine.execute_step(
            "Evaluate", ["$greeting.upper()"], sid,
            assign_to="loud", use_context=True,
        )
        assert r2["success"] is True
        assert sess.variables["${loud}"] == "HELLO"

    async def test_multi_assign_persists_all(self, engine):
        sid = "assign-multi"
        sess = engine.session_manager.get_or_create_session(sid)
        r = await engine.execute_step(
            "Evaluate", ["(1, 'two')"], sid,
            assign_to=["first", "second"], use_context=False,
        )
        assert r["success"] is True
        assert sess.variables.get("${first}") == 1
        assert sess.variables.get("${second}") == "two"

    async def test_failed_step_assigns_nothing(self, engine):
        sid = "assign-fail"
        sess = engine.session_manager.get_or_create_session(sid)
        r = await engine.execute_step(
            "Evaluate", ["1/0"], sid, assign_to="boom", use_context=False
        )
        assert r["success"] is False
        assert "${boom}" not in sess.variables
