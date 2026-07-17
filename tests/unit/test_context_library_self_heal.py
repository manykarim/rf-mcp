"""On-demand library import into an existing RF context
(post-archive fix, run-4 finding #2, 2026-06-11).

Run 4: the RF context was created before the session's library set was
final, so an unqualified ``Take Screenshot`` failed with "No keyword with
name 'Take Screenshot' found" until the agent explicitly re-imported
PlatynUI.BareMetal. The executor now ensures every session search-order
library is actually imported into the live namespace before each
context execution (idempotent).
"""

from __future__ import annotations

import pytest

from robotmcp.components.execution.execution_coordinator import ExecutionCoordinator


@pytest.fixture()
def engine():
    return ExecutionCoordinator()


@pytest.mark.asyncio
class TestContextLibrarySelfHeal:
    async def test_library_added_after_context_creation_resolves(self, engine):
        sid = "selfheal-1"
        sess = engine.session_manager.get_or_create_session(sid)
        # Context gets created with ONLY BuiltIn in the search order.
        sess.search_order = ["BuiltIn"]
        r1 = await engine.execute_step("Log", ["warmup"], sid, use_context=True)
        assert r1["success"] is True
        # The session gains a library AFTER the context exists (the run-4
        # shape: init/import_library/start_test ordering races).
        sess.search_order = ["BuiltIn", "String"]
        r2 = await engine.execute_step(
            "Replace String", ["hello world", "world", "rf"], sid,
            assign_to="out", use_context=True,
        )
        assert r2["success"] is True, r2.get("error")
        assert sess.variables.get("${out}") == "hello rf"

    async def test_unknown_library_in_order_is_tolerated(self, engine):
        # A bogus entry must not break execution of resolvable keywords.
        sid = "selfheal-2"
        sess = engine.session_manager.get_or_create_session(sid)
        sess.search_order = ["BuiltIn"]
        await engine.execute_step("Log", ["warmup"], sid, use_context=True)
        sess.search_order = ["BuiltIn", "NoSuchLibraryXyz"]
        r = await engine.execute_step("Log", ["still fine"], sid, use_context=True)
        assert r["success"] is True

    async def test_already_imported_library_not_reimported(self, engine, caplog):
        import logging

        sid = "selfheal-3"
        sess = engine.session_manager.get_or_create_session(sid)
        sess.search_order = ["BuiltIn", "String"]
        r1 = await engine.execute_step(
            "Replace String", ["a-b", "-", "+"], sid, use_context=True
        )
        assert r1["success"] is True
        with caplog.at_level(logging.INFO):
            r2 = await engine.execute_step(
                "Replace String", ["c-d", "-", "+"], sid, use_context=True
            )
        assert r2["success"] is True
        assert "On-demand import of session library 'String'" not in caplog.text
