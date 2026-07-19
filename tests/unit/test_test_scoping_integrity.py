"""Test-scoping integrity
(change: desktop-test-scoping-and-close-lifecycle, D1-D4).

Run 3 (2026-06-11): an early start_test failed with "No context for
session", the failure was swallowed as success, and 43 of 46 recorded
steps landed in suite_level_steps — invisible to build_test_suite. The
final end_test reported "No active test to end" from the wrong layer.

These tests drive the REAL manage_session / execute_step / build_test_suite
tools end to end.
"""

from __future__ import annotations
from robotmcp.compat.fastmcp_compat import get_tool_fn

import pytest

from robotmcp import server


async def _start_test(sid, name, **kwargs):
    return await get_tool_fn(server.manage_session)(
        action="start_test", session_id=sid, test_name=name, **kwargs
    )


async def _end_test(sid, **kwargs):
    return await get_tool_fn(server.manage_session)(
        action="end_test", session_id=sid, **kwargs
    )


@pytest.mark.asyncio
class TestStartTestBeforeAnyStep:
    async def test_start_test_immediately_after_init_succeeds(self):
        sid = "scope-early"
        server.execution_engine.session_manager.get_or_create_session(sid)
        res = await _start_test(sid, "Early Test")
        assert res["success"] is True, res
        assert "No context for session" not in str(res)
        # The context layer really started the test (no swallowed failure).
        assert res["context_result"].get("success") is True
        # And steps land in the test.
        step = await server.execution_engine.execute_step(
            "Log", ["hello"], sid, use_context=True
        )
        assert step["success"] is True
        session = server.execution_engine.session_manager.get_session(sid)
        current = session.test_registry.get_current_test()
        assert current is not None
        assert len(current.steps) == 1
        assert len(session.suite_level_steps) == 0
        ended = await _end_test(sid)
        assert ended["success"] is True

    async def test_atomic_failure_leaves_registry_untouched(self, monkeypatch):
        sid = "scope-atomic"
        server.execution_engine.session_manager.get_or_create_session(sid)
        from robotmcp.components.execution import rf_native_context_manager as m

        mgr = m.get_rf_native_context_manager()
        monkeypatch.setattr(
            mgr,
            "start_test_in_context",
            lambda *a, **k: {"success": False, "error": "boom (forced)"},
        )
        res = await _start_test(sid, "Doomed Test")
        assert res["success"] is False
        assert "boom" in res["error"]
        session = server.execution_engine.session_manager.get_session(sid)
        # Registry never half-activated: no current test, not multi-test mode.
        assert session.test_registry.get_current_test() is None
        assert session.test_registry.is_multi_test_mode() is False


@pytest.mark.asyncio
class TestRun3Interleaving:
    async def test_steps_stay_in_test_across_interleaved_builds(self):
        # start_test -> steps -> build -> steps -> build -> steps -> end_test
        # (run 3 called build_test_suite 10x stepwise; 43 steps escaped).
        sid = "scope-interleave"
        server.execution_engine.session_manager.get_or_create_session(sid)
        res = await _start_test(sid, "Interleaved Test")
        assert res["success"] is True, res

        async def _step(msg):
            r = await server.execution_engine.execute_step(
                "Log", [msg], sid, use_context=True
            )
            assert r["success"] is True

        await _step("one")
        b1 = await get_tool_fn(server.build_test_suite)(session_id=sid, test_name="")
        assert b1["success"] is True
        await _step("two")
        b2 = await get_tool_fn(server.build_test_suite)(session_id=sid, test_name="")
        assert b2["success"] is True
        await _step("three")
        ended = await _end_test(sid)
        assert ended["success"] is True, ended

        session = server.execution_engine.session_manager.get_session(sid)
        test = session.test_registry.tests["Interleaved Test"]
        assert len(test.steps) == 3
        assert len(session.suite_level_steps) == 0
        final = await get_tool_fn(server.build_test_suite)(session_id=sid, test_name="")
        assert final["success"] is True
        assert final.get("suite_level_step_count", 0) == 0
        rf_text = final.get("rf_text") or ""
        for msg in ("one", "two", "three"):
            assert msg in rf_text


@pytest.mark.asyncio
class TestRegistryFirstEndTest:
    async def test_context_layer_miss_is_warning_not_failure(self, monkeypatch):
        sid = "scope-endfirst"
        server.execution_engine.session_manager.get_or_create_session(sid)
        res = await _start_test(sid, "End Me")
        assert res["success"] is True
        from robotmcp.components.execution import rf_native_context_manager as m

        mgr = m.get_rf_native_context_manager()
        monkeypatch.setattr(
            mgr,
            "end_test_in_context",
            lambda *a, **k: {"success": False, "error": "No active test to end"},
        )
        ended = await _end_test(sid)
        assert ended["success"] is True, ended
        assert ended["test_name"] == "End Me"
        assert "warning" in ended
        assert "No active test" in ended["warning"]

    async def test_end_without_start_fails(self):
        sid = "scope-noend"
        server.execution_engine.session_manager.get_or_create_session(sid)
        ended = await _end_test(sid)
        assert ended["success"] is False
        assert "start_test" in ended["error"]


@pytest.mark.asyncio
class TestSuiteLevelVisibility:
    async def test_orphaned_steps_warned(self):
        from robotmcp.models.execution_models import ExecutionStep
        from robotmcp.models.session_models import ExecutionSession
        from unittest.mock import MagicMock
        from robotmcp.components.test_builder import TestBuilder

        sess = ExecutionSession(session_id="scope-orphans")
        sess.test_registry.start_test("Tiny Test")
        for i in range(3):
            st = ExecutionStep(step_id=f"t{i}", keyword="Log", arguments=[f"in{i}"])
            st.mark_success()
            sess.test_registry.tests["Tiny Test"].steps.append(st)
        sess.test_registry.end_test(status="pass")
        # 43 orphaned suite-level steps (the run-3 shape).
        for i in range(43):
            st = ExecutionStep(step_id=f"s{i}", keyword="Log", arguments=[f"out{i}"])
            st.mark_success()
            sess.suite_level_steps.append(st)

        engine = MagicMock()
        engine.sessions = {"scope-orphans": sess}
        builder = TestBuilder(execution_engine=engine)
        result = await builder.build_suite(session_id="scope-orphans", test_name="")
        assert result["success"] is True
        assert result["suite_level_step_count"] == 43
        warning = result.get("warning") or ""
        assert "OUTSIDE" in warning
        assert "43" in warning

    async def test_healthy_session_silent(self):
        from robotmcp.models.execution_models import ExecutionStep
        from robotmcp.models.session_models import ExecutionSession
        from unittest.mock import MagicMock
        from robotmcp.components.test_builder import TestBuilder

        sess = ExecutionSession(session_id="scope-healthy")
        sess.test_registry.start_test("Full Test")
        for i in range(5):
            st = ExecutionStep(step_id=f"t{i}", keyword="Log", arguments=[f"m{i}"])
            st.mark_success()
            sess.test_registry.tests["Full Test"].steps.append(st)
        sess.test_registry.end_test(status="pass")

        engine = MagicMock()
        engine.sessions = {"scope-healthy": sess}
        builder = TestBuilder(execution_engine=engine)
        result = await builder.build_suite(session_id="scope-healthy", test_name="")
        assert result["success"] is True
        assert result["suite_level_step_count"] == 0
        assert result.get("warning") is None


@pytest.mark.asyncio
class TestDottedLibraryImports:
    async def test_qualified_dotted_library_imported_fully(self):
        # Run-4 standalone replay failed: 'PlatynUI.BareMetal.Take Screenshot'
        # generated 'Library  PlatynUI' (the placeholder) instead of
        # 'Library  PlatynUI.BareMetal'. The prefix is everything before the
        # LAST dot (RF keyword names contain no dots).
        from unittest.mock import MagicMock

        from robotmcp.components.test_builder import TestBuilder
        from robotmcp.models.execution_models import ExecutionStep
        from robotmcp.models.session_models import ExecutionSession

        sess = ExecutionSession(session_id="dotted-lib")
        sess.test_registry.start_test("Dotted")
        for kw, args in [
            ("Start Process", ["soffice", "--writer"]),
            ("PlatynUI.BareMetal.Take Screenshot", ["${None}", "a.png"]),
            ("PlatynUI.BareMetal.Keyboard Type", ["${None}", "hello"]),
        ]:
            st = ExecutionStep(step_id=kw, keyword=kw, arguments=args)
            st.mark_success()
            sess.test_registry.tests["Dotted"].steps.append(st)
        sess.test_registry.end_test(status="pass")

        engine = MagicMock()
        engine.sessions = {"dotted-lib": sess}
        builder = TestBuilder(execution_engine=engine)
        result = await builder.build_suite(session_id="dotted-lib", test_name="")
        assert result["success"] is True
        rf_text = result.get("rf_text") or ""
        assert "Library         PlatynUI.BareMetal" in rf_text
        # The bare placeholder import must NOT appear.
        import re
        assert not re.search(r"Library\s+PlatynUI\s*$", rf_text, re.M)
