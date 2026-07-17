"""Stepwise suite feedback (change: platynui-visible-safe-targeting, I-1).

A stepwise desktop session can execute 20 steps of which 19 fail; only
successful + gate-passing steps are recorded, so the generated suite is
silently empty (LibreOffice validation run, 2026-06-11). These tests pin:

- ExecutionSession executed/failed counters + recorded_step_count()
- build_test_suite statistics: steps_executed / steps_recorded / steps_failed
- the top-level empty/near-empty suite warning (fires on the LibreOffice
  regression shape, silent on healthy sessions)
- TestBuilder._suite_body_is_launch_only classification
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from robotmcp.components.test_builder import TestBuilder
from robotmcp.models.execution_models import ExecutionStep
from robotmcp.models.session_models import ExecutionSession


def _add_recorded_step(sess, keyword, arguments=None, test_name=None):
    st = ExecutionStep(step_id=keyword, keyword=keyword, arguments=arguments or [])
    st.mark_success()
    if test_name:
        sess.test_registry.tests[test_name].steps.append(st)
    else:
        sess.steps.append(st)
    return st


class TestSessionStepAccounting:
    def test_counters_default_to_zero(self):
        sess = ExecutionSession(session_id="acct0")
        assert sess.executed_step_count == 0
        assert sess.failed_step_count == 0
        assert sess.recorded_step_count() == 0

    def test_recorded_step_count_sums_all_storage_modes(self):
        sess = ExecutionSession(session_id="acct1")
        # Legacy flat list
        _add_recorded_step(sess, "Click")
        # Suite-level step
        sl = ExecutionStep(step_id="s", keyword="Log", arguments=["x"])
        sl.mark_success()
        sess.suite_level_steps.append(sl)
        # Registry test steps
        sess.test_registry.start_test("T1")
        _add_recorded_step(sess, "Fill Text", test_name="T1")
        _add_recorded_step(sess, "Pointer Click", test_name="T1")
        assert sess.recorded_step_count() == 4

    def test_counters_track_divergence(self):
        # 5 executed, 2 failed, at most 3 recorded (spec scenario).
        sess = ExecutionSession(session_id="acct2")
        sess.executed_step_count = 5
        sess.failed_step_count = 2
        _add_recorded_step(sess, "Click")
        _add_recorded_step(sess, "Fill Text")
        assert sess.executed_step_count == 5
        assert sess.failed_step_count == 2
        assert sess.recorded_step_count() <= 3


class TestSuiteBodyLaunchOnly:
    def setup_method(self):
        self.builder = TestBuilder()

    def _suite_with(self, keywords):
        tc = MagicMock()
        tc.steps = [
            MagicMock(keyword=kw) for kw in keywords
        ]
        suite = MagicMock()
        suite.test_cases = [tc]
        return suite

    def test_only_start_process_is_launch_only(self):
        assert self.builder._suite_body_is_launch_only(
            self._suite_with(["Start Process"])
        ) is True

    def test_library_prefixed_launch_keyword(self):
        assert self.builder._suite_body_is_launch_only(
            self._suite_with(["Process.Start Process", "Sleep"])
        ) is True

    def test_interaction_step_is_substance(self):
        assert self.builder._suite_body_is_launch_only(
            self._suite_with(["Start Process", "Pointer Click"])
        ) is False

    def test_empty_suite_counts_as_launch_only(self):
        suite = MagicMock()
        suite.test_cases = []
        assert self.builder._suite_body_is_launch_only(suite) is True

    def test_template_data_row_counts_as_substance(self):
        # Empty-keyword steps are template data rows — real content.
        assert self.builder._suite_body_is_launch_only(
            self._suite_with([""])
        ) is False


def _desktop_session(session_id, *, executed, failed, recorded_keywords):
    sess = ExecutionSession(session_id=session_id)
    sess.configure_from_scenario(
        "Open LibreOffice Writer desktop application and edit a document",
        context="desktop",
    )
    sess.executed_step_count = executed
    sess.failed_step_count = failed
    sess.test_registry.start_test("Writer Test", tags=["desktop"])
    for kw, args in recorded_keywords:
        st = ExecutionStep(step_id=kw, keyword=kw, arguments=args)
        st.mark_success()
        sess.test_registry.tests["Writer Test"].steps.append(st)
    sess.test_registry.end_test(status="pass")
    return sess


@pytest.mark.asyncio
class TestBuildSuiteFeedback:
    async def _build(self, sess):
        engine = MagicMock()
        engine.sessions = {sess.session_id: sess}
        builder = TestBuilder(execution_engine=engine)
        return await builder.build_suite(
            session_id=sess.session_id, test_name="Writer Suite"
        )

    async def test_libreoffice_regression_shape_warns(self):
        # 20 executed, 19 failed, only Start Process recorded → warning names
        # the counts and states that failed steps are never recorded.
        sess = _desktop_session(
            "lo1",
            executed=20,
            failed=19,
            recorded_keywords=[("Start Process", ["soffice", "--writer"])],
        )
        result = await self._build(sess)
        assert result["success"] is True
        warning = result.get("warning")
        assert warning, "empty-suite warning must fire"
        assert "20" in warning
        assert "failed steps are never recorded" in warning
        assert result["statistics"]["steps_executed"] == 20
        assert result["statistics"]["steps_failed"] == 19
        assert result["statistics"]["steps_recorded"] == 1

    async def test_healthy_session_does_not_warn(self):
        sess = _desktop_session(
            "lo2",
            executed=10,
            failed=0,
            recorded_keywords=[
                ("Start Process", ["soffice", "--writer"]),
                ("Pointer Click", ["//control:Button[@Name='OK']"]),
                ("Keyboard Type", ["hello"]),
            ],
        )
        result = await self._build(sess)
        assert result["success"] is True
        assert result.get("warning") is None
        assert result["statistics"]["steps_executed"] == 10
        assert result["statistics"]["steps_failed"] == 0

    async def test_no_warning_below_executed_threshold(self):
        # 2 executed (< 3) — exploration-only session shape stays silent.
        sess = _desktop_session(
            "lo3",
            executed=2,
            failed=1,
            recorded_keywords=[("Start Process", ["soffice"])],
        )
        result = await self._build(sess)
        assert result["success"] is True
        assert result.get("warning") is None

    async def test_libreoffice_run2_shape_all_signals_fire(self):
        """Task 6.3: the full Run-2 transcript pattern (LibreOffice
        validation, 2026-06-11) now produces all three signals that were
        silent then: the empty-suite warning, the focus-unverifiable
        warning, and the Process `=`-argument hint."""
        from robotmcp.components.execution.platynui_focus import (
            FOCUS_UNVERIFIABLE_PREFIX,
            PlatynUIFocusManager,
        )
        from robotmcp.utils.hints import detect_process_eq_arg_misparse

        # 1) Suite feedback: 20 executed / 19 failed / launch-only body.
        launch_args = [
            "soffice", "--writer",
            "-env:UserInstallation=file:///tmp/lo-profile",
        ]
        sess = _desktop_session(
            "run2",
            executed=20,
            failed=19,
            recorded_keywords=[("Start Process", launch_args)],
        )
        result = await self._build(sess)
        assert result.get("warning")

        # 2) Process arg hint: the LibreOffice profile arg is flagged.
        flagged = detect_process_eq_arg_misparse("Start Process", launch_args)
        assert flagged == ["-env:UserInstallation=file:///tmp/lo-profile"]

        # 3) Focus verifiability: a Writer-like frame (no WindowSurface /
        # Focusable pattern, no working focus path) warns explicitly.
        class WriterFrame:
            runtime_id = "soffice-frame"

            def supported_patterns(self):
                return ["org.platynui.patterns.Element"]

            def attribute(self, name):
                raise KeyError(name)

            def top_level_or_self(self):
                return self

        class RT:
            def evaluate_single(self, descriptor):
                return WriterFrame()

            def desktop_info(self):
                return {}

            def clear_cache(self):
                pass

        mgr = PlatynUIFocusManager()
        mgr._runtime = RT()
        mgr._desktop_bounds = lambda: None
        mgr._window_surface = staticmethod(lambda w: None)
        mgr._x11_raise = lambda w: False
        oc = mgr.ensure_focused(
            "Keyboard Type", ["/app:*[@Name='soffice']//control:Frame", "text"]
        )
        expected = f"{FOCUS_UNVERIFIABLE_PREFIX} (no WindowSurface/Focusable pattern)"
        assert expected in oc.warnings

    async def test_substantive_suite_with_failures_does_not_warn(self):
        # Failures occurred but the suite has real content → no warning.
        sess = _desktop_session(
            "lo4",
            executed=8,
            failed=2,
            recorded_keywords=[
                ("Start Process", ["soffice"]),
                ("Pointer Click", ["//control:Button[@Name='OK']"]),
            ],
        )
        result = await self._build(sess)
        assert result["success"] is True
        assert result.get("warning") is None
