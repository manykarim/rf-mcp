"""Unit tests for the desktop-stepwise-execution-fidelity change.

Covers the maintainer-report re-run findings (docs/gnome-calculator-mcp-
maintainer-report.md + tests/e2e/gnome_calculator_mcp_stepwise_trace.robot):

- D5 suite-recording fidelity: OBS-11 propagation guard (finding #9)
- D1/D2 desktop launch alignment (findings #2, #4)
- D6 stepwise-suite hygiene (finding #10)
- D3 desktop isolation guidance (finding #3)
- D4 driveability guidance (findings #5-#8)
- D7 classification determinism (finding #1)
"""

from __future__ import annotations

import pytest


# ── D5: OBS-11 propagation guard (finding #9) ───────────────────────


class TestObs11PropagationGuard:
    def _tb(self):
        from robotmcp.components.test_builder import TestBuilder

        return TestBuilder()

    def _step(self, **kw):
        from robotmcp.components.test_builder import TestCaseStep

        kw.setdefault("arguments", [])
        return TestCaseStep(**kw)

    def test_evaluate_sourced_capture_not_substituted(self):
        # The exact GNOME-trace corruption: Keyboard Type ${None} 1 must NOT
        # become Keyboard Type ${None} ${active_desktop_override} because the
        # override value "1" was captured by an Evaluate probe.
        tb = self._tb()
        cap = self._step(
            keyword="Evaluate",
            arguments=["__import__('os')..."],
            assigned_variables=["active_desktop_override"],
            captured_value="1",
        )
        typ = self._step(keyword="Keyboard Type", arguments=["${None}", "1"])
        steps = [cap, typ]
        tb._propagate_assigned_variables_to_literal_args(steps)
        assert steps[1].arguments[1] == "1"

    def test_legit_single_char_data_dependency_still_propagates(self):
        # A real single-char data capture from a non-introspection keyword is
        # still propagated (the discriminator is the SOURCE, not value length).
        tb = self._tb()
        cap = self._step(
            keyword="Get Element Count",
            arguments=["css=.option"],
            assigned_variables=["COUNT"],
            captured_value="5",
        )
        use = self._step(keyword="Fill Text", arguments=["id=entryCount", "5"])
        steps = [cap, use]
        tb._propagate_assigned_variables_to_literal_args(steps)
        assert steps[1].arguments[1] == "COUNT"

    def test_multi_char_dependency_still_propagates(self):
        tb = self._tb()
        cap = self._step(
            keyword="Get Element Count",
            arguments=["css=.option"],
            assigned_variables=["ENTRY_COUNT"],
            captured_value="42",
        )
        use = self._step(keyword="Fill Text", arguments=["id=entryCount", "42"])
        steps = [cap, use]
        tb._propagate_assigned_variables_to_literal_args(steps)
        # OBS-11 benefit preserved for a real multi-char data dependency.
        assert steps[1].arguments[1] == "ENTRY_COUNT"

    def test_introspection_source_does_not_propagate(self):
        tb = self._tb()
        cap = self._step(
            keyword="Evaluate",
            arguments=["[n.name for n in $nodes]"],
            assigned_variables=["probe"],
            captured_value="hello",
        )
        use = self._step(keyword="Log", arguments=["msg", "hello"])
        steps = [cap, use]
        tb._propagate_assigned_variables_to_literal_args(steps)
        assert steps[1].arguments[1] == "hello"

    def test_query_source_does_not_propagate(self):
        tb = self._tb()
        cap = self._step(
            keyword="Query",
            arguments=["//control:Text"],
            assigned_variables=["nodes"],
            captured_value="something",
        )
        use = self._step(keyword="Log", arguments=["msg", "something"])
        steps = [cap, use]
        tb._propagate_assigned_variables_to_literal_args(steps)
        assert steps[1].arguments[1] == "something"

    def test_is_introspection_source_helper(self):
        from robotmcp.components.test_builder import TestBuilder

        assert TestBuilder._is_introspection_source("Evaluate") is True
        assert TestBuilder._is_introspection_source("PlatynUI.BareMetal.Query") is True
        assert TestBuilder._is_introspection_source("Get Text") is False
        assert TestBuilder._is_introspection_source(None) is False


# ── D1/D2: desktop launch alignment (findings #2, #4) ───────────────


class TestDesktopLaunchSanitizeWiring:
    def _executor(self):
        from robotmcp.components.execution.keyword_executor import KeywordExecutor

        return KeywordExecutor.__new__(KeywordExecutor)

    def _desktop_session(self, **attrs):
        class _S:
            def is_desktop_session(self):
                return True

        s = _S()
        for k, v in attrs.items():
            setattr(s, k, v)
        return s

    def test_sanitize_called_for_gui_start_process(self, monkeypatch):
        # The sanitizer must actually be invoked for a desktop GUI Start Process
        # (finding #2 — it was dead code). We assert it is called and its return
        # value is used.
        ke = self._executor()
        called = {}

        def _fake_sanitize(session, keyword, arguments):
            called["yes"] = (keyword, list(arguments))
            return list(arguments) + ["env:LD_LIBRARY_PATH=/clean"]

        monkeypatch.setattr(ke, "_maybe_sanitize_desktop_launch", _fake_sanitize, raising=False)
        out = ke._maybe_sanitize_desktop_launch(
            self._desktop_session(), "Start Process", ["/usr/bin/gnome-calculator"]
        )
        assert called.get("yes")
        assert out[-1] == "env:LD_LIBRARY_PATH=/clean"

    def test_real_sanitizer_known_gui_binary(self):
        # The real sanitizer recognizes a known GUI binary for a desktop session.
        from robotmcp.components.execution.desktop_launch_env import is_desktop_gui_launch

        assert is_desktop_gui_launch(["/usr/bin/gnome-calculator"]) == "gnome-calculator"
        assert is_desktop_gui_launch(["/usr/bin/some-random-tool"]) is None

    def test_real_sanitizer_wires_both_start_and_run_process(self):
        # Both Start Process AND Run Process GUI launches must be sanitized
        # (Kilo MiniMax-M3 review: Run Process branch was untested even though
        # the spec/tasks name it). The real method gates on the keyword name.
        ke = self._executor()
        sess = self._desktop_session()
        for kw in ("Start Process", "Run Process", "Process.Run Process"):
            out = ke._maybe_sanitize_desktop_launch(
                sess, kw, ["/usr/bin/gnome-calculator"]
            )
            assert len(out) > 1, f"{kw} should get env: overrides"
            assert any(str(a).startswith("env:") for a in out), kw

    def test_real_sanitizer_skips_non_process_keyword(self):
        ke = self._executor()
        sess = self._desktop_session()
        out = ke._maybe_sanitize_desktop_launch(
            sess, "Pointer Click", ["/usr/bin/gnome-calculator"]
        )
        assert out == ["/usr/bin/gnome-calculator"]


class TestDesktopExecutionSignals:
    def test_launch_liveness_dead_process_with_nodes(self):
        from robotmcp.components.execution.desktop_execution_signals import (
            launch_liveness_hint,
        )

        hint = launch_liveness_hint(process_running=False, discovery_node_count=3)
        assert hint is not None
        assert hint["type"] == "desktop_launch_not_running"
        assert "3 application node" in hint["message"]

    def test_launch_liveness_running_no_hint(self):
        from robotmcp.components.execution.desktop_execution_signals import (
            launch_liveness_hint,
        )

        assert launch_liveness_hint(process_running=True, discovery_node_count=3) is None

    def test_launch_liveness_unknown_no_hint(self):
        from robotmcp.components.execution.desktop_execution_signals import (
            launch_liveness_hint,
        )

        assert launch_liveness_hint(process_running=None, discovery_node_count=3) is None

    def test_input_effect_no_change_warns(self):
        from robotmcp.components.execution.desktop_execution_signals import (
            input_effect_hint,
        )

        hint = input_effect_hint(
            keyword="Pointer Click", success=True, state_before=0, state_after=0
        )
        assert hint is not None
        assert hint["type"] == "desktop_input_no_effect"

    def test_input_effect_changed_no_warn(self):
        from robotmcp.components.execution.desktop_execution_signals import (
            input_effect_hint,
        )

        assert (
            input_effect_hint(
                keyword="Keyboard Type", success=True, state_before=0, state_after=1
            )
            is None
        )

    def test_input_effect_missing_snapshot_no_warn(self):
        from robotmcp.components.execution.desktop_execution_signals import (
            input_effect_hint,
        )

        assert (
            input_effect_hint(
                keyword="Pointer Click", success=True, state_before=None, state_after=0
            )
            is None
        )

    def test_input_effect_non_interaction_no_warn(self):
        from robotmcp.components.execution.desktop_execution_signals import (
            input_effect_hint,
        )

        assert (
            input_effect_hint(
                keyword="Get Attribute", success=True, state_before=0, state_after=0
            )
            is None
        )


# ── D6: stepwise-suite hygiene (finding #10) ────────────────────────


class TestStepwiseSuiteHygiene:
    def _tb(self):
        from robotmcp.components.test_builder import TestBuilder

        return TestBuilder()

    def _suite(self, steps):
        from robotmcp.components.test_builder import (
            GeneratedTestCase,
            GeneratedTestSuite,
        )

        return GeneratedTestSuite(
            name="S", test_cases=[GeneratedTestCase(name="T", steps=steps)]
        )

    def _step(self, keyword, arguments=None, assigned=None):
        from robotmcp.components.test_builder import TestCaseStep

        return TestCaseStep(
            keyword=keyword,
            arguments=arguments or [],
            assigned_variables=assigned or [],
        )

    def test_exploratory_probes_filtered(self):
        tb = self._tb()
        steps = [
            self._step("Start Process", ["/usr/bin/gnome-calculator"], ["calc_handle"]),
            self._step("Query", ["//app:*[@Name='gnome-calculator']"], ["calc_app_nodes"]),
            self._step("Evaluate", ["[n.name for n in $text_nodes]"], ["text_name_role_pairs"]),
            self._step("Pointer Click", ["//control:Button[@Name='1']"]),
            self._step("Should Be Equal As Strings", ["${result}", "1"]),
        ]
        suite = self._suite(steps)
        removed = tb._filter_exploratory_introspection(suite)
        kept = [s.keyword for s in suite.test_cases[0].steps]
        assert removed == 2  # the unused Query + Evaluate probes
        assert "Query" not in kept
        assert "Evaluate" not in kept
        assert "Pointer Click" in kept
        assert "Should Be Equal As Strings" in kept

    def test_load_bearing_capture_retained_rf_var_format(self):
        tb = self._tb()
        # assigned_variables are stored in ${VAR} form by real execution
        # (_normalize_variable_name). The fixpoint must normalize before
        # comparing or a load-bearing Query is wrongly dropped (Codex review #1).
        steps = [
            self._step(
                "Query",
                ["//control:Label[@Name='56']", "${None}", "True"],
                ["${result_label}"],  # RF-format assigned var (real data shape)
            ),
            self._step("Should Be Equal As Strings", ["${result_label}", "56"]),
        ]
        suite = self._suite(steps)
        removed = tb._filter_exploratory_introspection(suite)
        kept = [s.keyword for s in suite.test_cases[0].steps]
        assert removed == 0, "load-bearing Query must survive with ${VAR}-format assignment"
        assert "Query" in kept

    def test_standalone_unassigned_probe_not_erased(self):
        tb = self._tb()
        # A standalone Query/Evaluate with NO assignment may be a side-effect /
        # existence assertion — it must not be silently erased (Codex review #2).
        steps = [
            self._step("Query", ["//app:*[@Name='gnome-calculator']"]),  # no assign
            self._step("Evaluate", ["1 + 1 == 2"]),  # no assign
            self._step("Pointer Click", ["//control:Button[@Name='1']"]),
        ]
        suite = self._suite(steps)
        removed = tb._filter_exploratory_introspection(suite)
        kept = [s.keyword for s in suite.test_cases[0].steps]
        assert removed == 0
        assert kept == ["Query", "Evaluate", "Pointer Click"]

    def test_bare_var_name_helper(self):
        from robotmcp.components.test_builder import TestBuilder

        assert TestBuilder._bare_var_name("${result}") == "result"
        assert TestBuilder._bare_var_name("$result") == "result"
        assert TestBuilder._bare_var_name("result") == "result"

    def test_transitive_dependency_retained(self):
        tb = self._tb()
        # Evaluate B consumes Query A's var; a retained Set Root consumes B's var.
        steps = [
            self._step("Query", ["//app:*"], ["calc_app_nodes"]),
            self._step("Evaluate", ["$calc_app_nodes[2]"], ["calc_app_target"]),
            self._step("Set Root", ["${calc_app_target}"]),
            self._step("Pointer Click", ["//control:Button[@Name='1']"]),
        ]
        suite = self._suite(steps)
        removed = tb._filter_exploratory_introspection(suite)
        kept = [s.keyword for s in suite.test_cases[0].steps]
        assert removed == 0  # both probes are transitively load-bearing
        assert kept == ["Query", "Evaluate", "Set Root", "Pointer Click"]

    def test_referenced_var_names_covers_both_forms(self):
        from robotmcp.components.test_builder import TestBuilder

        names = TestBuilder._referenced_var_names(["${FOO}", "[n for n in $bar]", "plain"])
        assert "FOO" in names
        assert "bar" in names


@pytest.mark.asyncio
class TestSuiteHygieneIntegration:
    async def test_desktop_build_suite_filters_and_reports(self):
        from unittest.mock import MagicMock
        from robotmcp.components.test_builder import TestBuilder
        from robotmcp.models.execution_models import ExecutionStep
        from robotmcp.models.session_models import ExecutionSession

        sess = ExecutionSession(session_id="hyg1")
        sess.configure_from_scenario(
            "Open GNOME Calculator desktop application and verify results",
            context="desktop",
        )
        sess.test_registry.start_test("Calc Test", tags=["desktop"])
        # The exploratory probes assign variables (as the real trace does) that
        # nothing downstream consumes → they are the noise to be filtered.
        for kw, args, assigned in [
            ("Start Process", ["/usr/bin/gnome-calculator"], []),
            ("Query", ["//app:*[@Name='gnome-calculator']"], ["${calc_app_nodes}"]),
            ("Evaluate", ["[n.name for n in $nodes]"], ["${probe_names}"]),
            ("Pointer Click", ["//control:Button[@Name='1']"], []),
        ]:
            st = ExecutionStep(step_id=kw, keyword=kw, arguments=args)
            st.mark_success()
            st.assigned_variables = assigned
            sess.test_registry.tests["Calc Test"].steps.append(st)
        sess.test_registry.end_test(status="pass")

        engine = MagicMock()
        engine.sessions = {"hyg1": sess}
        builder = TestBuilder(execution_engine=engine)
        result = await builder.build_suite(session_id="hyg1", test_name="Calc Suite")

        assert result["success"] is True
        assert result["introspection_filtered_count"] == 2
        assert result["introspection_summary"]
        rf_text = result.get("rf_text") or ""
        assert "Pointer Click" in rf_text
        # The exploratory introspection probes are gone from the body.
        assert "[n.name for n in" not in rf_text

    async def test_load_bearing_evaluate_survives_build_suite_real_var_form(self):
        # Regression for the cross-LLM review finding: a load-bearing Evaluate
        # capturing ${greeting} consumed by a later Pointer Click must survive
        # build_suite — production stores assigned vars in ${VAR} form, so the
        # generated suite must not reference an undefined variable.
        from unittest.mock import MagicMock
        from robotmcp.components.test_builder import TestBuilder
        from robotmcp.models.execution_models import ExecutionStep
        from robotmcp.models.session_models import ExecutionSession

        sess = ExecutionSession(session_id="lb1")
        sess.configure_from_scenario("Desktop calculator scenario", context="desktop")
        sess.test_registry.start_test("T1")
        ev = ExecutionStep(step_id="e1", keyword="Evaluate", arguments=["nodes[0].name"])
        ev.assigned_variables = ["${greeting}"]  # production ${VAR} form
        ev.mark_success()
        sess.test_registry.tests["T1"].steps.append(ev)
        cl = ExecutionStep(
            step_id="c1",
            keyword="Pointer Click",
            arguments=['//control:Button[@Name="${greeting}"]'],
        )
        cl.mark_success()
        sess.test_registry.tests["T1"].steps.append(cl)
        sess.test_registry.end_test(status="pass")

        engine = MagicMock()
        engine.sessions = {"lb1": sess}
        result = await TestBuilder(execution_engine=engine).build_suite(
            session_id="lb1", test_name="T1"
        )
        assert result["introspection_filtered_count"] == 0
        rf_text = result.get("rf_text") or ""
        # The Evaluate that defines ${greeting} must remain so the suite compiles.
        assert "Evaluate" in rf_text

    async def test_web_build_suite_not_filtered(self):
        # A non-desktop session must NOT have its Evaluate steps filtered.
        from unittest.mock import MagicMock
        from robotmcp.components.test_builder import TestBuilder
        from robotmcp.models.execution_models import ExecutionStep
        from robotmcp.models.session_models import ExecutionSession

        sess = ExecutionSession(session_id="web1")
        sess.configure_from_scenario("Open the website in a browser")
        sess.test_registry.start_test("Web Test")
        for kw, args, assigned in [
            ("Evaluate", ["1 + 1"], ["sum"]),
            ("Log", ["${sum}"], []),
        ]:
            st = ExecutionStep(step_id=kw, keyword=kw, arguments=args)
            st.mark_success()
            st.assigned_variables = assigned
            sess.test_registry.tests["Web Test"].steps.append(st)
        sess.test_registry.end_test(status="pass")

        engine = MagicMock()
        engine.sessions = {"web1": sess}
        builder = TestBuilder(execution_engine=engine)
        result = await builder.build_suite(session_id="web1", test_name="Web Suite")
        assert result["introspection_filtered_count"] == 0


# ── D3: active-desktop refusal returns an isolation recipe ──────────


class TestDesktopIsolationGuidance:
    @pytest.fixture(autouse=True)
    def _force_posix(self, monkeypatch):
        # Windows-CI Linux-model guard: these tests validate the Linux/X11
        # active-desktop refusal + Xephyr isolation recipe. On a Windows host,
        # classify_bound_display short-circuits to the WINDOWS branch (F4,
        # change fix-platynui-windows-runtime), which intentionally has no
        # Xephyr recipe. Force the non-Windows path so the Linux-model
        # assertions run on ANY host (Windows CI included).
        from robotmcp.components.execution import desktop_display_safety as dds

        monkeypatch.setattr(dds, "_is_windows", lambda: False)

    def _refuse_outcome(self):
        from robotmcp.components.execution.desktop_display_safety import evaluate_safety

        # An empty env => no isolation marker => classified non-isolated =>
        # refuse (no opt-in, no warn-mode).
        class _S:
            platynui_allow_active_desktop = False

        return evaluate_safety(_S(), environ={})

    def test_refusal_includes_isolation_recipe(self):
        outcome = self._refuse_outcome()
        assert outcome["allowed"] is False
        recipe = outcome.get("isolation_recipe")
        assert recipe is not None
        assert recipe["marker_env"] == "ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY"
        joined = " ".join(recipe["steps"]).lower()
        assert "xvfb" in joined
        assert recipe["marker_env"] in " ".join(recipe["steps"])

    def test_bypass_framed_as_escape_hatch(self):
        outcome = self._refuse_outcome()
        recipe = outcome["isolation_recipe"]
        assert recipe["bypass_env"] == "ROBOTMCP_PLATYNUI_ALLOW_ACTIVE_DESKTOP"
        assert "escape hatch" in recipe["bypass_note"].lower()
        # The reason itself no longer presents the bypass as a plain "override".
        assert "escape hatch" in outcome["reason"].lower()

    def test_build_isolation_recipe_shape(self):
        from robotmcp.components.execution.desktop_display_safety import (
            build_isolation_recipe,
        )

        r = build_isolation_recipe()
        assert set(r) >= {"summary", "steps", "marker_env", "bypass_env", "bypass_note"}
        assert isinstance(r["steps"], list) and r["steps"]


# ── D4: desktop driveability guidance (findings #5-#8) ──────────────


class TestDriveabilityGuidance:
    def _guidance(self):
        from robotmcp.utils.rf_native_type_converter import RobotFrameworkNativeConverter

        return RobotFrameworkNativeConverter().get_platynui_locator_guidance()

    def test_node_attribute_api_documented(self):
        g = self._guidance()
        assert "node_attribute_api" in g
        ns = g["node_attribute_api"]
        joined = " ".join(ns["supported"]) + " " + " ".join(ns["not_supported"])
        assert "attribute(" in joined
        assert "get_attribute" in " ".join(ns["not_supported"])
        assert "Get Attribute" in ns["prefer_keyword"]

    def test_duplicate_roots_documented(self):
        g = self._guidance()
        assert "duplicate_roots_and_controls" in g
        rules = " ".join(g["duplicate_roots_and_controls"]["rules"]).lower()
        assert "set root" in rules
        assert "interactable" in rules or "in-view" in rules

    def test_control_naming_documented(self):
        g = self._guidance()
        assert "control_naming" in g
        rules = " ".join(g["control_naming"]["rules"])
        assert "Button[@Name='+']" in rules
        assert "Label[@Name='plus']" in rules

    def test_display_state_reading_documented(self):
        g = self._guidance()
        assert "display_state_reading" in g
        rules = " ".join(g["display_state_reading"]["rules"])
        assert "native:Text.CharacterCount" in rules


# ── D7: explicit desktop context classifies deterministically ───────


@pytest.mark.asyncio
class TestDesktopClassificationDeterminism:
    PHRASINGS = [
        "Open GNOME Calculator desktop application and assert the result",
        "Automate the calculator app on my Linux desktop, verify each value",
        "Use the gnome-calculator application to compute 7 x 8 and check it",
        "Drive a native desktop GUI: launch Calculator, perform additions",
        "Calculator application: enter digits, press equals, assert the total",
    ]

    async def test_all_phrasings_classify_desktop(self):
        from robotmcp.components.nlp_processor import NaturalLanguageProcessor

        nlp = NaturalLanguageProcessor()
        for scenario in self.PHRASINGS:
            result = await nlp.analyze_scenario(scenario, context="desktop")
            assert (
                result["analysis"].get("detected_session_type") == "desktop_testing"
            ), scenario

    async def test_explicit_desktop_overrides_mobile_wording(self):
        # Even mobile-leaning wording is forced to desktop under explicit context.
        from robotmcp.components.nlp_processor import NaturalLanguageProcessor

        nlp = NaturalLanguageProcessor()
        result = await nlp.analyze_scenario(
            "Tap the app buttons and swipe", context="desktop"
        )
        assert result["analysis"].get("detected_session_type") == "desktop_testing"
