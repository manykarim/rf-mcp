"""OBS-11 — captured-variable propagation into subsequent literal args.

The 2026-05-17 post-OBS validation benchmark surfaced a generated-suite
fidelity gap on Obstacle 7 (And counting). Sonnet's call sequence:

    intent_action(intent="extract", mode="count",
                  target="css=.select2-results__option",
                  assign_to="ENTRY_COUNT")        # captured ${ENTRY_COUNT}=5
    execute_step(Fill Text, ["id=entryCount", "5"])

Generated suite output (BEFORE OBS-11):

    ${ENTRY_COUNT} =    Get Element Count    css=.select2-results__option
    Fill Text           id=entryCount        5     ← literal, not ${ENTRY_COUNT}

The suite isn't truly dynamic — it would silently mis-fill on a page
where the search string returns a different count. OBS-11 closes the
gap by post-processing each test case's steps before rendering and
rewriting literal args that match a recently captured variable's
value into ``${VAR}`` references.

These tests pin:
(1) Basic propagation: capture in step N → literal in step N+1 rewritten
(2) Captured-value comparison is exact-match (no substring false-positives)
(3) Arg 0 (locator slot) is left untouched to avoid false matches in
    locator strings
(4) Lookback window: rewrites only within ~10 steps of capture
(5) Already-substituted ``${VAR}`` references in literal slots are
    left alone (no double-wrap)
(6) Multiple captures: most recent match wins
(7) Empty / None captured values produce no substitutions
(8) Cross-test-case isolation: a capture in test case A doesn't
    bleed into test case B's literals
(9) End-to-end: build_test_suite's rf_text output shows the
    substituted reference (the headline acceptance from the story)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from robotmcp.components.test_builder import (
    GeneratedTestCase,
    GeneratedTestSuite,
    TestBuilder,
    TestCaseStep,
)


# ---------------------------------------------------------------------------
# Layer 1: the helper's transform behaviour
# ---------------------------------------------------------------------------


def _builder() -> TestBuilder:
    """A minimal TestBuilder for invoking the propagation helper —
    the helper doesn't touch the execution engine."""
    return TestBuilder(execution_engine=None)


def _capture_step(var: str, value: str, kw: str = "Get Element Count",
                  arg: str = "css=.x") -> TestCaseStep:
    return TestCaseStep(
        keyword=kw,
        arguments=[arg],
        assigned_variables=[var],
        assignment_type="single",
        captured_value=value,
    )


class TestBasicPropagation:
    """Capture in step N → literal in step N+1 rewritten."""

    def test_captures_and_rewrites_in_next_step(self):
        steps = [
            _capture_step("${COUNT}", "5"),
            TestCaseStep(keyword="Fill Text", arguments=["id=foo", "5"]),
        ]
        _builder()._propagate_assigned_variables_to_literal_args(steps)
        assert steps[1].arguments == ["id=foo", "${COUNT}"]

    def test_capture_step_itself_is_not_self_substituted(self):
        # The capturing step's own args (e.g. the locator at arg 0,
        # or any other args) must NOT be rewritten — a self-
        # substitution would corrupt the keyword call.
        steps = [
            _capture_step("${COUNT}", "5", arg="5"),  # locator literally "5"
        ]
        _builder()._propagate_assigned_variables_to_literal_args(steps)
        # The captured step's locator arg stays as "5".
        assert steps[0].arguments == ["5"]

    def test_arg0_is_skipped_in_subsequent_steps(self):
        # Arg 0 is the locator slot by convention. A literal matching
        # a captured value MUST NOT be rewritten there — locator
        # strings that happen to equal a captured value are
        # coincidental, not intentional references. Both arg slots
        # carry the matching value to prove the asymmetry: only the
        # later positions are rewritten.
        steps = [
            _capture_step("${X}", "5"),
            TestCaseStep(keyword="Click", arguments=["5", "5"]),
        ]
        _builder()._propagate_assigned_variables_to_literal_args(steps)
        # arg[0] preserved (locator slot rule); arg[1] rewritten.
        assert steps[1].arguments == ["5", "${X}"]


class TestExactMatchSemantics:
    """Substitution requires exact equality between captured value and
    the literal argument. Substring / containment matches must NOT
    trigger rewrites."""

    @pytest.mark.parametrize("literal,captured,should_rewrite", [
        ("5", "5", True),                          # exact match
        ("ORD-1007696", "ORD-1007696", True),      # exact match
        ("HELLO", "hello", False),                 # case-sensitive
        ("5 items", "5", False),                   # substring NOT match
        ("ID-5", "5", False),                      # substring NOT match
        (" 5", "5", False),                        # whitespace NOT match
        ("5 ", "5", False),                        # trailing space
    ])
    def test_exact_match_required(self, literal, captured, should_rewrite):
        steps = [
            _capture_step("${X}", captured),
            TestCaseStep(keyword="Log", arguments=["msg", literal]),
        ]
        _builder()._propagate_assigned_variables_to_literal_args(steps)
        if should_rewrite:
            assert steps[1].arguments == ["msg", "${X}"]
        else:
            assert steps[1].arguments == ["msg", literal]


class TestLookbackWindow:
    """Substitution is bounded to the lookback window after capture."""

    def test_substitutes_within_lookback(self):
        steps: list[TestCaseStep] = [_capture_step("${X}", "5")]
        # Filler steps with non-matching literals — these don't reset
        # the active capture but consume lookback steps.
        for i in range(8):
            steps.append(TestCaseStep(
                keyword="Log", arguments=["msg", f"filler-{i}"],
            ))
        # Step 9 (8 filler + capture) — within 10-step lookback.
        steps.append(
            TestCaseStep(keyword="Log", arguments=["msg", "5"]),
        )
        _builder()._propagate_assigned_variables_to_literal_args(steps)
        assert steps[-1].arguments == ["msg", "${X}"]

    def test_does_not_substitute_beyond_lookback(self):
        steps: list[TestCaseStep] = [_capture_step("${X}", "5")]
        # Fill out 11 steps after the capture so the literal is
        # past the default 10-step lookback.
        for i in range(11):
            steps.append(TestCaseStep(
                keyword="Log", arguments=["msg", f"filler-{i}"],
            ))
        steps.append(
            TestCaseStep(keyword="Log", arguments=["msg", "5"]),
        )
        _builder()._propagate_assigned_variables_to_literal_args(steps)
        # The trailing "5" is 12 steps after the capture — outside
        # the 10-step window. Must NOT be substituted.
        assert steps[-1].arguments == ["msg", "5"]


class TestExistingVariableReferences:
    """Args that already contain a ``${VAR}`` reference must NOT be
    re-wrapped or otherwise transformed."""

    def test_existing_variable_reference_preserved(self):
        steps = [
            _capture_step("${X}", "5"),
            TestCaseStep(
                keyword="Fill Text",
                arguments=["id=foo", "${OTHER}"],
            ),
        ]
        _builder()._propagate_assigned_variables_to_literal_args(steps)
        assert steps[1].arguments == ["id=foo", "${OTHER}"]

    def test_embedded_variable_reference_preserved(self):
        # Embedded refs in larger literals also preserve the original.
        steps = [
            _capture_step("${X}", "5"),
            TestCaseStep(
                keyword="Log",
                arguments=["msg", "Hello ${USER}"],
            ),
        ]
        _builder()._propagate_assigned_variables_to_literal_args(steps)
        assert steps[1].arguments == ["msg", "Hello ${USER}"]


class TestMultipleCaptures:
    """When multiple captures are active, the most-recent matching
    capture wins. This matches the intuition that recent captures are
    semantically closer to the literal usage."""

    def test_most_recent_capture_wins(self):
        steps = [
            _capture_step("${X}", "5"),
            _capture_step("${Y}", "5"),       # same value, newer
            TestCaseStep(keyword="Log", arguments=["msg", "5"]),
        ]
        _builder()._propagate_assigned_variables_to_literal_args(steps)
        # Most-recent capture (${Y}) wins.
        assert steps[2].arguments == ["msg", "${Y}"]

    def test_different_values_route_to_correct_vars(self):
        steps = [
            _capture_step("${COUNT}", "5"),
            _capture_step("${ID}", "ORD-1"),
            TestCaseStep(keyword="Log", arguments=["msg", "5"]),
            TestCaseStep(keyword="Log", arguments=["msg", "ORD-1"]),
        ]
        _builder()._propagate_assigned_variables_to_literal_args(steps)
        assert steps[2].arguments == ["msg", "${COUNT}"]
        assert steps[3].arguments == ["msg", "${ID}"]


class TestNoCaptureSemantics:
    """Steps without captures don't pollute the active-capture list,
    and captured-value=None / empty is treated as 'no capture'."""

    def test_none_captured_value_does_not_register(self):
        steps = [
            TestCaseStep(
                keyword="Get Text",
                arguments=["id=foo"],
                assigned_variables=["${X}"],
                captured_value=None,
            ),
            TestCaseStep(keyword="Log", arguments=["msg", "5"]),
        ]
        _builder()._propagate_assigned_variables_to_literal_args(steps)
        # No substitution — capture wasn't usable.
        assert steps[1].arguments == ["msg", "5"]

    def test_empty_captured_value_does_not_register(self):
        steps = [
            TestCaseStep(
                keyword="Get Text",
                arguments=["id=foo"],
                assigned_variables=["${X}"],
                captured_value="",
            ),
            TestCaseStep(keyword="Log", arguments=["msg", ""]),
        ]
        _builder()._propagate_assigned_variables_to_literal_args(steps)
        # Both empty — but capture is non-usable, no rewrite.
        assert steps[1].arguments == ["msg", ""]

    def test_whitespace_only_captured_value_does_not_register(self):
        steps = [
            TestCaseStep(
                keyword="Get Text",
                arguments=["id=foo"],
                assigned_variables=["${X}"],
                captured_value="   ",
            ),
            TestCaseStep(keyword="Log", arguments=["msg", "   "]),
        ]
        _builder()._propagate_assigned_variables_to_literal_args(steps)
        assert steps[1].arguments == ["msg", "   "]


class TestCrossTestCaseIsolation:
    """A capture in test case A must NOT bleed into test case B.
    The propagation pass is called per-test-case so captures are
    naturally scoped."""

    def test_captures_do_not_cross_test_case_boundary(self):
        # Two independent step lists — simulates two test cases.
        tc_a_steps = [
            _capture_step("${X}", "5"),
            TestCaseStep(keyword="Log", arguments=["msg", "5"]),
        ]
        tc_b_steps = [
            TestCaseStep(keyword="Log", arguments=["msg", "5"]),
        ]
        _builder()._propagate_assigned_variables_to_literal_args(tc_a_steps)
        _builder()._propagate_assigned_variables_to_literal_args(tc_b_steps)
        # tc_a: capture available, substitution happens.
        assert tc_a_steps[1].arguments == ["msg", "${X}"]
        # tc_b: NO capture available, literal preserved.
        assert tc_b_steps[0].arguments == ["msg", "5"]


class TestEdgeCases:
    """Defensive cases: empty / minimal inputs must not crash."""

    def test_empty_steps_list_is_noop(self):
        steps: list[TestCaseStep] = []
        _builder()._propagate_assigned_variables_to_literal_args(steps)
        assert steps == []

    def test_single_step_without_capture_is_noop(self):
        steps = [TestCaseStep(keyword="Log", arguments=["msg"])]
        _builder()._propagate_assigned_variables_to_literal_args(steps)
        assert steps[0].arguments == ["msg"]

    def test_step_with_no_args_does_not_crash(self):
        steps = [
            _capture_step("${X}", "5"),
            TestCaseStep(keyword="Get Title", arguments=[]),
        ]
        _builder()._propagate_assigned_variables_to_literal_args(steps)
        assert steps[1].arguments == []

    def test_non_string_args_are_skipped_gracefully(self):
        # ExecutionStep coerces args to strings on the wire, but
        # defensive: non-string args (int, None) must not crash.
        steps = [
            _capture_step("${X}", "5"),
            TestCaseStep(keyword="Log", arguments=["msg", 5]),
        ]
        _builder()._propagate_assigned_variables_to_literal_args(steps)
        # int 5 != string "5"; no substitution.
        assert steps[1].arguments == ["msg", 5]


# ---------------------------------------------------------------------------
# Layer 2: end-to-end via _generate_rf_text — the headline acceptance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestGenerateRfTextSubstitutesAssignedVars:
    """The story acceptance #2: build_test_suite output contains
    ``Fill Text id=entryCount ${ENTRY_COUNT}`` (NOT literal ``"5"``)."""

    async def test_obstacle_7_scenario_rf_text(self):
        # Synthesize the Obstacle 7 scenario directly via the test-
        # case + suite DTOs, skipping the upstream session machinery.
        capture = TestCaseStep(
            keyword="Get Element Count",
            arguments=["css=.select2-results__option"],
            assigned_variables=["${ENTRY_COUNT}"],
            assignment_type="single",
            captured_value="5",
        )
        fill = TestCaseStep(
            keyword="Fill Text",
            arguments=["id=entryCount", "5"],
        )
        tc = GeneratedTestCase(
            name="Obstacle 7 - And counting",
            steps=[capture, fill],
        )
        suite = GeneratedTestSuite(
            name="OBS-11 fixture",
            documentation="Variable-propagation acceptance",
            imports=["Browser"],
            test_cases=[tc],
        )
        builder = _builder()
        rf_text = await builder._generate_rf_text(suite)
        # The headline assertion — ${ENTRY_COUNT} appears at the Fill
        # Text site (not the literal "5").
        assert "${ENTRY_COUNT} =    Get Element Count" in rf_text
        assert "Fill Text    id=entryCount    ${ENTRY_COUNT}" in rf_text
        # Pin the negative: the literal "5" must NOT appear in the
        # Fill Text args slot.
        assert "Fill Text    id=entryCount    5\n" not in rf_text
        assert "Fill Text    id=entryCount    5    " not in rf_text


# ---------------------------------------------------------------------------
# Layer 3: ExecutionStep → TestCaseStep plumbing
# ---------------------------------------------------------------------------


class TestCapturedValuePlumbing:
    """``captured_value`` is populated from ``ExecutionStep.result``
    when the step assigned a variable, and is None otherwise."""

    def test_captures_string_result(self):
        exec_step = MagicMock()
        exec_step.assigned_variables = ["${X}"]
        exec_step.result = "hello"
        assert TestBuilder._captured_value_str(exec_step) == "hello"

    def test_captures_int_result_as_str(self):
        exec_step = MagicMock()
        exec_step.assigned_variables = ["${N}"]
        exec_step.result = 5
        # Coerced to string for arg-level equality comparison.
        assert TestBuilder._captured_value_str(exec_step) == "5"

    def test_no_assigned_variables_returns_none(self):
        exec_step = MagicMock()
        exec_step.assigned_variables = []
        exec_step.result = "hello"
        assert TestBuilder._captured_value_str(exec_step) is None

    def test_no_result_returns_none(self):
        exec_step = MagicMock()
        exec_step.assigned_variables = ["${X}"]
        exec_step.result = None
        assert TestBuilder._captured_value_str(exec_step) is None
