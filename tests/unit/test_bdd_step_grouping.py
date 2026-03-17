"""Tests for BDD step grouping (Phase 2 of BDD quality improvement).

Validates that bdd_group and bdd_intent fields are properly stored on
ExecutionStep and TestCaseStep, and survive round-trip through session steps.
"""

from robotmcp.models.execution_models import ExecutionStep, TestInfo, TestRegistry
from robotmcp.components.test_builder import TestCaseStep


class TestExecutionStepBddFields:
    """Test bdd_group/bdd_intent on ExecutionStep."""

    def test_default_values_are_none(self):
        step = ExecutionStep(step_id="s1", keyword="Click", arguments=["button"])
        assert step.bdd_group is None
        assert step.bdd_intent is None

    def test_set_bdd_group_and_intent(self):
        step = ExecutionStep(
            step_id="s1",
            keyword="Click",
            arguments=["button"],
            bdd_group="add product to cart",
            bdd_intent="when",
        )
        assert step.bdd_group == "add product to cart"
        assert step.bdd_intent == "when"

    def test_set_bdd_fields_after_construction(self):
        step = ExecutionStep(step_id="s1", keyword="Click", arguments=["button"])
        step.bdd_group = "login"
        step.bdd_intent = "given"
        assert step.bdd_group == "login"
        assert step.bdd_intent == "given"

    def test_bdd_fields_independent_of_other_fields(self):
        step = ExecutionStep(
            step_id="s1",
            keyword="Click",
            arguments=["button"],
            assigned_variables=["${result}"],
            assignment_type="single",
            bdd_group="verify result",
            bdd_intent="then",
        )
        assert step.assigned_variables == ["${result}"]
        assert step.assignment_type == "single"
        assert step.bdd_group == "verify result"
        assert step.bdd_intent == "then"


class TestTestCaseStepBddFields:
    """Test bdd_group/bdd_intent on TestCaseStep."""

    def test_default_values_are_none(self):
        step = TestCaseStep(keyword="Click", arguments=["button"])
        assert step.bdd_group is None
        assert step.bdd_intent is None

    def test_set_bdd_group_and_intent(self):
        step = TestCaseStep(
            keyword="Click",
            arguments=["button"],
            bdd_group="checkout",
            bdd_intent="when",
        )
        assert step.bdd_group == "checkout"
        assert step.bdd_intent == "when"


class TestBddFieldsRoundTrip:
    """Test that bdd_group/bdd_intent survive from ExecutionStep through session."""

    def test_session_steps_preserve_bdd_fields(self):
        """Steps added to a TestInfo preserve bdd_group/bdd_intent."""
        test_info = TestInfo(name="Test BDD")
        step = ExecutionStep(
            step_id="s1",
            keyword="Open Browser",
            arguments=["http://example.com"],
            bdd_group="setup browser",
            bdd_intent="given",
        )
        test_info.steps.append(step)

        retrieved = test_info.steps[0]
        assert retrieved.bdd_group == "setup browser"
        assert retrieved.bdd_intent == "given"

    def test_multi_test_registry_preserves_bdd_fields(self):
        """Steps in multi-test mode preserve bdd_group/bdd_intent."""
        registry = TestRegistry()
        test = registry.start_test("BDD Test")

        step1 = ExecutionStep(
            step_id="s1",
            keyword="Navigate To",
            arguments=["http://shop.com"],
            bdd_group="navigate to shop",
            bdd_intent="given",
        )
        step2 = ExecutionStep(
            step_id="s2",
            keyword="Click",
            arguments=["Add to Cart"],
            bdd_group="add product to cart",
            bdd_intent="when",
        )
        step3 = ExecutionStep(
            step_id="s3",
            keyword="Get Text",
            arguments=["cart-count"],
            bdd_group="verify cart",
            bdd_intent="then",
        )
        test.steps.extend([step1, step2, step3])

        all_steps = registry.all_steps_flat()
        assert len(all_steps) == 3
        assert all_steps[0].bdd_group == "navigate to shop"
        assert all_steps[0].bdd_intent == "given"
        assert all_steps[1].bdd_group == "add product to cart"
        assert all_steps[1].bdd_intent == "when"
        assert all_steps[2].bdd_group == "verify cart"
        assert all_steps[2].bdd_intent == "then"

    def test_mixed_bdd_and_non_bdd_steps(self):
        """Steps without bdd_group/bdd_intent coexist with annotated ones."""
        test_info = TestInfo(name="Mixed Test")
        step1 = ExecutionStep(
            step_id="s1",
            keyword="Log",
            arguments=["Starting test"],
        )
        step2 = ExecutionStep(
            step_id="s2",
            keyword="Click",
            arguments=["button"],
            bdd_group="perform action",
            bdd_intent="when",
        )
        test_info.steps.extend([step1, step2])

        assert test_info.steps[0].bdd_group is None
        assert test_info.steps[0].bdd_intent is None
        assert test_info.steps[1].bdd_group == "perform action"
        assert test_info.steps[1].bdd_intent == "when"

    def test_retroactive_annotation(self):
        """Simulate the server.py pattern of retroactively setting bdd fields."""
        test_info = TestInfo(name="Retroactive Test")
        step = ExecutionStep(
            step_id="s1",
            keyword="Click",
            arguments=["submit"],
        )
        test_info.steps.append(step)

        # Retroactively set (as server.py does after execution)
        test_info.steps[-1].bdd_group = "submit form"
        test_info.steps[-1].bdd_intent = "when"

        assert test_info.steps[0].bdd_group == "submit form"
        assert test_info.steps[0].bdd_intent == "when"

    def test_all_bdd_intents(self):
        """All five BDD intent values work correctly."""
        intents = ["given", "when", "then", "and", "but"]
        for intent in intents:
            step = ExecutionStep(
                step_id="s1",
                keyword="Click",
                arguments=[],
                bdd_intent=intent,
                bdd_group="test",
            )
            assert step.bdd_intent == intent


class TestTestCaseStepFromExecutionStep:
    """Test conversion from ExecutionStep to TestCaseStep preserves BDD fields."""

    def test_manual_conversion_preserves_bdd_fields(self):
        """Simulates what _build_test_case_from_test_info does."""
        exec_step = ExecutionStep(
            step_id="s1",
            keyword="Click",
            arguments=["button"],
            assigned_variables=["${result}"],
            assignment_type="single",
            bdd_group="click action",
            bdd_intent="when",
        )

        tc_step = TestCaseStep(
            keyword=exec_step.keyword,
            arguments=exec_step.arguments,
            assigned_variables=getattr(exec_step, "assigned_variables", []),
            assignment_type=getattr(exec_step, "assignment_type", None),
            bdd_group=getattr(exec_step, "bdd_group", None),
            bdd_intent=getattr(exec_step, "bdd_intent", None),
        )

        assert tc_step.bdd_group == "click action"
        assert tc_step.bdd_intent == "when"
        assert tc_step.assigned_variables == ["${result}"]

    def test_conversion_without_bdd_fields(self):
        """Conversion of step without BDD fields yields None."""
        exec_step = ExecutionStep(
            step_id="s1",
            keyword="Log",
            arguments=["hello"],
        )

        tc_step = TestCaseStep(
            keyword=exec_step.keyword,
            arguments=exec_step.arguments,
            bdd_group=getattr(exec_step, "bdd_group", None),
            bdd_intent=getattr(exec_step, "bdd_intent", None),
        )

        assert tc_step.bdd_group is None
        assert tc_step.bdd_intent is None
