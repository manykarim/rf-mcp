"""Tests for recovery domain entities."""
import pytest

from robotmcp.domains.recovery.entities import (
    RecoveryPlan,
    RecoveryPlanPhase,
)
from robotmcp.domains.recovery.value_objects import (
    ErrorClassification,
    RecoveryAction,
    RecoveryStrategy,
    RecoveryTier,
)


# ── RecoveryPlanPhase ────────────────────────────────────────────────


class TestRecoveryPlanPhase:
    def test_classify_value(self):
        assert RecoveryPlanPhase.CLASSIFY.value == "CLASSIFY"

    def test_strategize_value(self):
        assert RecoveryPlanPhase.STRATEGIZE.value == "STRATEGIZE"

    def test_execute_value(self):
        assert RecoveryPlanPhase.EXECUTE.value == "EXECUTE"

    def test_evaluate_value(self):
        assert RecoveryPlanPhase.EVALUATE.value == "EVALUATE"

    def test_completed_value(self):
        assert RecoveryPlanPhase.COMPLETED.value == "COMPLETED"

    def test_next_phase_classify(self):
        assert RecoveryPlanPhase.CLASSIFY.next_phase == RecoveryPlanPhase.STRATEGIZE

    def test_next_phase_strategize(self):
        assert RecoveryPlanPhase.STRATEGIZE.next_phase == RecoveryPlanPhase.EXECUTE

    def test_next_phase_execute(self):
        assert RecoveryPlanPhase.EXECUTE.next_phase == RecoveryPlanPhase.EVALUATE

    def test_next_phase_evaluate(self):
        assert RecoveryPlanPhase.EVALUATE.next_phase == RecoveryPlanPhase.COMPLETED

    def test_next_phase_completed_is_none(self):
        assert RecoveryPlanPhase.COMPLETED.next_phase is None

    def test_full_chain(self):
        phase = RecoveryPlanPhase.CLASSIFY
        chain = [phase]
        while phase.next_phase is not None:
            phase = phase.next_phase
            chain.append(phase)
        assert [p.value for p in chain] == [
            "CLASSIFY", "STRATEGIZE", "EXECUTE", "EVALUATE", "COMPLETED",
        ]


# ── RecoveryPlan ─────────────────────────────────────────────────────


def _make_strategy(name="wait_and_retry"):
    return RecoveryStrategy(
        name=name, tier=RecoveryTier.TIER_1,
        applicable_to=(ErrorClassification.ELEMENT_NOT_FOUND,),
        actions=(RecoveryAction(keyword="Sleep", args=("2s",)),),
    )


class TestRecoveryPlanCreate:
    def test_create_basic(self):
        plan = RecoveryPlan.create(
            session_id="s1", keyword="Click", args=["btn"],
            error_message="not found",
        )
        assert plan.session_id == "s1"
        assert plan.keyword == "Click"
        assert plan.args == ["btn"]
        assert plan.error_message == "not found"
        assert plan.phase == RecoveryPlanPhase.CLASSIFY

    def test_create_generates_id(self):
        plan = RecoveryPlan.create("s1", "K", [], "err")
        assert plan.plan_id.startswith("recovery_")

    def test_create_unique_ids(self):
        ids = {
            RecoveryPlan.create("s1", "K", [], "err").plan_id
            for _ in range(20)
        }
        assert len(ids) == 20

    def test_create_args_are_copied(self):
        original = ["a", "b"]
        plan = RecoveryPlan.create("s1", "K", original, "err")
        original.append("c")
        assert plan.args == ["a", "b"]


class TestRecoveryPlanLifecycle:
    def test_set_classification(self):
        plan = RecoveryPlan.create("s1", "K", [], "err")
        plan.set_classification(ErrorClassification.ELEMENT_NOT_FOUND)
        assert plan.classification == ErrorClassification.ELEMENT_NOT_FOUND
        assert plan.phase == RecoveryPlanPhase.STRATEGIZE

    def test_set_classification_wrong_phase(self):
        plan = RecoveryPlan.create("s1", "K", [], "err")
        plan.set_classification(ErrorClassification.ELEMENT_NOT_FOUND)
        with pytest.raises(ValueError, match="Cannot classify"):
            plan.set_classification(ErrorClassification.TIMEOUT_EXCEPTION)

    def test_set_strategy(self):
        plan = RecoveryPlan.create("s1", "K", [], "err")
        plan.set_classification(ErrorClassification.ELEMENT_NOT_FOUND)
        strategy = _make_strategy()
        plan.set_strategy(strategy)
        assert plan.selected_strategy is strategy
        assert plan.phase == RecoveryPlanPhase.EXECUTE

    def test_set_strategy_wrong_phase(self):
        plan = RecoveryPlan.create("s1", "K", [], "err")
        with pytest.raises(ValueError, match="Cannot set strategy"):
            plan.set_strategy(_make_strategy())

    def test_record_action(self):
        plan = RecoveryPlan.create("s1", "K", [], "err")
        plan.set_classification(ErrorClassification.ELEMENT_NOT_FOUND)
        plan.set_strategy(_make_strategy())
        action = RecoveryAction(keyword="Sleep", args=("2s",))
        plan.record_action(action)
        assert len(plan.actions_executed) == 1
        assert plan.phase == RecoveryPlanPhase.EXECUTE

    def test_record_action_wrong_phase(self):
        plan = RecoveryPlan.create("s1", "K", [], "err")
        with pytest.raises(ValueError, match="Cannot record action"):
            plan.record_action(RecoveryAction(keyword="Sleep"))

    def test_finish_execution(self):
        plan = RecoveryPlan.create("s1", "K", [], "err")
        plan.set_classification(ErrorClassification.ELEMENT_NOT_FOUND)
        plan.set_strategy(_make_strategy())
        plan.finish_execution()
        assert plan.phase == RecoveryPlanPhase.EVALUATE

    def test_finish_execution_wrong_phase(self):
        plan = RecoveryPlan.create("s1", "K", [], "err")
        with pytest.raises(ValueError, match="Cannot finish execution"):
            plan.finish_execution()

    def test_set_result_success(self):
        plan = RecoveryPlan.create("s1", "K", [], "err")
        plan.set_classification(ErrorClassification.ELEMENT_NOT_FOUND)
        plan.set_strategy(_make_strategy())
        plan.finish_execution()
        plan.set_result(True)
        assert plan.retry_succeeded is True
        assert plan.phase == RecoveryPlanPhase.COMPLETED

    def test_set_result_failure(self):
        plan = RecoveryPlan.create("s1", "K", [], "err")
        plan.set_classification(ErrorClassification.ELEMENT_NOT_FOUND)
        plan.set_strategy(_make_strategy())
        plan.finish_execution()
        plan.set_result(False)
        assert plan.retry_succeeded is False
        assert plan.phase == RecoveryPlanPhase.COMPLETED

    def test_set_result_wrong_phase(self):
        plan = RecoveryPlan.create("s1", "K", [], "err")
        with pytest.raises(ValueError, match="Cannot set result"):
            plan.set_result(True)

    def test_advance_from_completed_raises(self):
        plan = RecoveryPlan.create("s1", "K", [], "err")
        plan.set_classification(ErrorClassification.ELEMENT_NOT_FOUND)
        plan.set_strategy(_make_strategy())
        plan.finish_execution()
        plan.set_result(True)
        with pytest.raises(ValueError, match="Cannot advance"):
            plan.advance_phase()


class TestRecoveryPlanToDict:
    def test_initial_state(self):
        plan = RecoveryPlan.create("s1", "Click", ["btn"], "not found")
        d = plan.to_dict()
        assert d["keyword"] == "Click"
        assert d["error_message"] == "not found"
        assert d["phase"] == "CLASSIFY"
        assert "classification" not in d
        assert "strategy" not in d
        assert "actions" not in d
        assert "retry_succeeded" not in d

    def test_after_classification(self):
        plan = RecoveryPlan.create("s1", "K", [], "err")
        plan.set_classification(ErrorClassification.ELEMENT_NOT_FOUND)
        d = plan.to_dict()
        assert d["classification"] == "ElementNotFound"
        assert d["phase"] == "STRATEGIZE"

    def test_after_strategy(self):
        plan = RecoveryPlan.create("s1", "K", [], "err")
        plan.set_classification(ErrorClassification.ELEMENT_NOT_FOUND)
        plan.set_strategy(_make_strategy("my_strategy"))
        d = plan.to_dict()
        assert d["strategy"] == "my_strategy"

    def test_after_actions(self):
        plan = RecoveryPlan.create("s1", "K", [], "err")
        plan.set_classification(ErrorClassification.ELEMENT_NOT_FOUND)
        plan.set_strategy(_make_strategy())
        action = RecoveryAction(keyword="Sleep", args=("2s",), description="wait")
        plan.record_action(action)
        d = plan.to_dict()
        assert len(d["actions"]) == 1
        assert d["actions"][0]["keyword"] == "Sleep"

    def test_completed_with_result(self):
        plan = RecoveryPlan.create("s1", "K", [], "err")
        plan.set_classification(ErrorClassification.ELEMENT_NOT_FOUND)
        plan.set_strategy(_make_strategy())
        plan.finish_execution()
        plan.set_result(True)
        d = plan.to_dict()
        assert d["retry_succeeded"] is True
        assert d["phase"] == "COMPLETED"
