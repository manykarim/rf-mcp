"""Tests for recovery domain services."""
from unittest.mock import AsyncMock

import pytest

from robotmcp.domains.recovery.aggregates import RecoveryEngine
from robotmcp.domains.recovery.entities import RecoveryPlan, RecoveryPlanPhase
from robotmcp.domains.recovery.services import (
    ErrorClassifier,
    EvidenceCollector,
    Tier1RecoveryService,
    Tier2RecoveryService,
)
from robotmcp.domains.recovery.value_objects import (
    ErrorClassification,
    RecoveryAction,
    RecoveryStrategy,
    RecoveryTier,
)


# ── Helpers ──────────────────────────────────────────────────────────


def _make_plan_in_execute_phase():
    """Create a RecoveryPlan already in EXECUTE phase."""
    plan = RecoveryPlan.create("s1", "Click", ["btn"], "not found")
    plan.set_classification(ErrorClassification.ELEMENT_NOT_FOUND)
    plan.set_strategy(RecoveryStrategy(
        name="wait_and_retry", tier=RecoveryTier.TIER_1,
        applicable_to=(ErrorClassification.ELEMENT_NOT_FOUND,),
        actions=(RecoveryAction(keyword="Sleep", args=("2s",)),),
    ))
    return plan


def _make_strategy_with_actions(name="test_strategy", tier=RecoveryTier.TIER_1):
    return RecoveryStrategy(
        name=name, tier=tier,
        applicable_to=(ErrorClassification.ELEMENT_NOT_FOUND,),
        actions=(
            RecoveryAction(keyword="Sleep", args=("2s",), description="wait"),
            RecoveryAction(keyword="Scroll Down", args=(), description="scroll"),
        ),
    )


def _mock_keyword_runner(side_effects=None):
    runner = AsyncMock()
    if side_effects:
        runner.run_keyword = AsyncMock(side_effect=side_effects)
    else:
        runner.run_keyword = AsyncMock(return_value=None)
    return runner


def _mock_page_state(
    url="http://example.com",
    title="Example",
    screenshot="base64data",
    page_source="<html>",
):
    ps = AsyncMock()
    ps.capture_screenshot = AsyncMock(return_value=screenshot)
    ps.capture_page_source = AsyncMock(return_value=page_source)
    ps.get_current_url = AsyncMock(return_value=url)
    ps.get_page_title = AsyncMock(return_value=title)
    return ps


# ── ErrorClassifier ──────────────────────────────────────────────────


class TestErrorClassifier:
    def test_classify_delegates(self):
        engine = RecoveryEngine.with_defaults()
        classifier = ErrorClassifier(engine=engine)
        result = classifier.classify("Element not found")
        assert result == ErrorClassification.ELEMENT_NOT_FOUND

    def test_classify_unknown(self):
        engine = RecoveryEngine.with_defaults()
        classifier = ErrorClassifier(engine=engine)
        result = classifier.classify("random error xyz")
        assert result == ErrorClassification.UNKNOWN

    def test_classify_timeout(self):
        engine = RecoveryEngine.with_defaults()
        classifier = ErrorClassifier(engine=engine)
        result = classifier.classify("Connection timed out after 30s")
        assert result == ErrorClassification.TIMEOUT_EXCEPTION


# ── Tier1RecoveryService ─────────────────────────────────────────────


class TestTier1RecoveryService:
    @pytest.mark.asyncio
    async def test_execute_all_actions_succeed(self):
        runner = _mock_keyword_runner()
        service = Tier1RecoveryService(keyword_runner=runner)
        plan = _make_plan_in_execute_phase()
        strategy = _make_strategy_with_actions()

        actions = await service.execute("s1", strategy, plan)
        assert len(actions) == 2
        assert len(plan.actions_executed) == 2
        assert runner.run_keyword.call_count == 2

    @pytest.mark.asyncio
    async def test_execute_action_failure_continues(self):
        runner = _mock_keyword_runner(
            side_effects=[RuntimeError("fail"), None]
        )
        service = Tier1RecoveryService(keyword_runner=runner)
        plan = _make_plan_in_execute_phase()
        strategy = _make_strategy_with_actions()

        actions = await service.execute("s1", strategy, plan)
        # Both actions are attempted even if first fails
        assert len(actions) == 2
        assert len(plan.actions_executed) == 2

    @pytest.mark.asyncio
    async def test_execute_empty_actions(self):
        runner = _mock_keyword_runner()
        service = Tier1RecoveryService(keyword_runner=runner)
        plan = _make_plan_in_execute_phase()
        strategy = RecoveryStrategy(
            name="no_actions", tier=RecoveryTier.TIER_1, actions=(),
        )

        actions = await service.execute("s1", strategy, plan)
        assert actions == []
        assert runner.run_keyword.call_count == 0

    @pytest.mark.asyncio
    async def test_execute_passes_timeout(self):
        runner = _mock_keyword_runner()
        service = Tier1RecoveryService(keyword_runner=runner)
        plan = _make_plan_in_execute_phase()
        strategy = RecoveryStrategy(
            name="s", tier=RecoveryTier.TIER_1,
            actions=(RecoveryAction(keyword="Sleep", args=("2s",)),),
        )

        await service.execute("s1", strategy, plan)
        call_kwargs = runner.run_keyword.call_args
        assert call_kwargs.kwargs.get("timeout") == "5s" or call_kwargs[1].get("timeout") == "5s"

    @pytest.mark.asyncio
    async def test_execute_records_failed_action(self):
        runner = _mock_keyword_runner(side_effects=[RuntimeError("err")])
        service = Tier1RecoveryService(keyword_runner=runner)
        plan = _make_plan_in_execute_phase()
        strategy = RecoveryStrategy(
            name="s", tier=RecoveryTier.TIER_1,
            actions=(RecoveryAction(keyword="Fail", args=()),),
        )

        actions = await service.execute("s1", strategy, plan)
        # Failed action is still recorded
        assert len(actions) == 1
        assert len(plan.actions_executed) == 1


# ── Tier2RecoveryService ─────────────────────────────────────────────


class TestTier2RecoveryService:
    @pytest.mark.asyncio
    async def test_execute_with_evidence_capture(self):
        runner = _mock_keyword_runner()
        page_state = _mock_page_state(url="http://test.com", title="Test Page")
        service = Tier2RecoveryService(
            keyword_runner=runner, page_state=page_state,
        )
        plan = _make_plan_in_execute_phase()
        strategy = _make_strategy_with_actions(tier=RecoveryTier.TIER_2)

        actions = await service.execute("s1", strategy, plan)
        assert len(actions) == 2
        assert plan.evidence["current_url"] == "http://test.com"
        assert plan.evidence["page_title"] == "Test Page"

    @pytest.mark.asyncio
    async def test_execute_without_page_state(self):
        runner = _mock_keyword_runner()
        service = Tier2RecoveryService(keyword_runner=runner, page_state=None)
        plan = _make_plan_in_execute_phase()
        strategy = _make_strategy_with_actions(tier=RecoveryTier.TIER_2)

        actions = await service.execute("s1", strategy, plan)
        assert len(actions) == 2
        assert plan.evidence == {}

    @pytest.mark.asyncio
    async def test_execute_evidence_failure_continues(self):
        runner = _mock_keyword_runner()
        page_state = AsyncMock()
        page_state.get_current_url = AsyncMock(side_effect=RuntimeError("fail"))
        page_state.get_page_title = AsyncMock(side_effect=RuntimeError("fail"))
        service = Tier2RecoveryService(
            keyword_runner=runner, page_state=page_state,
        )
        plan = _make_plan_in_execute_phase()
        strategy = _make_strategy_with_actions(tier=RecoveryTier.TIER_2)

        # Should not raise despite evidence capture failure
        actions = await service.execute("s1", strategy, plan)
        assert len(actions) == 2

    @pytest.mark.asyncio
    async def test_execute_passes_timeout(self):
        runner = _mock_keyword_runner()
        service = Tier2RecoveryService(keyword_runner=runner)
        plan = _make_plan_in_execute_phase()
        strategy = RecoveryStrategy(
            name="s", tier=RecoveryTier.TIER_2,
            actions=(RecoveryAction(keyword="Reload Page", args=()),),
        )

        await service.execute("s1", strategy, plan)
        call_kwargs = runner.run_keyword.call_args
        assert call_kwargs.kwargs.get("timeout") == "10s" or call_kwargs[1].get("timeout") == "10s"

    @pytest.mark.asyncio
    async def test_execute_action_failure_continues(self):
        runner = _mock_keyword_runner(
            side_effects=[RuntimeError("fail"), None]
        )
        service = Tier2RecoveryService(keyword_runner=runner)
        plan = _make_plan_in_execute_phase()
        strategy = _make_strategy_with_actions(tier=RecoveryTier.TIER_2)

        actions = await service.execute("s1", strategy, plan)
        assert len(actions) == 2

    @pytest.mark.asyncio
    async def test_execute_records_null_url(self):
        runner = _mock_keyword_runner()
        page_state = AsyncMock()
        page_state.get_current_url = AsyncMock(return_value=None)
        page_state.get_page_title = AsyncMock(return_value=None)
        service = Tier2RecoveryService(
            keyword_runner=runner, page_state=page_state,
        )
        plan = _make_plan_in_execute_phase()
        strategy = RecoveryStrategy(
            name="s", tier=RecoveryTier.TIER_2, actions=(),
        )

        await service.execute("s1", strategy, plan)
        # None values should NOT be stored
        assert "current_url" not in plan.evidence
        assert "page_title" not in plan.evidence


# ── EvidenceCollector ────────────────────────────────────────────────


class TestEvidenceCollector:
    @pytest.mark.asyncio
    async def test_collect_all_evidence(self):
        page_state = _mock_page_state(
            url="http://x.com", title="X",
            screenshot="s==", page_source="<p>",
        )
        collector = EvidenceCollector(page_state=page_state)
        evidence = await collector.collect("s1")
        assert evidence["screenshot_base64"] == "s=="
        assert evidence["page_source_snippet"] == "<p>"
        assert evidence["current_url"] == "http://x.com"
        assert evidence["page_title"] == "X"

    @pytest.mark.asyncio
    async def test_collect_no_page_state(self):
        collector = EvidenceCollector(page_state=None)
        evidence = await collector.collect("s1")
        assert evidence == {}

    @pytest.mark.asyncio
    async def test_collect_partial_failure(self):
        page_state = AsyncMock()
        page_state.capture_screenshot = AsyncMock(
            side_effect=RuntimeError("no screenshot")
        )
        page_state.capture_page_source = AsyncMock(return_value="<html>")
        page_state.get_current_url = AsyncMock(return_value="http://x.com")
        page_state.get_page_title = AsyncMock(
            side_effect=RuntimeError("no title")
        )
        collector = EvidenceCollector(page_state=page_state)
        evidence = await collector.collect("s1")
        assert "screenshot_base64" not in evidence
        assert evidence["page_source_snippet"] == "<html>"
        assert evidence["current_url"] == "http://x.com"
        assert "page_title" not in evidence

    @pytest.mark.asyncio
    async def test_collect_null_values_omitted(self):
        page_state = AsyncMock()
        page_state.capture_screenshot = AsyncMock(return_value=None)
        page_state.capture_page_source = AsyncMock(return_value=None)
        page_state.get_current_url = AsyncMock(return_value=None)
        page_state.get_page_title = AsyncMock(return_value=None)
        collector = EvidenceCollector(page_state=page_state)
        evidence = await collector.collect("s1")
        assert evidence == {}

    @pytest.mark.asyncio
    async def test_collect_page_source_max_chars(self):
        page_state = _mock_page_state()
        collector = EvidenceCollector(page_state=page_state)
        await collector.collect("s1")
        # Verify max_chars=2000 is passed
        page_state.capture_page_source.assert_called_once_with("s1", max_chars=2000)
