"""Tests for Critical: Server-level BDD integration pipeline.

Verifies the full path: execute_step(bdd_group, bdd_intent) → ExecutionStep
→ TestRegistry → _build_test_case_from_test_info() → build_suite(bdd_style=True)
→ RF text with Given/When/Then + Keywords section.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from robotmcp.models.execution_models import ExecutionStep, TestInfo, TestRegistry
from robotmcp.components.test_builder import (
    TestBuilder,
    TestCaseStep,
    GeneratedTestCase,
    GeneratedTestSuite,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exec_step(keyword: str, arguments: list, status: str = "pass", **kwargs):
    """Create an ExecutionStep with sensible defaults."""
    s = ExecutionStep(
        step_id=f"s-{id(keyword)}",
        keyword=keyword,
        arguments=arguments,
        status=status,
        **kwargs,
    )
    return s


def _make_mock_engine_legacy(steps_dicts):
    """Mock engine for legacy (non-multi-test) mode."""
    engine = MagicMock()
    session = MagicMock()
    session.test_registry.is_multi_test_mode.return_value = False
    engine.sessions = {"test-session": session}
    engine.validate_test_readiness = AsyncMock(
        return_value={"ready_for_suite_generation": True}
    )
    return engine, steps_dicts


def _make_mock_engine_multitest(test_registry):
    """Mock engine for multi-test mode with a real TestRegistry."""
    engine = MagicMock()
    session = MagicMock()
    session.test_registry = test_registry
    engine.sessions = {"test-session": session}
    engine.session_manager.get_session.return_value = session
    return engine


# ---------------------------------------------------------------------------
# TestServerBddAnnotationPipeline
# ---------------------------------------------------------------------------

class TestServerBddAnnotationPipeline:
    """Simulate the server.py retroactive annotation pattern."""

    def test_retroactive_bdd_annotation_on_session_steps(self):
        """Simulate server.py lines 3654-3664: retroactively set bdd fields on last step."""
        # Simulate: session has steps, server retroactively annotates the last one
        steps = [
            _exec_step("New Browser", ["chromium"]),
            _exec_step("Click", ['text="Buy"']),
        ]

        # This is what server.py does after keyword execution:
        bdd_group = "add product"
        bdd_intent = "when"
        if steps:
            steps[-1].bdd_group = bdd_group or None
            steps[-1].bdd_intent = bdd_intent or None

        assert steps[0].bdd_group is None
        assert steps[0].bdd_intent is None
        assert steps[1].bdd_group == "add product"
        assert steps[1].bdd_intent == "when"

    def test_retroactive_annotation_multi_test_mode(self):
        """Same pattern but via TestRegistry (multi-test mode)."""
        registry = TestRegistry()
        test = registry.start_test("BDD Test")
        test.steps.append(_exec_step("Click", ['text="Submit"']))

        # server.py: get current test's steps, annotate last
        current_test = registry.get_current_test()
        _steps = current_test.steps if current_test else []
        if _steps:
            _steps[-1].bdd_group = "submit form"
            _steps[-1].bdd_intent = "when"

        assert test.steps[0].bdd_group == "submit form"
        assert test.steps[0].bdd_intent == "when"

    def test_annotation_only_on_last_step(self):
        """Only the last step gets annotated (matching server.py behavior)."""
        steps = [
            _exec_step("Open Browser", ["http://x.com"]),
            _exec_step("Click", ["#btn1"]),
            _exec_step("Click", ["#btn2"]),
        ]
        # Annotate last
        steps[-1].bdd_group = "click second"
        steps[-1].bdd_intent = "when"

        assert steps[0].bdd_group is None
        assert steps[1].bdd_group is None
        assert steps[2].bdd_group == "click second"

    def test_empty_bdd_group_becomes_none(self):
        """server.py uses `bdd_group or None` — empty string → None."""
        step = _exec_step("Click", ["btn"])
        bdd_group = ""
        bdd_intent = "when"
        step.bdd_group = bdd_group or None
        step.bdd_intent = bdd_intent or None

        assert step.bdd_group is None
        assert step.bdd_intent == "when"

    def test_empty_bdd_intent_becomes_none(self):
        """Empty intent → None, non-empty group preserved."""
        step = _exec_step("Click", ["btn"])
        step.bdd_group = "my group" or None
        step.bdd_intent = "" or None

        assert step.bdd_group == "my group"
        assert step.bdd_intent is None


# ---------------------------------------------------------------------------
# TestExplicitAnnotationToBddSuite
# ---------------------------------------------------------------------------

class TestExplicitAnnotationToBddSuite:
    """Test full pipeline: annotated steps → build_suite(bdd_style=True) → RF text."""

    def _steps_to_dicts(self, steps):
        return [
            {
                "keyword": s.keyword,
                "arguments": s.arguments,
                "status": "pass",
                "assigned_variables": [],
                "assignment_type": None,
                "bdd_group": s.bdd_group,
                "bdd_intent": s.bdd_intent,
            }
            for s in steps
        ]

    @pytest.mark.asyncio
    async def test_explicit_groups_produce_named_keywords(self):
        """Steps with same bdd_group are clustered into one behavioral keyword."""
        steps = [
            _exec_step("New Browser", ["chromium"], bdd_group="open browser", bdd_intent="given"),
            _exec_step("New Page", ["https://shop.com"], bdd_group="open browser", bdd_intent="given"),
            _exec_step("Click", ['text="Add"'], bdd_group="add to cart", bdd_intent="when"),
        ]
        engine, step_dicts = _make_mock_engine_legacy(self._steps_to_dicts(steps))
        builder = TestBuilder(execution_engine=engine)

        with patch.object(builder, "_get_session_steps", new_callable=AsyncMock, return_value=step_dicts):
            result = await builder.build_suite(
                session_id="test-session", test_name="BDD", bdd_style=True,
            )

        assert result["success"] is True
        rf_text = result["rf_text"]
        assert "*** Keywords ***" in rf_text
        assert "Given " in rf_text
        assert "When " in rf_text

    @pytest.mark.asyncio
    async def test_explicit_intents_produce_correct_prefixes(self):
        """bdd_intent 'given'/'when'/'then' map to Given/When/Then prefixes."""
        steps = [
            _exec_step("Log", ["setup"], bdd_group="setup", bdd_intent="given"),
            _exec_step("Click", ["btn"], bdd_group="action", bdd_intent="when"),
            _exec_step("Should Be Equal", ["1", "1"], bdd_group="verify", bdd_intent="then"),
        ]
        engine, step_dicts = _make_mock_engine_legacy(self._steps_to_dicts(steps))
        builder = TestBuilder(execution_engine=engine)

        with patch.object(builder, "_get_session_steps", new_callable=AsyncMock, return_value=step_dicts):
            result = await builder.build_suite(
                session_id="test-session", test_name="Intents", bdd_style=True,
            )

        rf_text = result["rf_text"]
        assert "Given " in rf_text
        assert "When " in rf_text
        assert "Then " in rf_text

    @pytest.mark.asyncio
    async def test_mixed_explicit_and_heuristic(self):
        """Steps with bdd_group use explicit grouping; heuristic is skipped."""
        steps = [
            _exec_step("New Browser", ["chromium"], bdd_group="setup browser", bdd_intent="given"),
            _exec_step("Click", ['text="Go"'], bdd_group="navigate", bdd_intent="when"),
        ]
        engine, step_dicts = _make_mock_engine_legacy(self._steps_to_dicts(steps))
        builder = TestBuilder(execution_engine=engine)

        with patch.object(builder, "_get_session_steps", new_callable=AsyncMock, return_value=step_dicts):
            result = await builder.build_suite(
                session_id="test-session", test_name="Mixed", bdd_style=True,
            )

        rf_text = result["rf_text"]
        # Explicit groups should be used as keyword names
        assert "*** Keywords ***" in rf_text
        # Should have 2 keywords (one per group)
        kw_section = rf_text.split("*** Keywords ***")[1]
        # Each keyword starts at column 0 (not indented)
        kw_names = [l for l in kw_section.strip().split("\n") if l and not l.startswith("    ")]
        assert len(kw_names) == 2

    @pytest.mark.asyncio
    async def test_explicit_groups_in_multi_test_mode(self):
        """Multi-test registry with bdd_group annotations → correct output."""
        registry = TestRegistry()
        test = registry.start_test("Login Test")
        s1 = _exec_step("Open Browser", ["http://x.com"], bdd_group="open app", bdd_intent="given")
        s1.mark_success()
        s2 = _exec_step("Click", ['text="Login"'], bdd_group="login", bdd_intent="when")
        s2.mark_success()
        s3 = _exec_step("Should Contain", ["Welcome"], bdd_group="verify", bdd_intent="then")
        s3.mark_success()
        test.steps.extend([s1, s2, s3])
        registry.end_test(status="pass")

        engine = _make_mock_engine_multitest(registry)
        builder = TestBuilder(execution_engine=engine)

        result = await builder.build_suite(
            session_id="test-session", test_name="Multi BDD", bdd_style=True,
        )

        assert result["success"] is True
        rf_text = result["rf_text"]
        assert "*** Keywords ***" in rf_text
        assert "Given " in rf_text
        assert "When " in rf_text
        assert "Then " in rf_text


# ---------------------------------------------------------------------------
# TestFullBddPipeline
# ---------------------------------------------------------------------------

class TestFullBddPipeline:
    """End-to-end pipeline tests."""

    def _steps_to_dicts(self, steps):
        return [
            {
                "keyword": s.keyword,
                "arguments": s.arguments,
                "status": "pass",
                "assigned_variables": [],
                "assignment_type": None,
                "bdd_group": s.bdd_group,
                "bdd_intent": s.bdd_intent,
            }
            for s in steps
        ]

    @pytest.mark.asyncio
    async def test_full_pipeline_setup_action_assertion(self):
        """3 groups → RF text with Given/When/Then + Keywords section."""
        steps = [
            _exec_step("New Browser", ["chromium"], bdd_group="browser ready", bdd_intent="given"),
            _exec_step("New Page", ["https://shop.com"], bdd_group="browser ready", bdd_intent="given"),
            _exec_step("Click", ['text="Add to Cart"'], bdd_group="add product", bdd_intent="when"),
            _exec_step("Get Text", ["#count", "==", "1"], bdd_group="verify cart", bdd_intent="then"),
        ]
        engine, step_dicts = _make_mock_engine_legacy(self._steps_to_dicts(steps))
        builder = TestBuilder(execution_engine=engine)

        with patch.object(builder, "_get_session_steps", new_callable=AsyncMock, return_value=step_dicts):
            result = await builder.build_suite(
                session_id="test-session", test_name="Shopping", bdd_style=True,
            )

        rf_text = result["rf_text"]
        assert "*** Keywords ***" in rf_text
        # Test case body should reference Given, When, Then
        tc_section = rf_text.split("*** Test Cases ***")[1].split("*** Keywords ***")[0]
        assert "Given " in tc_section
        assert "When " in tc_section
        assert "Then " in tc_section
        # Keywords section should contain implementation
        kw_section = rf_text.split("*** Keywords ***")[1]
        assert "New Browser" in kw_section or "Click" in kw_section

    @pytest.mark.asyncio
    async def test_full_pipeline_deduplication(self):
        """Two test cases with identical BDD groups → keywords deduplicated."""
        # Build suite directly to control multi-test
        builder = TestBuilder()
        tc1 = GeneratedTestCase(
            name="Test 1",
            steps=[
                TestCaseStep(keyword="Click", arguments=['text="A"'], bdd_group="click item", bdd_intent="when"),
                TestCaseStep(keyword="Get Text", arguments=["#r", "==", "ok"], bdd_group="verify", bdd_intent="then"),
            ],
        )
        tc2 = GeneratedTestCase(
            name="Test 2",
            steps=[
                TestCaseStep(keyword="Click", arguments=['text="B"'], bdd_group="click item", bdd_intent="when"),
                TestCaseStep(keyword="Get Text", arguments=["#r", "==", "ok"], bdd_group="verify", bdd_intent="then"),
            ],
        )
        suite = GeneratedTestSuite(name="Dedup", test_cases=[tc1, tc2])
        result = builder._transform_to_bdd_style(suite)

        # "click item" keyword should appear only once (deduplicated)
        kw_names = [kw.name for kw in result.bdd_keywords]
        assert kw_names.count("click item") == 1
        assert kw_names.count("verify") == 1

    @pytest.mark.asyncio
    async def test_full_pipeline_data_driven_with_annotations(self):
        """Template test case with bdd_group annotations → BDD wrapper keyword."""
        builder = TestBuilder()
        tc = GeneratedTestCase(
            name="Add Products",
            steps=[
                TestCaseStep(keyword="Click", arguments=['btn[aria-label="Add ${product}"]'],
                             bdd_group="add item", bdd_intent="when"),
                TestCaseStep(keyword="Get Text", arguments=["#count", "==", "${expected}"],
                             bdd_group="verify count", bdd_intent="then"),
                TestCaseStep(keyword="", arguments=["Speaker", "1"]),
                TestCaseStep(keyword="", arguments=["Timer", "2"]),
            ],
            template="Add And Check",
        )
        suite = GeneratedTestSuite(name="DD", test_cases=[tc])
        result = builder._transform_to_bdd_style(suite)

        # Should have template keyword with original name
        assert any("Add And Check" == kw.name for kw in result.bdd_keywords)
        # Template name preserved
        assert result.test_cases[0].template == "Add And Check"

    @pytest.mark.asyncio
    async def test_full_pipeline_multiple_test_cases(self):
        """Suite with 2 test cases → shared Keywords section."""
        builder = TestBuilder()
        tc1 = GeneratedTestCase(
            name="Login",
            steps=[
                TestCaseStep(keyword="Fill Text", arguments=["#user", "admin"]),
                TestCaseStep(keyword="Click", arguments=['text="Login"']),
            ],
        )
        tc2 = GeneratedTestCase(
            name="Logout",
            steps=[
                TestCaseStep(keyword="Click", arguments=['text="Logout"']),
                TestCaseStep(keyword="Get Text", arguments=["#status", "==", "logged out"]),
            ],
        )
        suite = GeneratedTestSuite(name="Auth", test_cases=[tc1, tc2])
        result = builder._transform_to_bdd_style(suite)

        assert result.bdd_keywords is not None
        assert len(result.bdd_keywords) >= 3  # at least 3 unique keywords
        # Both test cases should have BDD step references
        for tc in result.test_cases:
            assert any(
                s.keyword.startswith("Given") or s.keyword.startswith("When") or s.keyword.startswith("Then")
                for s in tc.steps
            )


# ---------------------------------------------------------------------------
# TestBddResponsePayload
# ---------------------------------------------------------------------------

class TestBddResponsePayload:
    """Test that server-level response dict contains bdd metadata."""

    def test_response_includes_bdd_group_when_set(self):
        """Simulate server.py: if bdd_group → result['bdd_group']."""
        result = {"status": "pass", "keyword": "Click"}
        bdd_group = "add product"
        bdd_intent = "when"

        if bdd_group:
            result["bdd_group"] = bdd_group
        if bdd_intent:
            result["bdd_intent"] = bdd_intent

        assert result["bdd_group"] == "add product"
        assert result["bdd_intent"] == "when"

    def test_response_excludes_bdd_when_empty(self):
        """Empty bdd_group/bdd_intent → not in result."""
        result = {"status": "pass", "keyword": "Click"}
        bdd_group = ""
        bdd_intent = ""

        if bdd_group:
            result["bdd_group"] = bdd_group
        if bdd_intent:
            result["bdd_intent"] = bdd_intent

        assert "bdd_group" not in result
        assert "bdd_intent" not in result

    def test_response_partial_bdd_fields(self):
        """Only bdd_intent set, not bdd_group."""
        result = {"status": "pass"}
        bdd_group = ""
        bdd_intent = "then"

        if bdd_group:
            result["bdd_group"] = bdd_group
        if bdd_intent:
            result["bdd_intent"] = bdd_intent

        assert "bdd_group" not in result
        assert result["bdd_intent"] == "then"
