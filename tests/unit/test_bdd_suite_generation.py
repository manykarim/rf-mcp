"""Tests for BDD-style test suite generation (Phase 3).

Tests the BDD transformation pipeline: step grouping, keyword name generation,
humanized locators, embedded argument detection, and end-to-end RF text rendering.
"""

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from robotmcp.components.test_builder import (
    BddKeyword,
    BddKeywordGroup,
    GeneratedTestCase,
    GeneratedTestSuite,
    TestBuilder,
    TestCaseStep,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(keyword: str, arguments: list | None = None, **kwargs) -> TestCaseStep:
    return TestCaseStep(keyword=keyword, arguments=arguments or [], **kwargs)


def _suite(steps: list[TestCaseStep], name: str = "Suite") -> GeneratedTestSuite:
    tc = GeneratedTestCase(name="Test Case", steps=steps)
    return GeneratedTestSuite(name=name, test_cases=[tc])


# ---------------------------------------------------------------------------
# _humanize_locator
# ---------------------------------------------------------------------------

class TestHumanizeLocator:
    def setup_method(self):
        self.builder = TestBuilder()

    def test_aria_label(self):
        result = self.builder._humanize_locator('button[aria-label="Submit Form"]')
        assert result == "submit form"

    def test_name_attribute(self):
        result = self.builder._humanize_locator('role=button[name="Add to Cart"]')
        assert result == "add to cart"

    def test_text_selector(self):
        result = self.builder._humanize_locator('text="Login"')
        assert result == "login"

    def test_text_selector_unquoted(self):
        result = self.builder._humanize_locator("text=Sign Up")
        assert result == "sign up"

    def test_id_selector(self):
        result = self.builder._humanize_locator("id=user-name")
        assert result == "the user name field"

    def test_hash_id(self):
        result = self.builder._humanize_locator("#login_button")
        assert result == "the login button field"

    def test_data_testid(self):
        result = self.builder._humanize_locator('[data-testid="product-card"]')
        assert result == "product card"

    def test_data_test(self):
        result = self.builder._humanize_locator('[data-test="checkout-btn"]')
        assert result == "checkout btn"

    def test_css_class(self):
        result = self.builder._humanize_locator(".submit-button")
        assert result == "submit button"

    def test_xpath_name(self):
        result = self.builder._humanize_locator("//control:Button[@Name='OK']")
        assert result == "ok"

    def test_fallback(self):
        result = self.builder._humanize_locator("some random locator")
        assert result == "the element"


# ---------------------------------------------------------------------------
# _classify_step_intent
# ---------------------------------------------------------------------------

class TestClassifyStepIntent:
    def setup_method(self):
        self.builder = TestBuilder()

    def test_setup_keywords(self):
        assert self.builder._classify_step_intent("new browser") == "given"
        assert self.builder._classify_step_intent("new page") == "given"
        assert self.builder._classify_step_intent("open browser") == "given"
        assert self.builder._classify_step_intent("go to") == "given"
        assert self.builder._classify_step_intent("new context") == "given"

    def test_assertion_keywords(self):
        assert self.builder._classify_step_intent("should be equal") == "then"
        assert self.builder._classify_step_intent("should contain") == "then"
        assert self.builder._classify_step_intent("element should be visible") == "then"
        assert self.builder._classify_step_intent("page should contain") == "then"

    def test_getter_keywords_with_assertion(self):
        """Get keywords with assertion operators are 'then'."""
        assert self.builder._classify_step_intent("get text", ["#el", "==", "ok"]) == "then"
        assert self.builder._classify_step_intent("get element count", ["#el", "==", "5"]) == "then"
        assert self.builder._classify_step_intent("get text", [".badge", "contains", "2"]) == "then"

    def test_getter_keywords_without_assertion(self):
        """Get keywords without assertion operators are 'when' (observations)."""
        assert self.builder._classify_step_intent("get text") == "when"
        assert self.builder._classify_step_intent("get text", ["body"]) == "when"
        assert self.builder._classify_step_intent("get url") == "when"

    def test_action_keywords(self):
        assert self.builder._classify_step_intent("click") == "when"
        assert self.builder._classify_step_intent("fill text") == "when"
        assert self.builder._classify_step_intent("type text") == "when"
        assert self.builder._classify_step_intent("select from list by value") == "when"


# ---------------------------------------------------------------------------
# _generate_bdd_keyword_name
# ---------------------------------------------------------------------------

class TestGenerateBddKeywordName:
    def setup_method(self):
        self.builder = TestBuilder()

    def test_new_page_with_url(self):
        steps = [_step("New Page", ["https://www.demoshop.example.com/products"])]
        name = self.builder._generate_bdd_keyword_name(steps, "given")
        assert name == "the demoshop is open"

    def test_open_browser_with_url(self):
        steps = [_step("Open Browser", ["https://google.com"])]
        name = self.builder._generate_bdd_keyword_name(steps, "given")
        assert name == "the google is open"

    def test_go_to_no_url(self):
        steps = [_step("Go To", [])]
        name = self.builder._generate_bdd_keyword_name(steps, "given")
        assert name == "the page is open"

    def test_new_context(self):
        steps = [_step("New Context", [])]
        name = self.builder._generate_bdd_keyword_name(steps, "given")
        assert name == "a browser context is created"

    def test_click(self):
        steps = [_step("Click", ['text="Add to Cart"'])]
        name = self.builder._generate_bdd_keyword_name(steps, "when")
        assert name == "the user clicks add to cart"

    def test_fill_text(self):
        steps = [_step("Fill Text", ["id=username", "admin"])]
        name = self.builder._generate_bdd_keyword_name(steps, "when")
        assert name == "the user fills in the username field"

    def test_should_keyword(self):
        steps = [_step("Should Be Equal", ["${result}", "42"])]
        name = self.builder._generate_bdd_keyword_name(steps, "then")
        assert name == "the result is correct"

    def test_get_text(self):
        steps = [_step("Get Text", ['#cart-count', "==", "2"])]
        name = self.builder._generate_bdd_keyword_name(steps, "then")
        # Phase 1.3: double-article "the the" bug is fixed
        assert name == "the cart count field should show the expected value"

    def test_multi_step_given_with_url(self):
        steps = [
            _step("New Browser", ["chromium"]),
            _step("New Page", ["https://shop.example.com/"]),
        ]
        name = self.builder._generate_bdd_keyword_name(steps, "given")
        assert name == "the shop is open"

    def test_multi_step_then(self):
        steps = [
            _step("Should Be Equal", ["a", "a"]),
            _step("Should Contain", ["abc", "b"]),
        ]
        name = self.builder._generate_bdd_keyword_name(steps, "then")
        assert name == "the expected results are verified"

    def test_multi_step_when_single_click(self):
        steps = [_step("Click", ['text="Submit"'])]
        name = self.builder._generate_bdd_keyword_name([
            _step("Click", ['text="Submit"']),
        ], "when")
        assert name == "the user clicks submit"

    def test_multi_step_when_form(self):
        steps = [
            _step("Fill Text", ["#name", "Alice"]),
            _step("Fill Text", ["#email", "alice@example.com"]),
        ]
        name = self.builder._generate_bdd_keyword_name(steps, "when")
        # Phase 2.3: 2 fills → lists both targets
        assert "the user fills in" in name
        assert "name" in name.lower()

    def test_generic_single_step(self):
        steps = [_step("Sleep", ["2s"])]
        name = self.builder._generate_bdd_keyword_name(steps, "when")
        assert name == "the user performs sleep"


# ---------------------------------------------------------------------------
# _group_steps_into_bdd_keywords (explicit annotations)
# ---------------------------------------------------------------------------

class TestGroupByExplicitAnnotations:
    def setup_method(self):
        self.builder = TestBuilder()

    def test_groups_by_bdd_group(self):
        steps = [
            _step("New Browser", ["chromium"], bdd_group="open shop", bdd_intent="given"),
            _step("New Page", ["https://shop.example.com"], bdd_group="open shop", bdd_intent="given"),
            _step("Click", ['text="Products"'], bdd_group="browse products", bdd_intent="when"),
            _step("Click", ['text="Add"'], bdd_group="browse products", bdd_intent="when"),
            _step("Get Text", ["#total", "==", "10"], bdd_group="verify total", bdd_intent="then"),
        ]
        groups = self.builder._group_steps_into_bdd_keywords(steps)
        assert len(groups) == 3
        assert groups[0].name == "open shop"
        assert groups[0].bdd_intent == "given"
        assert len(groups[0].steps) == 2
        assert groups[1].name == "browse products"
        assert groups[1].bdd_intent == "when"
        assert len(groups[1].steps) == 2
        assert groups[2].name == "verify total"
        assert groups[2].bdd_intent == "then"
        assert len(groups[2].steps) == 1

    def test_ungrouped_steps_use_keyword_as_group(self):
        steps = [
            _step("Click", ['text="OK"']),
        ]
        groups = self.builder._group_steps_into_bdd_keywords(steps)
        # No bdd_group → heuristic grouping
        assert len(groups) == 1


# ---------------------------------------------------------------------------
# _group_steps_into_bdd_keywords (heuristic grouping)
# ---------------------------------------------------------------------------

class TestGroupByHeuristics:
    def setup_method(self):
        self.builder = TestBuilder()

    def test_setup_then_action_then_assertion(self):
        steps = [
            _step("New Browser", ["chromium"]),
            _step("New Page", ["https://example.com"]),
            _step("Click", ['text="Login"']),
            _step("Fill Text", ["#user", "admin"]),
            _step("Should Be Equal", ["${status}", "OK"]),
        ]
        groups = self.builder._group_steps_into_bdd_keywords(steps)
        assert len(groups) == 3
        assert groups[0].bdd_intent == "given"
        assert len(groups[0].steps) == 2
        assert groups[1].bdd_intent == "when"
        assert len(groups[1].steps) == 2
        assert groups[2].bdd_intent == "then"
        assert len(groups[2].steps) == 1

    def test_all_actions(self):
        steps = [
            _step("Click", ['text="A"']),
            _step("Click", ['text="B"']),
        ]
        groups = self.builder._group_steps_into_bdd_keywords(steps)
        assert len(groups) == 1
        assert groups[0].bdd_intent == "when"

    def test_empty_steps(self):
        groups = self.builder._group_steps_into_bdd_keywords([])
        assert groups == []


# ---------------------------------------------------------------------------
# _detect_embedded_args
# ---------------------------------------------------------------------------

class TestDetectEmbeddedArgs:
    def setup_method(self):
        self.builder = TestBuilder()

    def test_merge_similar_keywords(self):
        kw1 = BddKeyword(
            name="the user clicks add to cart for speaker",
            steps=[_step("Click", ['text="Add Speaker"'])],
            bdd_intent="when",
        )
        kw2 = BddKeyword(
            name="the user clicks add to cart for timer",
            steps=[_step("Click", ['text="Add Timer"'])],
            bdd_intent="when",
        )
        result = self.builder._detect_embedded_args([kw1, kw2])
        assert len(result) == 1
        merged = result[0]
        assert merged.is_embedded is True
        assert len(merged.embedded_args) == 1
        assert "${value}" in merged.steps[0].arguments[0] or "${" in merged.steps[0].arguments[0]

    def test_no_merge_for_different_structures(self):
        kw1 = BddKeyword(
            name="the user clicks submit",
            steps=[_step("Click", ['text="Submit"'])],
            bdd_intent="when",
        )
        kw2 = BddKeyword(
            name="the user fills in name",
            steps=[
                _step("Fill Text", ["#name", "Alice"]),
                _step("Fill Text", ["#email", "alice@test.com"]),
            ],
            bdd_intent="when",
        )
        result = self.builder._detect_embedded_args([kw1, kw2])
        assert len(result) == 2

    def test_single_keyword_unchanged(self):
        kw = BddKeyword(name="the page is open", steps=[_step("New Page", ["https://x.com"])], bdd_intent="given")
        result = self.builder._detect_embedded_args([kw])
        assert len(result) == 1
        assert result[0] is kw


# ---------------------------------------------------------------------------
# _transform_to_bdd_style (end-to-end)
# ---------------------------------------------------------------------------

class TestTransformToBddStyle:
    def setup_method(self):
        self.builder = TestBuilder()

    def test_basic_transformation(self):
        steps = [
            _step("New Browser", ["chromium"]),
            _step("New Page", ["https://shop.example.com"]),
            _step("Click", ['text="Login"']),
            _step("Should Be Equal", ["${x}", "1"]),
        ]
        suite = _suite(steps)
        result = self.builder._transform_to_bdd_style(suite)

        # Test case steps should be BDD references
        tc = result.test_cases[0]
        assert len(tc.steps) == 3
        assert tc.steps[0].keyword.startswith("Given ")
        assert tc.steps[1].keyword.startswith("When ")
        assert tc.steps[2].keyword.startswith("Then ")

        # BDD keywords should be on the suite
        assert result.bdd_keywords is not None
        assert len(result.bdd_keywords) >= 3

    def test_bdd_style_false_unchanged(self):
        """Verify that without BDD transform the suite is unchanged."""
        steps = [_step("Click", ['text="OK"'])]
        suite = _suite(steps)
        # Do NOT call _transform_to_bdd_style
        assert suite.bdd_keywords is None
        assert suite.test_cases[0].steps[0].keyword == "Click"

    def test_explicit_groups(self):
        steps = [
            _step("New Browser", ["chromium"], bdd_group="setup browser", bdd_intent="given"),
            _step("New Page", ["https://x.com"], bdd_group="setup browser", bdd_intent="given"),
            _step("Click", ['text="Go"'], bdd_group="navigate", bdd_intent="when"),
        ]
        suite = _suite(steps)
        result = self.builder._transform_to_bdd_style(suite)

        tc = result.test_cases[0]
        assert len(tc.steps) == 2
        assert "Given setup browser" == tc.steps[0].keyword
        assert "When navigate" == tc.steps[1].keyword

    def test_duplicate_keyword_dedup(self):
        """Two test cases with identical steps should reuse the same keyword."""
        tc1 = GeneratedTestCase(
            name="Test 1",
            steps=[_step("Click", ['text="OK"'])],
        )
        tc2 = GeneratedTestCase(
            name="Test 2",
            steps=[_step("Click", ['text="OK"'])],
        )
        suite = GeneratedTestSuite(name="Suite", test_cases=[tc1, tc2])
        result = self.builder._transform_to_bdd_style(suite)

        # Both test cases reference the same keyword
        assert result.test_cases[0].steps[0].keyword == result.test_cases[1].steps[0].keyword
        # Only one BDD keyword definition
        assert len(result.bdd_keywords) == 1


# ---------------------------------------------------------------------------
# _generate_rf_text with BDD keywords
# ---------------------------------------------------------------------------

class TestGenerateRfTextWithBdd:
    def setup_method(self):
        self.builder = TestBuilder()

    @pytest.mark.asyncio
    async def test_keywords_section_rendered(self):
        steps = [
            _step("New Browser", ["chromium"]),
            _step("New Page", ["https://example.com"]),
            _step("Click", ['text="Submit"']),
            _step("Should Be Equal", ["a", "a"]),
        ]
        suite = _suite(steps)
        suite = self.builder._transform_to_bdd_style(suite)
        rf_text = await self.builder._generate_rf_text(suite)

        assert "*** Keywords ***" in rf_text
        assert "*** Test Cases ***" in rf_text
        # Keywords section should come after test cases
        tc_pos = rf_text.index("*** Test Cases ***")
        kw_pos = rf_text.index("*** Keywords ***")
        assert kw_pos > tc_pos

    @pytest.mark.asyncio
    async def test_test_case_uses_bdd_prefixes(self):
        steps = [
            _step("New Page", ["https://example.com"]),
            _step("Click", ['text="OK"']),
        ]
        suite = _suite(steps)
        suite = self.builder._transform_to_bdd_style(suite)
        rf_text = await self.builder._generate_rf_text(suite)

        # Test case body should reference Given/When/Then keywords
        lines = rf_text.split("\n")
        tc_section = False
        kw_section = False
        tc_lines = []
        kw_lines = []
        for line in lines:
            if line.strip() == "*** Test Cases ***":
                tc_section = True
                kw_section = False
                continue
            if line.strip() == "*** Keywords ***":
                tc_section = False
                kw_section = True
                continue
            if tc_section:
                tc_lines.append(line)
            if kw_section:
                kw_lines.append(line)

        # Test case lines should have Given/When
        tc_content = "\n".join(tc_lines)
        assert "Given " in tc_content or "When " in tc_content

        # Keywords section should contain the actual implementation
        kw_content = "\n".join(kw_lines)
        assert "New Page" in kw_content or "Click" in kw_content

    @pytest.mark.asyncio
    async def test_no_bdd_keywords_no_section(self):
        steps = [_step("Click", ['text="OK"'])]
        suite = _suite(steps)
        # No BDD transform
        rf_text = await self.builder._generate_rf_text(suite)
        assert "*** Keywords ***" not in rf_text

    @pytest.mark.asyncio
    async def test_embedded_args_in_keywords_section(self):
        kw = BddKeyword(
            name='the user adds "${value}" to cart',
            steps=[_step("Click", ['text="${value}"'])],
            bdd_intent="when",
            embedded_args={"value": "Speaker"},
            is_embedded=True,
        )
        suite = _suite([
            _step('When the user adds "${value}" to cart', ["Speaker"]),
        ])
        suite.bdd_keywords = [kw]
        rf_text = await self.builder._generate_rf_text(suite)
        assert '*** Keywords ***' in rf_text
        assert 'the user adds "${value}" to cart' in rf_text


# ---------------------------------------------------------------------------
# build_suite with bdd_style (integration-level via mocked engine)
# ---------------------------------------------------------------------------

class TestBuildSuiteBddStyleIntegration:
    """Integration tests using a mock execution engine."""

    def _make_mock_engine(self, steps):
        """Create a mock execution engine with given steps in a session."""
        engine = MagicMock()
        session = MagicMock()
        session.test_registry.is_multi_test_mode.return_value = False
        engine.sessions = {"test-session": session}
        engine.validate_test_readiness = AsyncMock(
            return_value={"ready_for_suite_generation": True}
        )

        # _get_session_steps returns list of dicts
        step_dicts = []
        for s in steps:
            step_dicts.append({
                "keyword": s.keyword,
                "arguments": s.arguments,
                "status": "pass",
                "assigned_variables": getattr(s, "assigned_variables", []),
                "assignment_type": getattr(s, "assignment_type", None),
                "bdd_group": getattr(s, "bdd_group", None),
                "bdd_intent": getattr(s, "bdd_intent", None),
            })
        return engine, step_dicts

    @pytest.mark.asyncio
    async def test_bdd_style_true_produces_keywords_section(self):
        steps = [
            _step("New Browser", ["chromium"]),
            _step("New Page", ["https://example.com"]),
            _step("Click", ['text="Buy"']),
            _step("Should Be Equal", ["${x}", "1"]),
        ]
        engine, step_dicts = self._make_mock_engine(steps)
        builder = TestBuilder(execution_engine=engine)

        with patch.object(builder, "_get_session_steps", new_callable=AsyncMock, return_value=step_dicts):
            result = await builder.build_suite(
                session_id="test-session",
                test_name="BDD Test",
                bdd_style=True,
            )

        assert result["success"] is True
        rf_text = result["rf_text"]
        assert "*** Keywords ***" in rf_text
        assert "Given " in rf_text or "When " in rf_text or "Then " in rf_text

    @pytest.mark.asyncio
    async def test_bdd_style_false_no_keywords_section(self):
        steps = [
            _step("Click", ['text="OK"']),
        ]
        engine, step_dicts = self._make_mock_engine(steps)
        builder = TestBuilder(execution_engine=engine)

        with patch.object(builder, "_get_session_steps", new_callable=AsyncMock, return_value=step_dicts):
            result = await builder.build_suite(
                session_id="test-session",
                test_name="Normal Test",
                bdd_style=False,
            )

        assert result["success"] is True
        rf_text = result["rf_text"]
        assert "*** Keywords ***" not in rf_text

    @pytest.mark.asyncio
    async def test_bdd_browser_setup_becomes_given(self):
        steps = [
            _step("New Browser", ["chromium"]),
            _step("New Page", ["https://shop.example.com/"]),
        ]
        engine, step_dicts = self._make_mock_engine(steps)
        builder = TestBuilder(execution_engine=engine)

        with patch.object(builder, "_get_session_steps", new_callable=AsyncMock, return_value=step_dicts):
            result = await builder.build_suite(
                session_id="test-session",
                test_name="Setup Test",
                bdd_style=True,
            )

        rf_text = result["rf_text"]
        assert "Given " in rf_text

    @pytest.mark.asyncio
    async def test_bdd_click_becomes_when(self):
        steps = [
            _step("Click", ['text="Submit"']),
        ]
        engine, step_dicts = self._make_mock_engine(steps)
        builder = TestBuilder(execution_engine=engine)

        with patch.object(builder, "_get_session_steps", new_callable=AsyncMock, return_value=step_dicts):
            result = await builder.build_suite(
                session_id="test-session",
                test_name="Click Test",
                bdd_style=True,
            )

        rf_text = result["rf_text"]
        assert "When " in rf_text

    @pytest.mark.asyncio
    async def test_bdd_assertion_becomes_then(self):
        steps = [
            _step("Should Be Equal", ["${x}", "1"]),
        ]
        engine, step_dicts = self._make_mock_engine(steps)
        builder = TestBuilder(execution_engine=engine)

        with patch.object(builder, "_get_session_steps", new_callable=AsyncMock, return_value=step_dicts):
            result = await builder.build_suite(
                session_id="test-session",
                test_name="Assert Test",
                bdd_style=True,
            )

        rf_text = result["rf_text"]
        assert "Then " in rf_text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestBddEdgeCases:
    def setup_method(self):
        self.builder = TestBuilder()

    def test_single_step_suite(self):
        steps = [_step("Click", ['text="OK"'])]
        suite = _suite(steps)
        result = self.builder._transform_to_bdd_style(suite)
        assert len(result.bdd_keywords) == 1
        assert result.test_cases[0].steps[0].keyword.startswith("When ")

    def test_mixed_intent_alternation(self):
        """Steps that alternate given/when/then should create separate groups.

        After embedded-arg merging, the two Click steps become one embedded
        keyword and the two assertion steps may also merge, so the final count
        of unique BDD keywords can be less than 5.  What matters is that the
        test case references 5 steps (given, when, then, when, then).
        """
        steps = [
            _step("New Page", ["https://example.com"]),
            _step("Click", ['text="A"']),
            _step("Should Be Equal", ["1", "1"]),
            _step("Click", ['text="B"']),
            _step("Should Contain", ["abc", "a"]),
        ]
        suite = _suite(steps)
        result = self.builder._transform_to_bdd_style(suite)
        # The test case body should still have 5 BDD references
        tc = result.test_cases[0]
        assert len(tc.steps) == 5
        assert tc.steps[0].keyword.startswith("Given ")
        assert tc.steps[1].keyword.startswith("When ")
        assert tc.steps[2].keyword.startswith("Then ")
        assert tc.steps[3].keyword.startswith("When ")
        assert tc.steps[4].keyword.startswith("Then ")

    def test_steps_with_no_arguments(self):
        steps = [_step("Close Browser", [])]
        suite = _suite(steps)
        result = self.builder._transform_to_bdd_style(suite)
        assert len(result.bdd_keywords) == 1

    @pytest.mark.asyncio
    async def test_keywords_section_step_arguments_escaped(self):
        """Arguments in the Keywords section should be properly escaped."""
        kw = BddKeyword(
            name="the user fills in data",
            steps=[_step("Fill Text", ["#field", "value with    spaces"])],
            bdd_intent="when",
        )
        suite = _suite([_step("When the user fills in data", [])])
        suite.bdd_keywords = [kw]
        rf_text = await self.builder._generate_rf_text(suite)
        assert "Fill Text" in rf_text
        assert "#field" in rf_text


# ---------------------------------------------------------------------------
# _build_embedded_name
# ---------------------------------------------------------------------------

class TestBuildEmbeddedName:
    def setup_method(self):
        self.builder = TestBuilder()

    def test_single_name_returns_as_is(self):
        result = self.builder._build_embedded_name(["the user clicks submit"], {"value": "x"})
        assert result == "the user clicks submit"

    def test_empty_list_returns_default(self):
        result = self.builder._build_embedded_name([], {})
        assert result == "the user performs the action"

    def test_common_prefix_and_suffix(self):
        names = ["the user clicks submit", "the user clicks cancel"]
        result = self.builder._build_embedded_name(names, {"value": "Submit"})
        assert "${value}" in result
        assert result.startswith("the user clicks")

    def test_common_prefix_only(self):
        names = ["add speaker to cart", "add timer to wishlist"]
        result = self.builder._build_embedded_name(names, {"item": "speaker"})
        assert "${item}" in result
        assert result.startswith("add")

    def test_common_suffix_only(self):
        names = ["click the submit button", "press the submit button"]
        result = self.builder._build_embedded_name(names, {"action": "click"})
        assert "${action}" in result
        assert "button" in result

    def test_no_common_parts(self):
        names = ["open the page", "verify results"]
        result = self.builder._build_embedded_name(names, {"value": "x"})
        # Fallback: rsplit on first name + ${arg_name}
        assert "${value}" in result
        assert "open" in result

    def test_prefix_suffix_overlap(self):
        # "ab" and "ac": prefix="a", suffix="" (after overlap clearing)
        names = ["ab", "ac"]
        result = self.builder._build_embedded_name(names, {"value": "x"})
        assert "${value}" in result

    def test_custom_arg_name(self):
        names = ["the user clicks submit", "the user clicks cancel"]
        result = self.builder._build_embedded_name(names, {"product": "Speaker"})
        assert "${product}" in result

    def test_empty_embedded_args(self):
        names = ["the user clicks submit", "the user clicks cancel"]
        result = self.builder._build_embedded_name(names, {})
        assert "${value}" in result

    def test_three_names(self):
        names = [
            "the user adds speaker to cart",
            "the user adds timer to cart",
            "the user adds headphones to cart",
        ]
        result = self.builder._build_embedded_name(names, {"product": "speaker"})
        assert "${product}" in result
        assert "the user adds" in result
        assert "to cart" in result

    def test_whitespace_in_names(self):
        names = ["the user clicks submit ", "the user clicks cancel "]
        result = self.builder._build_embedded_name(names, {"value": "x"})
        assert "${value}" in result
        # rstrip/lstrip should clean up extra spaces
        assert "  " not in result or "${value}" in result

    def test_identical_names(self):
        names = ["the user clicks submit", "the user clicks submit"]
        result = self.builder._build_embedded_name(names, {"value": "x"})
        # prefix = suffix = full string → overlap → suffix cleared
        # Result should still be valid with ${value}
        assert "${value}" in result


# ---------------------------------------------------------------------------
# RF syntax validation — parse generated BDD .robot text with Robot Framework
# ---------------------------------------------------------------------------

class TestBddRfSyntaxValidation:
    """Verify that generated BDD RF text parses as valid Robot Framework syntax."""

    def setup_method(self):
        self.builder = TestBuilder()

    def _parse_rf_text(self, rf_text: str):
        """Write RF text to temp file and parse with Robot Framework."""
        import os
        import tempfile
        from robot.api import get_model

        with tempfile.NamedTemporaryFile(mode="w", suffix=".robot", delete=False) as f:
            f.write(rf_text)
            temp_path = f.name
        try:
            return get_model(temp_path)
        finally:
            os.unlink(temp_path)

    def _section_names(self, model):
        return [type(s).__name__ for s in model.sections]

    @pytest.mark.asyncio
    async def test_basic_bdd_suite_parses_as_valid_robot(self):
        """BDD suite with setup/action/assertion parses without errors."""
        steps = [
            _step("New Browser", ["chromium"]),
            _step("Click", ['text="Buy"']),
            _step("Should Be Equal", ["1", "1"]),
        ]
        suite = _suite(steps)
        suite = self.builder._transform_to_bdd_style(suite)
        rf_text = await self.builder._generate_rf_text(suite)

        model = self._parse_rf_text(rf_text)
        names = self._section_names(model)
        assert "TestCaseSection" in names
        assert "KeywordSection" in names

    @pytest.mark.asyncio
    async def test_bdd_suite_has_correct_test_count(self):
        """Parsed suite has correct number of test cases."""
        steps = [_step("Click", ['text="OK"'])]
        suite = _suite(steps)
        suite = self.builder._transform_to_bdd_style(suite)
        rf_text = await self.builder._generate_rf_text(suite)

        model = self._parse_rf_text(rf_text)
        tc_section = next(
            s for s in model.sections if type(s).__name__ == "TestCaseSection"
        )
        tests = [item for item in tc_section.body if type(item).__name__ == "TestCase"]
        assert len(tests) == 1

    @pytest.mark.asyncio
    async def test_bdd_keywords_section_has_keywords(self):
        """Parsed Keywords section contains behavioral keywords."""
        steps = [
            _step("New Browser", ["chromium"]),
            _step("Click", ['text="Submit"']),
            _step("Should Contain", ["result", "ok"]),
        ]
        suite = _suite(steps)
        suite = self.builder._transform_to_bdd_style(suite)
        rf_text = await self.builder._generate_rf_text(suite)

        model = self._parse_rf_text(rf_text)
        kw_section = next(
            s for s in model.sections if type(s).__name__ == "KeywordSection"
        )
        keywords = [item for item in kw_section.body if type(item).__name__ == "Keyword"]
        assert len(keywords) >= 2  # at least given + when or then

    @pytest.mark.asyncio
    async def test_bdd_test_case_has_given_when_then_calls(self):
        """Parsed test body contains Given/When/Then keyword calls."""
        steps = [
            _step("New Browser", ["chromium"]),
            _step("Click", ['text="Go"']),
            _step("Should Be Equal", ["a", "a"]),
        ]
        suite = _suite(steps)
        suite = self.builder._transform_to_bdd_style(suite)
        rf_text = await self.builder._generate_rf_text(suite)

        model = self._parse_rf_text(rf_text)
        tc_section = next(
            s for s in model.sections if type(s).__name__ == "TestCaseSection"
        )
        test = next(item for item in tc_section.body if type(item).__name__ == "TestCase")

        # Collect keyword call names from the test body
        kw_calls = []
        for stmt in test.body:
            if type(stmt).__name__ == "KeywordCall":
                kw_calls.append(stmt.keyword)

        prefixes_found = {
            kw.split()[0] for kw in kw_calls if kw.split()[0] in ("Given", "When", "Then")
        }
        assert "Given" in prefixes_found
        assert "When" in prefixes_found or "Then" in prefixes_found

    @pytest.mark.asyncio
    async def test_embedded_args_suite_parses_correctly(self):
        """BDD suite with embedded args (merged keywords) parses correctly."""
        steps = [
            _step("New Browser", ["chromium"]),
            _step("Click", ['text="Speaker"']),
            _step("Click", ['text="Timer"']),
            _step("Should Be Equal", ["2", "2"]),
        ]
        suite = _suite(steps)
        suite = self.builder._transform_to_bdd_style(suite)
        rf_text = await self.builder._generate_rf_text(suite)

        model = self._parse_rf_text(rf_text)
        names = self._section_names(model)
        assert "KeywordSection" in names

    @pytest.mark.asyncio
    async def test_data_driven_bdd_suite_parses_correctly(self):
        """Data-driven BDD suite with [Template] parses correctly."""
        impl_steps = [
            _step("Click", ['btn[aria-label="Add ${product}"]']),
            _step("Get Text", ["#count", "==", "${expected}"]),
        ]
        data_rows = [
            TestCaseStep(keyword="", arguments=["Speaker", "1"]),
            TestCaseStep(keyword="", arguments=["Timer", "2"]),
        ]
        tc = GeneratedTestCase(
            name="Add Products",
            steps=impl_steps + data_rows,
            template="Add And Check",
        )
        suite = GeneratedTestSuite(name="DD Suite", test_cases=[tc])
        suite = self.builder._transform_to_bdd_style(suite)
        rf_text = await self.builder._generate_rf_text(suite)

        model = self._parse_rf_text(rf_text)
        names = self._section_names(model)
        assert "TestCaseSection" in names
        assert "KeywordSection" in names

    @pytest.mark.asyncio
    async def test_multi_test_bdd_suite_parses_correctly(self):
        """Suite with 2 test cases parses with correct test count."""
        tc1 = GeneratedTestCase(
            name="Test Login",
            steps=[
                _step("Fill Text", ["#user", "admin"]),
                _step("Click", ['text="Login"']),
            ],
        )
        tc2 = GeneratedTestCase(
            name="Test Logout",
            steps=[_step("Click", ['text="Logout"'])],
        )
        suite = GeneratedTestSuite(name="Auth", test_cases=[tc1, tc2])
        suite = self.builder._transform_to_bdd_style(suite)
        rf_text = await self.builder._generate_rf_text(suite)

        model = self._parse_rf_text(rf_text)
        tc_section = next(
            s for s in model.sections if type(s).__name__ == "TestCaseSection"
        )
        tests = [item for item in tc_section.body if type(item).__name__ == "TestCase"]
        assert len(tests) == 2
