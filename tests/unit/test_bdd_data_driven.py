"""Tests for Phase 4: BDD + Data-Driven Template Combination.

Verifies that data-driven (template) test cases are correctly transformed
into BDD style with a wrapper template keyword containing Given/When/Then
steps and a Keywords section.
"""

import pytest
from robotmcp.components.test_builder import (
    BddKeyword,
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


def _data_row(arguments: list) -> TestCaseStep:
    """A data-row step: no keyword, only arguments."""
    return TestCaseStep(keyword="", arguments=arguments)


def _template_suite(
    template_name: str,
    impl_steps: list[TestCaseStep],
    data_rows: list[TestCaseStep],
    test_name: str = "Add Product To Cart",
) -> GeneratedTestSuite:
    """Build a data-driven test suite with template steps and data rows."""
    # Mix implementation steps (keyword-bearing) with data rows
    tc = GeneratedTestCase(
        name=test_name,
        steps=impl_steps,  # Template body steps
        template=template_name,
    )
    return GeneratedTestSuite(name="Suite", test_cases=[tc])


# ---------------------------------------------------------------------------
# _transform_template_to_bdd
# ---------------------------------------------------------------------------

class TestTransformTemplateToBdd:
    def setup_method(self):
        self.builder = TestBuilder()

    def test_template_tc_creates_bdd_keyword(self):
        """Template test case should get a BDD keyword with original template name."""
        steps = [
            _step("Click", ['button[aria-label="Add ${product} to cart"]']),
            _step("Get Text", ["[data-cart-count]", "==", "${expected}"]),
        ]
        suite = _template_suite("Add And Verify", steps, [])
        result = self.builder._transform_to_bdd_style(suite)

        # Template keyword created with original name
        assert result.bdd_keywords is not None
        kw_names = [kw.name for kw in result.bdd_keywords]
        assert "Add And Verify" in kw_names

        # Template name preserved
        assert result.test_cases[0].template == "Add And Verify"

    def test_template_keyword_has_bdd_steps(self):
        """Template keyword should contain Given/When/Then-prefixed steps."""
        steps = [
            _step("New Browser", ["chromium"]),
            _step("Click", ['text="Buy"']),
            _step("Get Text", ["#result", "==", "OK"]),
        ]
        suite = _template_suite("Buy Flow", steps, [])
        result = self.builder._transform_to_bdd_style(suite)

        template_kw = next(kw for kw in result.bdd_keywords if kw.name == "Buy Flow")
        step_keywords = [s.keyword for s in template_kw.steps]
        assert any(s.startswith("Given") for s in step_keywords)
        assert any(s.startswith("When") for s in step_keywords)
        assert any(s.startswith("Then") for s in step_keywords)

    def test_template_keyword_has_arguments(self):
        """Template keyword should have arg_vars with ${var} from template steps."""
        steps = [
            _step("Click", ['button[aria-label="Add ${product} to cart"]']),
            _step("Get Text", ["[data-cart-count]", "==", "${expected_count}"]),
        ]
        suite = _template_suite("Verify Product", steps, [])
        result = self.builder._transform_to_bdd_style(suite)

        template_kw = next(kw for kw in result.bdd_keywords if kw.name == "Verify Product")
        assert "${product}" in template_kw.arg_vars
        assert "${expected_count}" in template_kw.arg_vars

    def test_data_rows_preserved(self):
        """Data rows (no keyword) should be preserved in the test case."""
        impl_steps = [
            _step("Click", ['button[aria-label="Add ${product} to cart"]']),
        ]
        data_rows = [
            _data_row(["Echo Speaker", "1"]),
            _data_row(["Loop Timer", "2"]),
        ]
        # Test case has both impl steps and data rows
        tc = GeneratedTestCase(
            name="Add Products",
            steps=impl_steps + data_rows,
            template="Add Product",
        )
        suite = GeneratedTestSuite(name="Suite", test_cases=[tc])
        result = self.builder._transform_to_bdd_style(suite)

        # Data rows should remain
        remaining_steps = result.test_cases[0].steps
        data_step_args = [s.arguments for s in remaining_steps if not (s.keyword and s.keyword.strip())]
        assert ["Echo Speaker", "1"] in data_step_args
        assert ["Loop Timer", "2"] in data_step_args

    def test_non_template_tc_unchanged(self):
        """Non-template test cases should follow normal BDD transformation."""
        steps = [
            _step("Click", ['text="Login"']),
            _step("Get Text", ["#result", "==", "OK"]),
        ]
        tc = GeneratedTestCase(name="Login Test", steps=steps)
        suite = GeneratedTestSuite(name="Suite", test_cases=[tc])
        result = self.builder._transform_to_bdd_style(suite)

        # Should have BDD keywords but no template wrapper
        assert result.bdd_keywords is not None
        assert all("Verify" not in kw.name for kw in result.bdd_keywords)
        # Test case steps should be BDD references
        assert any(s.keyword.startswith("When") or s.keyword.startswith("Then")
                    for s in result.test_cases[0].steps)

    def test_template_no_impl_steps_unchanged(self):
        """Template test case with only data rows should not crash."""
        data_rows = [
            _data_row(["Echo Speaker", "1"]),
            _data_row(["Loop Timer", "2"]),
        ]
        tc = GeneratedTestCase(
            name="Add Products",
            steps=data_rows,
            template="External Keyword",
        )
        suite = GeneratedTestSuite(name="Suite", test_cases=[tc])
        result = self.builder._transform_to_bdd_style(suite)

        # Template unchanged (no impl steps to transform)
        assert result.test_cases[0].template == "External Keyword"

    def test_mixed_template_and_regular(self):
        """Suite with both template and regular test cases."""
        template_tc = GeneratedTestCase(
            name="Data Test",
            steps=[
                _step("Click", ['text="${item}"']),
                _step("Get Text", ["#count", "==", "${n}"]),
            ],
            template="Check Item",
        )
        regular_tc = GeneratedTestCase(
            name="Smoke Test",
            steps=[
                _step("New Browser", ["chromium"]),
                _step("Click", ['text="Home"']),
            ],
        )
        suite = GeneratedTestSuite(
            name="Suite", test_cases=[template_tc, regular_tc]
        )
        result = self.builder._transform_to_bdd_style(suite)

        assert result.bdd_keywords is not None
        names = [kw.name for kw in result.bdd_keywords]
        # Template keyword with original name
        assert "Check Item" in names
        # Regular BDD keywords from smoke test
        assert len(names) >= 2


# ---------------------------------------------------------------------------
# _extract_template_arg_vars
# ---------------------------------------------------------------------------

class TestExtractTemplateArgVars:
    def setup_method(self):
        self.builder = TestBuilder()

    def test_extracts_vars_from_arguments(self):
        steps = [
            _step("Click", ['button[aria-label="Add ${product} to cart"]']),
            _step("Get Text", ["[data-cart-count]", "==", "${expected}"]),
        ]
        result = self.builder._extract_template_arg_vars(steps)
        assert "${product}" in result
        assert "${expected}" in result

    def test_no_duplicates(self):
        steps = [
            _step("Click", ['${locator}']),
            _step("Get Text", ['${locator}', "==", "ok"]),
        ]
        result = self.builder._extract_template_arg_vars(steps)
        assert result.count("${locator}") == 1

    def test_no_vars(self):
        steps = [_step("Click", ["#button"])]
        result = self.builder._extract_template_arg_vars(steps)
        assert result == []

    def test_preserves_order(self):
        steps = [
            _step("Fill Text", ["#name", "${name}"]),
            _step("Fill Text", ["#email", "${email}"]),
            _step("Click", ['text="${action}"']),
        ]
        result = self.builder._extract_template_arg_vars(steps)
        assert result == ["${name}", "${email}", "${action}"]


# ---------------------------------------------------------------------------
# RF Text rendering — data-driven BDD
# ---------------------------------------------------------------------------

class TestBddDataDrivenRfText:
    def setup_method(self):
        self.builder = TestBuilder()

    @pytest.mark.asyncio
    async def test_template_bdd_rf_text_has_arguments_line(self):
        """Generated RF text should have [Arguments] on wrapper keyword."""
        steps = [
            _step("Click", ['button[aria-label="Add ${product} to cart"]']),
            _step("Get Text", ["[data-cart-count]", "==", "${expected}"]),
        ]
        suite = _template_suite("Add And Verify", steps, [])
        suite = self.builder._transform_to_bdd_style(suite)
        rf_text = await self.builder._generate_rf_text(suite)

        assert "[Arguments]" in rf_text
        assert "${product}" in rf_text
        assert "${expected}" in rf_text

    @pytest.mark.asyncio
    async def test_template_bdd_rf_text_has_keywords_section(self):
        """Generated RF text should have *** Keywords *** section."""
        steps = [
            _step("New Browser", ["chromium"]),
            _step("Click", ['text="Buy"']),
            _step("Get Text", ["#result", "==", "OK"]),
        ]
        suite = _template_suite("Buy", steps, [])
        suite = self.builder._transform_to_bdd_style(suite)
        rf_text = await self.builder._generate_rf_text(suite)

        assert "*** Keywords ***" in rf_text
        # Template keyword uses original name
        assert "Buy" in rf_text

    @pytest.mark.asyncio
    async def test_template_bdd_rf_text_has_template_directive(self):
        """Test case should have [Template] pointing to template keyword."""
        steps = [
            _step("Click", ['text="${item}"']),
        ]
        suite = _template_suite("Check Item", steps, [])
        suite = self.builder._transform_to_bdd_style(suite)
        rf_text = await self.builder._generate_rf_text(suite)

        assert "[Template]    Check Item" in rf_text

    @pytest.mark.asyncio
    async def test_complete_data_driven_bdd_suite(self):
        """End-to-end: data-driven BDD suite with data rows."""
        impl_steps = [
            _step("Click", ['button[aria-label="Add ${product} to cart"]']),
            _step("Get Text", ["[data-cart-count]", "==", "${count}"]),
        ]
        data_rows = [
            _data_row(["Echo Speaker", "1"]),
            _data_row(["Loop Timer", "2"]),
        ]
        tc = GeneratedTestCase(
            name="Add Products",
            steps=impl_steps + data_rows,
            template="Add And Check",
        )
        suite = GeneratedTestSuite(name="DemoShop", test_cases=[tc])
        suite = self.builder._transform_to_bdd_style(suite)
        rf_text = await self.builder._generate_rf_text(suite)

        # Template keeps original name
        assert "[Template]    Add And Check" in rf_text
        # Should have Keywords section
        assert "*** Keywords ***" in rf_text
        # Should have [Arguments]
        assert "[Arguments]" in rf_text
        # Should have BDD-prefixed steps (When/Then)
        assert "When" in rf_text or "Then" in rf_text
        # Data rows should be present
        assert "Echo Speaker" in rf_text
        assert "Loop Timer" in rf_text
