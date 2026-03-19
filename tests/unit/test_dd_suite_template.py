"""Tests for data-driven Test Template / named-row support.

Validates the full workflow:
  start_test(template=...) → add_data_row(test_name=...) → build_test_suite(suite_template)
"""

import pytest
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

def _make_named_suite(
    template: str,
    impl_steps: list[TestCaseStep],
    named_rows: list[tuple[str, list[str]]],
) -> GeneratedTestSuite:
    """Build a suite with named data rows (simulates multi-test registry path)."""
    test_cases = []
    for row_name, row_values in named_rows:
        test_cases.append(GeneratedTestCase(
            name=row_name,
            steps=list(impl_steps) + [TestCaseStep(keyword="", arguments=list(row_values))],
            template=template,
        ))
    return GeneratedTestSuite(name="Suite", test_cases=test_cases)


# ---------------------------------------------------------------------------
# TestInfo named_data_rows
# ---------------------------------------------------------------------------

class TestNamedDataRows:
    def test_named_rows_stored_on_testinfo(self):
        test = TestInfo(name="Login", template="Verify Login")
        test.named_data_rows.append(("Valid", ["admin", "pass"]))
        test.named_data_rows.append(("Invalid", ["bad", "wrong"]))
        assert len(test.named_data_rows) == 2
        assert test.named_data_rows[0] == ("Valid", ["admin", "pass"])

    def test_named_rows_via_registry(self):
        registry = TestRegistry()
        test = registry.start_test("Login Tests", template="Verify Login")
        test.named_data_rows.append(("Valid", ["admin", "pass"]))
        test.named_data_rows.append(("Empty", ["${EMPTY}", "pass"]))
        registry.end_test()
        assert len(registry.tests["Login Tests"].named_data_rows) == 2

    def test_mixed_named_and_unnamed_rows(self):
        test = TestInfo(name="Mixed", template="Template")
        test.data_rows.append(["val1", "val2"])
        test.named_data_rows.append(("Named Row", ["val3", "val4"]))
        assert len(test.data_rows) == 1
        assert len(test.named_data_rows) == 1


# ---------------------------------------------------------------------------
# _apply_data_driven_mode
# ---------------------------------------------------------------------------

class TestApplyDataDrivenMode:
    def setup_method(self):
        self.builder = TestBuilder()

    def test_suite_template_promotion(self):
        """All TCs with same template → suite_template set."""
        suite = _make_named_suite("Verify Login", [], [
            ("Valid User", ["admin", "pass"]),
            ("Invalid User", ["bad", "wrong"]),
        ])
        self.builder._apply_data_driven_mode(suite, "suite_template")

        assert suite.suite_template == "Verify Login"
        assert all(tc.template is None for tc in suite.test_cases)

    def test_per_test_no_promotion(self):
        """per_test mode should not promote."""
        suite = _make_named_suite("Verify Login", [], [
            ("Valid", ["admin", "pass"]),
            ("Invalid", ["bad", "wrong"]),
        ])
        self.builder._apply_data_driven_mode(suite, "per_test")

        assert suite.suite_template is None
        assert all(tc.template == "Verify Login" for tc in suite.test_cases)

    def test_auto_promotes_with_named_rows(self):
        """Auto mode promotes when ≥2 TCs share template + ≤1 data row each."""
        suite = _make_named_suite("T", [], [
            ("Row 1", ["a"]),
            ("Row 2", ["b"]),
            ("Row 3", ["c"]),
        ])
        self.builder._apply_data_driven_mode(suite, "auto")
        assert suite.suite_template == "T"

    def test_auto_no_promote_single_tc(self):
        """Auto mode does NOT promote with only 1 TC."""
        suite = _make_named_suite("T", [], [("Only One", ["a"])])
        self.builder._apply_data_driven_mode(suite, "auto")
        assert suite.suite_template is None

    def test_column_headers_from_impl_steps(self):
        """Column headers extracted from ${var} in implementation steps."""
        impl = [
            TestCaseStep(keyword="Fill Text", arguments=["#user", "${username}"]),
            TestCaseStep(keyword="Fill Text", arguments=["#pass", "${password}"]),
        ]
        suite = _make_named_suite("Verify Login", impl, [
            ("Valid", ["admin", "pass"]),
            ("Invalid", ["bad", "wrong"]),
        ])
        self.builder._apply_data_driven_mode(suite, "suite_template")

        assert suite.column_headers == ["username", "password"]

    def test_column_headers_filter_builtins(self):
        """RF built-in vars (${EMPTY}) not included in column headers."""
        impl = [
            TestCaseStep(keyword="Fill Text", arguments=["#user", "${username}"]),
            TestCaseStep(keyword="Should Be Equal", arguments=["${EMPTY}", "${EMPTY}"]),
        ]
        suite = _make_named_suite("T", impl, [
            ("R1", ["a"]),
            ("R2", ["b"]),
        ])
        self.builder._apply_data_driven_mode(suite, "suite_template")
        assert suite.column_headers == ["username"]


# ---------------------------------------------------------------------------
# RF text rendering — suite_template mode
# ---------------------------------------------------------------------------

class TestSuiteTemplateRfText:
    def setup_method(self):
        self.builder = TestBuilder()

    @pytest.mark.asyncio
    async def test_test_template_in_settings(self):
        suite = _make_named_suite("Verify Login", [], [
            ("Valid", ["admin", "pass"]),
            ("Invalid", ["bad", "wrong"]),
        ])
        self.builder._apply_data_driven_mode(suite, "suite_template")
        rf = await self.builder._generate_rf_text(suite)

        assert "Test Template    Verify Login" in rf
        assert "[Template]" not in rf

    @pytest.mark.asyncio
    async def test_named_rows_on_same_line(self):
        suite = _make_named_suite("T", [], [
            ("Valid User", ["admin", "pass"]),
            ("Invalid User", ["bad", "wrong"]),
        ])
        self.builder._apply_data_driven_mode(suite, "suite_template")
        rf = await self.builder._generate_rf_text(suite)

        assert "Valid User    admin    pass" in rf
        assert "Invalid User    bad    wrong" in rf

    @pytest.mark.asyncio
    async def test_empty_values_as_rf_empty(self):
        suite = _make_named_suite("T", [], [
            ("Empty User", ["${EMPTY}", "pass"]),
        ])
        self.builder._apply_data_driven_mode(suite, "suite_template")
        rf = await self.builder._generate_rf_text(suite)

        assert "Empty User    ${EMPTY}    pass" in rf

    @pytest.mark.asyncio
    async def test_column_headers_in_test_cases_line(self):
        impl = [
            TestCaseStep(keyword="Fill Text", arguments=["#u", "${user}"]),
            TestCaseStep(keyword="Fill Text", arguments=["#p", "${pass}"]),
        ]
        suite = _make_named_suite("T", impl, [
            ("R1", ["a", "b"]),
            ("R2", ["c", "d"]),
        ])
        self.builder._apply_data_driven_mode(suite, "suite_template")
        rf = await self.builder._generate_rf_text(suite)

        assert "*** Test Cases ***    user    pass" in rf

    @pytest.mark.asyncio
    async def test_bdd_skips_suite_template_tcs(self):
        """BDD transformation should skip suite_template test cases."""
        suite = _make_named_suite("T", [], [
            ("R1", ["a"]),
            ("R2", ["b"]),
        ])
        self.builder._apply_data_driven_mode(suite, "suite_template")
        suite = self.builder._transform_to_bdd_style(suite)
        rf = await self.builder._generate_rf_text(suite)

        # Should have Test Template, no [Template], no Keywords section altering TCs
        assert "Test Template    T" in rf
        assert "R1    a" in rf
        assert "R2    b" in rf


# ---------------------------------------------------------------------------
# Full saucedemo workflow simulation
# ---------------------------------------------------------------------------

class TestSaucedemoWorkflow:
    """Reproduce the exact workflow from DATADRIVEN_SUITE_CREATION_REPORT."""

    def setup_method(self):
        self.builder = TestBuilder()

    @pytest.mark.asyncio
    async def test_full_saucedemo_dd_workflow(self):
        """Simulates: start_test → add_data_row(test_name) → end_test → build_suite."""
        registry = TestRegistry()
        test = registry.start_test(
            "SauceDemo Login Tests",
            template="Login With Credentials",
            documentation="Data-driven login tests",
            tags=["login", "saucedemo"],
        )

        # Simulate execute_step recording impl steps
        test.steps.extend([
            ExecutionStep(step_id="s1", keyword="Fill Text",
                          arguments=["id=user-name", "${username}"], status="pass"),
            ExecutionStep(step_id="s2", keyword="Fill Text",
                          arguments=["id=password", "${password}"], status="pass"),
            ExecutionStep(step_id="s3", keyword="Click",
                          arguments=["id=login-button"], status="pass"),
        ])

        # Simulate add_data_row with test_name
        test.named_data_rows.extend([
            ("Valid User Login", ["standard_user", "secret_sauce", "Products"]),
            ("Locked Out User", ["locked_out_user", "secret_sauce", "locked out"]),
            ("Invalid Credentials", ["invalid_user", "wrong_password", "Error"]),
            ("Empty Credentials", ["${EMPTY}", "${EMPTY}", "Required"]),
        ])

        registry.end_test()

        # Build test cases from registry (simulates build_suite multi-test path)
        test_cases = []
        for name, test_info in registry.tests.items():
            impl_steps = self.builder._build_impl_steps_from_test_info(test_info)
            for row_name, row_values in test_info.named_data_rows:
                test_cases.append(GeneratedTestCase(
                    name=row_name,
                    steps=impl_steps + [TestCaseStep(keyword="", arguments=list(row_values))],
                    template=test_info.template,
                ))

        suite = GeneratedTestSuite(name="SauceDemo", test_cases=test_cases)
        self.builder._apply_data_driven_mode(suite, "suite_template")

        rf = await self.builder._generate_rf_text(suite)

        # Verify Test Template in Settings
        assert "Test Template    Login With Credentials" in rf

        # Verify no per-test [Template]
        assert "[Template]" not in rf

        # Verify 4 named test cases
        assert "Valid User Login" in rf
        assert "Locked Out User" in rf
        assert "Invalid Credentials" in rf
        assert "Empty Credentials" in rf

        # Verify data values
        assert "standard_user    secret_sauce    Products" in rf
        assert "${EMPTY}    ${EMPTY}    Required" in rf

        # Verify column headers extracted from ${username}, ${password}
        assert "*** Test Cases ***    username    password" in rf
