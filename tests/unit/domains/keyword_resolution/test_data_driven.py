"""Tests for data-driven template support (ADR-019 Phase 3)."""
import pytest
from robotmcp.models.execution_models import TestInfo, TestRegistry


class TestTestInfoTemplate:
    def test_template_default_none(self):
        ti = TestInfo(name="Test")
        assert ti.template is None

    def test_template_set(self):
        ti = TestInfo(name="Test", template="Should Be Equal")
        assert ti.template == "Should Be Equal"


class TestTestRegistryTemplate:
    __test__ = True  # override pytest suppression from class name

    def test_start_test_with_template(self):
        reg = TestRegistry()
        test = reg.start_test(name="My Test", template="Log")
        assert test.template == "Log"

    def test_start_test_without_template(self):
        reg = TestRegistry()
        test = reg.start_test(name="My Test")
        assert test.template is None


class TestGeneratedTestCaseTemplate:
    def test_template_field(self):
        from robotmcp.components.test_builder import GeneratedTestCase, TestCaseStep
        tc = GeneratedTestCase(
            name="Test",
            steps=[TestCaseStep(keyword="Log", arguments=["hello"])],
            template="Log"
        )
        assert tc.template == "Log"

    def test_template_default_none(self):
        from robotmcp.components.test_builder import GeneratedTestCase, TestCaseStep
        tc = GeneratedTestCase(
            name="Test",
            steps=[TestCaseStep(keyword="Log", arguments=["hello"])]
        )
        assert tc.template is None


class TestGenerateRfTextTemplate:
    @pytest.mark.asyncio
    async def test_template_rendering(self):
        from robotmcp.components.test_builder import (
            TestBuilder, GeneratedTestSuite, GeneratedTestCase, TestCaseStep
        )
        builder = TestBuilder()
        suite = GeneratedTestSuite(
            name="Data Driven Suite",
            test_cases=[
                GeneratedTestCase(
                    name="Equality Checks",
                    template="Should Be Equal",
                    steps=[
                        # Data rows have empty keyword (C1 fix)
                        TestCaseStep(keyword="", arguments=["hello", "hello"]),
                        TestCaseStep(keyword="", arguments=["42", "42"]),
                    ],
                )
            ],
            imports=["BuiltIn"],
        )
        text = await builder._generate_rf_text(suite)
        assert "[Template]    Should Be Equal" in text
        assert "    hello    hello" in text
        assert "    42    42" in text
        # Should NOT have keyword name on data rows
        lines = text.split("\n")
        data_lines = [l for l in lines if l.strip().startswith("hello") or l.strip().startswith("42")]
        for line in data_lines:
            assert "Should Be Equal" not in line

    @pytest.mark.asyncio
    async def test_no_template_unchanged(self):
        from robotmcp.components.test_builder import (
            TestBuilder, GeneratedTestSuite, GeneratedTestCase, TestCaseStep
        )
        builder = TestBuilder()
        suite = GeneratedTestSuite(
            name="Normal Suite",
            test_cases=[
                GeneratedTestCase(
                    name="Normal Test",
                    steps=[
                        TestCaseStep(keyword="Log", arguments=["hello"]),
                    ],
                )
            ],
            imports=["BuiltIn"],
        )
        text = await builder._generate_rf_text(suite)
        assert "[Template]" not in text
        assert "Log    hello" in text

    @pytest.mark.asyncio
    async def test_template_with_tags(self):
        from robotmcp.components.test_builder import (
            TestBuilder, GeneratedTestSuite, GeneratedTestCase, TestCaseStep
        )
        builder = TestBuilder()
        suite = GeneratedTestSuite(
            name="Suite",
            test_cases=[
                GeneratedTestCase(
                    name="Tagged Template",
                    template="Should Be Equal",
                    tags=["smoke", "regression"],
                    steps=[
                        TestCaseStep(keyword="Should Be Equal", arguments=["a", "a"]),
                    ],
                )
            ],
            imports=["BuiltIn"],
        )
        text = await builder._generate_rf_text(suite)
        assert "[Tags]" in text
        assert "[Template]    Should Be Equal" in text

    @pytest.mark.asyncio
    async def test_mixed_suite(self):
        """One test with template, one without."""
        from robotmcp.components.test_builder import (
            TestBuilder, GeneratedTestSuite, GeneratedTestCase, TestCaseStep
        )
        builder = TestBuilder()
        suite = GeneratedTestSuite(
            name="Mixed",
            test_cases=[
                GeneratedTestCase(
                    name="Template Test",
                    template="Should Be Equal",
                    steps=[
                        TestCaseStep(keyword="Should Be Equal", arguments=["a", "a"]),
                    ],
                ),
                GeneratedTestCase(
                    name="Normal Test",
                    steps=[
                        TestCaseStep(keyword="Log", arguments=["hello"]),
                    ],
                ),
            ],
            imports=["BuiltIn"],
        )
        text = await builder._generate_rf_text(suite)
        assert "[Template]    Should Be Equal" in text
        assert "Log    hello" in text

    @pytest.mark.asyncio
    async def test_create_rf_suite_sets_template(self):
        from robotmcp.components.test_builder import (
            TestBuilder, GeneratedTestSuite, GeneratedTestCase, TestCaseStep
        )
        builder = TestBuilder()
        suite = GeneratedTestSuite(
            name="Suite",
            test_cases=[
                GeneratedTestCase(
                    name="Template Test",
                    template="Should Be Equal",
                    steps=[
                        TestCaseStep(keyword="Should Be Equal", arguments=["a", "a"]),
                    ],
                )
            ],
            imports=["BuiltIn"],
        )
        rf_suite = await builder._create_rf_suite(suite)
        assert rf_suite.tests[0].template == "Should Be Equal"

    @pytest.mark.asyncio
    async def test_create_rf_suite_no_template(self):
        from robotmcp.components.test_builder import (
            TestBuilder, GeneratedTestSuite, GeneratedTestCase, TestCaseStep
        )
        builder = TestBuilder()
        suite = GeneratedTestSuite(
            name="Suite",
            test_cases=[
                GeneratedTestCase(
                    name="Normal Test",
                    steps=[
                        TestCaseStep(keyword="Log", arguments=["hello"]),
                    ],
                )
            ],
            imports=["BuiltIn"],
        )
        rf_suite = await builder._create_rf_suite(suite)
        assert rf_suite.tests[0].template is None


class TestBuildTestCaseFromTestInfo:
    def test_copies_template(self):
        from robotmcp.components.test_builder import TestBuilder
        from robotmcp.models.execution_models import TestInfo, ExecutionStep

        builder = TestBuilder()
        test_info = TestInfo(
            name="Test",
            template="Should Be Equal",
            steps=[
                ExecutionStep(step_id=1, keyword="Should Be Equal", arguments=["a", "a"], status="pass"),
            ],
        )
        generated = builder._build_test_case_from_test_info(test_info)
        assert generated.template == "Should Be Equal"

    def test_no_template(self):
        from robotmcp.components.test_builder import TestBuilder
        from robotmcp.models.execution_models import TestInfo, ExecutionStep

        builder = TestBuilder()
        test_info = TestInfo(
            name="Test",
            steps=[
                ExecutionStep(step_id=1, keyword="Log", arguments=["hello"], status="pass"),
            ],
        )
        generated = builder._build_test_case_from_test_info(test_info)
        assert generated.template is None
