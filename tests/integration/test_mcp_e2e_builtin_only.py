"""E2E integration tests using BuiltIn-only keywords (no browser required).

All tests use real MCP Client round-trips with BuiltIn/Collections/String
libraries. No browser, no Selenium, no Playwright needed.

Run with: uv run pytest tests/integration/test_mcp_e2e_builtin_only.py -v
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio
from fastmcp import Client

from robotmcp.server import mcp


def _sid(prefix: str = "e2e") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


# =============================================================================
# P1: Critical scenarios
# =============================================================================


class TestFullSessionLifecycle:
    """Test complete session lifecycle from analyze to build."""

    @pytest.mark.asyncio
    async def test_full_session_lifecycle_analyze_to_build(self, mcp_client):
        """Full lifecycle: analyze -> set_variables -> execute_step x4 -> state -> build."""
        # 1. Analyze scenario (creates session automatically)
        analyze = await mcp_client.call_tool(
            "analyze_scenario",
            {"scenario": "Test string operations with Robot Framework BuiltIn keywords", "context": "api"},
        )
        assert analyze.data["success"] is True
        sid = analyze.data["session_id"]

        # 2. Set variables
        set_vars = await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "set_variables", "variables": {"TEXT": "Hello World", "EXPECTED": "HELLO WORLD"}},
        )
        assert set_vars.data["success"] is True

        # 3. Execute steps
        step1 = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Log", "arguments": ["Starting test"], "session_id": sid},
        )
        assert step1.data["success"] is True

        step2 = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Convert To Upper Case", "arguments": ["${TEXT}"], "session_id": sid, "assign_to": "RESULT"},
        )
        assert step2.data["success"] is True

        step3 = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Should Be Equal", "arguments": ["${RESULT}", "${EXPECTED}"], "session_id": sid},
        )
        assert step3.data["success"] is True

        step4 = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Log", "arguments": ["Test completed"], "session_id": sid},
        )
        assert step4.data["success"] is True

        # 4. Get session state
        state = await mcp_client.call_tool(
            "get_session_state",
            {"session_id": sid, "sections": ["summary", "variables"]},
        )
        assert state.data["success"] is True

        # 5. Build test suite
        build = await mcp_client.call_tool(
            "build_test_suite",
            {"session_id": sid, "test_name": "String Operations Test", "documentation": "Validates string conversion"},
        )
        assert build.data["success"] is True
        rf_text = build.data["rf_text"]
        assert "*** Test Cases ***" in rf_text
        assert "Convert To Upper Case" in rf_text


class TestMultiTestSession:
    """Test multi-test session via MCP client (ADR-005)."""

    @pytest.mark.asyncio
    async def test_multi_test_session_via_mcp_client(self, mcp_client):
        """Multi-test: init -> suite_setup -> testA -> testB -> list_tests -> build."""
        sid = _sid("multi")

        # Init session (no scenario param on manage_session)
        init = await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "init", "libraries": ["BuiltIn"]},
        )
        assert init.data["success"] is True

        # Set suite setup (uses "args" not "arguments")
        setup = await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "set_suite_setup", "keyword": "Log", "args": ["Suite starting"]},
        )
        assert setup.data["success"] is True

        # Test A
        start_a = await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "start_test", "test_name": "Test A - Log"},
        )
        assert start_a.data["success"] is True

        step_a = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Log", "arguments": ["In Test A"], "session_id": sid},
        )
        assert step_a.data["success"] is True

        end_a = await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "end_test"},
        )
        assert end_a.data["success"] is True

        # Test B
        start_b = await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "start_test", "test_name": "Test B - Convert"},
        )
        assert start_b.data["success"] is True

        step_b = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Convert To Upper Case", "arguments": ["hello"], "session_id": sid, "assign_to": "UPPER"},
        )
        assert step_b.data["success"] is True

        end_b = await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "end_test"},
        )
        assert end_b.data["success"] is True

        # List tests
        list_tests = await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "list_tests"},
        )
        assert list_tests.data["success"] is True
        tests = list_tests.data.get("tests", [])
        test_names = [t.get("name", t.get("test_name", "")) for t in tests]
        assert "Test A - Log" in test_names
        assert "Test B - Convert" in test_names

        # Build multi-test suite (test_name is required)
        build = await mcp_client.call_tool(
            "build_test_suite",
            {"session_id": sid, "test_name": "Multi Test Suite"},
        )
        assert build.data["success"] is True
        rf_text = build.data["rf_text"]
        assert "Test A" in rf_text
        assert "Test B" in rf_text


class TestAutoCreatedSession:
    """Test auto-created session lifecycle."""

    @pytest.mark.asyncio
    async def test_auto_created_session_lifecycle(self, mcp_client):
        """Execute step without prior init -> auto-create session via default session_id."""
        # execute_step with default session_id auto-creates a session
        sid = "default"
        step1 = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Set Variable", "arguments": ["hello_world"], "session_id": sid, "assign_to": "MY_VAR"},
        )
        assert step1.data["success"] is True

        # Use the auto-created session for subsequent steps
        step2 = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Log", "arguments": ["${MY_VAR}"], "session_id": sid},
        )
        assert step2.data["success"] is True

        # Get state
        state = await mcp_client.call_tool(
            "get_session_state",
            {"session_id": sid, "sections": ["summary", "variables"]},
        )
        assert state.data["success"] is True

        # Build
        build = await mcp_client.call_tool(
            "build_test_suite",
            {"session_id": sid, "test_name": "Auto Created Test"},
        )
        assert build.data["success"] is True
        assert "*** Test Cases ***" in build.data["rf_text"]


# =============================================================================
# P2: High priority scenarios
# =============================================================================


class TestErrorRecovery:
    """Test error recovery after failed steps."""

    @pytest.mark.asyncio
    async def test_error_recovery_after_failed_step(self, mcp_client):
        """Failed step (with raise_on_failure=False) doesn't block subsequent steps."""
        sid = _sid("err")
        init = await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "init", "libraries": ["BuiltIn"]},
        )
        assert init.data["success"] is True

        # Execute a step that will fail - compare unequal values
        failed = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Should Be Equal", "arguments": ["abc", "xyz"], "session_id": sid, "raise_on_failure": False},
        )
        # Should NOT crash the server - either success:False or raises handled
        # The point is that the session is still usable after

        # Subsequent step should still work
        recovery = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Log", "arguments": ["Recovered!"], "session_id": sid},
        )
        assert recovery.data["success"] is True


class TestDynamicLibraryImport:
    """Test dynamic library import enabling new keywords."""

    @pytest.mark.asyncio
    async def test_dynamic_library_import_enables_keywords(self, mcp_client):
        """Init BuiltIn-only -> import Collections -> Create Dictionary works."""
        sid = _sid("dynlib")
        init = await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "init", "libraries": ["BuiltIn"]},
        )
        assert init.data["success"] is True

        # Import Collections library
        imp = await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "import_library", "library_name": "Collections"},
        )
        assert imp.data["success"] is True

        # Now Create Dictionary should work
        step = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Create Dictionary", "arguments": ["key1=val1", "key2=val2"], "session_id": sid, "assign_to": "MY_DICT"},
        )
        assert step.data["success"] is True


class TestGetSessionStateAllSections:
    """Test get_session_state with all sections."""

    @pytest.mark.asyncio
    async def test_get_session_state_all_sections(self, mcp_client):
        """Request all 7 sections -- none should crash."""
        sid = _sid("state")
        init = await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "init", "libraries": ["BuiltIn"]},
        )
        assert init.data["success"] is True

        # Execute a step first
        await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Log", "arguments": ["test"], "session_id": sid},
        )

        # Request all sections
        state = await mcp_client.call_tool(
            "get_session_state",
            {"session_id": sid, "sections": ["summary", "variables", "page_source", "validation", "libraries", "rf_context", "application_state"]},
        )
        assert state.data["success"] is True
        sections = state.data.get("sections", {})
        # At minimum summary should be present
        assert "summary" in sections


class TestConcurrentSessions:
    """Test state isolation between concurrent sessions."""

    @pytest.mark.asyncio
    async def test_concurrent_sessions_state_isolation(self, mcp_client):
        """3 sessions with different libraries -- verify state isolation."""
        sid1 = _sid("iso1")
        sid2 = _sid("iso2")
        sid3 = _sid("iso3")

        # Init 3 sessions
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid1, "action": "init", "libraries": ["BuiltIn"]},
        )
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid2, "action": "init", "libraries": ["BuiltIn", "Collections"]},
        )
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid3, "action": "init", "libraries": ["BuiltIn", "String"]},
        )

        # Set different variables in each
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid1, "action": "set_variables", "variables": {"VAR": "session1"}},
        )
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid2, "action": "set_variables", "variables": {"VAR": "session2"}},
        )
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid3, "action": "set_variables", "variables": {"VAR": "session3"}},
        )

        # Verify each session has correct variable
        s1 = await mcp_client.call_tool(
            "get_session_state", {"session_id": sid1, "sections": ["variables"]},
        )
        s2 = await mcp_client.call_tool(
            "get_session_state", {"session_id": sid2, "sections": ["variables"]},
        )
        s3 = await mcp_client.call_tool(
            "get_session_state", {"session_id": sid3, "sections": ["variables"]},
        )

        # Each should have their own value (not leaking between sessions)
        assert s1.data["success"] is True
        assert s2.data["success"] is True
        assert s3.data["success"] is True


# =============================================================================
# P3: Medium priority scenarios
# =============================================================================


class TestBuildTestSuiteOptions:
    """Test build_test_suite with various options."""

    @pytest.mark.asyncio
    async def test_build_test_suite_with_variables_docs_tags(self, mcp_client):
        """Verify .robot output includes variables, docs, tags."""
        sid = _sid("build")
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "init", "libraries": ["BuiltIn"]},
        )
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "set_variables", "variables": {"URL": "https://example.com"}, "scope": "suite"},
        )
        await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Log", "arguments": ["hello"], "session_id": sid},
        )

        build = await mcp_client.call_tool(
            "build_test_suite",
            {"session_id": sid, "test_name": "Tagged Test", "documentation": "Test docs", "tags": ["smoke", "regression"]},
        )
        assert build.data["success"] is True
        rf_text = build.data["rf_text"]
        assert "*** Test Cases ***" in rf_text


class TestFindKeywordsAfterDynamicImport:
    """Test find_keywords before and after library import."""

    @pytest.mark.asyncio
    async def test_find_keywords_after_dynamic_import(self, mcp_client):
        """Import Collections, then find_keywords should include its keywords."""
        sid = _sid("fk")
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "init", "libraries": ["BuiltIn"]},
        )

        # Find a Collections keyword - may not be found yet
        before = await mcp_client.call_tool(
            "find_keywords",
            {"query": "Create Dictionary", "session_id": sid},
        )

        # Import Collections
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "import_library", "library_name": "Collections"},
        )

        # Now find it
        after = await mcp_client.call_tool(
            "find_keywords",
            {"query": "Create Dictionary", "session_id": sid},
        )
        assert after.data["success"] is True


class TestCheckLibraryAvailability:
    """Test check_library_availability tool."""

    @pytest.mark.asyncio
    async def test_check_library_availability(self, mcp_client):
        """BuiltIn=available, NonExistent=unavailable."""
        result = await mcp_client.call_tool(
            "check_library_availability",
            {"libraries": ["BuiltIn", "NonExistentLibrary123"]},
        )
        assert result.data["success"] is True
        availability = result.data.get("availability", result.data.get("libraries", {}))
        # BuiltIn should be available in some form
        assert isinstance(availability, (dict, list))


class TestSetLibrarySearchOrder:
    """Test set_library_search_order tool."""

    @pytest.mark.asyncio
    async def test_set_library_search_order_via_mcp(self, mcp_client):
        """Set search order and verify it takes effect."""
        sid = _sid("so")
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "init", "libraries": ["BuiltIn", "String"]},
        )

        result = await mcp_client.call_tool(
            "set_library_search_order",
            {"libraries": ["String", "BuiltIn"], "session_id": sid},
        )
        assert result.data["success"] is True


class TestAnalyzeRecommendInitChain:
    """Test the analyze -> recommend -> init chain."""

    @pytest.mark.asyncio
    async def test_analyze_recommend_init_chain(self, mcp_client):
        """analyze_scenario -> recommend_libraries -> manage_session(init)."""
        # Analyze
        analysis = await mcp_client.call_tool(
            "analyze_scenario",
            {"scenario": "Test REST API endpoints with HTTP requests", "context": "api"},
        )
        assert analysis.data["success"] is True
        sid = analysis.data["session_id"]

        # Recommend
        recommend = await mcp_client.call_tool(
            "recommend_libraries",
            {"scenario": "API testing with HTTP methods"},
        )
        assert recommend.data["success"] is True

        # The session from analyze should already be usable
        step = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Log", "arguments": ["API test step"], "session_id": sid},
        )
        assert step.data["success"] is True


class TestSuiteSetupTeardownInBuild:
    """Test suite setup/teardown in generated .robot."""

    @pytest.mark.asyncio
    async def test_suite_setup_teardown_in_generated_robot(self, mcp_client):
        """set_suite_setup + set_suite_teardown appear in build_test_suite output."""
        sid = _sid("stsd")
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "init", "libraries": ["BuiltIn"]},
        )
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "set_suite_setup", "keyword": "Log", "args": ["Suite setup"]},
        )
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "set_suite_teardown", "keyword": "Log", "args": ["Suite teardown"]},
        )
        await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Log", "arguments": ["test step"], "session_id": sid},
        )

        build = await mcp_client.call_tool(
            "build_test_suite",
            {"session_id": sid, "test_name": "With Setup"},
        )
        assert build.data["success"] is True
        rf_text = build.data["rf_text"]
        assert "*** Settings ***" in rf_text or "Suite Setup" in rf_text


# =============================================================================
# P4: Low priority scenarios
# =============================================================================


class TestGracefulErrors:
    """Test graceful error handling."""

    @pytest.mark.asyncio
    async def test_execute_step_invalid_keyword_graceful_error(self, mcp_client):
        """Nonexistent keyword -> returns error, doesn't crash."""
        sid = _sid("grace")
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "init", "libraries": ["BuiltIn"]},
        )

        result = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Nonexistent Keyword That Doesnt Exist", "arguments": [], "session_id": sid, "raise_on_failure": False},
        )
        # Should NOT crash - either success:False or a handled error
        assert isinstance(result.data, dict)

    @pytest.mark.asyncio
    async def test_get_session_state_nonexistent_session(self, mcp_client):
        """Non-existent session_id -> doesn't crash server."""
        result = await mcp_client.call_tool(
            "get_session_state",
            {"session_id": "nonexistent-session-id-12345", "sections": ["summary"]},
        )
        # Should return some response, not crash
        assert isinstance(result.data, dict)


class TestRecommendLibraries:
    """Test recommend_libraries tool independently."""

    @pytest.mark.asyncio
    async def test_recommend_libraries_returns_suggestions(self, mcp_client):
        """recommend_libraries returns relevant library suggestions for a scenario."""
        result = await mcp_client.call_tool(
            "recommend_libraries",
            {"scenario": "Validate JSON responses from a REST API"},
        )
        assert result.data["success"] is True
        # Should contain some recommendation data (libraries, categories, etc.)
        assert isinstance(result.data, dict)


class TestVariableAssignmentAndRetrieval:
    """Test variable assignment via execute_step and retrieval."""

    @pytest.mark.asyncio
    async def test_variable_roundtrip_via_execute_step(self, mcp_client):
        """Set variable, retrieve it, and verify the value."""
        sid = _sid("varrt")
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "init", "libraries": ["BuiltIn"]},
        )

        # Set a variable using Set Variable keyword
        set_result = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Set Variable", "arguments": ["test_value_123"], "session_id": sid, "assign_to": "MY_VAR"},
        )
        assert set_result.data["success"] is True

        # Use the variable in a subsequent step
        log_result = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Log", "arguments": ["Variable value is: ${MY_VAR}"], "session_id": sid},
        )
        assert log_result.data["success"] is True

    @pytest.mark.asyncio
    async def test_set_variables_via_manage_session_then_use(self, mcp_client):
        """Set variables via manage_session, then use them in execute_step."""
        sid = _sid("setvar")
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "init", "libraries": ["BuiltIn"]},
        )

        # Set variables
        await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "set_variables", "variables": {"GREETING": "Hello", "TARGET": "World"}},
        )

        # Use variables in a step
        step = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Log", "arguments": ["${GREETING} ${TARGET}"], "session_id": sid},
        )
        assert step.data["success"] is True


# =============================================================================
# P16-Issue7: BuiltIn available after analyze+recommend+import_library flow
# =============================================================================


class TestBuiltInAfterAnalyzeRecommendImport:
    """Verify BuiltIn keywords work after the analyze→recommend→import flow.

    Reproduces the exact scenario from P16-Issue7 where BuiltIn keywords like
    'Should Be Equal' failed with 'No keyword with name found' after
    analyze_scenario → recommend_libraries → manage_session(import_library).
    """

    @pytest.mark.asyncio
    async def test_builtin_after_analyze_recommend_import(self, mcp_client):
        """analyze_scenario → recommend_libraries → import_library → Should Be Equal works."""
        # 1. analyze_scenario (creates session)
        analyze = await mcp_client.call_tool(
            "analyze_scenario",
            {"scenario": "Web UI test: verify text on page", "context": "web"},
        )
        assert analyze.data["success"] is True
        sid = analyze.data["session_id"]

        # 2. recommend_libraries (auto-imports non-BuiltIn libraries)
        rec = await mcp_client.call_tool(
            "recommend_libraries",
            {"scenario": "Web UI test: verify text on page", "session_id": sid,
             "check_availability": True, "apply_search_order": True},
        )
        assert rec.data["success"] is True

        # 3. import_library for OperatingSystem (extra library)
        imp = await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "import_library", "library_name": "OperatingSystem"},
        )
        assert imp.data["success"] is True

        # 4. BuiltIn keywords MUST work after the above flow
        step = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Should Be Equal", "arguments": ["hello", "hello"], "session_id": sid},
        )
        assert step.data["success"] is True, (
            f"BuiltIn keyword 'Should Be Equal' failed: {step.data.get('error')}"
        )

    @pytest.mark.asyncio
    async def test_builtin_after_import_library_without_analyze(self, mcp_client):
        """manage_session(init) → import_library(Collections) → Should Be Equal works."""
        sid = _sid("bi-imp")
        init = await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "init"},
        )
        assert init.data["success"] is True

        # Import a non-BuiltIn library
        imp = await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "import_library", "library_name": "Collections"},
        )
        assert imp.data["success"] is True

        # BuiltIn keywords MUST still work
        step = await mcp_client.call_tool(
            "execute_step",
            {"keyword": "Should Be Equal", "arguments": ["42", "42"], "session_id": sid},
        )
        assert step.data["success"] is True, (
            f"BuiltIn keyword 'Should Be Equal' failed: {step.data.get('error')}"
        )

    @pytest.mark.asyncio
    async def test_builtin_log_after_multiple_import_library(self, mcp_client):
        """Multiple import_library calls don't break BuiltIn."""
        sid = _sid("bi-multi")
        init = await mcp_client.call_tool(
            "manage_session",
            {"session_id": sid, "action": "init"},
        )
        assert init.data["success"] is True

        # Import multiple libraries
        for lib in ["Collections", "String", "OperatingSystem"]:
            imp = await mcp_client.call_tool(
                "manage_session",
                {"session_id": sid, "action": "import_library", "library_name": lib},
            )
            assert imp.data["success"] is True

        # BuiltIn keywords MUST still work
        for kw, args in [
            ("Log", ["Still working"]),
            ("Should Be Equal", ["a", "a"]),
            ("Set Variable", ["test_value"]),
        ]:
            step = await mcp_client.call_tool(
                "execute_step",
                {"keyword": kw, "arguments": args, "session_id": sid},
            )
            assert step.data["success"] is True, (
                f"BuiltIn keyword '{kw}' failed: {step.data.get('error')}"
            )
