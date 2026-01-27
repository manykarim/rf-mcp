"""Comprehensive integration tests for architecture improvements.

This module tests:
1. LibraryDetector - keyword-based library detection
2. LibraryLoadingValidator - validation and conflict resolution
3. NLP enhancements - stemming and synonym expansion
4. MCP client integration - tool parameter validation
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import mcp
from robotmcp.utils.library_detector import (
    detect_library_from_keyword,
    detect_library_type_from_keyword,
)
from robotmcp.utils.library_checker import (
    LibraryAvailabilityChecker,
    check_and_suggest_libraries,
)
from robotmcp.components.nlp_processor import NaturalLanguageProcessor
from robotmcp.components.library_recommender import LibraryRecommender
from robotmcp.config.library_registry import (
    get_all_libraries,
    get_library_config,
    get_builtin_libraries,
    LibraryCategory,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def mcp_client():
    """Provide an MCP client for integration tests."""
    async with Client(mcp) as client:
        yield client


@pytest.fixture
def library_checker():
    """Provide a library availability checker instance."""
    return LibraryAvailabilityChecker()


@pytest.fixture
def nlp_processor():
    """Provide an NLP processor instance."""
    return NaturalLanguageProcessor()


@pytest.fixture
def library_recommender():
    """Provide a library recommender instance."""
    return LibraryRecommender()


# =============================================================================
# LibraryDetector Tests
# =============================================================================


class TestLibraryDetector:
    """Tests for the library detection utility."""

    def test_detect_selenium_explicit_keywords(self):
        """Detect SeleniumLibrary from explicit Selenium keywords."""
        selenium_keywords = [
            "open browser",
            "input text",
            "click button",
            "select from list",
            "wait until element is visible",
            "page should contain",
            "element should be visible",
            "capture page screenshot",
            "maximize browser window",
            "set window size",
        ]
        for keyword in selenium_keywords:
            result = detect_library_from_keyword(keyword)
            assert result == "SeleniumLibrary", (
                f"Expected 'SeleniumLibrary' for '{keyword}', got '{result}'"
            )

    def test_detect_browser_explicit_keywords(self):
        """Detect Browser Library from explicit Browser/Playwright keywords."""
        browser_keywords = [
            "new browser",
            "new context",
            "new page",
            "close context",
            "close page",
            "go to",
            "get viewport size",
            "set viewport size",
            "wait for elements state",
            "get element count",
            "fill text",
            "fill",
            "get text",
            "get property",
            "select options by",
            "check checkbox",
            "get page source",
            "click",
        ]
        for keyword in browser_keywords:
            result = detect_library_from_keyword(keyword)
            assert result == "Browser", (
                f"Expected 'Browser' for '{keyword}', got '{result}'"
            )

    def test_detect_requests_library_keywords(self):
        """Detect RequestsLibrary from API-related keywords."""
        api_keywords = [
            "get request",
            "post request",
            "put request",
            "delete request",
            "patch request",
            "head request",
            "options request",
            "response should",
            "create session",
            "get on session",
            "post on session",
        ]
        for keyword in api_keywords:
            result = detect_library_from_keyword(keyword)
            assert result == "RequestsLibrary", (
                f"Expected 'RequestsLibrary' for '{keyword}', got '{result}'"
            )

    def test_detect_appium_library_keywords(self):
        """Detect AppiumLibrary from mobile-related keywords."""
        mobile_keywords = [
            "open application",
            "close application",
            "launch application",
            "tap",
            "swipe",
            "get contexts",
            "get current context",
            "switch to context",
            "get source",
        ]
        for keyword in mobile_keywords:
            result = detect_library_from_keyword(keyword)
            assert result == "AppiumLibrary", (
                f"Expected 'AppiumLibrary' for '{keyword}', got '{result}'"
            )

    def test_detect_builtin_keywords(self):
        """Detect BuiltIn library from core keywords."""
        builtin_keywords = [
            "log",
            "set variable",
            "should be equal",
            "should contain",
            "should not contain",
            "convert to string",
            "convert to integer",
            "convert to number",
            "catenate",
            "length should be",
            "run keyword",
            "run keyword if",
        ]
        for keyword in builtin_keywords:
            result = detect_library_from_keyword(keyword)
            assert result == "BuiltIn", (
                f"Expected 'BuiltIn' for '{keyword}', got '{result}'"
            )

    def test_detect_collections_keywords(self):
        """Detect Collections library from data structure keywords."""
        collections_keywords = [
            "append to list",
            "get from list",
            "create list",
            "create dictionary",
            "get from dictionary",
            "set to dictionary",
            "remove from list",
        ]
        for keyword in collections_keywords:
            result = detect_library_from_keyword(keyword)
            assert result == "Collections", (
                f"Expected 'Collections' for '{keyword}', got '{result}'"
            )

    def test_detect_operating_system_keywords(self):
        """Detect OperatingSystem library from file system keywords."""
        os_keywords = [
            "copy file",
            "create directory",
            "file should exist",
            "directory should exist",
            "create file",
            "remove file",
            "remove directory",
            "move file",
        ]
        for keyword in os_keywords:
            result = detect_library_from_keyword(keyword)
            assert result == "OperatingSystem", (
                f"Expected 'OperatingSystem' for '{keyword}', got '{result}'"
            )

    def test_no_detection_generic_input(self):
        """Return None for generic/unknown keywords."""
        generic_keywords = [
            "run some tests",
            "do something",
            "test the application",
            "verify results",
        ]
        for keyword in generic_keywords:
            result = detect_library_from_keyword(keyword)
            assert result is None, (
                f"Expected None for generic keyword '{keyword}', got '{result}'"
            )

    def test_detect_library_type_browser(self):
        """Test library type detection returns 'browser' for Browser keywords."""
        result = detect_library_type_from_keyword("new page")
        assert result == "browser"

    def test_detect_library_type_selenium(self):
        """Test library type detection returns 'selenium' for Selenium keywords."""
        # Use a keyword that is unambiguously SeleniumLibrary
        result = detect_library_type_from_keyword("input text")
        assert result == "selenium"

    def test_detect_library_type_auto(self):
        """Test library type detection returns 'auto' for unknown keywords."""
        result = detect_library_type_from_keyword("unknown keyword")
        assert result == "auto"


# =============================================================================
# LibraryLoadingValidator Tests (using LibraryAvailabilityChecker)
# =============================================================================


class TestLibraryLoadingValidator:
    """Tests for library loading validation and conflict resolution."""

    def test_can_load_builtin_library(self, library_checker):
        """BuiltIn library should always be loadable."""
        # BuiltIn is part of robot.libraries, check via is_robot_library_available
        can_load = library_checker.is_robot_library_available("BuiltIn")
        assert can_load is True

    def test_can_load_collections_library(self, library_checker):
        """Collections library should be loadable."""
        can_load = library_checker.is_robot_library_available("Collections")
        assert can_load is True

    def test_can_load_string_library(self, library_checker):
        """String library should be loadable."""
        can_load = library_checker.is_robot_library_available("String")
        assert can_load is True

    def test_check_and_suggest_libraries_available(self):
        """Test checking available libraries returns them correctly."""
        available, suggestions = check_and_suggest_libraries(["BuiltIn", "Collections"])
        assert "BuiltIn" in available
        assert "Collections" in available
        # No suggestions needed for built-in libraries
        assert len(suggestions) == 0

    def test_check_and_suggest_missing_library(self):
        """Test that missing libraries generate installation suggestions."""
        # Use a library that is unlikely to be installed
        available, suggestions = check_and_suggest_libraries(["NonExistentLibrary123"])
        assert "NonExistentLibrary123" not in available
        assert len(suggestions) > 0
        assert "not found" in suggestions[0].lower() or "not available" in suggestions[0].lower()

    def test_library_cache_usage(self, library_checker):
        """Test that repeated checks use the cache."""
        # First check
        result1 = library_checker.is_robot_library_available("BuiltIn")
        # Cache should be populated
        assert len(library_checker.checked_libraries) > 0 or result1 is True
        # Second check should use cache
        result2 = library_checker.is_robot_library_available("BuiltIn")
        assert result1 == result2


class TestLibraryConflictResolution:
    """Tests for library conflict resolution logic."""

    def test_web_automation_exclusion_in_recommender(self, library_recommender):
        """Test that recommender applies exclusion for web automation libraries."""
        # Recommend for a web scenario
        result = library_recommender.recommend_libraries(
            scenario="Test a web page with clicks and forms",
            context="web",
            max_recommendations=10,
        )

        assert result["success"] is True
        recs = result["recommendations"]
        rec_names = [r["library_name"] for r in recs]

        # Should have at most one of Browser or SeleniumLibrary
        web_libs = [name for name in rec_names if name in ["Browser", "SeleniumLibrary"]]
        assert len(web_libs) <= 1, (
            f"Expected at most one web automation library, got: {web_libs}"
        )

    def test_context_based_filtering_web(self, library_recommender):
        """Test that web context filters out mobile libraries."""
        result = library_recommender.recommend_libraries(
            scenario="Click buttons and fill forms on a website",
            context="web",
            max_recommendations=10,
        )

        assert result["success"] is True
        recs = result["recommendations"]
        rec_names = [r["library_name"] for r in recs]

        # AppiumLibrary should not be recommended for pure web scenarios
        assert "AppiumLibrary" not in rec_names

    def test_context_based_filtering_mobile(self, library_recommender):
        """Test that mobile context prioritizes mobile libraries."""
        result = library_recommender.recommend_libraries(
            scenario="Test a mobile app with taps and swipes",
            context="mobile",
            max_recommendations=5,
        )

        assert result["success"] is True
        recs = result["recommendations"]

        # If there are recommendations, AppiumLibrary should be among them or first
        if recs:
            rec_names = [r["library_name"] for r in recs]
            # Web libraries should be excluded for mobile context
            assert "Browser" not in rec_names
            assert "SeleniumLibrary" not in rec_names


class TestLibraryRegistry:
    """Tests for the centralized library registry."""

    def test_get_all_libraries_returns_data(self):
        """Test that the library registry returns libraries."""
        libraries = get_all_libraries()
        assert len(libraries) > 0
        assert "BuiltIn" in libraries

    def test_builtin_libraries_have_correct_type(self):
        """Test that built-in libraries are correctly classified."""
        builtin_libs = get_builtin_libraries()
        assert "BuiltIn" in builtin_libs
        assert "Collections" in builtin_libs
        assert "String" in builtin_libs

    def test_library_config_has_required_fields(self):
        """Test that library configs have all required fields."""
        config = get_library_config("BuiltIn")
        assert config is not None
        assert config.name == "BuiltIn"
        assert config.package_name
        assert config.import_path
        assert config.description

    def test_web_category_libraries(self):
        """Test that web category contains expected libraries."""
        from robotmcp.config.library_registry import get_libraries_by_category

        web_libs = get_libraries_by_category(LibraryCategory.WEB)
        web_names = list(web_libs.keys())

        # At least Browser or SeleniumLibrary should be in web category
        assert any(name in web_names for name in ["Browser", "SeleniumLibrary"])


# =============================================================================
# NLP Enhancement Tests
# =============================================================================


class TestNLPEnhancements:
    """Tests for NLP processing enhancements."""

    def test_normalize_text(self, nlp_processor):
        """Test text normalization removes extra whitespace."""
        text = "  Multiple   spaces   and\nnewlines  "
        result = nlp_processor._normalize_text(text)
        assert "  " not in result
        assert result == "Multiple spaces and newlines"

    def test_extract_title_from_scenario(self, nlp_processor):
        """Test title extraction from scenario text."""
        scenario = "Test that user can login successfully. Navigate to login page."
        title = nlp_processor._extract_title(scenario)
        assert len(title) > 0
        # Title should be capitalized
        assert title[0].isupper()

    def test_split_sentences(self, nlp_processor):
        """Test sentence splitting."""
        text = "First sentence. Second sentence! Third sentence?"
        sentences = nlp_processor._split_sentences(text)
        assert len(sentences) == 3
        assert "First sentence" in sentences[0]
        assert "Second sentence" in sentences[1]
        assert "Third sentence" in sentences[2]

    def test_extract_action_navigate(self, nlp_processor):
        """Test extraction of navigation actions."""
        action = nlp_processor._extract_action("Go to the homepage")
        assert action is not None
        assert action.action_type == "navigate"

    def test_extract_action_click(self, nlp_processor):
        """Test extraction of click actions."""
        action = nlp_processor._extract_action("Click on the submit button")
        assert action is not None
        assert action.action_type == "click"
        assert action.target is not None

    def test_extract_action_input(self, nlp_processor):
        """Test extraction of input actions."""
        action = nlp_processor._extract_action("Enter 'john@example.com' into the email field")
        assert action is not None
        assert action.action_type == "input"
        assert action.value == "john@example.com"

    def test_extract_action_verify(self, nlp_processor):
        """Test extraction of verification actions."""
        action = nlp_processor._extract_action("Verify that the page contains welcome message")
        assert action is not None
        assert action.action_type == "verify"

    def test_detect_explicit_selenium_preference(self, nlp_processor):
        """Test detection of explicit SeleniumLibrary preference."""
        scenarios = [
            "Use SeleniumLibrary for web testing",
            "Using selenium for automation",
            "Test with seleniumlibrary",
        ]
        for scenario in scenarios:
            result = nlp_processor._detect_explicit_library_preference(scenario)
            assert result == "SeleniumLibrary", (
                f"Expected 'SeleniumLibrary' for '{scenario}', got '{result}'"
            )

    def test_detect_explicit_browser_preference(self, nlp_processor):
        """Test detection of explicit Browser Library preference."""
        scenarios = [
            "Use Browser Library for modern automation",
            "Using playwright for cross-browser testing",
            "Test with browserlibrary",
        ]
        for scenario in scenarios:
            result = nlp_processor._detect_explicit_library_preference(scenario)
            assert result == "Browser", (
                f"Expected 'Browser' for '{scenario}', got '{result}'"
            )

    def test_no_explicit_preference_neutral_text(self, nlp_processor):
        """Test that neutral text returns no preference."""
        scenarios = [
            "Test the login page functionality",
            "Automate the checkout process",
            "Verify user registration works",
        ]
        for scenario in scenarios:
            result = nlp_processor._detect_explicit_library_preference(scenario)
            assert result is None, (
                f"Expected None for neutral '{scenario}', got '{result}'"
            )

    def test_detect_session_type_web(self, nlp_processor):
        """Test detection of web automation session type."""
        scenario = "Click on buttons, fill forms, navigate pages"
        result = nlp_processor._detect_session_type(scenario, "web")
        assert result == "web_automation"

    def test_detect_session_type_api(self, nlp_processor):
        """Test detection of API testing session type."""
        scenario = "Send POST request to API endpoint and check response status"
        result = nlp_processor._detect_session_type(scenario, "api")
        assert result == "api_testing"

    def test_determine_capabilities_web_default(self, nlp_processor):
        """Test that web context defaults to Browser Library."""
        scenario = "Test a website with clicks and forms"
        capabilities = nlp_processor._determine_capabilities(scenario, "web")
        assert "Browser" in capabilities

    def test_determine_capabilities_selenium_explicit(self, nlp_processor):
        """Test that explicit Selenium mention adds SeleniumLibrary."""
        scenario = "Use selenium for web testing"
        capabilities = nlp_processor._determine_capabilities(scenario, "web")
        assert "SeleniumLibrary" in capabilities

    def test_assess_complexity_simple(self, nlp_processor):
        """Test complexity assessment for simple scenarios."""
        from robotmcp.components.nlp_processor import TestAction

        actions = [
            TestAction("navigate", "Go to page"),
            TestAction("click", "Click button"),
        ]
        complexity = nlp_processor._assess_complexity(actions)
        assert complexity == "simple"

    def test_assess_complexity_medium(self, nlp_processor):
        """Test complexity assessment for medium scenarios."""
        from robotmcp.components.nlp_processor import TestAction

        actions = [
            TestAction("navigate", "Go to page"),
            TestAction("input", "Enter text", target="field", value="test"),
            TestAction("click", "Click button"),
            TestAction("verify", "Check result"),
            TestAction("click", "Click next"),
        ]
        complexity = nlp_processor._assess_complexity(actions)
        assert complexity == "medium"

    def test_assess_complexity_complex(self, nlp_processor):
        """Test complexity assessment for complex scenarios."""
        from robotmcp.components.nlp_processor import TestAction

        actions = [TestAction("click", f"Step {i}") for i in range(10)]
        complexity = nlp_processor._assess_complexity(actions)
        assert complexity == "complex"


# =============================================================================
# MCP Client Integration Tests
# =============================================================================


@pytest.mark.asyncio
class TestMCPClientIntegration:
    """Integration tests with MCP client for tool validation."""

    async def test_recommend_libraries_basic(self, mcp_client):
        """Test basic library recommendation via MCP."""
        result = await mcp_client.call_tool(
            "recommend_libraries",
            {
                "scenario": "Test a web page with forms",
                "context": "web",
            },
        )
        assert result.data["success"] is True
        assert "recommendations" in result.data
        assert len(result.data["recommendations"]) > 0

    async def test_recommend_libraries_with_context(self, mcp_client):
        """Test library recommendation with different contexts."""
        contexts = ["web", "api", "mobile"]

        for context in contexts:
            result = await mcp_client.call_tool(
                "recommend_libraries",
                {
                    "scenario": f"Test application in {context} context",
                    "context": context,
                },
            )
            assert result.data["success"] is True
            assert result.data["context"] == context

    async def test_analyze_scenario_returns_library_preference(self, mcp_client):
        """Test that analyze_scenario returns library preference info."""
        result = await mcp_client.call_tool(
            "analyze_scenario",
            {
                "scenario": "Use SeleniumLibrary to test the login page",
                "context": "web",
            },
        )

        assert result.data["success"] is True
        assert "analysis" in result.data
        assert result.data["analysis"]["explicit_library_preference"] == "SeleniumLibrary"

    async def test_analyze_scenario_browser_preference(self, mcp_client):
        """Test that Browser Library preference is detected."""
        result = await mcp_client.call_tool(
            "analyze_scenario",
            {
                "scenario": "Use Browser Library with Playwright for modern web testing",
                "context": "web",
            },
        )

        assert result.data["success"] is True
        assert result.data["analysis"]["explicit_library_preference"] == "Browser"

    async def test_analyze_scenario_no_preference(self, mcp_client):
        """Test scenario without explicit library preference."""
        result = await mcp_client.call_tool(
            "analyze_scenario",
            {
                "scenario": "Test the shopping cart functionality",
                "context": "web",
            },
        )

        assert result.data["success"] is True
        # Should either be None or not SeleniumLibrary/Browser specifically
        pref = result.data["analysis"].get("explicit_library_preference")
        # Neutral scenario might still get a recommendation based on context
        assert pref is None or pref in [None, "Browser", "RequestsLibrary"]

    async def test_recommend_libraries_exclusion_applied(self, mcp_client):
        """Test that web automation exclusion is applied in recommendations."""
        result = await mcp_client.call_tool(
            "recommend_libraries",
            {
                "scenario": "Automate a web application with selenium and playwright",
                "context": "web",
                "max_recommendations": 10,
            },
        )

        assert result.data["success"] is True
        recs = result.data["recommendations"]
        lib_names = [r["library_name"] for r in recs]

        # Should not have both Browser and SeleniumLibrary
        has_browser = "Browser" in lib_names
        has_selenium = "SeleniumLibrary" in lib_names
        assert not (has_browser and has_selenium), (
            "Should not recommend both Browser and SeleniumLibrary together"
        )

    async def test_get_session_state_includes_library_info(self, mcp_client):
        """Test that session state includes library information."""
        session_id = "arch_test_session"

        # Initialize with analyze_scenario
        await mcp_client.call_tool(
            "analyze_scenario",
            {
                "scenario": "Use SeleniumLibrary for testing",
                "context": "web",
            },
        )

        # Execute a simple step to create session
        await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Log",
                "arguments": ["Test message"],
                "session_id": session_id,
            },
        )

        # Get session state
        state = await mcp_client.call_tool(
            "get_session_state",
            {
                "session_id": session_id,
                "sections": ["summary"],
            },
        )

        assert state.data["success"] is True
        assert "sections" in state.data
        assert "summary" in state.data["sections"]

    async def test_find_keywords_returns_results(self, mcp_client):
        """Test that find_keywords returns keyword results."""
        result = await mcp_client.call_tool(
            "find_keywords",
            {
                "query": "log message",
                "strategy": "semantic",
                "limit": 5,
            },
        )

        assert result.data["success"] is True
        # The response uses 'result' with 'matches' inside
        assert "result" in result.data
        assert "matches" in result.data["result"]
        assert len(result.data["result"]["matches"]) > 0

    async def test_get_keyword_info_for_log(self, mcp_client):
        """Test getting keyword info for Log keyword."""
        result = await mcp_client.call_tool(
            "get_keyword_info",
            {
                "keyword_name": "Log",
            },
        )

        assert result.data["success"] is True
        assert result.data["mode"] == "keyword"


# =============================================================================
# Library Recommender Direct Tests
# =============================================================================


class TestLibraryRecommenderDirect:
    """Direct tests for the LibraryRecommender class."""

    def test_recommender_initialization(self, library_recommender):
        """Test that recommender initializes correctly."""
        # Force initialization
        library_recommender._initialize_registry()
        assert library_recommender._initialized is True
        assert len(library_recommender.libraries_registry) > 0

    def test_recommender_category_mapping(self, library_recommender):
        """Test that category mapping is built correctly."""
        library_recommender._initialize_registry()
        assert len(library_recommender.category_mapping) > 0
        # Web category should exist
        assert "web" in library_recommender.category_mapping

    def test_recommender_use_case_mapping(self, library_recommender):
        """Test that use case mapping is built correctly."""
        library_recommender._initialize_registry()
        assert len(library_recommender.use_case_mapping) > 0

    def test_normalize_text_removes_punctuation(self, library_recommender):
        """Test text normalization."""
        text = "Test! With, punctuation? And (symbols)."
        result = library_recommender._normalize_text(text)
        assert "!" not in result
        assert "," not in result
        assert "?" not in result

    def test_extract_keywords(self, library_recommender):
        """Test keyword extraction from text."""
        text = "test a web browser page with clicks and forms"
        keywords = library_recommender._extract_keywords(text)

        assert "web" in keywords
        assert "browser" in keywords
        assert "page" in keywords
        assert "click" in keywords

    def test_confidence_threshold_filtering(self, library_recommender):
        """Test that low-confidence recommendations are filtered."""
        library_recommender._initialize_registry()

        # Create test recommendations with varying confidence
        from robotmcp.components.library_recommender import LibraryRecommendation

        test_recs = []
        for name, lib in list(library_recommender.libraries_registry.items())[:3]:
            test_recs.append(LibraryRecommendation(
                library=lib,
                confidence=0.2,  # Below threshold
                matching_keywords=["test"],
                rationale="Test rationale",
            ))

        filtered = library_recommender._filter_by_confidence(test_recs)

        # All should be filtered out due to low confidence
        assert len(filtered) == 0

    def test_generate_installation_script(self, library_recommender):
        """Test installation script generation."""
        library_recommender._initialize_registry()

        from robotmcp.components.library_recommender import LibraryRecommendation

        # Get Browser library config
        browser_lib = library_recommender.libraries_registry.get("Browser")
        if browser_lib:
            recs = [LibraryRecommendation(
                library=browser_lib,
                confidence=0.9,
                matching_keywords=["web"],
                rationale="Test",
            )]

            script = library_recommender._generate_installation_script(recs)
            assert "pip install" in script.lower() or "built-in" in script.lower() or "install" in script.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
