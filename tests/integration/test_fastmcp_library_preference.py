"""Integration tests for analyze_scenario library preference functionality using FastMCP Client."""

import pytest

pytestmark = pytest.mark.skip(reason="Session preference tools were consolidated; tests pending rewrite")

import pytest_asyncio
from fastmcp import Client

from robotmcp.server import mcp


class TestAnalyzeScenarioWithFastMCPClient:
    """Integration tests using FastMCP Client to test MCP tools end-to-end."""

    @pytest_asyncio.fixture
    async def mcp_client(self):
        """Create FastMCP client for testing."""
        async with Client(mcp) as client:
            yield client

    @pytest.mark.asyncio
    async def test_analyze_scenario_selenium_preference_with_client(self, mcp_client):
        """Test analyze_scenario with Selenium preference using FastMCP client."""
        scenario = """
        Use RobotMCP to create a TestSuite and execute it step wise.
        
        - Open https://www.saucedemo.com/
        - Login with valid user
        - Assert login was successful
        - Add item to cart
        - Assert item was added to cart
        - Add another item to cart
        - Assert another item was added to cart
        - Checkout
        - Assert checkout was successful
        
        Execute the test suite stepwise and build the final version afterwards.
        Use Selenium Library
        """

        # Call analyze_scenario via MCP client
        result = await mcp_client.call_tool(
            "analyze_scenario", {"scenario": scenario, "context": "web"}
        )

        # Verify the result structure
        assert result.data is not None
        data = result.data

        # Verify session info is included
        assert "session_info" in data
        session_info = data["session_info"]

        # Verify explicit library preference was detected
        assert session_info["explicit_library_preference"] == "SeleniumLibrary"
        assert session_info["session_type"] == "web_automation"
        assert session_info["auto_configured"] is True

        # Verify recommended libraries include SeleniumLibrary
        recommended_libs = session_info.get("recommended_libraries", [])
        assert "SeleniumLibrary" in recommended_libs

        # Browser Library should NOT be in recommended libraries when Selenium is explicitly preferred
        assert "Browser" not in recommended_libs

    @pytest.mark.asyncio
    async def test_analyze_scenario_browser_preference_with_client(self, mcp_client):
        """Test analyze_scenario with Browser Library preference using FastMCP client."""
        scenario = """
        Use RobotMCP to create a TestSuite and execute it step wise.
        
        - Open https://www.saucedemo.com/
        - Login with valid user
        - Assert login was successful
        
        Use Browser Library for modern web automation.
        """

        # Call analyze_scenario via MCP client
        result = await mcp_client.call_tool(
            "analyze_scenario", {"scenario": scenario, "context": "web"}
        )

        # Verify the result
        assert result.data is not None
        data = result.data

        # Verify session info
        assert "session_info" in data
        session_info = data["session_info"]

        # Verify explicit library preference was detected
        assert session_info["explicit_library_preference"] == "Browser"
        assert session_info["session_type"] == "web_automation"
        assert session_info["auto_configured"] is True

        # Verify recommended libraries include Browser Library
        recommended_libs = session_info.get("recommended_libraries", [])
        assert "Browser" in recommended_libs

        # SeleniumLibrary should NOT be in recommended libraries when Browser is explicitly preferred
        assert "SeleniumLibrary" not in recommended_libs

    @pytest.mark.asyncio
    async def test_analyze_scenario_no_preference_with_client(self, mcp_client):
        """Test analyze_scenario without explicit library preference using FastMCP client."""
        scenario = """
        Create a comprehensive web automation test suite.
        
        - Navigate to e-commerce website
        - Test user authentication
        - Validate product catalog functionality
        - Test shopping cart operations
        - Verify checkout process
        """

        # Call analyze_scenario via MCP client
        result = await mcp_client.call_tool(
            "analyze_scenario", {"scenario": scenario, "context": "web"}
        )

        # Verify the result
        assert result.data is not None
        data = result.data

        # Verify session info
        assert "session_info" in data
        session_info = data["session_info"]

        # Verify no explicit library preference
        assert session_info["explicit_library_preference"] is None
        # When no preference is specified, complex scenarios may be detected as "mixed"
        assert session_info["session_type"] in ["web_automation", "mixed"]
        assert session_info["auto_configured"] is True

        # Without explicit preference, recommended libraries may include Browser based on profiles
        recommended_libs = session_info.get("recommended_libraries", [])
        assert "BuiltIn" in recommended_libs
        assert "Collections" in recommended_libs

    @pytest.mark.asyncio
    async def test_execute_step_respects_session_preference(self, mcp_client):
        """Test that execute_step respects session library preferences."""
        # First, create a session with explicit Selenium preference
        selenium_scenario = "Use SeleniumLibrary for web automation testing"

        analyze_result = await mcp_client.call_tool(
            "analyze_scenario", {"scenario": selenium_scenario, "context": "web"}
        )

        assert analyze_result.data is not None
        session_id = analyze_result.data["session_info"]["session_id"]

        # Now execute a step in that session
        execute_result = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Log",
                "arguments": ["Testing Selenium Library preference"],
                "session_id": session_id,
                "detail_level": "standard",
            },
        )

        # Verify execution worked
        assert execute_result.data is not None
        exec_data = execute_result.data

        # Verify execution was successful
        assert exec_data["success"] is True
        assert exec_data["status"] == "pass"
        assert exec_data["keyword"] == "Log"

    @pytest.mark.asyncio
    async def test_get_session_info_shows_correct_configuration(self, mcp_client):
        """Test that get_session_info shows correct library configuration."""
        # Create session with Browser Library preference
        browser_scenario = "Use Browser Library for modern web testing"

        analyze_result = await mcp_client.call_tool(
            "analyze_scenario", {"scenario": browser_scenario, "context": "web"}
        )

        assert analyze_result.data is not None
        session_id = analyze_result.data["session_info"]["session_id"]

        # Get session info
        session_result = await mcp_client.call_tool(
            "get_session_info", {"session_id": session_id}
        )

        assert session_result.data is not None
        session_data = session_result.data[
            "session_info"
        ]  # get_session_info wraps data in session_info

        # Verify session configuration
        assert session_data["explicit_library_preference"] == "Browser"
        assert session_data["session_type"] == "web_automation"
        assert session_data["auto_configured"] is True

        # Verify search order prioritizes Browser Library
        search_order = session_data.get("search_order", [])
        if search_order:
            assert search_order[0] == "Browser"
            assert "SeleniumLibrary" not in search_order

    @pytest.mark.asyncio
    async def test_end_to_end_workflow_selenium_preference(self, mcp_client):
        """Test complete end-to-end workflow with Selenium Library preference."""
        # Step 1: Analyze scenario with explicit Selenium preference
        scenario = """
        Use SeleniumLibrary to automate testing on https://www.saucedemo.com
        - Open the website
        - Login with standard user
        - Verify successful login
        """

        analyze_result = await mcp_client.call_tool(
            "analyze_scenario", {"scenario": scenario, "context": "web"}
        )

        assert analyze_result.data is not None
        session_id = analyze_result.data["session_info"]["session_id"]

        # Step 2: Execute some steps in the session
        log_result = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Log",
                "arguments": ["Starting Selenium test"],
                "session_id": session_id,
            },
        )
        assert log_result.data is not None

        # Step 3: Check session validation status
        validation_result = await mcp_client.call_tool(
            "get_session_validation_status", {"session_id": session_id}
        )
        assert validation_result.data is not None

        # Step 4: Verify session maintains Selenium preference throughout
        final_session_result = await mcp_client.call_tool(
            "get_session_info", {"session_id": session_id}
        )

        assert final_session_result.data is not None
        final_data = final_session_result.data[
            "session_info"
        ]  # get_session_info wraps data

        # Verify preference persisted throughout workflow
        assert final_data["explicit_library_preference"] == "SeleniumLibrary"

        # Verify Browser Library was not added during execution
        search_order = final_data.get("search_order", [])
        assert "Browser" not in search_order
        assert "SeleniumLibrary" in search_order

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolation(self, mcp_client):
        """Test that multiple sessions with different preferences don't interfere."""
        # Create first session with Selenium preference
        selenium_scenario = "Use SeleniumLibrary for web testing"
        selenium_result = await mcp_client.call_tool(
            "analyze_scenario", {"scenario": selenium_scenario, "context": "web"}
        )
        assert selenium_result.data is not None
        selenium_session_id = selenium_result.data["session_info"]["session_id"]

        # Create second session with Browser preference
        browser_scenario = "Use Browser Library for modern automation"
        browser_result = await mcp_client.call_tool(
            "analyze_scenario", {"scenario": browser_scenario, "context": "web"}
        )
        assert browser_result.data is not None
        browser_session_id = browser_result.data["session_info"]["session_id"]

        # Verify sessions have different IDs
        assert selenium_session_id != browser_session_id

        # Execute steps in both sessions
        await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Log",
                "arguments": ["Selenium session"],
                "session_id": selenium_session_id,
            },
        )

        await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Log",
                "arguments": ["Browser session"],
                "session_id": browser_session_id,
            },
        )

        # Verify both sessions maintain their preferences
        selenium_info = await mcp_client.call_tool(
            "get_session_info", {"session_id": selenium_session_id}
        )
        browser_info = await mcp_client.call_tool(
            "get_session_info", {"session_id": browser_session_id}
        )

        assert selenium_info.data is not None
        assert browser_info.data is not None

        # Verify isolation - extract session_info from get_session_info responses
        selenium_session_data = selenium_info.data["session_info"]
        browser_session_data = browser_info.data["session_info"]

        assert selenium_session_data["explicit_library_preference"] == "SeleniumLibrary"
        assert browser_session_data["explicit_library_preference"] == "Browser"

        # Verify no cross-contamination
        selenium_search_order = selenium_session_data.get("search_order", [])
        browser_search_order = browser_session_data.get("search_order", [])

        assert "Browser" not in selenium_search_order
        assert "SeleniumLibrary" not in browser_search_order


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
