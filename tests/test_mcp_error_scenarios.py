"""Error scenario tests for MCP tools to ensure robust error handling."""

import pytest
import pytest_asyncio
import asyncio
import sys
import os
from typing import Dict, List, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastmcp import Client
from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    """Create FastMCP client connected to the server."""
    async with Client(mcp) as client:
        yield client


class TestMCPErrorScenarios:
    """Test error handling and edge cases for all MCP tools."""

    @pytest.mark.asyncio
    async def test_execute_step_invalid_keyword(self, mcp_client):
        """Test execute_step with invalid keyword - should raise exception."""
        with pytest.raises(Exception) as exc_info:
            await mcp_client.call_tool(
                "execute_step",
                {
                    "keyword": "NonExistentKeyword123",
                    "arguments": ["test"],
                    "session_id": "error_test",
                    "raise_on_failure": True
                }
            )
        
        # Should contain error information
        assert "NonExistentKeyword123" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_execute_step_invalid_keyword_no_raise(self, mcp_client):
        """Test execute_step with invalid keyword - should return error details."""
        result = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "NonExistentKeyword123",
                "arguments": ["test"],
                "session_id": "error_test",
                "raise_on_failure": False
            }
        )
        
        assert result.data["success"] is False
        assert "error" in result.data
        assert "NonExistentKeyword123" in result.data["error"]

    @pytest.mark.asyncio
    async def test_analyze_scenario_empty_string(self, mcp_client):
        """Test analyze_scenario with empty scenario."""
        result = await mcp_client.call_tool(
            "analyze_scenario",
            {"scenario": "", "context": "web"}
        )
        
        # Should handle gracefully and provide feedback
        assert isinstance(result.data, dict)
        # Tool handles empty scenarios gracefully - just check it returns something

    @pytest.mark.asyncio
    async def test_analyze_scenario_very_short(self, mcp_client):
        """Test analyze_scenario with very short scenario."""
        result = await mcp_client.call_tool(
            "analyze_scenario",
            {"scenario": "hi", "context": "web"}
        )
        
        # Should handle gracefully
        assert isinstance(result.data, dict)

    @pytest.mark.asyncio
    async def test_discover_keywords_empty_action(self, mcp_client):
        """Test discover_keywords with empty action description."""
        result = await mcp_client.call_tool(
            "discover_keywords",
            {"action_description": "", "context": "web"}
        )
        
        # Should handle gracefully
        assert isinstance(result.data, dict)

    @pytest.mark.asyncio
    async def test_get_keyword_documentation_nonexistent(self, mcp_client):
        """Test getting documentation for non-existent keyword."""
        result = await mcp_client.call_tool(
            "get_keyword_documentation",
            {"keyword_name": "NonExistentKeyword123"}
        )
        
        # Should return structured response indicating keyword not found
        assert isinstance(result.data, dict)
        assert (result.data.get("success") is False or "error" in result.data or "not found" in str(result.data).lower())

    @pytest.mark.asyncio
    async def test_search_keywords_no_matches(self, mcp_client):
        """Test searching for keywords with pattern that matches nothing."""
        result = await mcp_client.call_tool(
            "search_keywords",
            {"pattern": "XyZzQqWwEeRrTtYy123456789"}
        )
        
        # Should return empty list or no matches
        assert isinstance(result.data, list)
        assert len(result.data) == 0

    @pytest.mark.asyncio
    async def test_check_library_availability_invalid_libraries(self, mcp_client):
        """Test checking availability of completely invalid libraries."""
        result = await mcp_client.call_tool(
            "check_library_availability",
            {"libraries": ["FakeLibrary123", "AnotherFakeLib456", "NotReal789"]}
        )
        
        assert isinstance(result.data, dict)
        # Check for actual response structure
        assert ("missing_libraries" in result.data or "unavailable" in result.data)
        if "missing_libraries" in result.data:
            assert len(result.data["missing_libraries"]) == 3
            assert "FakeLibrary123" in result.data["missing_libraries"]
        elif "unavailable" in result.data:
            assert "FakeLibrary123" in result.data["unavailable"]

    @pytest.mark.asyncio
    async def test_get_library_status_nonexistent(self, mcp_client):
        pytest.skip("get_library_status tool is disabled in this build")
        """Test getting status of non-existent library."""
        result = await mcp_client.call_tool(
            "get_library_status",
            {"library_name": "NonExistentLibrary123"}
        )
        
        assert isinstance(result.data, dict)
        # Handle different possible response structures
        assert ("available" in result.data or "status" in result.data or "library_name" in result.data)

    @pytest.mark.asyncio
    async def test_get_application_state_invalid_session(self, mcp_client):
        """Test getting application state for invalid session."""
        result = await mcp_client.call_tool(
            "get_application_state",
            {"session_id": "definitely_nonexistent_session_123"}
        )
        
        # Should handle gracefully and create session or return appropriate error
        assert isinstance(result.data, dict)

    @pytest.mark.asyncio
    async def test_get_page_source_no_browser(self, mcp_client):
        """Test getting page source when no browser is open."""
        result = await mcp_client.call_tool(
            "get_page_source",
            {"session_id": "no_browser_session"}
        )
        
        # Should handle gracefully
        assert isinstance(result.data, dict)
        # Either returns empty/error state or creates a session

    @pytest.mark.asyncio
    async def test_get_page_source_attach_fallback(self, monkeypatch):
        """Ensure attach bridge failures fall back to local page source retrieval."""
        import robotmcp.server as server_module

        calls: List[str] = []

        class DummyClient:
            def run_keyword(self, keyword, args):
                calls.append(keyword)
                return {"success": False, "error": "keyword-not-available"}

        dummy_client = DummyClient()
        monkeypatch.setattr(
            server_module, "_get_external_client_if_configured", lambda: dummy_client
        )

        async def fake_get_page_source(
            session_id: str,
            full_source: bool,
            filtered: bool,
            filtering_level: str,
            include_reduced_dom: bool,
        ):
            return {
                "success": True,
                "session_id": session_id,
                "page_source": "<html></html>",
                "aria_snapshot": {"success": False, "skipped": not include_reduced_dom},
            }

        monkeypatch.setattr(
            server_module.execution_engine, "get_page_source", fake_get_page_source
        )

        result = await server_module.get_page_source.fn(
            session_id="attach_fallback_session", full_source=True
        )

        assert result["success"] is True
        assert result["page_source"] == "<html></html>"
        assert calls == ["Get Page Source", "Get Source"]

    @pytest.mark.asyncio
    async def test_build_test_suite_empty_session(self, mcp_client):
        """Test building test suite from session with no steps."""
        result = await mcp_client.call_tool(
            "build_test_suite",
            {
                "test_name": "Empty Test",
                "session_id": "empty_session_123"
            }
        )
        
        # Should handle empty session gracefully
        assert isinstance(result.data, dict)
        # Might succeed with empty suite or provide appropriate feedback

    @pytest.mark.asyncio
    async def test_validate_scenario_malformed_input(self, mcp_client):
        """Test validate_scenario with malformed input."""
        pytest.xfail("validate_scenario tool not listed in this build")
        malformed_scenario = {
            "invalid_structure": "this is not a proper scenario"
        }
        
        result = await mcp_client.call_tool(
            "validate_scenario",
            {"parsed_scenario": malformed_scenario}
        )
        
        # Should handle malformed input gracefully
        assert isinstance(result.data, dict)

    @pytest.mark.asyncio
    async def test_recommend_libraries_invalid_context(self, mcp_client):
        """Test library recommendations with invalid context."""
        result = await mcp_client.call_tool(
            "recommend_libraries",
            {
                "scenario": "Test something",
                "context": "invalid_context_type_123"
            }
        )
        
        # Should still provide recommendations or handle gracefully
        assert isinstance(result.data, dict)

    @pytest.mark.xfail(reason="suggest_next_step tool not listed in this build", strict=False)
    @pytest.mark.asyncio
    async def test_suggest_next_step_empty_state(self, mcp_client):
        """Test suggesting next step with empty/invalid state."""
        result = await mcp_client.call_tool(
            "suggest_next_step",
            {
                "current_state": {},
                "test_objective": "",
                "session_id": "empty_suggestion_test"
            }
        )
        
        # Should handle empty inputs gracefully
        assert isinstance(result.data, dict)

    # TOOL DISABLED: validate_step_before_suite
    # This tool has been disabled due to functional redundancy with execute_step().
    # See server.py lines 400-474 for detailed explanation.
    # 
    # @pytest.mark.asyncio
    # async def test_validate_step_before_suite_invalid_keyword(self, mcp_client):
    #     """Test validating invalid step before suite."""
    #     result = await mcp_client.call_tool(
    #         "validate_step_before_suite",
    #         {
    #             "keyword": "InvalidKeywordForValidation123",
    #             "arguments": ["test"],
    #             "session_id": "validation_error_test"
    #         }
    #     )
    #     
    #     # Should return validation failure
    #     assert isinstance(result.data, dict)
    #     # Check for failure indicators
    #     assert (result.data.get("validated") is False or 
    #             result.data.get("success") is False or 
    #             "error" in result.data)

    @pytest.mark.asyncio
    async def test_get_session_validation_status_nonexistent(self, mcp_client):
        """Test getting validation status for non-existent session."""
        result = await mcp_client.call_tool(
            "get_session_validation_status",
            {"session_id": "nonexistent_validation_session_123"}
        )
        
        # Should handle gracefully - may return error or default response
        assert isinstance(result.data, dict)
        assert ("total_steps" in result.data or "error" in result.data or "success" in result.data)

    @pytest.mark.asyncio
    async def test_validate_test_readiness_empty_session(self, mcp_client):
        """Test validating test readiness for empty session."""
        pytest.skip("validate_test_readiness tool is disabled in this build")
        result = await mcp_client.call_tool(
            "validate_test_readiness",
            {"session_id": "empty_readiness_session_123"}
        )
        
        # Should indicate not ready
        assert isinstance(result.data, dict)
        # Handle different possible response structures
        assert ("ready" in result.data or "ready_for_suite_generation" in result.data or "guidance" in result.data)

    @pytest.mark.asyncio
    async def test_execute_step_invalid_arguments(self, mcp_client):
        """Test execute_step with invalid argument types."""
        # FastMCP validates input types, so this will raise a validation error
        with pytest.raises(Exception) as exc_info:
            await mcp_client.call_tool(
                "execute_step",
                {
                    "keyword": "Log",
                    "arguments": [123, {"invalid": "dict"}, ["nested", "list"]],
                    "session_id": "invalid_args_test",
                    "raise_on_failure": False
                }
            )
        
        # Should contain validation error information
        assert "validation error" in str(exc_info.value).lower() or "not of type" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_selenium_locator_guidance_with_error(self, mcp_client):
        """Test Selenium locator guidance with specific error context."""
        result = await mcp_client.call_tool(
            "get_selenium_locator_guidance",
            {
                "error_message": "Element with locator 'id=nonexistent' not found",
                "keyword_name": "Click Element"
            }
        )
        
        assert isinstance(result.data, dict)
        # Check for guidance content (may not have error_analysis field specifically)
        assert ("locator_strategies" in result.data or "guidance" in result.data or "tips" in result.data)

    @pytest.mark.asyncio
    async def test_get_browser_locator_guidance_strict_mode_error(self, mcp_client):
        """Test Browser Library guidance with strict mode error."""
        result = await mcp_client.call_tool(
            "get_browser_locator_guidance",
            {
                "error_message": "strict mode violation: multiple elements found",
                "keyword_name": "Click"
            }
        )
        
        assert isinstance(result.data, dict)
        # Check for guidance content - may include strict mode info
        assert ("locator_strategies" in result.data or 
                "strict_mode" in str(result.data).lower() or 
                "selector_patterns" in result.data or
                "guidance" in result.data)


class TestMCPEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_execute_step_maximum_arguments(self, mcp_client):
        """Test execute_step with many arguments."""
        many_args = [f"arg_{i}" for i in range(50)]
        
        result = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Log Many",  # This will likely fail, but should handle gracefully
                "arguments": many_args,
                "session_id": "many_args_test",
                "raise_on_failure": False
            }
        )
        
        assert isinstance(result.data, dict)

    @pytest.mark.asyncio
    async def test_analyze_scenario_very_long(self, mcp_client):
        """Test analyze_scenario with very long scenario text."""
        long_scenario = "Test user login functionality. " * 1000  # Very long scenario
        
        result = await mcp_client.call_tool(
            "analyze_scenario",
            {"scenario": long_scenario, "context": "web"}
        )
        
        # Should handle long text gracefully
        assert isinstance(result.data, dict)

    @pytest.mark.asyncio
    async def test_recommend_libraries_max_recommendations(self, mcp_client):
        """Test library recommendations with maximum limit."""
        result = await mcp_client.call_tool(
            "recommend_libraries",
            {
                "scenario": "Complex testing scenario requiring many libraries",
                "context": "web",
                "max_recommendations": 100  # Very high number
            }
        )
        
        assert isinstance(result.data, dict)
        # Check for actual response structure
        assert ("recommendations" in result.data or "recommended_libraries" in result.data)
        # Should be limited to reasonable number even if max is high

    @pytest.mark.asyncio
    async def test_get_available_keywords_specific_library_filter(self, mcp_client):
        """Test getting keywords with very specific library filter."""
        result = await mcp_client.call_tool(
            "get_available_keywords",
            {"library_name": "Collections"}  # Specific standard library
        )
        
        if isinstance(result.data, list) and len(result.data) > 0:
            # Check only valid dict keywords (skip Root() objects)
            valid_keywords = [k for k in result.data if isinstance(k, dict) and "library" in k]
            if valid_keywords:
                # All valid keywords should be from Collections library
                for keyword in valid_keywords:
                    assert keyword["library"] == "Collections"

    @pytest.mark.asyncio
    async def test_search_keywords_special_characters(self, mcp_client):
        """Test keyword search with special characters."""
        result = await mcp_client.call_tool(
            "search_keywords",
            {"pattern": "log*"}  # Pattern with special characters
        )
        
        # Should handle special characters in search
        assert isinstance(result.data, list)

    @pytest.mark.asyncio
    async def test_execute_step_variable_assignment_multiple(self, mcp_client):
        """Test variable assignment with multiple variables."""
        result = await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Create List",
                "arguments": ["item1", "item2", "item3"],
                "session_id": "multi_assign_test",
                "assign_to": ["list_var", "backup_var"]  # Multiple assignment
            }
        )
        
        # Should handle multiple variable assignment
        assert isinstance(result.data, dict)
        # Check if step succeeded (variable info may not be in response)
        assert result.data.get("success") is True

    @pytest.mark.asyncio
    async def test_build_test_suite_special_characters_in_name(self, mcp_client):
        """Test building test suite with special characters in name."""
        # First add a step
        await mcp_client.call_tool(
            "execute_step",
            {
                "keyword": "Log",
                "arguments": ["Test with special chars"],
                "session_id": "special_chars_session"
            }
        )
        
        result = await mcp_client.call_tool(
            "build_test_suite",
            {
                "test_name": "Test with Spëciål Çhårs & Symbols!",
                "session_id": "special_chars_session"
            }
        )
        
        # Should handle special characters in test names
        assert isinstance(result.data, dict)


# Test runner for manual execution
if __name__ == "__main__":
    async def run_error_scenario_tests():
        """Run error scenario tests manually."""
        print("Running MCP error scenario tests...")
        
        from robotmcp.server import mcp
        
        async with Client(mcp) as client:
            # Test invalid keyword execution
            try:
                await client.call_tool(
                    "execute_step",
                    {
                        "keyword": "NonExistentKeyword",
                        "arguments": ["test"],
                        "raise_on_failure": True
                    }
                )
                print("Invalid keyword test: FAIL (should have raised exception)")
            except Exception:
                print("Invalid keyword test: PASS (correctly raised exception)")
            
            # Test error handling without raising
            result = await client.call_tool(
                "execute_step",
                {
                    "keyword": "NonExistentKeyword",
                    "arguments": ["test"],
                    "raise_on_failure": False
                }
            )
            print(f"Error handling test: {'PASS' if not result.data.get('success') else 'FAIL'}")
            
            # Test empty scenario
            result = await client.call_tool(
                "analyze_scenario",
                {"scenario": ""}
            )
            print(f"Empty scenario test: {'PASS' if isinstance(result.data, dict) else 'FAIL'}")
            
            # Test invalid library check
            result = await client.call_tool(
                "check_library_availability",
                {"libraries": ["FakeLibrary123"]}
            )
            print(f"Invalid library test: {'PASS' if 'FakeLibrary123' in result.data.get('unavailable', []) else 'FAIL'}")
            
            print("Error scenario tests completed!")
    
    asyncio.run(run_error_scenario_tests())
