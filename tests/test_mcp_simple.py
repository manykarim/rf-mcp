"""Simple MCP tools test to verify client connection and response structure."""

import pytest
import pytest_asyncio
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastmcp import Client
from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    """Create FastMCP client connected to the server."""
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_simple_log_execution(mcp_client):
    """Test simple Log keyword execution to verify client connection."""
    result = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Log",
            "arguments": ["Hello World"],
            "session_id": "simple_test"
        }
    )
    
    print(f"Result type: {type(result)}")
    print(f"Result: {result}")
    print(f"Result data: {result.data if hasattr(result, 'data') else 'No data attribute'}")
    
    # Check basic response structure
    assert hasattr(result, 'data'), "Result should have data attribute"
    assert isinstance(result.data, dict), "Result data should be a dictionary"


@pytest.mark.asyncio
async def test_analyze_scenario_structure(mcp_client):
    """Test analyze_scenario to understand response structure."""
    result = await mcp_client.call_tool(
        "analyze_scenario",
        {"scenario": "Test login functionality", "context": "web"}
    )
    
    print(f"Analyze scenario result: {result}")
    print(f"Analyze scenario data: {result.data}")
    
    assert hasattr(result, 'data'), "Result should have data attribute"
    assert isinstance(result.data, dict), "Result data should be a dictionary"


@pytest.mark.asyncio
async def test_find_keywords_structure(mcp_client):
    """Test find_keywords to understand response structure."""
    result = await mcp_client.call_tool(
        "find_keywords",
        {"query": "Log", "strategy": "pattern"}
    )
    
    print(f"Keywords result: {result}")
    print(f"Keywords data type: {type(result.data)}")
    if isinstance(result.data.get("results"), list) and len(result.data["results"]) > 0:
        print(f"First keyword: {result.data['results'][0]}")
    
    assert hasattr(result, 'data'), "Result should have data attribute"


@pytest.mark.asyncio
async def test_locator_guidance_structure(mcp_client):
    """Test locator guidance to understand response structure."""
    result = await mcp_client.call_tool(
        "get_locator_guidance",
        {"library": "selenium", "keyword_name": "Click Element"}
    )
    
    print(f"Selenium guidance result: {result}")
    print(f"Selenium guidance data: {result.data}")
    
    assert hasattr(result, 'data'), "Result should have data attribute"
    assert isinstance(result.data, dict), "Result data should be a dictionary"


if __name__ == "__main__":
    import asyncio
    
    async def manual_test():
        """Manual test runner to understand the MCP response structure."""
        print("Testing MCP client connection...")
        
        async with Client(mcp) as client:
            # Test execute_step
            print("\n=== Testing execute_step ===")
            result = await client.call_tool(
                "execute_step",
                {
                    "keyword": "Log",
                    "arguments": ["Manual test message"],
                    "session_id": "manual_test"
                }
            )
            print(f"Execute step result: {result}")
            print(f"Execute step data: {result.data}")
            
            # Test analyze_scenario
            print("\n=== Testing analyze_scenario ===")
            result = await client.call_tool(
                "analyze_scenario",
                {"scenario": "Test user login", "context": "web"}
            )
            print(f"Analyze scenario result: {result}")
            print(f"Analyze scenario data: {result.data}")
            
            # Test find_keywords
            print("\n=== Testing find_keywords ===")
            result = await client.call_tool(
                "find_keywords",
                {"query": "Log", "strategy": "pattern", "limit": 2}
            )
            print(f"Keywords result type: {type(result.data)}")
            if isinstance(result.data.get("results"), list):
                print(f"Keywords count: {len(result.data['results'])}")
                if len(result.data["results"]) > 0:
                    print(f"First keyword: {result.data['results'][0]}")
            
            print("\nManual tests completed!")
    
    asyncio.run(manual_test())
