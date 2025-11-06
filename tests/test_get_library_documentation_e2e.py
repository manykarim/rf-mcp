"""
End-to-end tests for the get_library_documentation MCP tool using FastMCP Client.
Tests the complete integration from MCP tool call to Robot Framework native libdoc API.
"""

import pytest

pytestmark = pytest.mark.skip(reason="get_library_documentation superseded by get_keyword_info")

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
    """FastMCP test client for end-to-end testing."""
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_get_library_documentation_builtin_library(mcp_client):
    """Test getting documentation for BuiltIn library (always available)."""
    result = await mcp_client.call_tool("get_library_documentation", {
        "library_name": "BuiltIn"
    })
    
    assert result.data["success"] == True
    library_info = result.data["library"]
    
    # Verify basic library information
    assert library_info["name"] == "BuiltIn"
    assert library_info["type"] == "LIBRARY"
    assert "doc" in library_info
    assert library_info["keyword_count"] > 0
    assert "data_source" in library_info
    
    # Verify keywords list structure
    assert isinstance(library_info["keywords"], list)
    assert len(library_info["keywords"]) > 0
    
    # Verify first keyword has all required fields
    first_keyword = library_info["keywords"][0]
    required_fields = [
        "name", "library", "args", "arg_types", "doc", 
        "short_doc", "tags", "is_deprecated", "source", "lineno"
    ]
    
    for field in required_fields:
        assert field in first_keyword, f"Missing field: {field}"
    
    # Verify the keyword belongs to BuiltIn
    assert first_keyword["library"] == "BuiltIn"


@pytest.mark.asyncio
async def test_get_library_documentation_collections_library(mcp_client):
    """Test getting documentation for Collections library."""
    result = await mcp_client.call_tool("get_library_documentation", {
        "library_name": "Collections"
    })
    
    assert result.data["success"] == True
    library_info = result.data["library"]
    
    # Verify Collections-specific information
    assert library_info["name"] == "Collections"
    assert library_info["type"] == "LIBRARY"
    assert library_info["keyword_count"] > 0
    
    # Check for some known Collections keywords
    keyword_names = [kw["name"] for kw in library_info["keywords"]]
    collections_keywords = ["Append To List", "Get From List", "Create List"]
    
    # At least one of these should be present
    assert any(kw in keyword_names for kw in collections_keywords), f"Expected Collections keywords not found in {keyword_names[:10]}"


@pytest.mark.asyncio
async def test_get_library_documentation_string_library(mcp_client):
    """Test getting documentation for String library."""
    result = await mcp_client.call_tool("get_library_documentation", {
        "library_name": "String"
    })
    
    assert result.data["success"] == True
    library_info = result.data["library"]
    
    # Verify String-specific information
    assert library_info["name"] == "String"
    assert library_info["keyword_count"] > 0
    
    # Check for some known String keywords
    keyword_names = [kw["name"] for kw in library_info["keywords"]]
    string_keywords = ["Get Length", "Replace String", "Split String"]
    
    # At least one of these should be present
    assert any(kw in keyword_names for kw in string_keywords), f"Expected String keywords not found in {keyword_names[:10]}"


@pytest.mark.asyncio
async def test_get_library_documentation_keyword_details(mcp_client):
    """Test that keyword details are properly populated."""
    result = await mcp_client.call_tool("get_library_documentation", {
        "library_name": "BuiltIn"
    })
    
    assert result.data["success"] == True
    library_info = result.data["library"]
    
    # Find the "Log" keyword which should have detailed information
    log_keyword = None
    for keyword in library_info["keywords"]:
        if keyword["name"] == "Log":
            log_keyword = keyword
            break
    
    assert log_keyword is not None, "Log keyword not found in BuiltIn library"
    
    # Verify detailed keyword information
    assert log_keyword["library"] == "BuiltIn"
    assert isinstance(log_keyword["args"], list)
    assert len(log_keyword["args"]) > 0  # Log should have at least 'message' argument
    assert isinstance(log_keyword["doc"], str)
    assert len(log_keyword["doc"]) > 0
    assert isinstance(log_keyword["short_doc"], str)
    assert isinstance(log_keyword["tags"], list)
    assert isinstance(log_keyword["is_deprecated"], bool)


@pytest.mark.asyncio
async def test_get_library_documentation_nonexistent_library(mcp_client):
    """Test behavior with non-existent library."""
    result = await mcp_client.call_tool("get_library_documentation", {
        "library_name": "NonExistentLibrary"
    })
    
    assert result.data["success"] == False
    assert "error" in result.data
    assert "NonExistentLibrary" in result.data["error"]
    assert "not found" in result.data["error"].lower()


@pytest.mark.asyncio
async def test_get_library_documentation_data_source_indication(mcp_client):
    """Test that data source is properly indicated (libdoc vs inspection)."""
    result = await mcp_client.call_tool("get_library_documentation", {
        "library_name": "BuiltIn"
    })
    
    assert result.data["success"] == True
    library_info = result.data["library"]
    
    # Should indicate data source
    assert "data_source" in library_info
    assert library_info["data_source"] in ["libdoc", "inspection"]


@pytest.mark.asyncio
async def test_get_library_documentation_keyword_count_accuracy(mcp_client):
    """Test that keyword_count matches actual number of keywords returned."""
    result = await mcp_client.call_tool("get_library_documentation", {
        "library_name": "Collections"
    })
    
    assert result.data["success"] == True
    library_info = result.data["library"]
    
    # Verify keyword count matches actual list length
    actual_count = len(library_info["keywords"])
    reported_count = library_info["keyword_count"]
    
    assert actual_count == reported_count, f"Keyword count mismatch: reported {reported_count}, actual {actual_count}"


@pytest.mark.asyncio
async def test_get_library_documentation_args_and_arg_types(mcp_client):
    """Test that keyword arguments and argument types are properly extracted."""
    result = await mcp_client.call_tool("get_library_documentation", {
        "library_name": "BuiltIn"
    })
    
    assert result.data["success"] == True
    library_info = result.data["library"]
    
    # Find a keyword with known arguments (e.g., "Set Variable")
    set_var_keyword = None
    for keyword in library_info["keywords"]:
        if keyword["name"] == "Set Variable":
            set_var_keyword = keyword
            break
    
    if set_var_keyword:  # Only test if keyword is found
        assert isinstance(set_var_keyword["args"], list)
        assert isinstance(set_var_keyword["arg_types"], list)
        assert len(set_var_keyword["args"]) > 0  # Set Variable should have arguments
        
        # Arguments should be strings
        for arg in set_var_keyword["args"]:
            assert isinstance(arg, str)


@pytest.mark.asyncio 
async def test_get_library_documentation_comprehensive_structure(mcp_client):
    """Test the complete structure of the returned library documentation."""
    result = await mcp_client.call_tool("get_library_documentation", {
        "library_name": "String"
    })
    
    assert result.data["success"] == True
    library_info = result.data["library"]
    
    # Verify top-level library fields
    required_library_fields = [
        "name", "doc", "version", "type", "scope", 
        "source", "keywords", "keyword_count", "data_source"
    ]
    
    for field in required_library_fields:
        assert field in library_info, f"Missing library field: {field}"
    
    # Verify each keyword has all required fields
    required_keyword_fields = [
        "name", "library", "args", "arg_types", "doc", 
        "short_doc", "tags", "is_deprecated", "source", "lineno"
    ]
    
    for keyword in library_info["keywords"][:3]:  # Check first 3 keywords
        for field in required_keyword_fields:
            assert field in keyword, f"Missing keyword field: {field} in keyword: {keyword['name']}"
        
        # Verify field types
        assert isinstance(keyword["name"], str)
        assert isinstance(keyword["library"], str)
        assert isinstance(keyword["args"], list)
        assert isinstance(keyword["arg_types"], list)
        assert isinstance(keyword["doc"], str)
        assert isinstance(keyword["short_doc"], str)
        assert isinstance(keyword["tags"], list)
        assert isinstance(keyword["is_deprecated"], bool)
        assert isinstance(keyword["source"], str)
        assert isinstance(keyword["lineno"], int)


@pytest.mark.asyncio
async def test_get_library_documentation_robot_framework_integration(mcp_client):
    """Test integration with Robot Framework native libdoc system."""
    result = await mcp_client.call_tool("get_library_documentation", {
        "library_name": "BuiltIn"
    })
    
    assert result.data["success"] == True
    library_info = result.data["library"]
    
    # If using libdoc, should have proper Robot Framework metadata
    if library_info["data_source"] == "libdoc":
        # Should have proper Robot Framework library scope
        assert library_info["scope"] in ["GLOBAL", "SUITE", "TEST"]
        
        # Should have version information
        assert library_info["version"] != "Unknown"
        
        # Keywords should have proper libdoc metadata
        sample_keyword = library_info["keywords"][0]
        assert isinstance(sample_keyword["is_deprecated"], bool)
        
        # May have source file information
        if sample_keyword["source"]:
            assert isinstance(sample_keyword["source"], str)
            assert isinstance(sample_keyword["lineno"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
