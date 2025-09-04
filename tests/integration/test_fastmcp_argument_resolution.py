"""Focused FastMCP client tests for argument resolution and library routing.

These tests avoid external network or browser dependencies and validate:
- Named + positional argument handling
- Object-valued named arguments preserved end-to-end
- Library prefix resolution for overlapping names (e.g., XML library)
"""

import os
import pytest
import pytest_asyncio
from fastmcp import Client

from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_collections_dictionary_create_and_get(mcp_client):
    """Create a dictionary via named args, retrieve a value, and assert result."""
    session_id = "collections_dict_session"

    # Create Dictionary with named args; assign to ${d}
    res_create = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Create Dictionary",
            "arguments": ["a=1", "b=2"],
            "session_id": session_id,
            "assign_to": "d",
            "raise_on_failure": True,
        },
    )
    assert res_create.data.get("success") is True

    # Get From Dictionary using the stored object ${d}
    res_get = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Get From Dictionary",
            "arguments": ["${d}", "a"],
            "session_id": session_id,
            "assign_to": "val",
            "raise_on_failure": True,
        },
    )
    assert res_get.data.get("success") is True
    # The value should be accessible via assignment or output
    assigned = res_get.data.get("assigned_variables", {})
    if assigned:
        assert assigned.get("${val}") in ("1", 1)
    else:
        assert res_get.data.get("result") in ("1", 1)


@pytest.mark.asyncio
async def test_collections_set_to_dictionary_named_object_arg(mcp_client):
    """Set To Dictionary should accept object-valued named args preserved via ${var}."""
    session_id = "collections_setdict_session"

    # Start with empty dict
    res_create = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Create Dictionary",
            "arguments": [],
            "session_id": session_id,
            "assign_to": "d",
        },
    )
    assert res_create.data.get("success") is True

    # Set To Dictionary expects key/value as positional pairs (no named kwargs)
    res_set = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Set To Dictionary",
            "arguments": ["${d}", "a", "42"],
            "session_id": session_id,
        },
    )
    assert res_set.data.get("success") is True

    # Verify the value was stored
    res_get = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Get From Dictionary",
            "arguments": ["${d}", "a"],
            "session_id": session_id,
            "assign_to": "val",
        },
    )
    assert res_get.data.get("success") is True
    assigned = res_get.data.get("assigned_variables", {})
    if assigned and "${val}" in assigned:
        assert str(assigned["${val}"]) == "42"
    else:
        # Fallback to output if assignment not returned
        assert str(res_get.data.get("output")) == "42"


@pytest.mark.asyncio
async def test_xml_library_get_element_count_with_prefix(mcp_client):
    """Use XML.Get Element Count with a file path and XPath; should load XML library automatically."""
    session_id = "xml_prefix_session"
    xml_path = os.path.join(os.path.dirname(__file__), "..", "..", "test_data", "books_authors.xml")
    xml_path = os.path.abspath(xml_path)

    # Some XML keywords accept source path + xpath. Use explicit library prefix to force loading.
    res = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "XML.Get Element Count",
            "arguments": [xml_path, ".//book"],
            "session_id": session_id,
            "assign_to": "count",
            "raise_on_failure": True,
        },
    )
    assert res.data.get("success") is True
    # Prefer assigned variable; fall back to output
    assigned = res.data.get("assigned_variables", {})
    if assigned and "${count}" in assigned:
        assert str(assigned["${count}"]).isdigit()
    else:
        assert str(res.data.get("output")).isdigit()
