"""
E2E tests for RF context keyword listing and documentation tools.

Covers:
- import_resource
- import_custom_library
- list_available_keywords
- get_session_keyword_documentation
- diagnose_rf_context (sanity)
"""

import os
import sys
import pytest
import pytest_asyncio

# Ensure src is on path for test runtime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fastmcp import Client
from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_list_keywords_after_context_creation(mcp_client):
    session_id = "kw_docs_s1"

    # Create RF context by executing a simple BuiltIn keyword
    res = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "Log",
            "arguments": ["hello"],
            "session_id": session_id,
            "use_context": True,
            "raise_on_failure": False,
        },
    )
    assert res.data["success"] is True

    # List available keywords
    lst = await mcp_client.call_tool(
        "list_available_keywords", {"session_id": session_id}
    )
    assert lst.data["success"] is True
    assert lst.data["libraries_count"] >= 1
    libs = lst.data.get("library_keywords", [])

    # BuiltIn keywords should be present (e.g., Log)
    names = {k["name"] for k in libs}
    assert any(name.lower() == "log" for name in names)


@pytest.mark.asyncio
async def test_imports_and_keyword_listing_and_docs(mcp_client):
    session_id = "kw_docs_s2"
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    resource_path = os.path.join(base, "test_data", "resources", "sample.resource")
    custom_lib_path = os.path.join(base, "test_data", "libs", "custom_lib.py")

    # Create context implicitly by importing a resource
    res_imp = await mcp_client.call_tool(
        "import_resource", {"session_id": session_id, "path": resource_path}
    )
    assert res_imp.data["success"] is True

    # Import custom library by file path
    lib_imp = await mcp_client.call_tool(
        "import_custom_library",
        {"session_id": session_id, "name_or_path": custom_lib_path},
    )
    assert lib_imp.data["success"] is True

    # List keywords should include resource keyword and custom library keyword
    lst = await mcp_client.call_tool(
        "list_available_keywords", {"session_id": session_id}
    )
    assert lst.data["success"] is True
    lib_kws = lst.data.get("library_keywords", [])
    res_kws = lst.data.get("resource_keywords", [])

    lib_kw_names = {k["name"] for k in lib_kws}
    res_kw_names = {k["name"] for k in res_kws}

    # CustomLib keyword names use Robot casing with spaces
    assert any(name.lower() == "kw no args" for name in (n.lower() for n in lib_kw_names))
    assert "Hello Message" in res_kw_names

    # Keyword documentation lookups
    doc_res = await mcp_client.call_tool(
        "get_session_keyword_documentation",
        {"session_id": session_id, "keyword_name": "Hello Message"},
    )
    assert doc_res.data["success"] is True
    assert doc_res.data["type"] == "resource"
    assert doc_res.data["name"] == "Hello Message"
    assert isinstance(doc_res.data.get("args", []), list)

    doc_lib = await mcp_client.call_tool(
        "get_session_keyword_documentation",
        {"session_id": session_id, "keyword_name": "Kw With Args"},
    )
    assert doc_lib.data["success"] is True
    assert doc_lib.data["type"] == "library"
    assert doc_lib.data["name"].lower() == "kw with args"
    assert isinstance(doc_lib.data.get("args", []), list)

    # Diagnose context for sanity
    diag = await mcp_client.call_tool(
        "diagnose_rf_context", {"session_id": session_id}
    )
    assert diag.data.get("context_exists") is True
    assert diag.data.get("variable_count") >= 0
