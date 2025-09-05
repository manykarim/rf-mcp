"""E2E tests to validate named argument parsing for Appium and locator-like args.

Uses the debug_parse_keyword_arguments tool to confirm that name=value pairs are
treated as named kwargs for keywords that accept **kwargs, and that selector-like
positional strings are preserved as positional unless they are true parameters.
"""

import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import mcp


@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_appium_open_application_named_kwargs(mcp_client):
    # Simulate typical Open Application capabilities
    args = [
        "http://localhost:4723",
        "platformName=Android",
        "deviceName=emulator-5554",
        "app=/abs/path/to/SauceLabs.apk",
        "automationName=UiAutomator2",
        "newCommandTimeout=300",
    ]

    res = await mcp_client.call_tool(
        "debug_parse_keyword_arguments",
        {
            "keyword_name": "Open Application",
            "arguments": args,
            "library_name": "AppiumLibrary",
        },
    )
    assert res.data.get("success") is True
    parsed = res.data.get("parsed", {})
    pos = parsed.get("positional", [])
    named = parsed.get("named", {})

    # First arg (server URL) positional; capabilities go to named kwargs
    assert pos and pos[0] == "http://localhost:4723"
    for key in [
        "platformName",
        "deviceName",
        "app",
        "automationName",
        "newCommandTimeout",
    ]:
        assert key in named, f"Expected named arg '{key}' in parsed kwargs"


@pytest.mark.asyncio
async def test_locator_like_strings_treated_correctly(mcp_client):
    # For a Browser keyword with a 'selector' param, this should be named
    res_named = await mcp_client.call_tool(
        "debug_parse_keyword_arguments",
        {
            "keyword_name": "Click",
            "arguments": ["selector=id=username"],
            "library_name": "Browser",
        },
    )
    assert res_named.data.get("success") is True
    parsed_named = res_named.data.get("parsed", {})
    assert parsed_named.get("named", {}).get("selector") == "id=username"

    # Plain 'id=username' without a matching param name should be positional
    res_pos = await mcp_client.call_tool(
        "debug_parse_keyword_arguments",
        {
            "keyword_name": "Click",
            "arguments": ["id=username"],
            "library_name": "Browser",
        },
    )
    assert res_pos.data.get("success") is True
    parsed_pos = res_pos.data.get("parsed", {})
    assert parsed_pos.get("positional", [])[0] == "id=username"

