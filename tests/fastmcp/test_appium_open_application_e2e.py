"""E2E test for AppiumLibrary Open Application via FastMCP Client.

This test validates that execute_step passes capabilities as named kwargs by
asserting the Appium server no longer complains about missing 'automationName'
or 'platformName' even if the server is not running.

If a local Appium server is running at http://localhost:4723 and an Android
emulator is available, this call may succeed; otherwise we assert that any
failure is NOT due to missing capabilities.
"""

import pytest
import pytest_asyncio

from fastmcp import Client
from robotmcp.server import mcp

from tests.utils.dependency_matrix import requires_extras

pytestmark = [
    requires_extras("mobile"),
    pytest.mark.optional_dependency("mobile"),
    pytest.mark.optional_mobile,
]



@pytest_asyncio.fixture
async def mcp_client():
    async with Client(mcp) as client:
        yield client


@pytest.mark.asyncio
async def test_appium_open_application_e2e_kwargs_passed(mcp_client):
    session_id = "appium_e2e_args"

    # Ensure AppiumLibrary is first in search order for this session
    await mcp_client.call_tool(
        "set_library_search_order",
        {"libraries": ["AppiumLibrary", "BuiltIn", "Collections", "String"], "session_id": session_id},
    )

    # Execute Open Application with typical capabilities (non-prefixed for this test)
    args = [
        "http://localhost:4723",
        "platformName=Android",
        "deviceName=emulator",
        "app=C:/workspace/rf-mcp/tests/appium/SauceLabs.apk",
        "automationName=UiAutomator2",
    ]

    res = await mcp_client.call_tool(
        "execute_step",
        {
            "keyword": "AppiumLibrary.Open Application",
            "arguments": args,
            "session_id": session_id,
            "raise_on_failure": False,
        },
    )

    data = res.data
    # If it succeeds, great.
    if data.get("success") is True:
        assert True
        return

    # If it fails, ensure it's not due to missing capabilities (kwargs got dropped)
    err = (data.get("error") or "").lower()
    assert "automationname" not in err, f"Unexpected missing capability in error: {data.get('error')}"
    assert "platformname" not in err, f"Unexpected missing capability in error: {data.get('error')}"

