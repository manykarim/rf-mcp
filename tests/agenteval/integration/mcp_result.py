"""Support keywords for porting rf-mcp black-box integration tests to the agenteval
harness (change: agenteval-port-blackbox-integration, Phase 2a).

rf-mcp's MCP tools return their payload as JSON inside a text content block, so the
pytest originals read ``result.data`` (fastmcp auto-parses it). agenteval's
``MCP.Call Tool`` returns the raw ``MCPToolResult.content``, so these keywords turn
that into a dict and assert on it. ``Rf Tool`` calls + parses in one step; the call
still lands in MCPLibrary's recorded tool-call trace (it goes through the same
``call_tool`` method), so the trace readers keep working.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from robot.api.deco import keyword, library
from robot.libraries.BuiltIn import BuiltIn


@library(scope="GLOBAL")
class mcp_result:
    @keyword("Parse Tool Result")
    def parse_tool_result(self, result: Any) -> Dict[str, Any]:
        """Parse the JSON payload from an rf-mcp ``MCPToolResult``'s text content."""
        content = getattr(result, "content", None) or []
        for block in content:
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if text:
                try:
                    return json.loads(text)
                except (ValueError, TypeError):
                    return {"text": text}
        return {}

    @keyword("Rf Tool")
    def rf_tool(self, handle: Any, tool: str, arguments: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Call an rf-mcp MCP tool (arguments as a dict) and return the parsed payload.
        The call is recorded in MCPLibrary's tool-call trace."""
        mcplib = BuiltIn().get_library_instance("MCPLibrary")
        result = mcplib.call_tool(handle, tool, arguments=arguments or {})
        data = self.parse_tool_result(result)
        if isinstance(data, dict):
            data.setdefault("is_error", getattr(result, "is_error", False))
        return data

    @keyword("Result Field Should Be")
    def result_field_should_be(self, data: Dict[str, Any], field: str, expected: Any) -> None:
        actual = data.get(field)
        if str(actual) != str(expected):
            raise AssertionError(f"result[{field!r}] = {actual!r}, expected {expected!r}")

    @keyword("Result Should Contain Field")
    def result_should_contain_field(self, data: Dict[str, Any], field: str) -> None:
        if field not in data:
            raise AssertionError(f"result is missing field {field!r}; keys={list(data)}")
