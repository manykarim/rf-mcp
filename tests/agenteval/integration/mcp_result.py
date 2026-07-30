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

    # -- Live tool-schema inspection (for ADR-009 schema-constraint tests) ------
    # These operate on the ``list[MCPTool]`` returned by ``MCP.List Tools`` — the
    # live, FastMCP-generated inputSchema the server advertises — mirroring the
    # pytest originals that read ``client.list_tools()[i].inputSchema``.

    @staticmethod
    def _tool_schema(tools: Any, tool_name: str) -> Dict[str, Any]:
        for t in tools:
            if getattr(t, "name", None) == tool_name:
                schema = getattr(t, "input_schema", None)
                return schema if isinstance(schema, dict) else {}
        available = [getattr(t, "name", "?") for t in tools]
        raise AssertionError(f"Tool {tool_name!r} not found. Available: {available}")

    @classmethod
    def _param_schema(cls, tools: Any, tool_name: str, param: str) -> Dict[str, Any]:
        props = cls._tool_schema(tools, tool_name).get("properties", {})
        if param not in props:
            raise AssertionError(
                f"Param {param!r} not in tool {tool_name!r}; params={list(props)}"
            )
        return props[param]

    @staticmethod
    def _schema_has_enum(schema: Dict[str, Any]) -> bool:
        if "enum" in schema:
            return True
        if "anyOf" in schema:
            return any(isinstance(v, dict) and "enum" in v for v in schema["anyOf"])
        return False

    @keyword("Param Enum Should Be")
    def param_enum_should_be(self, tools: Any, tool_name: str, param: str, expected: list) -> None:
        """Assert the live param schema's enum equals ``expected`` (order-insensitive,
        nulls dropped). Handles flat ``enum`` and Optional ``anyOf`` schemas."""
        schema = self._param_schema(tools, tool_name, param)
        path = f"{tool_name}.{param}"
        if "enum" in schema:
            enum = schema["enum"]
        elif "anyOf" in schema:
            variants = [v for v in schema["anyOf"] if isinstance(v, dict) and "enum" in v]
            if not variants:
                raise AssertionError(f"{path}: anyOf has no enum variant: {schema}")
            enum = variants[0]["enum"]
        else:
            raise AssertionError(f"{path}: schema has neither 'enum' nor 'anyOf': {schema}")
        non_null = sorted(v for v in enum if v is not None)
        if non_null != sorted(expected):
            raise AssertionError(f"{path}: enum {non_null}, expected {sorted(expected)}")

    @keyword("Param Should Have Enum")
    def param_should_have_enum(self, tools: Any, tool_name: str, param: str) -> None:
        schema = self._param_schema(tools, tool_name, param)
        if not self._schema_has_enum(schema):
            raise AssertionError(f"{tool_name}.{param}: no enum constraint: {schema}")

    @keyword("Param Should Be Required")
    def param_should_be_required(self, tools: Any, tool_name: str, param: str) -> None:
        required = self._tool_schema(tools, tool_name).get("required", [])
        if param not in required:
            raise AssertionError(f"{tool_name}.{param} not required; required={required}")

    @keyword("Count Enum Constrained Params")
    def count_enum_constrained_params(self, tools: Any) -> int:
        count = 0
        for t in tools:
            schema = getattr(t, "input_schema", None) or {}
            for prop_schema in schema.get("properties", {}).values():
                if self._schema_has_enum(prop_schema):
                    count += 1
        return count

    @keyword("Enum Constrained Param Names")
    def enum_constrained_param_names(self, tools: Any) -> list:
        names = []
        for t in tools:
            schema = getattr(t, "input_schema", None) or {}
            for prop_name, prop_schema in schema.get("properties", {}).items():
                if self._schema_has_enum(prop_schema):
                    names.append(f"{getattr(t, 'name', '?')}.{prop_name}")
        return names

    @keyword("Tools With Unconstrained Action Param")
    def tools_with_unconstrained_action_param(self, tools: Any) -> list:
        offenders = []
        for t in tools:
            schema = getattr(t, "input_schema", None) or {}
            props = schema.get("properties", {})
            if "action" in props and not self._schema_has_enum(props["action"]):
                offenders.append(getattr(t, "name", "?"))
        return offenders

    @keyword("Assigned Variable")
    def assigned_variable(self, data: Dict[str, Any], name: str) -> str:
        """Return an execute_step result's ``assigned_variables['${name}']`` (or '')."""
        assigned = data.get("assigned_variables", {}) if isinstance(data, dict) else {}
        return assigned.get("${" + name + "}", "")

    @keyword("Scroll Top")
    def scroll_top(self, scroll: Any):
        """Extract the ``top`` value from a Get Scroll Position result's ``output``
        (a ``{top,left,bottom,right}`` dict or its stringified form). Returns float or None.
        Mirrors the pytest ``_extract_scroll_top`` helper."""
        if isinstance(scroll, dict):
            v = scroll.get("top")
            if isinstance(v, (int, float)):
                return float(v)
        if isinstance(scroll, str):
            import re
            m = re.search(r"['\"]top['\"]\s*:\s*([\d.]+)", scroll)
            if m:
                return float(m.group(1))
        return None
