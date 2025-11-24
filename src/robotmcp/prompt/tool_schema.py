"""Utilities for translating MCP tool schemas into OpenAI tool definitions."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from mcp.types import Tool


def get_openai_tools(tools: Iterable[Tool]) -> List[Dict[str, Any]]:
    """Expose every MCP tool to the LLM as a callable function."""

    openai_tools: List[Dict[str, Any]] = []
    for tool in tools:
        parameters = _coerce_schema(tool)
        description = tool.description or f"RobotMCP tool '{tool.name}'"
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": description,
                    "parameters": parameters,
                },
            }
        )
    return openai_tools


def _coerce_schema(tool: Tool) -> Dict[str, Any]:
    schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
    if schema is None:
        return {"type": "object", "properties": {}, "additionalProperties": True}
    if hasattr(schema, "model_dump"):
        try:
            data = schema.model_dump(mode="json")
        except Exception:  # pragma: no cover - defensive path
            data = schema.model_dump()
        if isinstance(data, dict):
            return data
        try:
            return json.loads(json.dumps(data))
        except Exception:  # pragma: no cover - fallback for non-serializable objects
            return {"type": "object", "properties": {}, "additionalProperties": True}
    if isinstance(schema, dict):
        return schema
    return {"type": "object", "properties": {}, "additionalProperties": True}

__all__ = ["get_openai_tools"]
