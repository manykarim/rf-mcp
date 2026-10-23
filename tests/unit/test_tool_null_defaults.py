"""Regression: tool parameters that default to ``None`` must accept ``null``.

A parameter written ``name: str = None`` advertises ``"default": null`` in
``tools/list`` while its schema type excludes null. A client that sends the
advertised default back explicitly (``{"session_id": null}``) then gets a
validation error instead of the default behaviour.

Only the schema and per-parameter argument validation are checked; no tool is
called.

Run with: uv run pytest tests/unit/test_tool_null_defaults.py -v
"""

__test__ = True

import inspect

import pytest
from pydantic import TypeAdapter


async def _listed_tools():
    from fastmcp import Client

    from robotmcp.server import mcp

    async with Client(mcp) as client:
        return await client.list_tools()


def _schema_allows_null(schema: dict) -> bool:
    if schema.get("type") == "null" or "null" in (schema.get("type") or []):
        return True
    return any(
        _schema_allows_null(s)
        for key in ("anyOf", "oneOf")
        for s in schema.get(key, [])
    )


@pytest.mark.asyncio
async def test_null_default_parameters_accept_null():
    from robotmcp import server

    offenders = []
    for tool in await _listed_tools():
        fn = getattr(server, tool.name, None)
        params = inspect.signature(fn).parameters if fn else {}
        for name, schema in (tool.inputSchema.get("properties") or {}).items():
            if "default" not in schema or schema["default"] is not None:
                continue
            if not _schema_allows_null(schema):
                offenders.append(f"{tool.name}.{name}: schema excludes null")
            if name in params:
                try:
                    TypeAdapter(params[name].annotation).validate_python(None)
                except Exception:
                    offenders.append(f"{tool.name}.{name}: rejects explicit null")

    assert not offenders, "\n".join(offenders)
