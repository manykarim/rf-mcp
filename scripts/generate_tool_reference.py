"""Generate the MCP tool reference (docs/rf-mcp-<version>-libdoc.{json,html}).

The rf-mcp counterpart of Robot Framework's libdoc: one entry per MCP tool, with its
description, parameter schema, and whether it is enabled by default. Run via
``uv run invoke libdoc`` (which also regenerates the McpAttach RF libdoc).

Usage: uv run python scripts/generate_tool_reference.py [--out-dir docs]
"""

from __future__ import annotations

import argparse
import asyncio
import html
import json
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List


async def _collect() -> List[Dict[str, Any]]:
    from robotmcp.server import _DISABLED_TOOL_NAMES, mcp

    # The local provider lists every registered tool; mcp.list_tools() would omit the
    # ones disabled by default, which the reference must still document.
    tools = await mcp._local_provider.list_tools()
    entries = []
    for tool in sorted(tools, key=lambda t: t.name):
        params = (tool.parameters or {}).get("properties", {}) or {}
        required = set((tool.parameters or {}).get("required", []) or [])
        entries.append(
            {
                "name": tool.name,
                "description": (tool.description or "").strip(),
                "enabled_by_default": tool.name not in _DISABLED_TOOL_NAMES,
                "parameters": {
                    name: {
                        "type": _type_of(schema),
                        "required": name in required,
                        **({"default": schema["default"]} if "default" in schema else {}),
                        **({"enum": schema["enum"]} if "enum" in schema else {}),
                    }
                    for name, schema in params.items()
                },
            }
        )
    return entries


def _type_of(schema: Dict[str, Any]) -> str:
    if "type" in schema:
        return str(schema["type"])
    variants = schema.get("anyOf") or schema.get("oneOf") or []
    types = [str(v.get("type", "object")) for v in variants if isinstance(v, dict)]
    return " | ".join(dict.fromkeys(types)) or "any"


def _render_html(version: str, entries: List[Dict[str, Any]]) -> str:
    enabled = sum(e["enabled_by_default"] for e in entries)
    out = [
        '<!DOCTYPE html><html><head><meta charset="utf-8">',
        f"<title>rf-mcp {version} - Tool Reference</title>",
        "<style>body{font-family:system-ui,sans-serif;max-width:960px;margin:2em auto;"
        "padding:0 1em;color:#1a1a1a}h1{border-bottom:2px solid #2563eb;padding-bottom:.3em}"
        ".tool{margin:1.5em 0;padding:1em;border:1px solid #e5e7eb;border-radius:8px}"
        ".tool h3{margin:0 0 .5em;color:#2563eb}.tool.disabled h3{color:#9ca3af}"
        ".badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.75em;"
        "margin-left:.5em}.enabled{background:#dcfce7;color:#166534}"
        ".disabled{background:#f3f4f6;color:#6b7280}pre{background:#f8fafc;padding:.75em;"
        "border-radius:4px;overflow-x:auto;font-size:.85em;white-space:pre-wrap}"
        "table{border-collapse:collapse;font-size:.85em;margin-top:.5em}"
        "td,th{border:1px solid #e5e7eb;padding:2px 8px;text-align:left}</style></head><body>",
        f"<h1>rf-mcp v{html.escape(version)} - MCP Tool Reference</h1>",
        f"<p><strong>{len(entries)}</strong> tools total, <strong>{enabled}</strong> "
        "enabled by default</p><hr>",
    ]
    for e in entries:
        state = "enabled" if e["enabled_by_default"] else "disabled"
        cls = "tool" if e["enabled_by_default"] else "tool disabled"
        out.append(
            f'<div class="{cls}" id="{html.escape(e["name"])}"><h3>{html.escape(e["name"])}'
            f'<span class="badge {state}">{state}</span></h3>'
            f"<pre>{html.escape(e['description'] or 'No description')}</pre>"
        )
        if e["parameters"]:
            out.append("<table><tr><th>parameter</th><th>type</th><th>required</th>"
                       "<th>default</th></tr>")
            for name, p in e["parameters"].items():
                default = json.dumps(p["default"]) if "default" in p else ""
                out.append(
                    f"<tr><td>{html.escape(name)}</td><td>{html.escape(p['type'])}</td>"
                    f"<td>{'yes' if p['required'] else ''}</td>"
                    f"<td>{html.escape(default)}</td></tr>"
                )
            out.append("</table>")
        out.append("</div>")
    out.append("</body></html>")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="docs")
    args = parser.parse_args()

    version = metadata.version("rf-mcp")
    entries = asyncio.run(_collect())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Build names explicitly: Path.with_suffix() would treat ".0-libdoc" in
    # "rf-mcp-0.36.0-libdoc" as the suffix and write "rf-mcp-0.36.json".
    json_path = out_dir / f"rf-mcp-{version}-libdoc.json"
    html_path = out_dir / f"rf-mcp-{version}-libdoc.html"

    payload = {
        "name": "rf-mcp",
        "version": version,
        "total_tools": len(entries),
        "enabled_tools": sum(e["enabled_by_default"] for e in entries),
        "tools": entries,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(_render_html(version, entries), encoding="utf-8")
    print(f"wrote {json_path} / {html_path.name} - {payload['total_tools']} tools, "
          f"{payload['enabled_tools']} enabled by default")


if __name__ == "__main__":
    main()
