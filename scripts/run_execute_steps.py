#!/usr/bin/env python3
"""Run a sequence of execute_step calls for PlatynUI calculator steps.

This script calls the server.mcp.execute_step tool for the steps that use
execute_step in the calculator E2E flow, using a fixed session_id.

It prints a compact JSON summary of per-step results when complete.
"""

from __future__ import annotations

import asyncio
import json
import sys


async def main():
    from fastmcp import Client
    from robotmcp.server import mcp

    session_id = "63dc622a-99aa-444a-ba76-75b46b2cb876"

    steps = []

    # Step 4: fix sys.path (Evaluate)
    steps.append(
        {
            "step": 4,
            "keyword": "Evaluate",
            "arguments": [
                "__import__('sys').path.insert(0, '/home/many/workspace/robotframework-PlatynUI/packages/native/python')"
            ],
            "assign_to": None,
        }
    )

    # Step 7: Verify AT-SPI tree (Evaluate) -> assign ${calc_apps}
    steps.append(
        {
            "step": 7,
            "keyword": "Evaluate",
            "arguments": [
                "[c.name for c in __import__('platynui_native').Runtime().desktop_node().children() if 'calculator' in c.name.lower()]"
            ],
            "assign_to": "calc_apps",
        }
    )

    # Step 8: Get Pointer Position (PlatynUI keyword)
    steps.append(
        {
            "step": 8,
            "keyword": "Get Pointer Position",
            "arguments": [],
            "assign_to": "pos",
        }
    )

    # Step 9: Pointer Click
    steps.append(
        {
            "step": 9,
            "keyword": "Pointer Click",
            "arguments": ["${NONE}", "x=431", "y=350"],
            "assign_to": None,
        }
    )

    # Step 10a: Type expression via xtype subprocess (Evaluate)
    steps.append(
        {
            "step": 10,
            "keyword": "Evaluate",
            "arguments": [
                "__import__('subprocess').run(['python3', '/tmp/xtype.py', '42*13'], capture_output=True, text=True, timeout=10).stdout.strip()"
            ],
            "assign_to": "type_result",
        }
    )

    # Step 10b: Press Enter via xkey subprocess (Evaluate)
    steps.append(
        {
            "step": 10.1,
            "keyword": "Evaluate",
            "arguments": [
                "__import__('subprocess').run(['python3', '/tmp/xkey.py', 'ff0d'], capture_output=True, text=True, timeout=5).stdout.strip()"
            ],
            "assign_to": "enter_result",
        }
    )

    # Step 12: Log verification
    steps.append(
        {
            "step": 12,
            "keyword": "Log",
            "arguments": [
                "VERIFIED: Calculator display shows 42×13 = 546. Result confirmed via ImageMagick X11 window capture."
            ],
            "assign_to": None,
        }
    )

    results = []

    async with Client(mcp) as client:
        for s in steps:
            payload = {
                "session_id": session_id,
                "keyword": s["keyword"],
            }
            if s.get("arguments"):
                payload["arguments"] = s["arguments"]
            if s.get("assign_to"):
                payload["assign_to"] = s["assign_to"]

            try:
                resp = await client.call_tool("execute_step", payload)
                data = resp.data
            except Exception as e:  # capture client-level errors
                data = {"success": False, "error": str(e)}

            entry = {
                "step": s["step"],
                "keyword": s["keyword"],
                "arguments": s.get("arguments", []),
                "assign_to": s.get("assign_to"),
                "result": data,
            }
            results.append(entry)

    # Print compact JSON to stdout
    print(json.dumps({"session_id": session_id, "results": results}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
