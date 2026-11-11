"""End-to-end tests for the FastMCP server tools (`rf_*`) in both `server` and `client` modes."""

from __future__ import annotations

from pathlib import Path

from fastmcp.client import Client

from robotmcp.config import ServerConfig
from robotmcp.server import create_server


async def test_server_exposes_8_rf_tools_and_skills_resources(tmp_path: Path) -> None:
    cfg = ServerConfig(execution_mode="client", workspace_root=tmp_path)
    mcp = create_server(cfg)

    async with Client(mcp) as client:
        tools = await client.list_tools()
        tool_names = {t.name for t in tools}
        expected = {
            "rf_discover",
            "rf_analyze",
            "rf_doc",
            "rf_run",
            "rf_results",
            "rf_repl",
            "rf_debug",
            "rf_diagnose",
            "list_resources",
            "read_resource",
        }
        assert expected.issubset(tool_names)

        # Call rf_discover in client mode
        disc = await client.call_tool("rf_discover", {"action": "tests"})
        assert disc.data["mode"] == "client"
        assert "discover tests" in disc.data["shell_command"]

        # Call rf_repl in client mode
        repl_res = await client.call_tool(
            "rf_repl",
            {
                "action": "send",
                "session_id": "repl-11223344",
                "statement": "Log    ${user_id}",
            },
        )
        assert repl_res.data["mode"] == "client"
        assert "'Log    ${user_id}'" in repl_res.data["shell_command"]
        assert "fallback_command" not in repl_res.data

        # Call rf_diagnose
        diag = await client.call_tool(
            "rf_diagnose",
            {
                "keyword": "Evaluate JavaScript    () => document.title",
                "error_text": "Unexpected token '='",
            },
        )
        assert diag.data["hints_count"] >= 1
        assert "Evaluate JavaScript    ${None}    () => document.title" in diag.data["hints"][0]["fix_examples"]


async def test_server_mode_end_to_end_workflow(tmp_path: Path) -> None:
    cfg = ServerConfig(execution_mode="server", workspace_root=tmp_path)
    mcp = create_server(cfg)

    async with Client(mcp) as client:
        # 1. rf_analyze with inline file sync (Mode 2 remote runner pattern)
        an_res = await client.call_tool(
            "rf_analyze",
            {
                "files": {
                    "tests/demo.robot": (
                        "*** Settings ***\n"
                        "Metadata    Issue    4409\n\n"
                        "*** Test Cases ***\n"
                        "Passing Case\n"
                        "    Log    Synced via MCP\n"
                    )
                },
                "paths": ["tests/demo.robot"],
            },
        )
        assert an_res.data["mode"] == "server"
        assert an_res.data["synced_files"] == ["tests/demo.robot"]
        assert an_res.data["data"]["summary"]["errors"] == 0

        # 2. rf_discover metadata --values
        meta_res = await client.call_tool(
            "rf_discover",
            {"action": "metadata", "include_values": True, "paths": ["tests/demo.robot"]},
        )
        assert meta_res.data["exit_code"] == 0
        assert "stdout" not in meta_res.data
        assert "4409" in str(meta_res.data["data"])

        # 3. rf_doc keyword lookup
        doc_res = await client.call_tool(
            "rf_doc",
            {"action": "keyword", "library": "BuiltIn", "pattern": "Should Be Equal"},
        )
        assert doc_res.data["exit_code"] == 0

        # 4. rf_run + rf_results summary
        run_res = await client.call_tool(
            "rf_run",
            {"paths": ["tests/demo.robot"], "by_longname": ["Demo.Passing Case"]},
        )
        assert run_res.data["exit_code"] == 0

        res_summary = await client.call_tool(
            "rf_results",
            {"action": "summary"},
        )
        assert res_summary.data["exit_code"] == 0

        # 5. rf_repl start -> send -> stop via MCP tool calls
        start_res = await client.call_tool("rf_repl", {"action": "start"})
        sid = start_res.data["session_id"]
        assert sid and sid.startswith("repl-")

        try:
            send_res = await client.call_tool(
                "rf_repl",
                {
                    "action": "send",
                    "session_id": sid,
                    "statement": "${greeting}=    Set Variable    Hello from MCP tool",
                },
            )
            assert send_res.data["exit_code"] == 0
            assert "${greeting} = Hello from MCP tool" in send_res.data["output"]
        finally:
            await client.call_tool(
                "rf_repl", {"action": "stop", "session_id": sid}
            )
