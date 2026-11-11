"""End-to-end tests for the FastMCP server tools (`rf_*`) in both `server` and `client` modes."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
from fastmcp.client import Client
from fastmcp.exceptions import ToolError

from robotmcp.client_roots import choose_root, file_uri_to_path
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


# --- the client's MCP roots (project-independent-server, task 1.2) ---------------------------

_LEGACY = {"mode": "legacy"} if "mode" in inspect.signature(Client.__init__).parameters else {}


def _project(path: Path, name: str) -> Path:
    path.mkdir(parents=True)
    (path / "robot.toml").write_text("", encoding="utf-8")
    (path / "which.txt").write_text(name, encoding="utf-8")
    return path


async def _which(client: Client) -> str:
    res = await client.call_tool("rf_discover", {"action": "read", "paths": ["which.txt"]})
    return res.data["files"]["which.txt"]


@pytest.fixture
def projects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    found = {"api": _project(tmp_path / "api", "api"), "web": _project(tmp_path / "web", "web"),
             "home": _project(tmp_path / "home", "home")}
    monkeypatch.chdir(found["home"])
    return found


def _unpinned_server():
    return create_server(ServerConfig(execution_mode="server"))


async def test_the_clients_single_root_is_used(projects: dict[str, Path]) -> None:
    async with Client(_unpinned_server(), roots=[projects["api"].as_uri()], **_LEGACY) as client:
        assert await _which(client) == "api"


async def test_the_working_directory_picks_one_of_several_roots(projects, monkeypatch) -> None:
    monkeypatch.chdir(projects["web"])
    async with Client(_unpinned_server(), roots=[projects["api"].as_uri(), projects["web"].as_uri()], **_LEGACY) as client:
        assert await _which(client) == "web"


async def test_several_roots_without_a_match_ask_for_root(projects: dict[str, Path]) -> None:
    async with Client(_unpinned_server(), roots=[projects["api"].as_uri(), projects["web"].as_uri()], **_LEGACY) as client:
        with pytest.raises(ToolError, match="several workspace folders") as err:
            await _which(client)
        assert str(projects["api"]) in str(err.value) and str(projects["web"]) in str(err.value)
        res = await client.call_tool("rf_discover", {"action": "read", "paths": ["which.txt"], "root": str(projects["web"])})
        assert res.data["files"]["which.txt"] == "web"


async def test_a_client_without_roots_falls_back_to_the_working_directory(projects: dict[str, Path]) -> None:
    async with Client(_unpinned_server(), **_LEGACY) as client:
        assert await _which(client) == "home"


async def test_a_pinned_root_beats_the_clients_roots(projects: dict[str, Path]) -> None:
    server = create_server(ServerConfig(execution_mode="server", workspace_root=projects["web"]))
    async with Client(server, roots=[projects["api"].as_uri()], **_LEGACY) as client:
        assert await _which(client) == "web"


@pytest.mark.skipif(not _LEGACY, reason="FastMCP 3 has only the handshake-era protocol")
async def test_new_protocol_clients_are_not_asked_for_roots(projects: dict[str, Path]) -> None:
    async with Client(_unpinned_server(), roots=[projects["api"].as_uri()]) as client:
        assert await _which(client) == "home"


async def test_client_mode_never_injects_a_server_side_root(projects: dict[str, Path]) -> None:
    async with Client(create_server(ServerConfig(execution_mode="client")), roots=[projects["api"].as_uri()], **_LEGACY) as client:
        res = await client.call_tool("rf_run", {"by_longname": ["S.T"]})
        assert "-r" not in res.data["command"]


def test_only_file_uris_become_roots(tmp_path: Path) -> None:
    assert file_uri_to_path(tmp_path.as_uri()) == tmp_path
    assert file_uri_to_path("https://example.com/repo") is None
    assert choose_root([], tmp_path) is None


@pytest.mark.skipif(sys.platform != "win32", reason="drive letters are Windows paths")
def test_vscode_root_uri_with_encoded_drive_colon() -> None:
    # VS Code 1.145 sends workspace folders as file:///c%3A/... (lowercase drive, encoded colon).
    assert file_uri_to_path("file:///c%3A/work/my%20shop") == Path("C:/work/my shop")
