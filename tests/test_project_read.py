"""15.1: rf_discover action='read' returns several project files in one call."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastmcp.client import Client
from fastmcp.exceptions import ToolError
from robotmcp.config import ServerConfig
from robotmcp.runner.workspace_sync import read_project_files
from robotmcp.server import create_server
from robotmcp.skills import build_server_instructions
from robotmcp.skills.provider import INSTRUCTIONS_MAX_CHARS


def _project(root: Path) -> Path:
    files = {
        "robot.toml": 'output-dir = "results"\n',
        "resources/shop.resource": "*** Keywords ***\nOpen Shop\n    No Operation\n",
        "resources/api.resource": "*** Keywords ***\nCall Api\n    No Operation\n",
        "docs/conventions.md": "# Conventions\nOne test per criterion.\n",
        "results/output.xml": "<robot/>",
        ".venv/lib/x.resource": "*** Keywords ***\nHidden\n",
        "data/blob.bin": "\x00\x01binary",
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


def test_reads_paths_and_globs_once_in_request_order(tmp_path: Path) -> None:
    res = read_project_files(_project(tmp_path), ["robot.toml", "resources/*.resource", "resources/shop.resource",
                                                  "docs/*.md"], 20000)
    assert list(res["files"]) == ["robot.toml", "resources/api.resource", "resources/shop.resource", "docs/conventions.md"]
    assert res["files"]["docs/conventions.md"].startswith("# Conventions")
    assert "omitted" not in res and "note" not in res


def test_globs_skip_environments_and_result_files_are_refused(tmp_path: Path) -> None:
    res = read_project_files(_project(tmp_path), ["**/*.resource", "results/output.xml", "data/blob.bin", "nope/*.md"], 20000)
    assert ".venv/lib/x.resource" not in res["files"]
    assert {s["path"] for s in res["skipped"]} == {"results/output.xml", "data/blob.bin"}
    assert "rf_results" in res["skipped"][0]["reason"]
    assert res["unmatched"] == ["nope/*.md"]


def test_budget_omits_files_that_do_not_fit_and_cuts_only_a_first_oversized_file(tmp_path: Path) -> None:
    root = _project(tmp_path)
    (root / "docs/big.md").write_text("x" * 500, encoding="utf-8")
    res = read_project_files(root, ["robot.toml", "docs/big.md", "docs/conventions.md"], 100)
    assert list(res["files"]) == ["robot.toml", "docs/conventions.md"]
    assert res["omitted"] == [{"path": "docs/big.md", "chars": 500}]
    assert "another call" in res["note"]
    first = read_project_files(root, ["docs/big.md"], 100)
    assert first["files"]["docs/big.md"] == "x" * 100 and first["truncated"] == ["docs/big.md"]


@pytest.mark.parametrize("bad", ["../outside.txt", "resources/../../outside.txt"])
def test_paths_outside_the_root_are_refused(tmp_path: Path, bad: str) -> None:
    root = _project(tmp_path / "p")
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the project root"):
        read_project_files(root, [bad], 20000)


async def test_rf_discover_read_tool(tmp_path: Path) -> None:
    root = _project(tmp_path)
    async with Client(create_server(ServerConfig(execution_mode="server", workspace_root=root))) as client:
        res = await client.call_tool("rf_discover", {"action": "read", "paths": "robot.toml,resources/*.resource"})
        assert list(res.data["files"]) == ["robot.toml", "resources/api.resource", "resources/shop.resource"]
        assert "command" not in res.data
        with pytest.raises(ToolError, match="needs paths"):
            await client.call_tool("rf_discover", {"action": "read"})
        with pytest.raises(ToolError, match="outside the project root"):
            await client.call_tool("rf_discover", {"action": "read", "paths": ["../x"]})
        tool = next(t for t in await client.list_tools() if t.name == "rf_discover")
        assert "action='read'" in tool.description and len(tool.description) <= 600
    async with Client(create_server(ServerConfig(execution_mode="client", workspace_root=root))) as client:
        with pytest.raises(ToolError, match="client mode"):
            await client.call_tool("rf_discover", {"action": "read", "paths": ["robot.toml"]})


def test_instructions_route_orientation_to_one_read_in_server_mode(tmp_path: Path) -> None:
    server_text = create_server(ServerConfig(execution_mode="server", workspace_root=tmp_path)).instructions or ""
    client_text = create_server(ServerConfig(execution_mode="client", workspace_root=tmp_path)).instructions or ""
    assert "rf_discover action=read" in server_text and "not one Read per file" in server_text
    assert "1. Read every file you need in one rf_discover action=read call" in server_text
    assert "action=read" not in client_text and "1. Start one rf_repl session" in client_text


def test_instructions_route_ui_work_to_rf_locators_when_the_skill_is_there(tmp_path: Path) -> None:
    cfg = ServerConfig(execution_mode="server", workspace_root=tmp_path)
    text = create_server(cfg).instructions or ""
    assert "| write or fix UI steps (Browser, Selenium, Appium) | read skill://rf-locators/SKILL.md first |" in text
    assert len(text) <= INSTRUCTIONS_MAX_CHARS
    assert "UI steps" not in build_server_instructions(cfg, [{"name": "rf-robotcode", "uri": "skill://rf-robotcode/SKILL.md",
                                                              "description": "x"}])
