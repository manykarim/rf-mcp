"""Server-level validation fixes: responses, schemas, descriptions, instructions, tolerance, telemetry."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pytest
from fastmcp.client import Client
from fastmcp.exceptions import ToolError
from robotmcp.config import ServerConfig
from robotmcp.runner.cli_runner import CliRunner
from robotmcp.server import create_server
from robotmcp.skills.provider import INSTRUCTIONS_MAX_CHARS

RF_TOOLS = ("rf_discover", "rf_analyze", "rf_doc", "rf_run", "rf_results", "rf_repl", "rf_debug", "rf_diagnose")


def _server(tmp_path: Path, mode: str = "client"):
    return create_server(ServerConfig(execution_mode=mode, workspace_root=tmp_path))


# --- 8.7 no deprecation warnings at startup ------------------------------------------------


def test_create_server_emits_no_deprecation_warning(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        create_server(ServerConfig(execution_mode="client", workspace_root=tmp_path))


# --- 8.8 non-cp1252 characters survive results log ------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="cp1252 console encoding is a Windows concern")
def test_results_log_keeps_non_cp1252_characters(tmp_path: Path) -> None:
    text = "Pfeil → Haken ✓ 日本語 ☃"
    (tmp_path / "u.robot").write_text(
        f"*** Test Cases ***\nUnicode Failure\n    Fail    {text}\n", encoding="utf-8"
    )
    runner = CliRunner(ServerConfig(execution_mode="server", workspace_root=tmp_path))
    runner.execute(runner.build_run_args(paths=["u.robot"]))
    res = runner.execute(runner.build_results_args("log", failed_only=True))
    assert res.exit_code == 0, res.stderr
    assert text in json.dumps(res.data, ensure_ascii=False)


# --- 10.1 each payload once ------------------------------------------------------------------


async def test_server_mode_responses_have_no_duplicates(tmp_path: Path) -> None:
    (tmp_path / "robot.toml").write_text('paths = ["a.robot"]\n', encoding="utf-8")
    (tmp_path / "a.robot").write_text("*** Test Cases ***\nA\n    Log    x\n", encoding="utf-8")
    async with Client(_server(tmp_path, "server")) as client:
        res = (await client.call_tool("rf_discover", {"action": "tests"})).data
        assert "command" not in res  # a passed server-mode call does not echo the argv
        assert "shell_command" not in res
        assert "stdout" not in res
        assert res["data"]["items"][0]["longname"] == "A.A"

        run = (await client.call_tool("rf_run", {"by_longname": ["A.A"]})).data
        assert "shell_command" not in run
        assert run["exit_code"] == 0
        assert next(iter(run)) == "outcome" and run["outcome"] == "PASS: 1 test, 1 passed, 0 failed"

        analyze = (await client.call_tool("rf_analyze", {})).data
        assert analyze["has_errors"] is False
        assert "stdout" not in analyze
        assert "rf_run" in analyze["note"]  # clean analysis is not a passing run

        missing = (await client.call_tool("rf_run", {"by_longname": ["A.Missing"]})).data
        assert missing["exit_code"] != 0 and missing["command"][0] == "robotcode"  # failures keep it
        assert missing["outcome"].startswith("FAIL: robot exited")

    async with Client(_server(tmp_path, "client")) as client:
        res = (await client.call_tool("rf_discover", {"action": "tests"})).data
        assert "shell_command" in res


# --- 11.2 result files only via output_path ------------------------------------------------


async def test_result_file_outside_output_path_is_rejected(tmp_path: Path) -> None:
    async with Client(_server(tmp_path)) as client:
        tool = next(t for t in await client.list_tools() if t.name == "rf_results")
        assert "paths" not in tool.inputSchema["properties"]
        with pytest.raises(ToolError, match="output_path"):
            await client.call_tool("rf_results", {"action": "log", "search": "results/output.xml"})
        with pytest.raises(ToolError, match="output_path"):
            await client.call_tool("rf_results", {"by_longname": ["output.xml"]})
        ok = await client.call_tool("rf_results", {"output_path": "results/output.xml"})
        assert ok.data["command"][-2:] == ["-o", "results/output.xml"]


# --- 11.4 / 12.2 schema consistency, descriptions, annotations ---------------------------


async def test_tool_schemas_share_parameter_names_and_describe_every_parameter(tmp_path: Path) -> None:
    async with Client(_server(tmp_path)) as client:
        tools = {t.name: t for t in await client.list_tools()}
    for name in RF_TOOLS:
        tool = tools[name]
        props = tool.inputSchema["properties"]
        assert "kind" not in props and "mode" not in props and "execution_target" not in props, name
        if name != "rf_diagnose":
            assert "action" in props or name in ("rf_analyze", "rf_run"), name
            assert props["target"].get("enum") == ["server", "client"] or any(
                o.get("enum") == ["server", "client"] for o in props["target"].get("anyOf", [])
            ), name
            assert "root" in props and "profile" in props, name
        for pname, prop in props.items():
            assert prop.get("description"), f"{name}.{pname} has no description"
        desc = tool.description or ""
        assert desc.startswith("Use when"), name
        assert "Mistake to avoid" in desc, name
        assert len(desc) <= 600, (name, len(desc))
        assert tool.annotations is not None and tool.annotations.readOnlyHint is not None, name
    assert tools["rf_doc"].inputSchema["properties"]["library"]
    assert "grep" in tools["rf_discover"].description
    assert "output.xml" in tools["rf_results"].description and "why did" in tools["rf_results"].description
    assert "before any rf_run" in tools["rf_analyze"].description
    assert "existing test fails" in tools["rf_debug"].description
    assert "no existing test" in tools["rf_repl"].description
    for read_only in ("rf_discover", "rf_doc", "rf_results", "rf_diagnose"):
        assert tools[read_only].annotations.readOnlyHint is True
    assert tools["rf_repl"].annotations.destructiveHint is True


# --- 12.1 / 14.4 / 14.10 instructions -----------------------------------------------------------


def test_instructions_route_by_intent_and_state_the_loop(tmp_path: Path) -> None:
    text = _server(tmp_path).instructions or ""
    assert len(text) <= INSTRUCTIONS_MAX_CHARS
    for needle in (
        "| Intent | Tool |",
        "never grep .robot files",
        "never read or grep output.xml",
        "by_longname",
        "rf_analyze until clean",
        "never repeat the same failing call",
        "Done only when rf_analyze is clean and rf_run passes",
        "Stop the sessions you started",
        "passes alone",
        "Set Suite Variable",
        "skill://rf-robotcode/SKILL.md",
    ):
        assert needle in text, needle


# --- 14.1 tolerant arguments (replaying the 2026-11-05 probe calls) --------------------------


@pytest.mark.parametrize(
    ("tool", "args", "expected"),
    [
        ("rf_discover", {"include_tags": "smoke"}, ["-i", "smoke"]),
        ("rf_discover", {"include_tags": '["smoke"]'}, ["-i", "smoke"]),
        ("rf_discover", {"include_tags": "smoke,api"}, ["-i", "smoke", "-i", "api"]),
        ("rf_results", {"action": "Summary"}, ["results", "summary"]),
        ("rf_doc", {"action": "KEYWORDS", "library": "Browser", "pattern": "upload,file,attach"}, ["Browser", "upload", "file", "attach"]),
        ("rf_run", {"by_longname": "Tests.A", "target": "CLIENT"}, ["-bl", "Tests.A"]),
    ],
)
async def test_weak_model_argument_shapes_are_accepted(
    tmp_path: Path, tool: str, args: dict, expected: list[str]
) -> None:
    async with Client(_server(tmp_path)) as client:
        res = await client.call_tool(tool, args)
    command = res.data["command"]
    joined = " ".join(command)
    assert " ".join(expected) in joined, command


async def test_published_schema_stays_plain(tmp_path: Path) -> None:
    async with Client(_server(tmp_path)) as client:
        tools = {t.name: t for t in await client.list_tools()}
    tags = tools["rf_discover"].inputSchema["properties"]["include_tags"]
    assert tags["anyOf"][0] == {"items": {"type": "string"}, "type": "array"}
    action = tools["rf_results"].inputSchema["properties"]["action"]
    assert action["enum"] == ["summary", "show", "log", "stats", "diff", "merge"]


# --- 14.2 rf_repl description teaches batching and inspection ------------------------------


async def test_repl_description_teaches_batching_and_inspection(tmp_path: Path) -> None:
    async with Client(_server(tmp_path)) as client:
        repl = next(t for t in await client.list_tools() if t.name == "rf_repl")
    desc = repl.description or ""
    for needle in (
        "several lines",
        "stop at the first failure",
        "FOR/IF/TRY",
        "Get Aria Snapshot",
        ".kw",
        ".doc",
        ".vars --user",
        ".imports",
        "Run Keywords cannot assign",
    ):
        assert needle in desc, needle


# --- 14.3 rf_doc accepts several patterns -------------------------------------------------------


async def test_doc_keywords_with_several_patterns(tmp_path: Path) -> None:
    async with Client(_server(tmp_path)) as client:
        res = await client.call_tool(
            "rf_doc", {"action": "keywords", "library": "Browser", "pattern": ["upload", "file", "attach"]}
        )
        tool = next(t for t in await client.list_tools() if t.name == "rf_doc")
    assert res.data["command"][-5:] == ["keywords", "Browser", "upload", "file", "attach"]
    assert "by name, not meaning" in (tool.description or "")
    assert "synonyms" in (tool.description or "")


async def test_debug_description_teaches_batching(tmp_path: Path) -> None:
    async with Client(_server(tmp_path)) as client:
        tool = next(t for t in await client.list_tools() if t.name == "rf_debug")
        assert "several lines in one send" in tool.description
        assert "one probe per send" in tool.description
        assert len(tool.description) <= 600


async def test_results_accepts_path_as_output_path(tmp_path: Path) -> None:
    async with Client(_server(tmp_path)) as client:
        res = await client.call_tool("rf_results", {"action": "summary", "path": "results/saucedemo.xml"})
    assert res.data["command"][-2:] == ["-o", "results/saucedemo.xml"]


async def test_empty_choice_arguments_mean_unset(tmp_path: Path) -> None:
    # models behind Claude Code on OpenRouter send unset optional arguments as "" (2026-11-09)
    async with Client(_server(tmp_path)) as client:
        res = await client.call_tool("rf_results", {"action": "summary", "path": "results/x.xml", "target": "", "shell": ""})
    assert res.data["command"][-2:] == ["-o", "results/x.xml"]


async def test_python_lists_and_empty_literals_are_tolerated(tmp_path: Path) -> None:
    # 2026-11-09 Qwen3-Coder: by_longname="['A.A']", rerunfailed="[]", files="", output=[]; DeepSeek: profile="[]"
    async with Client(_server(tmp_path)) as client:
        res = await client.call_tool("rf_run", {"by_longname": "['A.A', 'A.B']", "rerunfailed": "[]", "profile": "[]",
                                                "files": "", "output": [], "wrapper": [], "variables": ""})
    cmd = res.data["command"]
    assert cmd[cmd.index("-bl") + 1] == "A.A" and "A.B" in cmd
    assert "--rerunfailed" not in cmd and "-p" not in cmd and "[]" not in cmd


def test_results_log_is_compact(tmp_path: Path) -> None:
    (tmp_path / "c.robot").write_text(
        "*** Test Cases ***\nCompact Failure\n    IF    $False\n        Log    never\n    ELSE\n        Log    shown\n    END\n"
        "    Fail    boom\n",
        encoding="utf-8",
    )
    runner = CliRunner(ServerConfig(execution_mode="server", workspace_root=tmp_path))
    runner.execute(runner.build_run_args(paths=["c.robot"]))
    res = runner.execute(runner.build_results_args("log", failed_only=True))
    text = json.dumps(res.data["tests"])
    assert "boom" in text and "shown" in text
    assert "startTime" not in text and "timestamp" not in text and '"isHtml": false' not in text
    assert "NOT RUN" not in text and "never" not in text and str(tmp_path) not in text  # relSource only
    assert text.count("elapsedSeconds") == 1 and "elapsedSeconds" in res.data["tests"][0]


def test_failure_streak_points_to_rf_debug_after_three_failing_runs() -> None:
    from robotmcp.server import FailureStreaks

    s = FailureStreaks()
    fail = ["Order Total: 90.85 != 79.00"]
    assert s.update(fail, 1) is None and s.update(fail, 1) is None
    third = s.update(fail, 1) or ""
    assert "Order Total failed in 3 runs in a row" in third and "rf_debug" in third
    assert s.update(fail, 1) is None  # then every 3rd
    assert s.update(["Other: x"], 1) is None  # a different failure starts its own count
    assert s.update([], 0) is None and s.streaks == {}


def test_truncate_data_cuts_one_dominant_string_before_lists() -> None:
    from robotmcp.runner.cli_runner import truncate_data

    data = {"name": "XML", "markdown": "x" * 83000, "keywords": [{"n": i} for i in range(5)]}
    trimmed, note = truncate_data(data, 20000)
    assert len(json.dumps(trimmed)) <= 20000 and len(trimmed["keywords"]) == 5
    assert "cut text: markdown" in (note or "") and "characters cut" in trimmed["markdown"]


# --- verdict first: weak models claimed success after a failed rf_run ---------------------


async def test_run_outcome_leads_with_the_verdict(tmp_path: Path) -> None:
    (tmp_path / "robot.toml").write_text('paths = ["b.robot"]\n', encoding="utf-8")
    (tmp_path / "b.robot").write_text(
        "*** Test Cases ***\nOk\n    Log    x\nBroken\n    Fail    boom\n", encoding="utf-8"
    )
    async with Client(_server(tmp_path, "server")) as client:
        run = (await client.call_tool("rf_run", {})).data
        assert next(iter(run)) == "outcome"
        assert run["outcome"].startswith("FAIL: 2 tests, 1 passed, 1 failed. The tests do not pass yet")
        dry = (await client.call_tool("rf_run", {"dryrun": True})).data
        assert dry["outcome"].startswith("PASS (dry run")
        analyze = (await client.call_tool("rf_analyze", {"paths": ["b.robot"]})).data
        assert "note" in analyze
    async with Client(_server(tmp_path, "client")) as client:
        assert "outcome" not in (await client.call_tool("rf_run", {})).data
