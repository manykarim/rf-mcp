"""Validation fixes for CliRunner: timeouts, client roots, compact responses, new CLI options."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from robotmcp.config import ServerConfig
from robotmcp.runner import cli_runner as cr
from robotmcp.runner.cli_runner import (
    CliRunner,
    decode_analyze_exit_code,
    parse_console_failures,
    project_discover_data,
    truncate_data,
)
from robotmcp.runner.environment import EnvironmentResolver, missing_robotcode_instruction
from robotmcp.runner.shell_escape import build_agent_env


def _runner(tmp_path: Path, mode: str = "server", **kw) -> CliRunner:
    return CliRunner(ServerConfig(execution_mode=mode, workspace_root=tmp_path, **kw))


def _pid_alive(pid: int) -> bool:
    if sys.platform == "win32":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            check=False,
        ).stdout
        return str(pid).encode() in (out or b"")
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


# --- 8.3 timeouts and missing executable -------------------------------------------------


def test_run_timeout_kills_the_whole_process_tree(tmp_path: Path) -> None:
    pid_file = tmp_path / "pid.txt"
    (tmp_path / "slow.robot").write_text(
        "*** Settings ***\nLibrary    OperatingSystem\n\n"
        "*** Test Cases ***\nSlow\n"
        "    ${pid}=    Evaluate    str(os.getpid())    modules=os\n"
        f"    Create File    {pid_file.as_posix()}    ${{pid}}\n"
        "    Sleep    20s\n",
        encoding="utf-8",
    )
    runner = _runner(tmp_path)
    started = time.monotonic()
    res = runner.execute(runner.build_run_args(paths=["slow.robot"]), timeout_seconds=6)
    assert time.monotonic() - started < 18
    assert res.status == "timeout"
    assert "timeout_seconds" in (res.instruction or "")
    robot_pid = int(pid_file.read_text(encoding="utf-8").strip())
    for _ in range(20):
        if not _pid_alive(robot_pid):
            break
        time.sleep(0.25)
    assert not _pid_alive(robot_pid), "robot child process survived the timeout"


def test_missing_robotcode_returns_actionable_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("robotmcp.runner.environment.own_robotcode_command", lambda: ("robotcode-missing-xyz",))
    runner = _runner(tmp_path)
    res = runner.execute(runner.build_discover_args("tests"))
    assert res.status == "error"
    assert "robotcode-missing-xyz" in res.stderr
    assert "rf-mcp doctor" in (res.instruction or "") and str(tmp_path) in (res.instruction or "")


# --- project-independent-server 3.1/3.2: resolved environment, fallback warning, install-kind fixes ---


def _isolated(tmp_path: Path, **kw) -> EnvironmentResolver:
    (tmp_path / "_no_path").mkdir(exist_ok=True)
    return EnvironmentResolver(own_prefix=tmp_path / "_rf_mcp_tool_env", environ={"PATH": str(tmp_path / "_no_path")}, **kw)


def test_a_layered_prefix_reaches_run_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "p"
    (project / ".venv" / "Scripts").mkdir(parents=True)
    (project / ".venv" / "Scripts" / "python.exe").write_text("", encoding="utf-8")
    seen: list[list[str]] = []
    real_run = cr.run_process

    def fake_run(cmd, **kwargs):
        seen.append(list(cmd))
        return real_run([sys.executable, "-c", "print('{}')"], **kwargs)

    monkeypatch.setattr(cr, "run_process", fake_run)
    resolver = _isolated(tmp_path, uv="uv", pin="2.9.0.dev115", version_probe=lambda _p: None,
                         command_probe=lambda _prefix: (True, "robotcode, version 2.9.0-dev.115"))
    runner = CliRunner(ServerConfig(execution_mode="server", workspace_root=project), environments=resolver)
    runner.execute(runner.build_discover_args("tests"))
    assert seen[0][:9] == ["uv", "run", "--quiet", "--no-project", "--python",
                           str(project / ".venv" / "Scripts" / "python.exe"), "--with", "robotcode[all]==2.9.0.dev115",
                           "robotcode"]
    assert seen[0][9:11] == ["-r", str(project.resolve())]


def test_the_fallback_warning_reaches_the_result(tmp_path: Path) -> None:
    project = tmp_path / "p"
    project.mkdir()
    (project / "t.robot").write_text(
        "*** Settings ***\nLibrary    NoSuchProjectLibrary\n\n*** Test Cases ***\nT\n    Log    x\n", encoding="utf-8"
    )
    runner = CliRunner(ServerConfig(execution_mode="server", workspace_root=project), environments=_isolated(tmp_path))
    res = runner.execute(runner.build_analyze_args(), hint_kind="static")
    assert "project's libraries are not visible" in (res.warning or "")
    messages = [d["message"] for items in res.data["diagnostics"].values() for d in items]
    assert any(m.startswith(cr.FALLBACK_IMPORT_NOTE + "Importing library 'NoSuchProjectLibrary'") for m in messages)


def test_missing_robotcode_instruction_per_install_kind(tmp_path: Path) -> None:
    project = tmp_path / "p"
    (project / ".venv").mkdir(parents=True)
    env = _isolated(tmp_path).resolve(project)
    project_install = EnvironmentResolver(own_prefix=project / ".venv")
    no_uv = _isolated(tmp_path, uv=False)
    assert "Add robotcode to the project environment" in missing_robotcode_instruction(project, env, project_install)
    text = missing_robotcode_instruction(project, env, no_uv)
    assert "Install uv" in text and 'uv add --dev "robotcode[all]' in text and str(project) in text


# --- 9.1 client mode passes the agent's root verbatim ------------------------------------


def test_client_mode_root_is_verbatim_and_optional(tmp_path: Path) -> None:
    runner = _runner(tmp_path, "client")
    args = runner.build_run_args(root="/home/alice/proj", by_longname=["Tests.A"])
    assert args[:3] == ["robotcode", "-r", "/home/alice/proj"]
    assert args[-1] == "Tests.A"  # no server-side path detection

    no_root = runner.build_discover_args("tests")
    assert "-r" not in no_root
    assert no_root[-1] == "tests"
    assert "." not in no_root


# --- 10.1 / 10.2 compact responses ---------------------------------------------------------


def _seven_test_project(root: Path) -> None:
    (root / "robot.toml").write_text('paths = ["tests"]\n', encoding="utf-8")
    (root / "tests" / "api").mkdir(parents=True)
    (root / "tests" / "calc").mkdir(parents=True)
    (root / "tests" / "api" / "booking.robot").write_text(
        "*** Test Cases ***\n"
        "Ping Service\n    [Tags]    api    smoke\n    Log    x\n\n"
        "Get Booking Ids\n    [Tags]    api\n    Log    x\n\n"
        "Create Booking\n    [Tags]    api    regression\n    Log    x\n",
        encoding="utf-8",
    )
    (root / "tests" / "calc" / "calc.robot").write_text(
        "*** Test Cases ***\n"
        "Add Two Numbers\n    [Tags]    calc    smoke\n    Log    x\n\n"
        "Ten Percent Discount\n    [Tags]    calc    regression\n    Log    x\n\n"
        "Format Money In Euro\n    [Tags]    smoke\n    Log    x\n\n"
        "Cart Contains Items\n    [Tags]    regression\n    Log    x\n",
        encoding="utf-8",
    )


def test_discover_tests_response_is_compact(tmp_path: Path) -> None:
    _seven_test_project(tmp_path)
    runner = _runner(tmp_path)
    res = runner.execute(runner.build_discover_args("tests"))
    res.data = project_discover_data("tests", res.data)
    response = res.to_response()

    assert "stdout" not in response
    assert "shell_command" not in response
    items = response["data"]["items"]
    assert len(items) == 7
    assert items[0] == {
        "longname": "Tests.Api.Booking.Ping Service",
        "source": "tests/api/booking.robot:2",
        "tags": ["api", "smoke"],
    }
    text = json.dumps(response)
    for noise in ('"id"', '"uri"', '"range"', str(tmp_path / "tests")):
        assert noise not in text.replace("\\\\", "\\")

    exe = cr.resolve_robotcode_executable()
    plain = subprocess.run(
        [*exe, "-r", str(tmp_path), "discover", "tests"],
        cwd=tmp_path,
        env=build_agent_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    ).stdout
    assert "command" not in response  # a passed server-mode call does not echo the argv
    assert len(text) <= 2 * len(plain), (len(text), len(plain))


def test_tags_projection_maps_tags_to_longnames() -> None:
    data = {"tags": {"smoke": [{"longname": "A.B", "id": "x"}, {"longname": "A.C"}]}}
    assert project_discover_data("tags", data) == {"tags": {"smoke": ["A.B", "A.C"]}}


def test_large_data_is_truncated_with_a_stated_limit() -> None:
    data = {"items": [{"longname": f"Suite.Test {i}", "message": "x" * 50} for i in range(500)]}
    trimmed, note = truncate_data(data, 5000)
    assert len(json.dumps(trimmed)) <= 5000
    assert note and "5000" in note and "items" in note


# --- 10.3 hint gating ---------------------------------------------------------------------


def test_analyze_exit_code_is_a_bitmask() -> None:
    assert decode_analyze_exit_code(3) == {
        "has_errors": True,
        "has_warnings": True,
        "has_infos": False,
        "has_hints": False,
    }
    assert decode_analyze_exit_code(0)["has_errors"] is False


def test_static_variable_not_found_gets_no_shell_quoting_hint(tmp_path: Path) -> None:
    (tmp_path / "v.robot").write_text(
        "*** Test Cases ***\nT\n    Log    ${undefined_var}\n", encoding="utf-8"
    )
    runner = _runner(tmp_path)
    res = runner.execute(
        runner.build_analyze_args(paths=["v.robot"]), hint_kind="static"
    )
    ids = [h.id for h in res.hints]
    assert "rf-variable-not-found-static" in ids
    assert "rf-variable-not-found" not in ids
    assert all("single quotes" not in h.message for h in res.hints)


def test_console_fail_lines_become_failures() -> None:
    console = (
        "==============================================================================\n"
        "F\n"
        "==============================================================================\n"
        "Fails Here                                                            | FAIL |\n"
        "a != b\n"
        "------------------------------------------------------------------------------\n"
        "Passes                                                                | PASS |\n"
        "------------------------------------------------------------------------------\n"
        "F                                                                     | FAIL |\n"
        "2 tests, 1 passed, 1 failed\n"
        "==============================================================================\n"
    )
    assert parse_console_failures(console) == ["Fails Here: a != b"]


def test_run_reports_failures_and_no_tests_hint(tmp_path: Path) -> None:
    (tmp_path / "f.robot").write_text(
        "*** Test Cases ***\nFails Here\n    Should Be Equal    a    b\n", encoding="utf-8"
    )
    runner = _runner(tmp_path)
    res = runner.execute(runner.build_run_args(paths=["f.robot"], outputdir="out"))
    assert res.failures == ["Fails Here: a != b"]

    none = runner.execute(
        runner.build_run_args(paths=["f.robot"], by_longname=["F.No Such Test"], outputdir="out")
    )
    assert any(h.id == "rf-no-tests-after-modifiers" for h in none.hints), none.stdout


# --- 11.1 discover options ------------------------------------------------------------------


def test_discover_search_files_profile_show_and_tags(tmp_path: Path) -> None:
    runner = _runner(tmp_path, "client")
    assert runner.build_discover_args("tests", search="Open Browser")[-3:] == [
        "tests",
        "--search",
        "Open Browser",
    ]
    assert runner.build_discover_args("files")[-2:] == ["discover", "files"]
    assert runner.build_discover_args("tags")[-2:] == ["discover", "tags"]
    show = runner.build_discover_args("profile_show", profile="ci")
    assert show == ["robotcode", "-p", "ci", "-f", "json", "profiles", "show"]
    assert runner.build_discover_args("config")[-2:] == ["config", "show"]


# --- 11.2 results options -------------------------------------------------------------------


def test_results_options_are_passed(tmp_path: Path) -> None:
    runner = _runner(tmp_path, "client")
    log = runner.build_results_args(
        "log", search="invalid credentials", max_depth=2, level="WARN", failed_only=True
    )
    assert log[log.index("log") + 1 :] == [
        "--failed",
        "--search",
        "invalid credentials",
        "--level",
        "warn",
        "--max-depth",
        "2",
    ]
    assert "--sort" in runner.build_results_args("show", sort="elapsed")
    stats = runner.build_results_args("stats", by=["tag", "suite"])
    assert stats[-4:] == ["--by", "tag", "--by", "suite"]
    assert runner.build_results_args("show", message_chars=80)[-2:] == ["--message-chars", "80"]
    assert runner.build_results_args("summary", output_path="results/output.xml")[-2:] == [
        "-o",
        "results/output.xml",
    ]
    diff = runner.build_results_args("diff", baseline="baseline/output.xml", only=["new-failures"])
    assert diff[-4:] == ["diff", "baseline/output.xml", "--only", "new-failures"]
    merge = runner.build_results_args("merge", merge_inputs=["full.xml", "rerun1.xml"])
    assert merge[merge.index("rebot") :] == [
        "rebot",
        "--merge",
        "--outputdir",
        ".",
        "--output",
        "merged.xml",
        "--log",
        "NONE",
        "--report",
        "NONE",
        "full.xml",
        "rerun1.xml",
    ]
    with pytest.raises(ValueError):
        runner.build_results_args("diff")
    with pytest.raises(ValueError):
        runner.build_results_args("merge", merge_inputs=["only-one.xml"])


def test_live_pinned_run_rerun_merge_and_diff(tmp_path: Path) -> None:
    (tmp_path / "robot.toml").write_text('paths = ["tests"]\noutput-dir = "results"\n', encoding="utf-8")
    (tmp_path / "tests").mkdir()
    suite = tmp_path / "tests" / "orders.robot"
    suite.write_text(
        "*** Test Cases ***\nPasses\n    Log    ok\n\nFails\n    Should Be Equal    1    2\n",
        encoding="utf-8",
    )
    runner = _runner(tmp_path)
    full = runner.execute(runner.build_run_args(output="full.xml"))
    assert full.exit_code == 1
    assert (tmp_path / "results" / "full.xml").is_file()

    suite.write_text(
        "*** Test Cases ***\nPasses\n    Log    ok\n\nFails\n    Should Be Equal    1    1\n",
        encoding="utf-8",
    )
    rerun = runner.execute(
        runner.build_run_args(rerunfailed="results/full.xml", output="rerun1.xml")
    )
    assert rerun.exit_code == 0, rerun.stdout

    merged = runner.execute(
        runner.build_results_args(
            "merge",
            merge_inputs=["results/full.xml", "results/rerun1.xml"],
            output_path="results/merged.xml",
        )
    )
    assert merged.exit_code == 0, merged.stdout + merged.stderr
    summary = runner.execute(runner.build_results_args("summary", output_path="results/merged.xml"))
    assert summary.data["counts"]["failed"] == 0
    assert summary.data["counts"]["passed"] == 2

    diff = runner.execute(
        runner.build_results_args(
            "diff", baseline="results/full.xml", output_path="results/merged.xml"
        )
    )
    assert diff.exit_code in (0, 1), diff.stderr
    assert "Fails" in json.dumps(diff.data)


# --- 11.3 run output, selection guard, collect_unused ---------------------------------------


def test_run_output_collect_unused_and_selection_guard(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    run_args = runner.build_run_args(output="full.xml")
    assert run_args[run_args.index("--output") + 1] == "full.xml"
    assert "--collect-unused" in runner.build_analyze_args(collect_unused=True)
    # robotcode reports only the listed severities; rf_analyze means "this or worse" (2026-11-09: severity=info
    # returned 0 diagnostics for a file with KeywordNotFound errors)
    info = runner.build_analyze_args(severity="info")
    assert info[info.index("--severity") + 1] == "error,warning,info"
    hint = runner.build_analyze_args(severity="hint")
    assert hint[hint.index("--severity") + 1] == "error,warning,info,hint"
    err = runner.build_analyze_args(severity="error")
    assert err[err.index("--severity") + 1] == "error"

    (tmp_path / "robot.toml").write_text('paths = ["tests"]\n', encoding="utf-8")
    (tmp_path / "tests" / "auth").mkdir(parents=True)
    (tmp_path / "tests" / "__init__.robot").write_text(
        "*** Settings ***\nSuite Setup    Log    parent setup\n", encoding="utf-8"
    )
    (tmp_path / "tests" / "auth" / "login.robot").write_text(
        "*** Test Cases ***\nValid Login\n    Log    ok\n", encoding="utf-8"
    )
    warning = runner.selection_warning(
        root=None, paths=["tests/auth/login.robot"], by_longname=None
    )
    assert warning and "__init__.robot" in warning
    assert "Tests.Auth.Login.Valid Login" in warning

    combined = runner.selection_warning(
        root=None, paths=["tests/auth/login.robot"], by_longname=["Login.Valid Login"]
    )
    assert combined and "by_longname" in combined

    assert runner.selection_warning(root=None, paths=None, by_longname=["Tests.Auth.Login.Valid Login"]) is None
    outside = tmp_path / "scratch.robot"
    outside.write_text("*** Test Cases ***\nT\n    Log    x\n", encoding="utf-8")
    assert runner.selection_warning(root=None, paths=["scratch.robot"], by_longname=None) is None
