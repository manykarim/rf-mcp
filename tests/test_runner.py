"""Unit and integration tests for shell_escape and CliRunner."""

from __future__ import annotations

from pathlib import Path

from robotmcp.config import ServerConfig
from robotmcp.runner.cli_runner import CliRunner
from robotmcp.runner.shell_escape import (
    format_cli_command,
    quote_bash_arg,
    quote_powershell_arg,
)


def test_powershell_quoting_protects_rf_variables_and_bom() -> None:
    cmd = format_cli_command(
        ["robotcode", "-f", "json", "session", "send", "repl-1234", "${items}=    Create List    a    b"],
        shell="powershell",
    )
    assert '$env:PYTHONIOENCODING="utf-8"' in cmd
    assert '$env:PYTHONUTF8="1"' in cmd
    assert '$env:ROBOTCODE_FORCE_AI_AGENT="1"' in cmd
    assert "'${items}=    Create List    a    b'" in cmd


def test_powershell_multiline_quoting_and_stdin_pipe() -> None:
    multiline = "FOR    ${x}    IN    a    b\n    Log    ${x}\nEND"
    quoted = quote_powershell_arg(multiline)
    assert "[char]10" in quoted
    assert "'FOR    ${x}    IN    a    b'" in quoted

    piped = format_cli_command(
        ["robotcode", "repl", "--plain"],
        shell="powershell",
        stdin_lines=["Import Library    Collections", "${x}=    Create List    1    2"],
    )
    assert "UTF8Encoding]::new($false)" in piped
    assert "@('Import Library    Collections', '${x}=    Create List    1    2') -join [char]10 |" in piped


def test_bash_quoting_protects_rf_variables() -> None:
    arg = quote_bash_arg("Log    ${response.json()}[status]")
    assert arg == "'Log    ${response.json()}[status]'"

    cmd = format_cli_command(
        ["robotcode", "robot", "-v", "TOKEN:${secret}", "-btm", "Issue:4409"],
        shell="bash",
    )
    assert "PYTHONIOENCODING=utf-8" in cmd
    assert "'TOKEN:${secret}'" in cmd


def test_cli_runner_client_mode_and_builders(tmp_path: Path) -> None:
    cfg = ServerConfig(
        execution_mode="client",
        workspace_root=tmp_path,
        profile="ci",
        default_shell="powershell",
    )
    runner = CliRunner(cfg)

    discover_args = runner.build_discover_args(
        "metadata",
        by_test_metadata=["Issue:4409"],
        include_values=True,
    )
    # client mode: no server-side root resolution and no path detection
    assert discover_args == [
        "robotcode",
        "-p",
        "ci",
        "-f",
        "json",
        "discover",
        "metadata",
        "--values",
        "-btm",
        "Issue:4409",
    ]

    res = runner.execute(discover_args)
    assert res.mode == "client"
    assert res.exit_code is None
    assert "discover metadata --values" in res.shell_command
    assert res.instruction is not None


def test_cli_runner_server_mode_live_discover_and_analyze(tmp_path: Path) -> None:
    suite = tmp_path / "sample.robot"
    suite.write_text(
        """*** Settings ***
Metadata    Issue    4409

*** Test Cases ***
Valid Demo
    [Documentation]    Simple test
    Log    Hello RobotCode

Broken Demo
    Nonexistent Keyword XYZ
""",
        encoding="utf-8",
    )

    cfg = ServerConfig(execution_mode="server", workspace_root=tmp_path)
    runner = CliRunner(cfg)

    disc_res = runner.execute(runner.build_discover_args("tests"))
    assert disc_res.mode == "server"
    assert disc_res.exit_code == 0
    assert isinstance(disc_res.data, dict)

    analyze_res = runner.execute(runner.build_analyze_args(paths=[str(suite)]))
    assert analyze_res.mode == "server"
    assert isinstance(analyze_res.data, dict)
    assert analyze_res.data["summary"]["errors"] >= 1


def test_build_doc_args_and_pyproject_testpaths(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n',
        encoding="utf-8",
    )
    cfg = ServerConfig(execution_mode="server", workspace_root=tmp_path)
    runner = CliRunner(cfg)

    # Should still append "." when pyproject.toml only defines pytest testpaths
    disc_args = runner.build_discover_args("suites")
    assert disc_args[-1] == "."

    # Should map the library positionally and split pipe-delimited patterns
    doc_args = runner.build_doc_args(
        "keywords",
        library="XML",
        pattern="Parse Xml|Get Element",
    )
    assert doc_args[-3:] == ["XML", "Parse Xml", "Get Element"]

