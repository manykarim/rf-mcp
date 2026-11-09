"""Unit and live integration tests for SessionRunner (`robotcode -f json session`)."""

from __future__ import annotations

from pathlib import Path

from robotmcp.config import ServerConfig
from robotmcp.runner.session_runner import SessionRunner


def test_session_runner_client_mode_repl_and_debug_with_fallback(tmp_path: Path) -> None:
    cfg = ServerConfig(
        execution_mode="client",
        workspace_root=tmp_path,
        default_shell="powershell",
    )
    runner = SessionRunner(cfg)

    start_dbg = runner.run_debug(
        "start",
        by_longname=["Suite.Failing Test"],
        breakpoints=["tests/login.robot:12"],
    )
    assert start_dbg.mode == "client"
    assert "--output NONE --log NONE --report NONE" in start_dbg.shell_command
    assert "--break tests/login.robot:12" in start_dbg.shell_command
    assert "-bl 'Suite.Failing Test'" in start_dbg.shell_command
    assert start_dbg.fallback_command is not None
    assert "robot-debug --plain" in start_dbg.fallback_command

    send_repl = runner.run_repl(
        "send",
        session_id="repl-abcd1234",
        statement="${items}=    Create List    alpha    beta",
    )
    assert send_repl.mode == "client"
    assert "'${items}=    Create List    alpha    beta'" in send_repl.shell_command
    # a one-shot REPL cannot reproduce earlier turns, so there is no fallback
    assert send_repl.fallback_command is None
    assert start_dbg.one_shot is True
    assert "escape" in (start_dbg.instruction or "")


def test_session_runner_live_repl_multi_turn_restart_and_save(tmp_path: Path) -> None:
    cfg = ServerConfig(execution_mode="server", workspace_root=tmp_path)
    runner = SessionRunner(cfg)

    start_res = runner.run_repl("start")
    assert start_res.mode == "server"
    assert start_res.exit_code == 0
    assert start_res.session_id is not None
    assert start_res.session_id.startswith("repl-")
    sid = start_res.session_id

    try:
        # Turn 1: Import Collections and create a list
        t1 = runner.run_repl("send", session_id=sid, statement="Import Library    Collections")
        assert t1.exit_code == 0
        assert t1.status == "passed"

        t2 = runner.run_repl(
            "send",
            session_id=sid,
            statement="${items}=    Create List    first    second",
        )
        assert t2.exit_code == 0
        assert t2.status == "passed"

        # Turn 2: Verify variable persists across turns
        t3 = runner.run_repl("send", session_id=sid, statement="Log    ${items}[1]")
        assert t3.exit_code == 0
        assert "second" in t3.output

        # Turn 3: Intentional error triggers HintEngine without killing the session
        t_err = runner.run_repl("send", session_id=sid, statement="*** Settings ***")
        assert t_err.exit_code == 1
        assert t_err.status == "failed"
        assert len(t_err.hints) >= 1
        assert t_err.hints[0].id == "repl-section-header"

        # Turn 4: Save runnable .robot file via .save
        saved_file = tmp_path / "exported.robot"
        t_save = runner.run_repl(
            "save",
            session_id=sid,
            save_path=str(saved_file),
            test_name="Exported REPL Test",
        )
        assert t_save.exit_code == 0
        assert saved_file.is_file()
        robot_text = saved_file.read_text(encoding="utf-8")
        assert "Library    Collections" in robot_text
        assert "Exported REPL Test" in robot_text
        assert "*** Settings ***" not in robot_text.split("*** Test Cases ***")[1]

        # Check status & list
        st = runner.run_repl("status", session_id=sid)
        assert st.status == "waits"

        lst = runner.run_repl("list")
        assert lst.sessions is not None
        assert any(s.get("id") == sid for s in lst.sessions)

        # Restart session (keeps same ID in a fresh OS process)
        rst = runner.run_repl("restart", session_id=sid)
        assert rst.exit_code == 0
        assert rst.session_id == sid
    finally:
        stop_res = runner.run_repl("stop", session_id=sid)
        assert stop_res.exit_code == 0


def test_session_runner_live_debug_preserves_output_xml(tmp_path: Path) -> None:
    suite = tmp_path / "orders.robot"
    suite.write_text(
        """*** Test Cases ***
Order Total Fails
    ${actual}=    Set Variable    42
    Should Be Equal As Integers    ${actual}    99
""",
        encoding="utf-8",
    )
    sentinel_output = tmp_path / "output.xml"
    sentinel_output.write_text("<robot>recorded-run-sentinel</robot>", encoding="utf-8")

    cfg = ServerConfig(execution_mode="server", workspace_root=tmp_path)
    runner = SessionRunner(cfg)

    dbg_start = runner.run_debug(
        "start",
        by_longname=["Orders.Order Total Fails"],
        paths=[str(suite)],
    )
    assert dbg_start.exit_code == 0
    assert dbg_start.session_id is not None
    assert dbg_start.session_id.startswith("debug-")
    assert "exception" in dbg_start.output or "Should Be Equal As Integers" in dbg_start.output
    sid = dbg_start.session_id

    try:
        where_res = runner.run_debug("send", session_id=sid, command=".where")
        assert where_res.exit_code == 0
        assert "Order Total Fails" in where_res.output

        vars_res = runner.run_debug("send", session_id=sid, command=".vars --user")
        assert vars_res.exit_code == 0
        assert "${actual}" in vars_res.output
        assert "42" in vars_res.output
    finally:
        runner.run_debug("stop", session_id=sid)

    # Confirm the recorded output.xml was NOT overwritten by the debug session
    assert sentinel_output.read_text(encoding="utf-8") == "<robot>recorded-run-sentinel</robot>"
