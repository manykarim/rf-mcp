"""Validation fixes for SessionRunner: debug flags, run exit-code status, timeouts, client fallbacks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robotmcp.config import ServerConfig
from robotmcp.runner import session_runner as sr
from robotmcp.runner.process import ProcessOutcome
from robotmcp.runner.session_runner import SessionRunner


def _runner(tmp_path: Path, mode: str = "server") -> SessionRunner:
    return SessionRunner(ServerConfig(execution_mode=mode, workspace_root=tmp_path))


def test_debug_break_options_map_to_real_robot_debug_flags(tmp_path: Path) -> None:
    args = _runner(tmp_path).build_debug_args(
        "start",
        no_break_on_exception=True,
        break_on_failed_test=True,
        break_on_failed_suite=True,
        break_on_all_exceptions=True,
        paths=["t.robot"],
    )
    for flag in (
        "--no-break-on-exception",
        "--break-on-failed-test",
        "--break-on-failed-suite",
        "--break-on-all-exceptions",
    ):
        assert flag in args
    assert "--no-stop-on-failure" not in args


def test_live_debug_start_with_no_break_on_exception_is_accepted(tmp_path: Path) -> None:
    suite = tmp_path / "s.robot"
    suite.write_text(
        "*** Test Cases ***\nT\n    Log    one\n    Log    two\n", encoding="utf-8"
    )
    runner = _runner(tmp_path)
    res = runner.run_debug(
        "start", paths=[str(suite)], no_break_on_exception=True, stop_on_entry=True
    )
    try:
        assert res.exit_code != 252, res.output
        assert "not recognized" not in res.output
        assert res.status not in ("error", "aborted"), res.output
    finally:
        if res.session_id:
            runner.run_debug("stop", session_id=res.session_id)


@pytest.mark.parametrize(
    ("payload", "returncode", "expected"),
    [
        ({"id": "debug-1", "status": "passed", "exit_code": 252, "output": "[ ERROR ] option x not recognized"}, 0, "error"),
        ({"id": "debug-1", "status": "passed", "exit_code": 253, "output": "aborted"}, 0, "aborted"),
        ({"id": "debug-1", "status": "passed", "exit_code": 0, "output": "done"}, 0, "passed"),
        ({"id": "debug-1", "status": "failed", "output": "x != y"}, 1, "failed"),
    ],
)
def test_status_reflects_run_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict, returncode: int, expected: str
) -> None:
    monkeypatch.setattr(
        sr,
        "run_process",
        lambda *a, **k: ProcessOutcome("ok", returncode, json.dumps(payload), ""),
    )
    res = _runner(tmp_path).run_debug("start", paths=["t.robot"])
    assert res.status == expected
    assert res.run_exit_code == payload.get("exit_code")
    if expected == "error":
        assert "not recognized" in res.output


def test_session_command_missing_robotcode_returns_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sr, "resolve_robotcode_executable", lambda: ["robotcode-missing-xyz"])
    res = _runner(tmp_path).run_repl("list")
    assert res.status == "error"
    assert "not found" in res.output.lower()
    assert res.instruction and "doctor" in res.instruction


def test_client_mode_repl_has_no_fallback_and_debug_fallback_is_one_shot(tmp_path: Path) -> None:
    runner = _runner(tmp_path, "client")
    send = runner.run_repl("send", session_id="repl-1", statement="Log    hi")
    assert send.fallback_command is None
    assert send.one_shot is None
    assert "rf_run" in (send.instruction or "")

    start = runner.run_debug("start", by_longname=["Suite.Test"], no_break_on_exception=True)
    assert start.one_shot is True
    assert "robot-debug --plain" in (start.fallback_command or "")
    assert "--no-break-on-exception" in (start.fallback_command or "")
    assert "escape the sandbox" in (start.instruction or "")
    # a later send cannot fall back: the one-shot run has ended
    later = runner.run_debug("send", session_id="debug-1", command=".where")
    assert later.fallback_command is None


def test_client_mode_session_root_is_passed_verbatim(tmp_path: Path) -> None:
    runner = _runner(tmp_path, "client")
    res = runner.run_repl("start", root="/home/alice/proj")
    assert res.command[:3] == ["robotcode", "-r", "/home/alice/proj"]
    no_root = runner.run_debug("start", by_longname=["S.T"])
    assert "-r" not in no_root.command
    assert "." not in no_root.command


def test_save_keep_failed(tmp_path: Path) -> None:
    args = _runner(tmp_path, "client").build_repl_args(
        "save", session_id="repl-1", save_path="t.robot", test_name="It's Saved", keep_failed=True
    )
    assert args[-1] == '.save --keep-failed -t "It\'s Saved" t.robot'


def test_contract_multi_line_send_runs_in_order_and_stops_at_first_failure(tmp_path: Path) -> None:
    """Pins `robotcode session send` behaviour that the rf_repl description teaches."""
    runner = _runner(tmp_path)
    start = runner.run_repl("start")
    sid = start.session_id
    assert sid
    try:
        batch = "\n".join(
            [
                "${a}=    Set Variable    first",
                "${b}=    Set Variable    ${a}-second",
                "Should Be Equal    ${b}    wrong",
                "${c}=    Set Variable    never",
            ]
        )
        res = runner.run_repl("send", session_id=sid, statement=batch)
        assert res.status == "failed"
        assert "first-second != wrong" in res.output

        check = runner.run_repl("send", session_id=sid, statement="Log    ${b}")
        assert check.status == "passed"
        missing = runner.run_repl("send", session_id=sid, statement="Log    ${c}")
        assert missing.status == "failed"
        assert "Variable '${c}' not found" in missing.output

        group = runner.run_repl(
            "send",
            session_id=sid,
            statement="GROUP    setup\n    ${g}=    Set Variable    grouped\nEND\nShould Be Equal    ${g}    grouped",
        )
        assert group.status == "passed", group.output
    finally:
        runner.run_repl("stop", session_id=sid)


# --- 15.2 batched debug sends ----------------------------------------------------------------


def test_split_debug_batch_groups_keywords_and_isolates_debugger_commands() -> None:
    from robotmcp.runner.session_runner import split_debug_batch

    batch = "${y}=    Set Variable    ${x}-b\nLog    ${y}\n.vars --user\nwhere\nFOR    ${i}    IN RANGE    2\n    Log    ${i}\nEND\n\n.c"
    assert split_debug_batch(batch) == [
        "${y}=    Set Variable    ${x}-b\nLog    ${y}",
        ".vars --user",
        "where",
        "FOR    ${i}    IN RANGE    2\n    Log    ${i}\nEND",
        ".c",
    ]
    assert split_debug_batch("Log    a\nLog    b") == ["Log    a\nLog    b"]
    assert split_debug_batch("List Should Contain Value    ${l}    a\n.where") == [
        "List Should Contain Value    ${l}    a", ".where"]


def test_client_mode_debug_batch_chains_commands(tmp_path: Path) -> None:
    res = _runner(tmp_path, mode="client").run_debug("send", session_id="debug-1", command="Log    a\n.vars --user")
    assert res.shell_command and res.shell_command.count("session") == 2 and " && " in res.shell_command


def test_contract_debug_batch_mixes_keywords_and_dot_commands(tmp_path: Path) -> None:
    """Pins the 2026-11-06 probe: keyword lines batch in a debug session, dot-commands go alone."""
    (tmp_path / "robot.toml").write_text('output-dir = "results"\n', encoding="utf-8")
    (tmp_path / "probe.robot").write_text(
        "*** Test Cases ***\nProbe\n    ${x}=    Set Variable    one\n    Should Be Equal    ${x}    two\n",
        encoding="utf-8",
    )
    runner = _runner(tmp_path)
    start = runner.run_debug("start", paths=["probe.robot"], timeout_seconds=60)
    sid = start.session_id
    assert start.status == "passed", start.output
    try:
        res = runner.run_debug(
            "send", session_id=sid, command="${y}=    Set Variable    ${x}-b\nLog    ${y}\n.vars --user\n.where"
        )
        assert res.status == "passed", res.output
        assert "one-b" in res.output and "No keyword with name" not in res.output
        assert ">>> .vars --user" in res.output and "${y}" in res.output
        failed = runner.run_debug("send", session_id=sid, command="Should Be Equal    1    2\n.vars --user\nLog    never")
        assert failed.status == "failed"
        assert ">>> .vars" not in failed.output and "not sent" in (failed.instruction or "")
    finally:
        runner.run_debug("stop", session_id=sid)


def test_repl_nudges_batching_after_three_one_line_sends(tmp_path: Path) -> None:
    runner = SessionRunner(ServerConfig(execution_mode="server", workspace_root=tmp_path))

    def send(statement: str, status: str = "passed") -> sr.SessionResult:
        res = sr.SessionResult(mode="server", kind="repl", action="send", session_id="repl-1", status=status, command=["x"])
        return runner._batching_nudge(res, statement)

    assert send("Click    id=a").instruction is None
    assert send("Click    id=b").instruction is None
    assert "send the next known steps together" in (send("Click    id=c").instruction or "")
    assert send("Click    id=d").instruction is None  # once, then every 10th
    assert send("Click    id=e\nClick    id=f").instruction is None  # a batch resets the count
    assert send(".kw click").instruction is None and send("Get Url").instruction is None
    assert send("Click    id=g", status="failed").instruction is None  # a failure resets it too
    assert "command" not in send("Click    id=h").to_response()
    assert sr.SessionResult(mode="server", kind="repl", action="send", status="failed", command=["x"]).to_response()["command"] == ["x"]


def test_split_repl_batch_isolates_dot_commands_only() -> None:
    from robotmcp.runner.session_runner import split_debug_batch

    assert split_debug_batch(".doc XML\n.kw Parse XML\n.kw Get Element", repl=True) == [".doc XML", ".kw Parse XML", ".kw Get Element"]
    assert split_debug_batch("Library    XML\n${d}=    Parse XML    a.xml\n.vars --user\nwhere", repl=True) == [
        "Library    XML\n${d}=    Parse XML    a.xml", ".vars --user", "where",
    ]  # in the REPL a bare `where` is a keyword line, not a debugger command
    assert split_debug_batch("Log    a\nLog    b", repl=True) == ["Log    a\nLog    b"]


def test_contract_repl_batch_with_dot_commands(tmp_path: Path) -> None:
    """2026-11-07 Codex/GPT-5.6-Luna sent `.doc XML` + four `.kw` lines in one send and got
    `Dot-commands can only be sent on their own.`, then spent 11 one-line sends on the lookups."""
    (tmp_path / "robot.toml").write_text('output-dir = "results"\n', encoding="utf-8")
    runner = _runner(tmp_path)
    start = runner.run_repl("start", timeout_seconds=60)
    sid = start.session_id
    assert start.status == "passed", start.output
    try:
        res = runner.run_repl("send", session_id=sid, statement="${x}=    Set Variable    one\n.kw Should Be Equal\n.vars --user")
        assert res.status == "passed", res.output
        assert "on their own" not in res.output
        assert ">>> .kw Should Be Equal" in res.output and ">>> .vars --user" in res.output and "one" in res.output
        failed = runner.run_repl("send", session_id=sid, statement="Should Be Equal    1    2\n.vars --user")
        assert failed.status == "failed" and "not sent" in (failed.instruction or "")
    finally:
        runner.run_repl("stop", session_id=sid)


def test_session_output_is_capped_with_a_doc_tip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = SessionRunner(ServerConfig(execution_mode="server", workspace_root=tmp_path, max_output_chars=1000))
    payload = json.dumps({"status": "passed", "output": "x" * 5000, "session_id": "repl-1"})
    monkeypatch.setattr(sr, "run_process", lambda *a, **k: ProcessOutcome(status="ok", returncode=0, stdout=payload, stderr=""))
    res = runner.run_repl("send", session_id="repl-1", statement=".doc XML")
    assert len(res.output) < 1100 and "characters cut" in res.output
    assert "Output exceeded 1000 characters" in (res.instruction or "") and ".kw <name>" in (res.instruction or "")


def test_batched_session_output_is_capped_as_a_whole(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = SessionRunner(ServerConfig(execution_mode="server", workspace_root=tmp_path, max_output_chars=1000))
    payload = json.dumps({"status": "passed", "output": "y" * 900, "session_id": "repl-1"})
    monkeypatch.setattr(sr, "run_process", lambda *a, **k: ProcessOutcome(status="ok", returncode=0, stdout=payload, stderr=""))
    res = runner.run_repl("send", session_id="repl-1", statement=".doc XML\n.kw Parse XML")
    assert len(res.output) < 1100 and "characters cut" in res.output and "Output exceeded" in (res.instruction or "")
