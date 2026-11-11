"""Stateless `robotcode -f json session` runner for `rf_repl` and `rf_debug`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Literal, Sequence

from pydantic import BaseModel, Field

from robotmcp.config import ResolvedMode, ServerConfig, ShellKind
from robotmcp.hints import Hint, HintContext, HintEngine
from robotmcp.runner.cli_runner import (
    CliRunner,
    _longname_flag,
    truncate_text,
)
from robotmcp.runner.environment import missing_robotcode_instruction
from robotmcp.runner.process import run_process
from robotmcp.runner.shell_escape import build_agent_env, format_cli_command

_BARE_DEBUG_DOT_COMMANDS = {
    "c",
    "continue",
    "n",
    "next",
    "s",
    "step",
    "o",
    "out",
    "w",
    "where",
    "l",
    "list",
    "vars",
    "abort",
    "q",
    "quit",
    "exit",
}

# Run exit codes that mean the session did not run normally, whatever the payload says.
_RUN_EXIT_STATUS = {252: "error", 253: "aborted"}


def _normalize_debug_command(command: str) -> str:
    """Normalize bare pdb-style debugger commands ('c', 'n', 's', 'where', 'vars --user') to robotcode dot-commands."""
    stripped = command.strip()
    if not stripped or stripped.startswith((".", "!")):
        return command
    first_word = stripped.split()[0].lower()
    if stripped.lower() in _BARE_DEBUG_DOT_COMMANDS or (
        first_word in {"vars", "break", "clear", "list"} and "  " not in stripped
    ):
        return f".{stripped}"
    return command


def split_debug_batch(command: str, *, repl: bool = False) -> list[str]:
    """Split a multi-line debug (or, with `repl`, REPL) `send` into parts `robotcode session send` can run.

    In a debug session keyword lines batch like in the REPL (in order, stop at the first failure), but a
    debugger command inside a batch is run as a keyword (`No keyword with name '.vars --user' found`). So
    consecutive keyword lines - including indented block bodies - stay one part, and every debugger command
    (`.vars`, `.where`, or a bare `where` / `c` that `_normalize_debug_command` turns into one) is its own part.
    The REPL refuses dot-commands inside a batch (`Dot-commands can only be sent on their own.`), so there every
    line starting with `.` (`.kw`, `.doc`, `.vars`) is its own part; bare words stay keywords.
    """
    parts: list[str] = []
    keywords: list[str] = []
    for line in command.splitlines():
        if not line.strip():
            continue
        head = line.lstrip() if repl else _normalize_debug_command(line).lstrip()
        if not line[:1].isspace() and head.startswith((".",) if repl else (".", "!")):
            if keywords:
                parts.append("\n".join(keywords))
                keywords = []
            parts.append(line.strip())
        else:
            keywords.append(line)
    if keywords:
        parts.append("\n".join(keywords))
    return parts


ReplAction = Literal[
    "start",
    "send",
    "wait",
    "status",
    "result",
    "interrupt",
    "restart",
    "list",
    "save",
    "stop",
]

DebugAction = Literal[
    "start",
    "send",
    "wait",
    "status",
    "result",
    "interrupt",
    "restart",
    "list",
    "stop",
]


class SessionResult(BaseModel):
    """Structured response from a `robotcode session` operation (`repl` or `debug`)."""

    mode: ResolvedMode
    kind: Literal["repl", "debug"]
    action: str
    session_id: str | None = None
    status: str | None = None
    output: str = ""
    exit_code: int | None = None
    run_exit_code: int | None = None
    sessions: list[dict[str, Any]] | None = None
    command: list[str]
    shell_command: str | None = None
    fallback_command: str | None = None
    one_shot: bool | None = None
    hints: list[Hint] = Field(default_factory=list)
    warning: str | None = None
    instruction: str | None = None

    def to_response(self) -> dict[str, Any]:
        """Return the tool response without empty fields (and without `command` after a passed server-mode call)."""
        out = self.model_dump(exclude_none=True)
        if not out.get("hints"):
            out.pop("hints", None)
        if self.mode == "server" and self.status == "passed":
            out.pop("command", None)
        return out


def format_dot_save_command(
    save_path: str,
    *,
    test_name: str | None = None,
    append: bool = False,
    keep_failed: bool = False,
) -> str:
    """Format a `.save [-a] [--keep-failed] [-t \"Name\"] <path>` REPL dot command."""
    parts = [".save"]
    if append:
        parts.append("-a")
    if keep_failed:
        parts.append("--keep-failed")
    if test_name:
        escaped_name = test_name.replace('"', '\\"')
        parts.append(f'-t "{escaped_name}"')
    parts.append(save_path)
    return " ".join(parts)


class SessionRunner:
    """Drives `robotcode -f json session` for multi-turn `repl` and `debug` workflows."""

    def __init__(
        self,
        config: ServerConfig,
        cli_runner: CliRunner | None = None,
        hint_engine: HintEngine | None = None,
    ) -> None:
        self.config = config
        self.hint_engine = hint_engine or HintEngine(
            hints_file=config.resolve_hints_file()
        )
        self.cli_runner = cli_runner or CliRunner(config, self.hint_engine)
        self._single_sends: dict[str, int] = {}  # REPL session id -> passed one-line sends in a row
        self._session_roots: set[Path] = set()  # every root a session was started in, stopped on shutdown

    def _batching_nudge(self, res: SessionResult, statement: str | None) -> SessionResult:
        """After 3 passed one-line keyword sends in a row (and every 10th after), say to batch the next steps.

        Weak models ignore the batching advice in the tool description (and Codex never shows it): the 2026-11-07
        Codex/GPT-5.6-Luna web run made 27 one-line sends. The nudge travels in the response, which every client shows.
        """
        if res.mode != "server" or not res.session_id:
            return res
        lines = [ln for ln in (statement or "").splitlines() if ln.strip()]
        if res.status != "passed" or len(lines) != 1 or lines[0].lstrip().startswith((".", "!")):
            self._single_sends[res.session_id] = 0
            return res
        n = self._single_sends.get(res.session_id, 0) + 1
        self._single_sends[res.session_id] = n
        if n == 3 or (n > 3 and n % 10 == 0):
            nudge = (f"{n} one-line sends in a row: send the next known steps together, one keyword per line "
                     "(they run in order and stop at the first failure).")
            return res.model_copy(update={"instruction": f"{res.instruction} {nudge}" if res.instruction else nudge})
        return res

    def _build_session_prefix(
        self,
        *,
        root: str | Path | None = None,
        profile: str | None = None,
        json_format: bool = True,
        mode: str | None = None,
    ) -> list[str]:
        return self.cli_runner.build_global_args(
            root=root, profile=profile, json_format=json_format, mode=mode
        ) + ["session"]

    def _timeouts(
        self, timeout_seconds: int | None, idle_timeout_minutes: int | None
    ) -> tuple[int, int]:
        return (
            timeout_seconds
            if timeout_seconds is not None
            else self.config.session_command_timeout_seconds,
            idle_timeout_minutes
            if idle_timeout_minutes is not None
            else self.config.session_idle_timeout_minutes,
        )

    @staticmethod
    def _common_tail(
        args: list[str],
        action: str,
        *,
        session_id: str | None,
        timeout: int,
        stop_scope: str,
    ) -> list[str]:
        if action in ("wait", "restart"):
            if not session_id:
                raise ValueError(f"`session_id` is required for action='{action}'.")
            args.extend([action, "--timeout", str(timeout), session_id])
            return args
        if action in ("status", "result", "interrupt"):
            if not session_id:
                raise ValueError(f"`session_id` is required for action='{action}'.")
            args.extend([action, session_id])
            return args
        if action == "list":
            args.append("list")
            return args
        if action == "stop":
            args.append("stop")
            if stop_scope == "project":
                args.append("--project")
            elif stop_scope == "all":
                args.append("--all")
            elif session_id:
                args.append(session_id)
            else:
                args.append("--project")
            return args
        raise ValueError(f"Unsupported session action: {action}")

    def build_repl_args(
        self,
        action: ReplAction = "start",
        *,
        session_id: str | None = None,
        statement: str | None = None,
        save_path: str | None = None,
        test_name: str | None = None,
        append: bool = False,
        keep_failed: bool = False,
        variables: Sequence[str] | None = None,
        variable_files: Sequence[str] | None = None,
        source: str | None = None,
        timeout_seconds: int | None = None,
        idle_timeout_minutes: int | None = None,
        stop_scope: Literal["id", "project", "all"] = "id",
        root: str | Path | None = None,
        profile: str | None = None,
        mode: str | None = None,
    ) -> list[str]:
        """Build `robotcode [-r <root>] [-p <prof>] [-f json] session ...` args for `rf_repl`."""
        timeout, idle = self._timeouts(timeout_seconds, idle_timeout_minutes)
        use_json = action not in ("interrupt", "stop")
        args = self._build_session_prefix(
            root=root, profile=profile, json_format=use_json, mode=mode
        )

        if action == "start":
            args.extend(
                ["start", "repl", "--idle-timeout", str(idle), "--timeout", str(timeout)]
            )
            for v in variables or []:
                args.extend(["-v", v])
            for vf in variable_files or []:
                args.extend(["-V", vf])
            if source:
                args.extend(["-s", source])
            return args

        if action == "save":
            if not session_id:
                raise ValueError("`session_id` is required for action='save'.")
            if not save_path:
                raise ValueError("`save_path` is required for action='save'.")
            dot_cmd = format_dot_save_command(
                save_path, test_name=test_name, append=append, keep_failed=keep_failed
            )
            args.extend(["send", "--timeout", str(timeout), session_id, dot_cmd])
            return args

        if action == "send":
            if not session_id:
                raise ValueError("`session_id` is required for action='send'.")
            if statement is None:
                raise ValueError("`statement` is required for action='send'.")
            args.extend(["send", "--timeout", str(timeout), session_id, statement])
            return args

        return self._common_tail(
            args, action, session_id=session_id, timeout=timeout, stop_scope=stop_scope
        )

    def build_debug_args(
        self,
        action: DebugAction = "start",
        *,
        session_id: str | None = None,
        command: str | None = None,
        by_longname: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        by_test_metadata: Sequence[str] | None = None,
        breakpoints: Sequence[str] | None = None,
        stop_on_entry: bool = False,
        no_break_on_exception: bool = False,
        break_on_all_exceptions: bool = False,
        break_on_failed_test: bool = False,
        break_on_failed_suite: bool = False,
        keep_outputs: bool = False,
        variables: Sequence[str] | None = None,
        paths: Sequence[str] | None = None,
        timeout_seconds: int | None = None,
        idle_timeout_minutes: int | None = None,
        stop_scope: Literal["id", "project", "all"] = "id",
        root: str | Path | None = None,
        profile: str | None = None,
        mode: str | None = None,
    ) -> list[str]:
        """Build `robotcode [-r <root>] [-p <prof>] [-f json] session ...` args for `rf_debug`."""
        timeout, idle = self._timeouts(timeout_seconds, idle_timeout_minutes)
        use_json = action not in ("interrupt", "stop")
        args = self._build_session_prefix(
            root=root, profile=profile, json_format=use_json, mode=mode
        )

        if action == "start":
            args.extend(
                ["start", "debug", "--idle-timeout", str(idle), "--timeout", str(timeout)]
            )
            args.extend(
                self._debug_run_options(
                    by_longname=by_longname,
                    include_tags=include_tags,
                    exclude_tags=exclude_tags,
                    by_test_metadata=by_test_metadata,
                    breakpoints=breakpoints,
                    stop_on_entry=stop_on_entry,
                    no_break_on_exception=no_break_on_exception,
                    break_on_all_exceptions=break_on_all_exceptions,
                    break_on_failed_test=break_on_failed_test,
                    break_on_failed_suite=break_on_failed_suite,
                    keep_outputs=keep_outputs,
                    variables=variables,
                )
            )
            args.extend(self.cli_runner._effective_paths(root, paths, mode=mode))
            return args

        if action == "send":
            if not session_id:
                raise ValueError("`session_id` is required for action='send'.")
            if command is None:
                raise ValueError("`command` is required for action='send'.")
            args.extend(
                ["send", "--timeout", str(timeout), session_id, _normalize_debug_command(command)]
            )
            return args

        return self._common_tail(
            args, action, session_id=session_id, timeout=timeout, stop_scope=stop_scope
        )

    @staticmethod
    def _debug_run_options(
        *,
        by_longname: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        by_test_metadata: Sequence[str] | None = None,
        breakpoints: Sequence[str] | None = None,
        stop_on_entry: bool = False,
        no_break_on_exception: bool = False,
        break_on_all_exceptions: bool = False,
        break_on_failed_test: bool = False,
        break_on_failed_suite: bool = False,
        keep_outputs: bool = False,
        variables: Sequence[str] | None = None,
    ) -> list[str]:
        """Options shared by `session start debug` and `robot-debug` (real `robot-debug` flags only)."""
        opts: list[str] = []
        if not keep_outputs:
            opts.extend(["--output", "NONE", "--log", "NONE", "--report", "NONE"])
        if stop_on_entry:
            opts.append("--stop-on-entry")
        if no_break_on_exception:
            opts.append("--no-break-on-exception")
        if break_on_all_exceptions:
            opts.append("--break-on-all-exceptions")
        if break_on_failed_test:
            opts.append("--break-on-failed-test")
        if break_on_failed_suite:
            opts.append("--break-on-failed-suite")
        for bp in breakpoints or []:
            opts.extend(["--break", bp])
        for ln in by_longname or []:
            opts.extend([_longname_flag(ln), ln])
        for tag in include_tags or []:
            opts.extend(["-i", tag])
        for tag in exclude_tags or []:
            opts.extend(["-e", tag])
        for btm in by_test_metadata or []:
            opts.extend(["-btm", btm])
        for v in variables or []:
            opts.extend(["-v", v])
        return opts

    def _build_debug_fallback(
        self,
        *,
        shell: ShellKind,
        root: str | Path | None,
        profile: str | None,
        paths: Sequence[str] | None,
        run_options: Sequence[str],
    ) -> str:
        """One-shot `robot-debug --plain` run for agent shells that kill background sessions."""
        dbg_args = self.cli_runner.build_global_args(
            root=root, profile=profile, json_format=False, mode="client"
        ) + ["robot-debug", "--plain", *run_options]
        dbg_args.extend(self.cli_runner._effective_paths(root, paths, mode="client"))
        return format_cli_command(
            dbg_args,
            shell=shell,
            stdin_lines=[".where", ".list", ".vars --user", ".abort"],
        )

    def _execute_session_cmd(
        self,
        args: Sequence[str],
        *,
        kind: Literal["repl", "debug"],
        action: str,
        session_id: str | None = None,
        hint_keyword: str = "",
        root: str | Path | None = None,
        mode_override: str | None = None,
        shell: ShellKind | None = None,
        timeout_seconds: int = 120,
        fallback_command: str | None = None,
    ) -> SessionResult:
        mode = self.config.resolved_mode(mode_override)
        cmd_list = list(args)

        if mode == "client":
            effective_shell = shell or self.config.default_shell
            if fallback_command:
                instruction = (
                    "Run `shell_command` locally to drive the `robotcode session`. Only if your shell kills "
                    "background processes between commands (the next call says `No session '<id>' is open`), run "
                    "`fallback_command` once instead: it is a one-shot run that prints the stop location, source and "
                    "variables, then ends. Report only what its output shows; do not start servers, detach "
                    "processes or otherwise escape the sandbox to keep a session alive."
                )
            else:
                instruction = (
                    "Run `shell_command` locally to drive the `robotcode session`. A REPL session needs a shell "
                    "that keeps background processes alive between commands; if `send` says `No session '<id>' is "
                    "open`, write the steps into a `.robot` file and use `rf_run` instead. "
                    "On any keyword failure (exit code 1), call `rf_diagnose` with the output."
                )
            return SessionResult(
                mode="client",
                kind=kind,
                action=action,
                session_id=session_id,
                command=cmd_list,
                shell_command=format_cli_command(cmd_list, shell=effective_shell),
                fallback_command=fallback_command,
                one_shot=True if fallback_command else None,
                instruction=instruction,
            )

        resolved_root = self.config.resolve_root(root)
        exec_cmd, environment = self.cli_runner.robotcode_command(cmd_list, resolved_root)
        if action == "start":
            self._session_roots.add(resolved_root)

        outcome = run_process(
            exec_cmd,
            cwd=str(resolved_root),
            env=build_agent_env(),
            timeout_seconds=timeout_seconds,
        )
        if outcome.status == "error":
            return SessionResult(
                mode="server",
                kind=kind,
                action=action,
                session_id=session_id,
                status="error",
                output=outcome.error or "",
                command=cmd_list,
                warning=environment.warning,
                instruction=missing_robotcode_instruction(resolved_root, environment, self.cli_runner.environments),
            )
        if outcome.status == "timeout":
            return SessionResult(
                mode="server",
                kind=kind,
                action=action,
                session_id=session_id,
                status="timeout",
                output=(outcome.stdout or outcome.stderr).lstrip("﻿"),
                command=cmd_list,
                instruction=(
                    f"The `robotcode session {action}` call did not return within {timeout_seconds} s and was "
                    "stopped. The session itself may still be running: check it with action='status', collect "
                    "output with action='wait', or end it with action='stop'."
                ),
            )

        stdout = outcome.stdout.lstrip("﻿")
        stderr = outcome.stderr.lstrip("﻿")
        returncode = outcome.returncode

        resolved_sid = session_id
        status_str: str | None = None
        output_text = stdout
        run_exit_code: int | None = None
        sessions_list: list[dict[str, Any]] | None = None

        if "-f" in cmd_list and "json" in cmd_list and stdout.strip():
            try:
                payload = json.loads(stdout)
                if isinstance(payload, dict):
                    if payload.get("id"):
                        resolved_sid = str(payload["id"])
                    if "status" in payload:
                        status_str = str(payload["status"])
                    elif "state" in payload:
                        status_str = str(payload["state"])
                    if payload.get("output") is not None:
                        output_text = str(payload["output"])
                    if payload.get("exit_code") is not None:
                        run_exit_code = int(payload["exit_code"])
                elif isinstance(payload, list):
                    sessions_list = payload
                    status_str = "ok"
            except json.JSONDecodeError:
                pass

        if status_str is None:
            if returncode == 0:
                status_str = "stopped" if action == "stop" else "passed"
            elif returncode == 124:
                status_str = "running"
            elif returncode in _RUN_EXIT_STATUS:
                status_str = _RUN_EXIT_STATUS[returncode]
            else:
                status_str = "failed"
        if run_exit_code in _RUN_EXIT_STATUS:
            status_str = _RUN_EXIT_STATUS[run_exit_code]

        if stderr and not output_text:
            output_text = stderr
        elif stderr and returncode != 0:
            output_text = f"{output_text}\n{stderr}".strip()

        hints: list[Hint] = []
        if (
            returncode not in (0, 124)
            or status_str in ("failed", "error", "aborted")
            or "[ FAIL ]" in output_text
            or "[ ERROR ]" in output_text
        ):
            hints = self.hint_engine.diagnose(
                HintContext(keyword=hint_keyword or action, error_text=output_text or stderr),
                kind="runtime",
            )

        instruction: str | None = None
        if returncode == 124 or status_str == "running":
            instruction = (
                f"Session '{resolved_sid}' is still running after timeout. "
                f"Call with action='wait' and session_id='{resolved_sid}' to collect remaining output, "
                f"or action='interrupt' to cancel/pause."
            )
        elif status_str == "error" and run_exit_code == 252:
            instruction = (
                "The run could not start (robot exit code 252: invalid options or no tests selected). "
                "Read `output`, fix the selection or options, and start again."
            )

        output_text, cut = truncate_text(output_text, self.config.max_output_chars)
        if cut:
            if (hint_keyword or "").lstrip().startswith(".doc"):
                cut += " For one keyword use `.kw <name>`, or rf_doc action='keywords' with name patterns."
            instruction = f"{instruction} {cut}" if instruction else cut

        return SessionResult(
            mode="server",
            kind=kind,
            action=action,
            session_id=resolved_sid,
            status=status_str,
            output=output_text,
            exit_code=returncode,
            run_exit_code=run_exit_code,
            sessions=sessions_list,
            command=cmd_list,
            hints=hints,
            warning=environment.warning,
            instruction=instruction,
        )

    def run_repl(
        self,
        action: ReplAction = "start",
        *,
        session_id: str | None = None,
        statement: str | None = None,
        save_path: str | None = None,
        test_name: str | None = None,
        append: bool = False,
        keep_failed: bool = False,
        variables: Sequence[str] | None = None,
        variable_files: Sequence[str] | None = None,
        source: str | None = None,
        timeout_seconds: int | None = None,
        idle_timeout_minutes: int | None = None,
        stop_scope: Literal["id", "project", "all"] = "id",
        root: str | Path | None = None,
        profile: str | None = None,
        mode_override: str | None = None,
        shell: ShellKind | None = None,
    ) -> SessionResult:
        """Execute or synthesize a `robotcode session` REPL action."""
        if action == "send" and statement and "\n" in statement.strip():
            parts = split_debug_batch(statement, repl=True)
            if len(parts) > 1:
                kw = dict(session_id=session_id, timeout_seconds=timeout_seconds, idle_timeout_minutes=idle_timeout_minutes,
                          root=root, profile=profile, mode_override=mode_override, shell=shell)
                res = self._send_batch(parts, lambda part: self.run_repl("send", statement=part, **kw))
                return self._batching_nudge(res, statement)
        args = self.build_repl_args(
            action,
            session_id=session_id,
            statement=statement,
            save_path=save_path,
            test_name=test_name,
            append=append,
            keep_failed=keep_failed,
            variables=variables,
            variable_files=variable_files,
            source=source,
            timeout_seconds=timeout_seconds,
            idle_timeout_minutes=idle_timeout_minutes,
            stop_scope=stop_scope,
            root=root,
            profile=profile,
            mode=mode_override,
        )
        res = self._execute_session_cmd(
            args,
            kind="repl",
            action=action,
            session_id=session_id,
            hint_keyword=statement or action,
            root=root,
            mode_override=mode_override,
            shell=shell,
            timeout_seconds=(timeout_seconds or self.config.session_command_timeout_seconds) + 15,
        )
        return self._batching_nudge(res, statement) if action == "send" else res

    def run_debug(
        self,
        action: DebugAction = "start",
        *,
        session_id: str | None = None,
        command: str | None = None,
        by_longname: Sequence[str] | None = None,
        include_tags: Sequence[str] | None = None,
        exclude_tags: Sequence[str] | None = None,
        by_test_metadata: Sequence[str] | None = None,
        breakpoints: Sequence[str] | None = None,
        stop_on_entry: bool = False,
        no_break_on_exception: bool = False,
        break_on_all_exceptions: bool = False,
        break_on_failed_test: bool = False,
        break_on_failed_suite: bool = False,
        keep_outputs: bool = False,
        variables: Sequence[str] | None = None,
        paths: Sequence[str] | None = None,
        timeout_seconds: int | None = None,
        idle_timeout_minutes: int | None = None,
        stop_scope: Literal["id", "project", "all"] = "id",
        root: str | Path | None = None,
        profile: str | None = None,
        mode_override: str | None = None,
        shell: ShellKind | None = None,
    ) -> SessionResult:
        """Execute or synthesize a `robotcode session` debug action."""
        if action == "send" and command and "\n" in command.strip():
            parts = split_debug_batch(command)
            if len(parts) > 1:
                kw = dict(session_id=session_id, timeout_seconds=timeout_seconds, idle_timeout_minutes=idle_timeout_minutes,
                          root=root, profile=profile, mode_override=mode_override, shell=shell)
                return self._send_batch(parts, lambda part: self.run_debug("send", command=part, **kw))
        run_options = dict(
            by_longname=by_longname,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            by_test_metadata=by_test_metadata,
            breakpoints=breakpoints,
            stop_on_entry=stop_on_entry,
            no_break_on_exception=no_break_on_exception,
            break_on_all_exceptions=break_on_all_exceptions,
            break_on_failed_test=break_on_failed_test,
            break_on_failed_suite=break_on_failed_suite,
            keep_outputs=keep_outputs,
            variables=variables,
        )
        args = self.build_debug_args(
            action,
            session_id=session_id,
            command=command,
            paths=paths,
            timeout_seconds=timeout_seconds,
            idle_timeout_minutes=idle_timeout_minutes,
            stop_scope=stop_scope,
            root=root,
            profile=profile,
            mode=mode_override,
            **run_options,
        )
        fallback: str | None = None
        if action == "start" and self.config.resolved_mode(mode_override) == "client":
            fallback = self._build_debug_fallback(
                shell=shell or self.config.default_shell,
                root=root,
                profile=profile,
                paths=paths,
                run_options=self._debug_run_options(**run_options),
            )
        return self._execute_session_cmd(
            args,
            kind="debug",
            action=action,
            session_id=session_id,
            hint_keyword=command or action,
            root=root,
            mode_override=mode_override,
            shell=shell,
            timeout_seconds=(timeout_seconds or self.config.session_command_timeout_seconds) + 15,
            fallback_command=fallback,
        )

    def _send_batch(self, parts: list[str], send: Callable[[str], SessionResult]) -> SessionResult:
        """Send the parts of a mixed REPL/debug batch in order; stop at the first part that does not pass."""
        results: list[SessionResult] = []
        for part in parts:
            res = send(part)
            results.append(res)
            if res.mode == "server" and res.status != "passed":
                break
        last = results[-1]
        if last.mode == "client":
            # both supported shells (PowerShell 7+, bash) stop a `&&` chain at the first failing command
            return last.model_copy(update={
                "shell_command": " && ".join(r.shell_command or "" for r in results),
                "command": results[0].command,
            })
        output = "\n".join(f">>> {part.splitlines()[0]}{' ...' if chr(10) in part else ''}\n{r.output.rstrip()}"
                           for part, r in zip(parts, results))
        output, cut = truncate_text(output, self.config.max_output_chars)  # each part is capped, the sum too
        skipped = len(parts) - len(results)
        update: dict[str, Any] = {
            "output": output,
            "command": results[0].command,
            "hints": [h for r in results for h in r.hints],
        }
        notes = [n for r in results if r.instruction for n in [r.instruction]]
        if cut:
            notes.append(cut)
        if skipped:
            notes.append(f"Stopped at the first failed part; the remaining {skipped} part(s) were not sent.")
        if notes:
            update["instruction"] = " ".join(dict.fromkeys(notes))
        return last.model_copy(update=update)

    def stop_project_sessions(self, root: str | Path | None = None) -> None:
        """Stop the open `robotcode session` instances of every project this server started one in (on shutdown)."""
        if self.config.resolved_mode() != "server":
            return
        roots: list[str | Path | None] = [root] if root is not None else [*sorted(self._session_roots), None]
        for project in dict.fromkeys(roots):
            try:
                self.run_repl("stop", stop_scope="project", root=project)
            except Exception:
                pass
