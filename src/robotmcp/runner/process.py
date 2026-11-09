"""Subprocess execution that never leaks child processes on timeout."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass
class ProcessOutcome:
    """Result of `run_process`: `status` is `ok`, `timeout`, or `error` (executable missing)."""

    status: str
    returncode: int | None
    stdout: str
    stderr: str
    error: str | None = None


def kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill `proc` and every process it started."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        proc.kill()
    except OSError:
        pass


def run_process(
    args: Sequence[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    timeout_seconds: float,
    input_text: str | None = None,
) -> ProcessOutcome:
    """Run `args`, capturing UTF-8 output; on timeout kill the whole process tree."""
    popen_kwargs: dict = {}
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(
            list(args),
            cwd=cwd,
            env=dict(env),
            stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kwargs,
        )
    except FileNotFoundError as exc:
        return ProcessOutcome("error", None, "", "", error=f"Executable not found: {exc.filename or args[0]}")
    try:
        stdout, stderr = proc.communicate(input=input_text, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        kill_process_tree(proc)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        return ProcessOutcome("timeout", None, stdout or "", stderr or "")
    return ProcessOutcome("ok", proc.returncode, stdout or "", stderr or "")
