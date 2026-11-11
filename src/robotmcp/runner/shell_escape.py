"""Shell-safe quoting and command synthesis for PowerShell and Bash."""

from __future__ import annotations

import os
import re
from typing import Mapping, Sequence

from robotmcp.config import ShellKind

AGENT_ENV_VARS: dict[str, str] = {
    "ROBOTCODE_FORCE_AI_AGENT": "1",
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}

_SAFE_BARE_TOKEN_RE = re.compile(r"^[A-Za-z0-9_./:\\-]+$")


def build_agent_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return environment dictionary with RobotCode AI-agent and UTF-8 invariants enforced."""
    env = dict(os.environ if base_env is None else base_env)
    env.update(AGENT_ENV_VARS)
    return env


def quote_powershell_arg(arg: str) -> str:
    """Single-quote a CLI argument for PowerShell so `${var}` and special chars are never expanded."""
    if not arg:
        return "''"
    if "\n" in arg or "\r" in arg:
        lines = arg.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        escaped_lines = ["'" + line.replace("'", "''") + "'" for line in lines]
        return "(" + " + [char]10 + ".join(escaped_lines) + ")"
    if _SAFE_BARE_TOKEN_RE.match(arg):
        return arg
    return "'" + arg.replace("'", "''") + "'"


def quote_bash_arg(arg: str) -> str:
    """Single-quote a CLI argument for Bash/sh so `${var}` and special chars are never expanded."""
    if not arg:
        return "''"
    if _SAFE_BARE_TOKEN_RE.match(arg) and "\\" not in arg:
        return arg
    return "'" + arg.replace("'", "'\\''") + "'"


def format_cli_command(
    args: Sequence[str],
    *,
    shell: ShellKind = "powershell",
    stdin_lines: Sequence[str] | None = None,
    include_env: bool = True,
) -> str:
    """Format a shell-safe command string for PowerShell or Bash.

    Protects Robot Framework `${var}`, `@{list}`, and `&{dict}` syntax from shell interpolation
    and configures UTF-8 output encoding without BOM corruption.
    """
    if shell == "powershell":
        env_prefix = (
            '$env:ROBOTCODE_FORCE_AI_AGENT="1"; '
            '$env:PYTHONUTF8="1"; '
            '$env:PYTHONIOENCODING="utf-8"; '
            if include_env
            else ""
        )
        cmd_str = " ".join(quote_powershell_arg(str(a)) for a in args)
        if stdin_lines:
            quoted_lines = ", ".join(
                "'" + line.replace("'", "''") + "'" for line in stdin_lines
            )
            return (
                f"{env_prefix}[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); "
                f"@({quoted_lines}) -join [char]10 | {cmd_str}"
            )
        return f"{env_prefix}{cmd_str}"

    env_prefix_bash = (
        "ROBOTCODE_FORCE_AI_AGENT=1 PYTHONUTF8=1 PYTHONIOENCODING=utf-8 "
        if include_env
        else ""
    )
    cmd_str_bash = " ".join(quote_bash_arg(str(a)) for a in args)
    if stdin_lines:
        escaped_printf = "\\n".join(
            line.replace("\\", "\\\\").replace("'", "'\\''").replace("%", "%%")
            for line in stdin_lines
        )
        return f"printf '{escaped_printf}\\n' | {env_prefix_bash}{cmd_str_bash}"
    return f"{env_prefix_bash}{cmd_str_bash}"
