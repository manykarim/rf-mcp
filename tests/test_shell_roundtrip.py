"""Execute synthesized client-mode commands in real shells (PowerShell 7 and bash).

Each case runs `session start repl`, a `send` with `${items}`, a multi-line FOR block and
`.save -t "It's Saved"` exactly as `rf_repl` returns them in client mode. A shell that is not
installed is skipped. Windows PowerShell 5.1 is not covered: its native-argument quoting differs.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from robotmcp.config import ServerConfig
from robotmcp.runner.session_runner import SessionRunner


def _bash() -> str | None:
    if sys.platform == "win32":
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        if git_bash.is_file():
            return str(git_bash)
        found = shutil.which("bash")
        # WSL's System32 bash runs Linux, not this environment
        return found if found and "system32" not in found.lower() else None
    return shutil.which("bash")


SHELLS = {
    "powershell": lambda: [p, "-NoProfile", "-NonInteractive", "-Command"] if (p := shutil.which("pwsh")) else None,
    "bash": lambda: [b, "-c"] if (b := _bash()) else None,
}


def _run(shell_argv: list[str], command: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*shell_argv, command],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize("shell", ["powershell", "bash"])
def test_client_session_commands_round_trip(shell: str, tmp_path: Path) -> None:
    shell_argv = SHELLS[shell]()
    if shell_argv is None:
        pytest.skip(f"{shell} is not installed")
    exe_dir = Path(sys.executable).parent
    if shutil.which("robotcode", path=str(exe_dir)) is None and shutil.which("robotcode") is None:
        pytest.skip("robotcode not on PATH")

    runner = SessionRunner(
        ServerConfig(execution_mode="client", workspace_root=tmp_path, default_shell=shell)
    )
    root = tmp_path.as_posix() if shell == "bash" else str(tmp_path)

    start = _run(shell_argv, runner.run_repl("start", root=root).shell_command or "", tmp_path)
    assert start.returncode == 0, start.stdout + start.stderr
    sid = json.loads(start.stdout)["id"]
    try:
        steps = [
            "${items}=    Create List    alpha    beta",
            "FOR    ${item}    IN    @{items}\n    Log    ${item}\nEND",
            "Should Be Equal    ${items}[1]    beta",
        ]
        for step in steps:
            res = runner.run_repl("send", session_id=sid, statement=step, root=root)
            out = _run(shell_argv, res.shell_command or "", tmp_path)
            assert out.returncode == 0, (step, out.stdout, out.stderr)
            assert json.loads(out.stdout)["status"] == "passed", out.stdout

        saved = tmp_path / "saved.robot"
        save = runner.run_repl(
            "save", session_id=sid, save_path=saved.as_posix(), test_name="It's Saved", root=root
        )
        out = _run(shell_argv, save.shell_command or "", tmp_path)
        assert out.returncode == 0, out.stdout + out.stderr
        text = saved.read_text(encoding="utf-8")
        assert "It's Saved" in text
        assert "${items}=    Create List    alpha    beta" in text
        assert "FOR    ${item}    IN    @{items}" in text
        assert "Should Be Equal    ${items}[1]    beta" in text
    finally:
        stop = runner.run_repl("stop", session_id=sid, root=root)
        _run(shell_argv, stop.shell_command or "", tmp_path)
