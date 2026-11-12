"""`rf-mcp init` scopes, path-free entries and merging (project-independent-server, tasks 4.1-4.3)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import tomlkit

from robotmcp.onboarding import agent_config as ac
from robotmcp.onboarding import generate_agent_mcp_configs, run_onboarding_cli

URL = "http://mcp.internal:8000/mcp"


# --- 4.1 commands per scope and install kind -----------------------------------------------------


def test_global_install_project_scope_starts_rf_mcp_by_name(tmp_path: Path) -> None:
    configs = generate_agent_mcp_configs(agent="copilot", root=tmp_path, install_kind="global")
    entry = json.loads(configs["copilot"]["content"])["servers"]["rf-mcp"]
    assert entry["command"] == "rf-mcp" and entry["args"] == []
    assert entry["type"] == "stdio" and entry["env"]["ROBOTCODE_FORCE_AI_AGENT"] == "1"


@pytest.mark.parametrize("kind", ["project", "global"])
def test_no_entry_carries_a_path_or_flags(tmp_path: Path, kind: str) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    for item in generate_agent_mcp_configs(agent="all", root=tmp_path, install_kind=kind).values():
        for flag in ("--directory", '"-r"', "--mode", "--transport", str(tmp_path.resolve())):
            assert flag not in item["content"], (flag, item["content"])


def test_url_gives_an_http_entry_in_both_scopes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = generate_agent_mcp_configs(agent="claude", root=tmp_path, url=URL)
    assert json.loads(project["claude"]["content"])["mcpServers"]["rf-mcp"] == {"type": "http", "url": URL}
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    user = generate_agent_mcp_configs(agent="all", root=tmp_path, url=URL, scope="user")
    assert user["claude"]["command"][-3:] == ["http", "rf-mcp", URL]
    assert user["codex"]["command"][-2:] == ["--url", URL]
    assert json.loads(user["copilot"]["command"][2]) == {"name": "rf-mcp", "type": "http", "url": URL}
    assert user["opencode"]["command"] == ["opencode", "mcp", "add", "--global", "rf-mcp", "--url", URL]


def test_install_kind_is_project_only_when_running_from_the_projects_venv(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".venv").mkdir()
    monkeypatch.setattr(ac.sys, "prefix", str(tmp_path / ".venv"))
    assert ac.detect_install_kind(tmp_path) == "project"
    monkeypatch.setattr(ac.sys, "prefix", str(tmp_path / "elsewhere"))
    assert ac.detect_install_kind(tmp_path) == "global"


# --- 4.2 merging project files ---------------------------------------------------------------------


def test_other_servers_survive_and_the_old_entry_is_replaced(tmp_path: Path) -> None:
    old = {"mcpServers": {"github": {"command": "gh-mcp", "args": ["serve"]},
                          "rf-mcp": {"command": "uv", "args": ["run", "--directory", "C:/old", "rf-mcp", "-r", "C:/old"]}},
           "other": 1}
    (tmp_path / ".mcp.json").write_text(json.dumps(old), encoding="utf-8")
    generate_agent_mcp_configs(agent="claude", root=tmp_path, install_kind="global", write_files=True)
    data = json.loads((tmp_path / ".mcp.json").read_text(encoding="utf-8"))
    assert data["mcpServers"]["github"] == old["mcpServers"]["github"] and data["other"] == 1
    assert data["mcpServers"]["rf-mcp"]["command"] == "rf-mcp" and data["mcpServers"]["rf-mcp"]["args"] == []


def test_codex_comments_and_other_servers_survive(tmp_path: Path) -> None:
    path = tmp_path / ".codex" / "config.toml"
    path.parent.mkdir()
    path.write_text('# team settings\nmodel = "gpt-5.6"\n\n[mcp_servers.github]\ncommand = "gh-mcp"  # keep me\n',
                    encoding="utf-8")
    generate_agent_mcp_configs(agent="codex", root=tmp_path, install_kind="global", write_files=True)
    text = path.read_text(encoding="utf-8")
    assert "# team settings" in text and "# keep me" in text
    data = tomlkit.loads(text).unwrap()
    assert data["model"] == "gpt-5.6" and data["mcp_servers"]["github"]["command"] == "gh-mcp"
    assert data["mcp_servers"]["rf-mcp"]["command"] == "rf-mcp"
    assert "rf-mcp sync-skills --agent codex" in text


def test_invalid_toml_is_left_byte_identical_and_init_fails(tmp_path: Path) -> None:
    path = tmp_path / ".codex" / "config.toml"
    path.parent.mkdir()
    path.write_bytes(b"[mcp_servers\nbroken = \n")
    rc = run_onboarding_cli(["init", "--agent", "codex", "-r", str(tmp_path), "--write"])
    assert rc == 1 and path.read_bytes() == b"[mcp_servers\nbroken = \n"


def test_a_jsonc_file_with_comments_is_not_rewritten(tmp_path: Path) -> None:
    path = tmp_path / ".vscode" / "mcp.json"
    path.parent.mkdir()
    original = '{\n  // our servers\n  "servers": {"github": {"command": "gh-mcp"}}\n}\n'
    path.write_text(original, encoding="utf-8")
    report = generate_agent_mcp_configs(agent="copilot", root=tmp_path, install_kind="global", write_files=True)
    assert path.read_text(encoding="utf-8") == original
    assert report["copilot"]["status"] == "manual" and "comments" in report["copilot"]["message"]
    assert json.loads(report["copilot"]["content"])["servers"]["rf-mcp"]["command"] == "rf-mcp"
    assert ac.init_exit_code(report) == 1


def test_a_url_with_slashes_in_a_string_is_not_taken_for_a_comment(tmp_path: Path) -> None:
    path = tmp_path / ".vscode" / "mcp.json"
    path.parent.mkdir()
    path.write_text('{"servers": {"remote": {"type": "http", "url": "http://x/y"}}}', encoding="utf-8")
    generate_agent_mcp_configs(agent="copilot", root=tmp_path, install_kind="global", write_files=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["servers"]["remote"]["url"] == "http://x/y" and "rf-mcp" in data["servers"]


# --- 4.3 user scope ---------------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    home.mkdir()
    for var in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(var, str(home))
    monkeypatch.setenv("APPDATA", str(home / "AppData" / "Roaming"))
    monkeypatch.setenv("CODEX_HOME", str(home / ".codex"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.chdir(project)
    return home


class Recorder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, cmd):
        self.calls.append([Path(cmd[0]).stem, *cmd[1:]])
        return subprocess.CompletedProcess(cmd, 0, stdout="added", stderr="")


def test_user_scope_registers_with_each_agents_cli_and_writes_nothing_here(home: Path, monkeypatch) -> None:
    monkeypatch.setattr(ac.shutil, "which", lambda name: f"/stub/{name}" if name != "rf-mcp" else None)
    rec = Recorder()
    report = generate_agent_mcp_configs(agent="all", scope="user", write_files=True, run=rec)
    assert list(Path.cwd().iterdir()) == []
    exe = ac._rf_mcp_by_path_or_name()
    env = ["PYTHONUTF8=1", "PYTHONIOENCODING=utf-8", "ROBOTCODE_FORCE_AI_AGENT=1"]
    assert ["claude", "mcp", "remove", "--scope", "user", "rf-mcp"] in rec.calls
    assert ["claude", "mcp", "add", "--scope", "user", "rf-mcp", "-e", env[0], "-e", env[1], "-e", env[2], "--", exe] in rec.calls
    assert ["codex", "mcp", "add", "rf-mcp", "--env", env[0], "--env", env[1], "--env", env[2], "--", exe] in rec.calls
    assert ["agy", "mcp", "add", "-e", env[0], "-e", env[1], "-e", env[2], "rf-mcp", exe] in rec.calls
    assert ["copilot", "mcp", "add", "--env", env[0], "--env", env[1], "--env", env[2], "rf-mcp", "--", exe] in rec.calls
    vscode = next(c for c in rec.calls if c[0] == "code")
    assert vscode[1] == "--add-mcp" and json.loads(vscode[2])["command"] == exe
    assert ["opencode", "mcp", "add", "--global", "rf-mcp", "--env", env[0], "--env", env[1], "--env", env[2], "--",
            exe] in rec.calls
    assert all(item["status"] == "written" for item in report.values())


def test_user_scope_names_rf_mcp_by_full_path_when_it_is_not_on_path(home: Path, monkeypatch, tmp_path: Path) -> None:
    exe = tmp_path / "toolenv" / "rf-mcp.exe"
    exe.parent.mkdir()
    exe.write_text("", encoding="utf-8")
    monkeypatch.setattr(ac.sys, "executable", str(exe.parent / "python.exe"))
    monkeypatch.setattr(ac.shutil, "which", lambda name: None)
    report = generate_agent_mcp_configs(agent="claude", scope="user")
    assert report["claude"]["command"][-1] == str(exe) and report["claude"]["status"] == "preview"


def test_user_scope_preview_runs_nothing(home: Path, monkeypatch) -> None:
    rec = Recorder()
    report = generate_agent_mcp_configs(agent="all", scope="user", run=rec)
    assert rec.calls == [] and not (home / ".config" / "opencode").exists()
    assert {item["status"] for item in report.values()} == {"preview"}


def test_a_missing_agent_cli_is_reported_with_the_command_to_run(home: Path, monkeypatch) -> None:
    monkeypatch.setattr(ac.shutil, "which", lambda name: None)
    report = generate_agent_mcp_configs(agent="codex", scope="user", write_files=True, run=Recorder())
    assert report["codex"]["status"] == "manual" and report["codex"]["command"][:3] == ["codex", "mcp", "add"]
    assert ac.init_exit_code(report) == 1


def test_a_failing_agent_cli_is_an_error(home: Path, monkeypatch) -> None:
    monkeypatch.setattr(ac.shutil, "which", lambda name: f"/stub/{name}")

    def fail(cmd):
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="config is read-only")

    report = generate_agent_mcp_configs(agent="agy", scope="user", write_files=True, run=fail)
    assert report["agy"]["status"] == "error" and "read-only" in report["agy"]["message"]


def test_an_older_rf_mcp_earlier_on_path_is_not_named(tmp_path: Path, monkeypatch) -> None:
    """Found with OpenCode: plain `rf-mcp` started an rf-mcp 0.34 that came first on PATH."""
    exe = tmp_path / "toolenv" / "rf-mcp.exe"
    exe.parent.mkdir()
    exe.write_text("", encoding="utf-8")
    monkeypatch.setattr(ac.sys, "executable", str(exe.parent / "python.exe"))
    monkeypatch.setattr(ac.shutil, "which", lambda name: "/old/bin/rf-mcp" if name == "rf-mcp" else None)
    monkeypatch.setattr(ac, "_same_version_on_path", lambda found: False)
    assert ac._rf_mcp_by_path_or_name() == str(exe)
    monkeypatch.setattr(ac, "_same_version_on_path", lambda found: True)
    assert ac._rf_mcp_by_path_or_name() == "rf-mcp"
