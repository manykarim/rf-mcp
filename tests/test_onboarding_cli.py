"""Unit tests for rf-mcp onboarding CLI (`doctor`, `init`, `sync-skills`)."""

from __future__ import annotations

import json
from pathlib import Path

from robotmcp.onboarding import (
    generate_agent_mcp_configs,
    run_doctor,
    run_onboarding_cli,
    sync_skills_to_directory,
)


def test_run_doctor_verifies_robotcode_2_9_and_session_extra(tmp_path: Path) -> None:
    report = run_doctor(root=tmp_path, as_json=True)
    assert report["ok"] is True
    assert report["robotcode"]["installed"] is True
    assert report["robotcode"]["meets_min_version"] is True
    assert report["robotcode"]["commands"]["session"] is True
    assert any(s["name"] == "rf-robotcode" for s in report["skills"])


def test_generate_agent_mcp_configs_writes_files(tmp_path: Path) -> None:
    configs = generate_agent_mcp_configs(
        agent="all",
        mode="client",
        root=tmp_path,
        url="http://mcp.internal:8000/mcp",
        write_files=True,
    )
    assert set(configs.keys()) == {"claude", "codex", "agy", "copilot", "opencode"}
    assert (tmp_path / ".mcp.json").is_file()
    assert (tmp_path / ".codex" / "config.toml").is_file()
    assert (tmp_path / ".gemini" / "settings.json").is_file()
    assert (tmp_path / ".vscode" / "mcp.json").is_file()
    assert (tmp_path / "opencode.json").is_file()


def test_sync_skills_copies_builtin_and_project_skills(tmp_path: Path) -> None:
    custom = tmp_path / ".robotframework" / "skills" / "custom-team"
    custom.mkdir(parents=True)
    (custom / "SKILL.md").write_text(
        "---\nname: custom-team\ndescription: Team rules\n---\n# Rules\n",
        encoding="utf-8",
    )

    outcome = sync_skills_to_directory(root=tmp_path, agent="claude")
    assert len(outcome["synced"]) >= 2
    assert (tmp_path / ".claude" / "skills" / "rf-robotcode" / "SKILL.md").is_file()
    assert (tmp_path / ".claude" / "skills" / "custom-team" / "SKILL.md").is_file()

    rc = run_onboarding_cli(["doctor", "-r", str(tmp_path), "--json"])
    assert rc == 0


def _rf_mcp_checkout() -> Path:
    import robotmcp

    return Path(robotmcp.__file__).resolve().parents[2]


def test_init_runs_from_the_project_uv_environment(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\n', encoding="utf-8")
    configs = generate_agent_mcp_configs(agent="all", root=tmp_path)
    checkout = str(_rf_mcp_checkout())
    for agent, item in configs.items():
        assert checkout not in item["content"].replace("/", "\\"), agent
        assert checkout.replace("\\", "/") not in item["content"], agent
    claude = json.loads(configs["claude"]["content"])["mcpServers"]["rf-mcp"]
    assert claude["command"] == "uv"
    assert claude["args"][:4] == ["run", "--directory", str(tmp_path.resolve()), "rf-mcp"]


def test_init_without_uv_project_uses_the_current_interpreter(tmp_path: Path) -> None:
    import sys

    configs = generate_agent_mcp_configs(agent="claude", root=tmp_path)
    claude = json.loads(configs["claude"]["content"])["mcpServers"]["rf-mcp"]
    assert claude["command"] == sys.executable
    assert claude["args"][:2] == ["-m", "robotmcp"]
    assert "--directory" not in claude["args"]


def test_sync_skills_refuses_to_overwrite_a_changed_skill(tmp_path: Path) -> None:
    first = sync_skills_to_directory(root=tmp_path, agent="claude")
    target = tmp_path / ".claude" / "skills" / "rf-robotcode" / "SKILL.md"
    assert str(target.parent) in first["synced"]

    again = sync_skills_to_directory(root=tmp_path, agent="claude")
    assert str(target.parent) in again["unchanged"]

    target.write_text(target.read_text(encoding="utf-8") + "\nLocal team note.\n", encoding="utf-8")
    rc = run_onboarding_cli(["sync-skills", "--agent", "claude", "-r", str(tmp_path)])
    assert rc == 1
    assert "Local team note." in target.read_text(encoding="utf-8")

    rc = run_onboarding_cli(["sync-skills", "--agent", "claude", "-r", str(tmp_path), "--force"])
    assert rc == 0
    assert "Local team note." not in target.read_text(encoding="utf-8")


def test_doctor_reports_subcommands_and_hints_status(tmp_path: Path) -> None:
    report = run_doctor(root=tmp_path, as_json=True)
    assert report["robotcode"]["commands"]["rebot"] is True
    assert all(report["robotcode"]["session_commands"].values())
    assert set(report["robotcode"]["session_commands"]) >= {"start", "send", "stop", "restart"}
    assert report["hints_status"] == "not present"

    hints = tmp_path / ".robotframework" / "hints.toml"
    hints.parent.mkdir(parents=True)
    hints.write_text('[[hints]]\nid = "ok"\nerror_pattern = "x"\nmessage = "y"\n', encoding="utf-8")
    assert run_doctor(root=tmp_path, as_json=True)["hints_status"] == "ok"


def test_codex_config_says_how_to_route_and_approve(tmp_path: Path) -> None:
    import tomllib

    content = generate_agent_mcp_configs(agent="codex", root=tmp_path)["codex"]["content"]
    server = tomllib.loads(content)["mcp_servers"]["rf-mcp"]
    assert "default_tools_approval_mode" not in server  # opt-in only: the tools run outside Codex's sandbox
    assert "rf-mcp sync-skills --agent codex" in content and '# default_tools_approval_mode = "approve"' in content
