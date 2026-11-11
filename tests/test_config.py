"""Unit tests for src/robotmcp/config.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from robotmcp.config import ServerConfig


def test_default_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = ServerConfig.from_env_and_args(argv=[], env={})
    assert cfg.execution_mode == "auto"
    assert cfg.workspace_root == tmp_path.resolve()
    assert cfg.profile is None
    assert cfg.skills_dirs == []
    assert cfg.hints_file is None
    assert cfg.otel_endpoint is None
    assert cfg.transport == "stdio"
    assert cfg.port == 8000


def test_cli_flags_override_env(tmp_path: Path) -> None:
    skill_dir_env = tmp_path / "env_skills"
    skill_dir_cli = tmp_path / "cli_skills"
    hints_path = tmp_path / "custom_hints.toml"
    hints_path.write_text("[[hints]]\nid = 'test'\n", encoding="utf-8")

    env = {
        "RF_MCP_MODE": "server",
        "RF_MCP_ROOT": str(tmp_path),
        "RF_MCP_PROFILE": "default-prof",
        "RF_MCP_SKILLS_DIR": str(skill_dir_env),
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://localhost:4317",
        "RF_MCP_TRANSPORT": "stdio",
        "RF_MCP_PORT": "8000",
    }
    argv = [
        "--mode",
        "client",
        "--profile",
        "ci",
        "--skills-dir",
        str(skill_dir_cli),
        "--hints-file",
        str(hints_path),
        "--otel-endpoint",
        "http://collector:4318",
        "--shell",
        "bash",
        "--transport",
        "http",
        "--host",
        "0.0.0.0",
        "--port",
        "9100",
        "--idle-timeout",
        "30",
        "--timeout",
        "120",
    ]
    cfg = ServerConfig.from_env_and_args(argv=argv, env=env)
    assert cfg.execution_mode == "client"
    assert cfg.resolved_mode() == "client"
    assert cfg.resolved_mode(override="server") == "server"
    assert cfg.profile == "ci"
    assert cfg.skills_dirs == [skill_dir_env.resolve(), skill_dir_cli.resolve()]
    assert cfg.resolve_hints_file() == hints_path.resolve()
    assert cfg.otel_endpoint == "http://collector:4318"
    assert cfg.default_shell == "bash"
    assert cfg.transport == "http"
    assert cfg.host == "0.0.0.0"
    assert cfg.port == 9100
    assert cfg.session_idle_timeout_minutes == 30
    assert cfg.session_command_timeout_seconds == 120


def test_workspace_hints_file_discovery(tmp_path: Path) -> None:
    rf_dir = tmp_path / ".robotframework"
    rf_dir.mkdir(parents=True)
    hints_file = rf_dir / "hints.toml"
    hints_file.write_text("[[hints]]\nid = 'ws'\n", encoding="utf-8")

    cfg = ServerConfig(workspace_root=tmp_path)
    assert cfg.resolve_hints_file() == hints_file.resolve()


def test_auto_mode_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = ServerConfig(execution_mode="auto")
    monkeypatch.setattr("shutil.which", lambda _cmd: "/usr/bin/robotcode")
    assert cfg.resolved_mode() == "server"

    monkeypatch.setattr("shutil.which", lambda _cmd: None)
    assert cfg.resolved_mode() == "client"
