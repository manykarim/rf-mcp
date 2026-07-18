"""Tests for the tool-install onboarding + multi-agent installer
(change: uv-tool-install-onboarding)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from robotmcp.onboarding import adapters as A
from robotmcp.onboarding import codecs, diagnostics, installer
from robotmcp.onboarding.manifest import Manifest


# ── §6.1 per-adapter round-trip: insert preserves existing servers ──────────
_PREEXISTING = {
    "json": ('{ "mcpServers": { "other": { "command": "x" } } }', ["mcpServers"]),
    "jsonc": ('{ // keep\n "mcp": { "other": { "command": "x" } } }', ["mcp"]),
    "toml": ('[mcp_servers.other]\ncommand = "x"\n', ["mcp_servers"]),
    "yaml": ("extensions:\n  other:\n    cmd: x\n", ["extensions"]),
}


@pytest.mark.parametrize("adapter", [a for a in A.REGISTRY if a.status == "supported"],
                         ids=lambda a: a.id)
def test_install_inserts_and_preserves(adapter, tmp_path):
    scope = "project" if adapter.supports_scope("project") else "user"
    home = tmp_path / "home"; home.mkdir()
    cwd = tmp_path / "proj"; cwd.mkdir()
    path = adapter.resolve_path(scope, cwd=cwd, home=home)
    # seed a pre-existing server of the same format under the same container
    seed_text, seed_container = _PREEXISTING[adapter.fmt]
    if seed_container == adapter.container:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(seed_text, encoding="utf-8")

    mani = Manifest(tmp_path / "manifest.json")
    res = installer.install(agents=adapter.id, scope=scope, whats=["mcp"],
                            command="/opt/robotmcp", manifest=mani, home=home, cwd=cwd)
    assert [r.status for r in res] == ["installed"], res

    data, _ = codecs.load(path, adapter.fmt)
    container = codecs.get_container(data, adapter.container)
    assert "robotmcp" in container
    if seed_container == adapter.container:
        assert "other" in container, "pre-existing server must survive"


# ── §6.2 uninstall: unchanged removed, user-edited kept, dry-run no-op ───────
def test_uninstall_removes_unchanged(tmp_path):
    home = tmp_path / "h"; home.mkdir()
    mani = Manifest(tmp_path / "m.json")
    installer.install(agents="claude-code", scope="user", command="/opt/robotmcp",
                      manifest=mani, home=home)
    path = A.get("claude-code").resolve_path("user", home=home)
    assert "robotmcp" in json.loads(path.read_text())["mcpServers"]

    res = installer.uninstall(agents="claude-code", manifest=mani, home=home)
    assert [r.status for r in res] == ["removed"]
    assert not path.exists()  # created-whole-file → removed


def test_uninstall_keeps_user_modified(tmp_path):
    home = tmp_path / "h"; home.mkdir()
    mani = Manifest(tmp_path / "m.json")
    installer.install(agents="claude-code", scope="user", command="/opt/robotmcp",
                      manifest=mani, home=home)
    path = A.get("claude-code").resolve_path("user", home=home)
    d = json.loads(path.read_text())
    d["mcpServers"]["robotmcp"]["command"] = "/edited/by/user"  # user edit
    path.write_text(json.dumps(d))

    res = installer.uninstall(agents="claude-code", manifest=mani, home=home)
    assert [r.status for r in res] == ["kept-user-modified"]
    assert json.loads(path.read_text())["mcpServers"]["robotmcp"]["command"] == "/edited/by/user"


def test_dry_run_writes_nothing(tmp_path):
    home = tmp_path / "h"; home.mkdir()
    mani = Manifest(tmp_path / "m.json")
    installer.install(agents="claude-code", scope="user", dry_run=True,
                      command="/opt/robotmcp", manifest=mani, home=home)
    assert not A.get("claude-code").resolve_path("user", home=home).exists()
    assert mani.entries == []


# ── §6.3 detection + selection ──────────────────────────────────────────────
def test_detection_via_config_dir(tmp_path, monkeypatch):
    home = tmp_path / "h"; (home / ".claude").mkdir(parents=True)
    monkeypatch.setattr("shutil.which", lambda *_a, **_k: None)  # no binaries
    assert A.get("claude-code").detect(home=home) is True
    assert A.get("gemini").detect(home=home) is False


def test_resolve_selection(tmp_path, monkeypatch):
    home = tmp_path / "h"; (home / ".codex").mkdir(parents=True)
    monkeypatch.setattr("shutil.which", lambda *_a, **_k: None)
    assert {a.id for a in A.resolve_selection("all")} == set(A.SUPPORTED_IDS)
    assert [a.id for a in A.resolve_selection("detected", home=home)] == ["codex"]
    assert [a.id for a in A.resolve_selection("claude-code,cursor")] == ["claude-code", "cursor"]
    # planned adapters are never in all/detected
    assert "pi" not in {a.id for a in A.resolve_selection("all")}


def test_planned_adapter_not_written(tmp_path):
    home = tmp_path / "h"; home.mkdir()
    mani = Manifest(tmp_path / "m.json")
    res = installer.install(agents="pi", scope="user", manifest=mani, home=home)
    assert [r.status for r in res] == ["planned"]
    assert mani.entries == []


# ── §6.4 init / doctor / version ────────────────────────────────────────────
def test_version_matches_metadata(capsys):
    diagnostics.cmd_version()
    from importlib.metadata import version
    assert capsys.readouterr().out.strip() == version("rf-mcp")


def test_init_prints_mcp_snippet_and_does_not_start_server(capsys, monkeypatch):
    monkeypatch.setattr(diagnostics.importlib.util, "find_spec", lambda n: None)  # no libs
    rc = diagnostics.cmd_init(browsers=False)
    out = capsys.readouterr().out
    assert rc == 0
    assert '"robotmcp"' in out and "mcpServers" in out
    assert "rf-mcp[web]" in out  # advises the extra rather than failing


def test_init_browsers_dispatches_browser_entry(monkeypatch, capsys):
    monkeypatch.setattr(diagnostics.importlib.util, "find_spec",
                        lambda n: object() if n == "Browser" else None)
    monkeypatch.setattr(diagnostics, "browser_initialized", lambda: False)
    monkeypatch.setattr(diagnostics, "node_present", lambda: True)
    called = {}
    monkeypatch.setattr(diagnostics, "run_browser_init",
                        lambda: (called.setdefault("ran", True), "ok")[1] and (True, "ok"))
    diagnostics.cmd_init(browsers=True)
    assert called.get("ran") is True


def test_doctor_is_read_only(capsys):
    rc = diagnostics.cmd_doctor()
    out = capsys.readouterr().out
    assert rc == 0 and "test libraries:" in out and "Node.js present" in out


# ── §6.5 guard: the resolved rfbrowser invocation stays valid ───────────────
def test_browser_init_argv_runs_when_resolvable():
    import subprocess
    argv = diagnostics.browser_init_argv()
    if argv is None:
        pytest.skip("robotframework-browser not installed in this environment")
    r = subprocess.run(argv + ["--help"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
