"""official-skills-source: the official skills from the bundled package or from a snapshot of a ref."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest

from robotmcp.config import ServerConfig
from robotmcp.server import create_server
from robotmcp.skills import BUILTIN_SKILLS_ROOT, official
from robotmcp.skills.catalog import catalog_entries
from robotmcp.skills.official import load_official_skills
from robotmcp.skills.provider import resolve_skill_roots
from tests.conftest import OFFICIAL_PLUGINS, FakeRemote, sha


def test_fake_remote_serves_a_snapshot_and_the_cache_is_isolated(remote: FakeRemote, official_cache: Path) -> None:
    from robotframework_plugins import bundle

    remote.publish("latest", sha(1))
    snap = bundle.fetch("latest")
    assert (snap.ref, snap.commit) == ("latest", sha(1))
    assert set(snap.skills_dirs()) == set(OFFICIAL_PLUGINS)
    assert (snap.skills_dirs()["rf-web"] / "rf-browser" / "SKILL.md").is_file()
    assert remote.calls == ["latest"]
    remote.fail("main")
    with pytest.raises(bundle.FetchError):
        bundle.fetch("main")
    assert official.cache_base() == official_cache
    real = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    assert official_cache.resolve() != real.resolve()


# --- the state file (task 2.1, design D2) ----------------------------------------------------


def test_state_roundtrip(official_cache: Path, tmp_path: Path) -> None:
    from tests.conftest import FakeSnapshot

    assert official.read_state() == {}
    official.record_check("latest")
    assert set(official.read_state()["latest"]) == {"checked_at"}
    snap = FakeSnapshot(ref="latest", commit=sha(7), path=tmp_path)
    entry = official.record_fetch("latest", snap, catalog_format=1, package_version="2026.11.0.dev1")
    state = official.read_state()
    assert state["latest"] == entry
    assert entry["commit"] == sha(7) and entry["path"] == str(tmp_path.resolve())
    assert entry["catalog_format"] == 1 and entry["package_version"] == "2026.11.0.dev1"
    assert entry["fetched_at"] == entry["checked_at"]
    assert official.state_path() == official_cache / "rf-mcp" / "official-skills.json"


@pytest.mark.parametrize("content", ["not json", "[]", '{"refs": 3}', '{"refs": {"latest": "x"}}'])
def test_an_invalid_state_file_reads_as_empty(content: str) -> None:
    official.state_path().parent.mkdir(parents=True, exist_ok=True)
    official.state_path().write_text(content, encoding="utf-8")
    assert official.read_state() == {}


def test_concurrent_writers_leave_a_valid_state_file(tmp_path: Path) -> None:
    import threading

    from tests.conftest import FakeSnapshot

    def write(n: int) -> None:
        for i in range(10):
            snap = FakeSnapshot(ref=f"ref-{n}", commit=sha(n * 100 + i), path=tmp_path)
            official.record_fetch(f"ref-{n}", snap, catalog_format=1, package_version=None)
            official.record_check(f"ref-{n}")

    threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    state = official.read_state()  # valid JSON, whichever writer won
    assert state and all(entry.get("checked_at") for entry in state.values())
    assert not list(official.state_path().parent.glob(".official-skills-*"))  # no temporary files left over


# --- startup: a stored snapshot or the bundled content (task 2.2, design D3) -------------------

SNAPSHOT_PLUGINS = {
    name: {**spec, "body": "SNAPSHOT-CONTENT"} for name, spec in OFFICIAL_PLUGINS.items()
}


def _fetched(remote: FakeRemote, ref: str, commit: str, **record) -> Path:
    """Publish `ref` at `commit`, fetch it and record it as rf-mcp's update-skills would."""
    from robotframework_plugins import bundle

    remote.publish(ref, commit, SNAPSHOT_PLUGINS)
    snap = bundle.fetch(ref)
    official.record_fetch(ref, snap, catalog_format=record.get("catalog_format", 1), package_version="2026.11.0")
    official.load_official_skills.cache_clear()
    return snap.path


def _cfg(channel: str, **kwargs) -> ServerConfig:
    return ServerConfig(execution_mode="server", official_skills=channel, **kwargs)


def test_bundled_is_served_by_default(fake_bundle, remote: FakeRemote) -> None:
    served = load_official_skills()
    assert served.channel == "bundled" and not served.from_snapshot and served.fallback_reason is None
    assert served.skills_dirs == {k: v.resolve() for k, v in fake_bundle.items()}
    assert remote.calls == []


def test_a_fetched_snapshot_is_served_without_the_network(fake_bundle, remote: FakeRemote) -> None:
    path = _fetched(remote, "latest", sha(1))
    calls = list(remote.calls)
    served = load_official_skills("latest")
    assert (served.channel, served.commit, served.path) == ("latest", sha(1), path)
    assert served.fallback_reason is None and served.warning is None
    assert all(path in d.parents for d in served.skills_dirs.values())
    roots = resolve_skill_roots(_cfg("latest"))
    assert served.skills_dirs["rf-web"] in roots and fake_bundle["rf-web"].resolve() not in roots
    assert remote.calls == calls  # nothing fetched at startup


def test_a_ref_never_fetched_serves_the_bundled_skills(fake_bundle, remote: FakeRemote, caplog) -> None:
    with caplog.at_level(logging.WARNING):
        served = load_official_skills("main")
    assert served.channel == "main" and not served.from_snapshot
    assert served.skills_dirs == {k: v.resolve() for k, v in fake_bundle.items()}
    assert "no snapshot of 'main'" in served.fallback_reason
    assert "rf-mcp update-skills --ref main" in served.warning and "rf-mcp update-skills --ref main" in caplog.text
    assert remote.calls == []


def test_a_removed_snapshot_folder_falls_back(fake_bundle, remote: FakeRemote) -> None:
    import shutil

    path = _fetched(remote, "latest", sha(2))
    shutil.rmtree(path)
    served = load_official_skills("latest")
    assert not served.from_snapshot and "no longer in" in served.fallback_reason
    assert served.skills_dirs == {k: v.resolve() for k, v in fake_bundle.items()}


def test_a_snapshot_in_another_catalogue_format_falls_back(fake_bundle, remote: FakeRemote) -> None:
    _fetched(remote, "latest", sha(3), catalog_format=2)
    served = load_official_skills("latest")
    assert not served.from_snapshot and "catalogue format 2" in served.fallback_reason


def test_a_commit_sha_channel_serves_its_snapshot(fake_bundle, remote: FakeRemote) -> None:
    path = _fetched(remote, sha(4), sha(4))
    served = load_official_skills(sha(4))
    assert served.commit == sha(4) and served.path == path
    assert official.is_commit(sha(4)) and not official.is_commit("latest")


def test_no_package_and_a_ref_channel_serve_no_official_skills() -> None:
    served = load_official_skills("latest")
    assert served.skills_dirs == {} and not served.installed and served.channel == "latest"


def test_the_snapshot_takes_the_official_layer(fake_bundle, remote: FakeRemote, plugin_repo) -> None:
    """mcp-skills-provider, "Snapshot read like the bundled content": between --skills-dir and built-in."""
    snap_path = _fetched(remote, "latest", sha(5))
    repo = plugin_repo({"acme": {"skills": {"acme-robot": ()}, "libraries": []}})
    roots = resolve_skill_roots(_cfg("latest", skills_dirs=[repo]))
    company = roots.index((repo / "plugins" / "acme" / "skills").resolve())
    snapshot = [i for i, r in enumerate(roots) if snap_path in r.parents]
    assert company < min(snapshot) and max(snapshot) < roots.index(BUILTIN_SKILLS_ROOT)
    entries = catalog_entries(_cfg("latest"))
    assert {e.plugin for e in entries if e.layer == "official"} == set(SNAPSHOT_PLUGINS)


def test_create_server_never_fetches(fake_bundle, remote: FakeRemote) -> None:
    create_server(_cfg("latest"))
    create_server(_cfg("main"))
    assert remote.calls == []


async def test_a_served_snapshot_behaves_like_the_bundled_layer(
    fake_bundle, remote: FakeRemote, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task 2.3: resources, project_skills and hint links all come from the snapshot."""
    import inspect
    import json

    from fastmcp.client import Client

    from robotmcp.runner.environment import EnvironmentResolver

    legacy = {"mode": "legacy"} if "mode" in inspect.signature(Client.__init__).parameters else {}
    _fetched(remote, "latest", sha(6))
    shop = tmp_path / "shop"
    shop.mkdir()
    (shop / "robot.toml").write_text("", encoding="utf-8")
    monkeypatch.setattr(EnvironmentResolver, "installed_distributions",
                        lambda self, root: {"robotframework-browser": "20.1.0"})
    calls = list(remote.calls)

    async with Client(create_server(_cfg("latest")), **legacy) as client:
        uris = {str(r.uri) for r in await client.list_resources()}
        assert "skill://rf-browser/SKILL.md" in uris
        text = (await client.read_resource("skill://rf-browser/SKILL.md"))[0].text
        assert "SNAPSHOT-CONTENT" in text
        first = await client.call_tool("rf_discover", {"action": "read", "paths": ["robot.toml"], "root": str(shop)})
        assert first.structured_content["project_skills"][0]["skill"] == "skill://rf-browser/SKILL.md"
        assert json.loads(first.content[0].text)["project_skills"] == first.structured_content["project_skills"]
        diag = await client.call_tool("rf_diagnose", {"error_text": "strict mode violation: x resolved to 2 elements"})
        link = diag.structured_content["hints"][0]["skill"]
        assert link == "skill://rf-browser/references/locators.md"
        assert (await client.read_resource(link))[0].text.startswith("# locators.md")
    assert remote.calls == calls


# --- background refresh (task 3.1, design D4) -------------------------------------------------


def _checked_hours_ago(ref: str, hours: float) -> None:
    import datetime as dt

    refs = official.read_state()
    when = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)  # noqa: UP017
    refs.setdefault(ref, {})["checked_at"] = when.isoformat(timespec="seconds")
    official._write_state(refs)


def test_a_due_refresh_fetches_and_the_next_start_serves_it(fake_bundle, remote: FakeRemote) -> None:
    _fetched(remote, "latest", sha(10))
    running = load_official_skills("latest")  # the server started with this
    _checked_hours_ago("latest", 30)
    remote.publish("latest", sha(11), SNAPSHOT_PLUGINS)
    assert official.refresh("latest", 24) == "fetched"
    assert official.read_state()["latest"]["commit"] == sha(11)
    assert load_official_skills("latest").commit == running.commit == sha(10)  # unchanged until the next start
    official.load_official_skills.cache_clear()
    assert load_official_skills("latest").commit == sha(11)


def test_a_refresh_that_is_not_due_makes_no_request(fake_bundle, remote: FakeRemote) -> None:
    _fetched(remote, "latest", sha(12))
    calls = list(remote.calls)
    _checked_hours_ago("latest", 2)
    assert official.refresh("latest", 24) == "skipped"
    assert remote.calls == calls


def test_an_offline_refresh_keeps_the_snapshot_and_logs_once(fake_bundle, remote: FakeRemote, caplog) -> None:
    _fetched(remote, "latest", sha(13))
    _checked_hours_ago("latest", 30)
    remote.fail("latest")
    with caplog.at_level(logging.WARNING):
        assert official.refresh("latest", 24) == "failed"
    state = official.read_state()["latest"]
    assert state["commit"] == sha(13)
    assert not official.refresh_due("latest", 24)  # checked now: an offline machine retries after one interval
    assert caplog.text.count("Could not refresh the official skills channel 'latest'") == 1
    official.load_official_skills.cache_clear()
    assert load_official_skills("latest").commit == sha(13)


def test_a_ref_never_checked_is_due_and_a_failed_first_fetch_serves_bundled(fake_bundle, remote: FakeRemote) -> None:
    assert official.refresh_due("main", 24)
    assert official.refresh("main", 24) == "failed"  # the fake remote knows no `main`
    assert load_official_skills("main").fallback_reason


@pytest.mark.parametrize("channel", ["bundled", sha(14)])
def test_bundled_and_commits_are_never_refreshed(fake_bundle, remote: FakeRemote, channel: str) -> None:
    assert official.refresh(channel, 24) == "skipped"
    assert official.start_background_refresh(channel, 24) is None
    assert remote.calls == []


def test_refresh_interval_zero_turns_the_background_refresh_off(fake_bundle, remote: FakeRemote) -> None:
    """official-skills-channel, "Refresh off"."""
    assert official.refresh("latest", 0) == "skipped"
    assert official.start_background_refresh("latest", 0) is None
    assert remote.calls == []


def test_main_starts_the_refresh_without_waiting(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    from robotmcp import entry

    release, started = threading.Event(), threading.Event()

    def slow_refresh(channel: str, hours: float) -> str:
        started.set()
        release.wait(10)
        return "fetched"

    class Server:
        def run(self, **_kwargs) -> None:
            assert started.wait(5)  # the refresh runs while the server is serving

    monkeypatch.setattr(official, "refresh", slow_refresh)
    monkeypatch.setattr(entry, "create_server", lambda config: Server())
    monkeypatch.setattr(entry, "configure_otel", lambda **_kwargs: None)
    monkeypatch.setenv("RF_MCP_OFFICIAL_SKILLS", "latest")
    try:
        assert entry.main([]) == 0
        assert [t for t in threading.enumerate() if t.name == "rf-mcp-official-skills"]
    finally:
        release.set()


# --- rf-mcp update-skills (task 3.2, design D7) ------------------------------------------------


def test_update_skills_seeds_a_snapshot_for_the_next_start(fake_bundle, remote: FakeRemote, capsys) -> None:
    import json

    from robotmcp.entry import main

    remote.publish("latest", sha(20), SNAPSHOT_PLUGINS)
    assert main(["update-skills", "--ref", "latest"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ref"] == "latest" and out["commit"] == sha(20) and out["skills"] == 4
    assert Path(out["path"]).is_dir()
    calls = list(remote.calls)
    official.load_official_skills.cache_clear()
    served = load_official_skills("latest")  # what a server in the seeded image serves, offline
    assert served.commit == sha(20) and str(served.path) == out["path"]
    create_server(_cfg("latest"))
    assert remote.calls == calls


def test_update_skills_uses_the_configured_channel(fake_bundle, remote: FakeRemote, monkeypatch, capsys) -> None:
    from robotmcp.entry import main

    remote.publish("main", sha(21))
    monkeypatch.setenv("RF_MCP_OFFICIAL_SKILLS", "main")
    assert main(["update-skills"]) == 0
    assert official.read_state()["main"]["commit"] == sha(21)


def test_update_skills_failure_leaves_the_state_alone(fake_bundle, remote: FakeRemote, capsys) -> None:
    from robotmcp.entry import main

    _fetched(remote, "latest", sha(22))
    before = official.read_state()
    assert main(["update-skills", "--ref", "no-such-branch"]) == 1
    assert "no-such-branch" in capsys.readouterr().err
    assert official.read_state() == before


@pytest.mark.parametrize("env", [{}, {"RF_MCP_OFFICIAL_SKILLS": "bundled"}])
def test_update_skills_needs_a_ref(fake_bundle, remote: FakeRemote, monkeypatch, capsys, env) -> None:
    from robotmcp.entry import main

    monkeypatch.delenv("RF_MCP_OFFICIAL_SKILLS", raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert main(["update-skills"]) == 1
    assert "--ref" in capsys.readouterr().err
    assert remote.calls == []


# --- rf-mcp init --official-skills (task 4.1) --------------------------------------------------


def _project_envs(configs: dict) -> dict[str, dict]:
    import tomlkit

    envs = {
        "claude": json.loads(configs["claude"]["content"])["mcpServers"]["rf-mcp"]["env"],
        "codex": tomlkit.parse(configs["codex"]["content"])["mcp_servers"]["rf-mcp"]["env"].unwrap(),
        "agy": json.loads(configs["agy"]["content"])["mcpServers"]["rf-mcp"]["env"],
        "copilot": json.loads(configs["copilot"]["content"])["servers"]["rf-mcp"]["env"],
        "opencode": json.loads(configs["opencode"]["content"])["mcp"]["rf-mcp"]["environment"],
    }
    return envs


def test_init_writes_the_channel_into_every_project_entry(tmp_path: Path) -> None:
    from robotmcp.onboarding import generate_agent_mcp_configs

    configs = generate_agent_mcp_configs(agent="all", root=tmp_path, install_kind="global", official_skills="latest")
    for agent, env in _project_envs(configs).items():
        assert env["RF_MCP_OFFICIAL_SKILLS"] == "latest", agent
        assert env["ROBOTCODE_FORCE_AI_AGENT"] == "1", agent


def test_init_writes_the_channel_into_every_user_entry(tmp_path: Path, monkeypatch, capsys) -> None:
    """official-skills-channel, "User-level entry for the latest skills"."""
    from robotmcp.onboarding import generate_agent_mcp_configs, run_onboarding_cli

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    user = generate_agent_mcp_configs(agent="all", root=tmp_path, scope="user", official_skills="latest")
    for agent in ("claude", "codex", "agy", "copilot-cli", "opencode"):
        assert "RF_MCP_OFFICIAL_SKILLS=latest" in user[agent]["command"], agent
    assert json.loads(user["copilot"]["command"][2])["env"]["RF_MCP_OFFICIAL_SKILLS"] == "latest"
    run_onboarding_cli(["init", "--scope", "user", "--agent", "claude", "--official-skills", "latest", "-r", str(tmp_path)])
    preview = json.loads(capsys.readouterr().out)
    assert "RF_MCP_OFFICIAL_SKILLS=latest" in preview["claude"]["command"]


def test_init_without_the_option_sets_no_channel(tmp_path: Path, monkeypatch) -> None:
    from robotmcp.onboarding import generate_agent_mcp_configs

    project = generate_agent_mcp_configs(agent="all", root=tmp_path, install_kind="global")
    assert all("RF_MCP_OFFICIAL_SKILLS" not in env for env in _project_envs(project).values())
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    user = generate_agent_mcp_configs(agent="all", root=tmp_path, scope="user")
    assert "RF_MCP_OFFICIAL_SKILLS" not in json.dumps(user)


# --- rf-mcp doctor (task 4.2) --------------------------------------------------------------------


def test_doctor_reports_the_served_snapshot(fake_bundle, remote: FakeRemote, tmp_path: Path, capsys) -> None:
    from robotmcp.onboarding.cli import run_doctor

    path = _fetched(remote, "latest", sha(30))
    calls = list(remote.calls)
    report = run_doctor(root=tmp_path, as_json=True, official_skills="latest")["official_skills"]
    assert report["channel"] == "latest" and report["served"] == "snapshot"
    assert report["commit"] == sha(30) and report["path"] == str(path) and report["fetched_at"]
    assert report["skills"] == 4 and report["version"] == "2026.11.0" and report["fallback_reason"] is None
    capsys.readouterr()
    run_doctor(root=tmp_path, official_skills="latest")
    text = capsys.readouterr().out
    assert f"snapshot of 'latest' at {sha(30)[:12]}" in text and "4 skills" in text
    assert "Fallback     : robotframework-plugins 2026.11.0" in text
    assert remote.calls == calls


def test_doctor_reports_a_fallback(fake_bundle, remote: FakeRemote, tmp_path: Path, monkeypatch, capsys) -> None:
    from robotmcp.onboarding.cli import run_doctor

    monkeypatch.setenv("RF_MCP_OFFICIAL_SKILLS", "main")
    report = run_doctor(root=tmp_path, as_json=True)["official_skills"]
    assert report["channel"] == "main" and report["served"] == "bundled"
    assert "no snapshot of 'main'" in report["fallback_reason"]
    assert "rf-mcp update-skills --ref main" in report["warning"]
    capsys.readouterr()
    run_doctor(root=tmp_path)
    text = capsys.readouterr().out
    assert "Channel      : main (no usable snapshot" in text and "rf-mcp update-skills --ref main" in text
    assert remote.calls == []


def test_doctor_reports_bundled_by_default(fake_bundle, tmp_path: Path, monkeypatch) -> None:
    from robotmcp.onboarding.cli import run_doctor

    monkeypatch.delenv("RF_MCP_OFFICIAL_SKILLS", raising=False)
    report = run_doctor(root=tmp_path, as_json=True)["official_skills"]
    assert (report["channel"], report["served"], report["commit"]) == ("bundled", "bundled", None)
