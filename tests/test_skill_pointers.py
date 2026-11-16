"""library-aware-skill-retrieval: `project_skills` on the first result of a session and root."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from fastmcp.client import Client
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from robotmcp import telemetry
from robotmcp.config import ServerConfig
from robotmcp.runner.environment import EnvironmentResolver
from robotmcp.server import create_server
from robotmcp.skills.pointers import SkillPointers

_LEGACY = {"mode": "legacy"} if "mode" in inspect.signature(Client.__init__).parameters else {}

BROWSER_AND_REQUESTS = {"robotframework-browser": "20.1.0", "robotframework-requests": "0.9.7", "robotframework": "7.3"}


def _project(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "robot.toml").write_text("", encoding="utf-8")
    return path


@pytest.fixture
def inventories(monkeypatch: pytest.MonkeyPatch) -> dict[Path, dict[str, str] | None]:
    """Per-root installed packages, instead of probing real environments."""
    table: dict[Path, dict[str, str] | None] = {}
    calls: list[Path] = []

    def fake(self: EnvironmentResolver, root: Path) -> dict[str, str] | None:
        calls.append(root.resolve())
        return table.get(root.resolve())

    monkeypatch.setattr(EnvironmentResolver, "installed_distributions", fake)
    table["__calls__"] = calls  # type: ignore[index]
    return table


async def _read(client: Client, root: Path) -> dict:
    res = await client.call_tool("rf_discover", {"action": "read", "paths": ["robot.toml"], "root": str(root)})
    assert json.loads(res.content[0].text) == res.structured_content  # text and structured content agree
    return res.structured_content


def _server(**kwargs):
    return create_server(ServerConfig(execution_mode="server", **kwargs))


async def test_first_call_of_a_session(tmp_path: Path, fake_bundle, inventories) -> None:
    shop = _project(tmp_path / "shop")
    inventories[shop.resolve()] = BROWSER_AND_REQUESTS
    async with Client(_server(), **_LEGACY) as client:
        data = await _read(client, shop)
    assert data["project_skills"] == [
        {"library": "Browser", "installed": "20.1.0", "skill": "skill://rf-browser/SKILL.md", "supported": True},
        {"library": "RequestsLibrary", "installed": "0.9.7", "skill": "skill://rf-requests/SKILL.md", "supported": True},
    ]
    assert "files" in data or len(data) > 1  # the tool's own result is still there


async def test_later_calls_and_another_root(tmp_path: Path, fake_bundle, inventories) -> None:
    shop, api = _project(tmp_path / "shop"), _project(tmp_path / "api")
    inventories[shop.resolve()] = BROWSER_AND_REQUESTS
    inventories[api.resolve()] = {"robotframework-requests": "0.9.7"}
    async with Client(_server(), **_LEGACY) as client:
        assert "project_skills" in await _read(client, shop)
        assert "project_skills" not in await _read(client, shop)
        api_first = await _read(client, api)
        assert [e["skill"] for e in api_first["project_skills"]] == ["skill://rf-requests/SKILL.md"]
        assert "project_skills" not in await _read(client, api)
    async with Client(_server(), **_LEGACY) as client:  # a new session gets them again
        assert "project_skills" in await _read(client, shop)


async def test_a_new_session_on_the_same_server_gets_them_again(tmp_path: Path, fake_bundle, inventories) -> None:
    shop = _project(tmp_path / "shop")
    inventories[shop.resolve()] = BROWSER_AND_REQUESTS
    server = _server()
    async with Client(server, **_LEGACY) as client:
        assert "project_skills" in await _read(client, shop)
    async with Client(server, **_LEGACY) as client:
        assert "project_skills" in await _read(client, shop)


async def test_many_matches_are_capped_at_three(tmp_path: Path, plugin_repo, inventories) -> None:
    libs = [{"name": f"Lib{i}", "package": f"lib-{i}", "skills": [f"skill-{i}"]} for i in range(5)]
    repo = plugin_repo({"many": {"skills": {f"skill-{i}": () for i in range(5)}, "libraries": libs}})
    shop = _project(tmp_path / "shop")
    inventories[shop.resolve()] = {f"lib-{i}": "1.0" for i in range(5)}
    async with Client(_server(skills_dirs=[repo]), **_LEGACY) as client:
        data = await _read(client, shop)
    assert [e["library"] for e in data["project_skills"]] == ["Lib0", "Lib1", "Lib2"]


async def test_unsupported_version_names_the_range(tmp_path: Path, fake_bundle, inventories) -> None:
    shop = _project(tmp_path / "shop")
    inventories[shop.resolve()] = {"robotframework-browser": "19.4.0"}
    async with Client(_server(), **_LEGACY) as client:
        [entry] = (await _read(client, shop))["project_skills"]
    assert entry == {"library": "Browser", "installed": "19.4.0", "skill": "skill://rf-browser/SKILL.md",
                     "supported": False, "versions": ">=20,<21"}


async def test_client_mode_has_no_project_skills(tmp_path: Path, fake_bundle, inventories) -> None:
    shop = _project(tmp_path / "shop")
    inventories[shop.resolve()] = BROWSER_AND_REQUESTS
    server = create_server(ServerConfig(execution_mode="client", workspace_root=shop))
    async with Client(server, **_LEGACY) as client:
        res = await client.call_tool("rf_diagnose", {"error_text": "strict mode violation"})
    assert "project_skills" not in res.structured_content
    assert inventories["__calls__"] == []  # type: ignore[index]


async def test_switched_off(tmp_path: Path, fake_bundle, inventories) -> None:
    shop = _project(tmp_path / "shop")
    inventories[shop.resolve()] = BROWSER_AND_REQUESTS
    cfg = ServerConfig.from_env_and_args(["--mode", "server"], env={"RF_MCP_SKILL_POINTERS": "0"})
    assert cfg.skill_pointers is False
    async with Client(create_server(cfg), **_LEGACY) as client:
        assert "project_skills" not in await _read(client, shop)


async def test_libraries_not_visible(tmp_path: Path, fake_bundle, inventories) -> None:
    shop = _project(tmp_path / "shop")
    inventories[shop.resolve()] = None
    async with Client(_server(), **_LEGACY) as client:
        assert "project_skills" not in await _read(client, shop)


async def test_a_failed_first_call_does_not_use_up_the_pointers(tmp_path: Path, fake_bundle, inventories) -> None:
    shop = _project(tmp_path / "shop")
    inventories[shop.resolve()] = BROWSER_AND_REQUESTS
    async with Client(_server(), **_LEGACY) as client:
        failed = await client.call_tool(
            "rf_discover", {"action": "read", "root": str(shop)}, raise_on_error=False,  # read without paths fails
        )
        assert failed.is_error
        assert "project_skills" in await _read(client, shop)


async def test_a_failure_in_the_middleware_returns_the_original_result(
    tmp_path: Path, fake_bundle, inventories, monkeypatch: pytest.MonkeyPatch
) -> None:
    shop = _project(tmp_path / "shop")
    inventories[shop.resolve()] = BROWSER_AND_REQUESTS

    def boom(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(SkillPointers, "_rewrite", boom)
    async with Client(_server(), **_LEGACY) as client:
        data = await _read(client, shop)
    assert "project_skills" not in data and data


async def test_a_failure_while_matching_gives_no_project_skills(
    tmp_path: Path, fake_bundle, inventories, monkeypatch: pytest.MonkeyPatch
) -> None:
    import robotmcp.skills.pointers as pointers

    shop = _project(tmp_path / "shop")
    inventories[shop.resolve()] = BROWSER_AND_REQUESTS
    monkeypatch.setattr(pointers, "match", lambda *_a: 1 / 0)
    async with Client(_server(), **_LEGACY) as client:
        assert "project_skills" not in await _read(client, shop)


async def test_pointers_are_counted(tmp_path: Path, fake_bundle, inventories, monkeypatch: pytest.MonkeyPatch) -> None:
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    monkeypatch.setattr(telemetry.metrics, "get_meter_provider", lambda: provider)
    monkeypatch.setattr(telemetry.metrics, "get_meter", lambda name, *a, **k: provider.get_meter(name))
    shop = _project(tmp_path / "shop")
    inventories[shop.resolve()] = {"robotframework-browser": "20.1.0"}
    async with Client(_server(), **_LEGACY) as client:
        await _read(client, shop)
    points = [
        (dict(p.attributes), p.value)
        for rm in reader.get_metrics_data().resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics if m.name == "rf_mcp.skills.pointed"
        for p in m.data.data_points
    ]
    assert points == [({"skill": "rf-browser", "source": "project_skills"}, 1)]
