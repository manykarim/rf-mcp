"""Unit and integration tests for Skills over MCP (`SkillsDirectoryProvider` + `ResourcesAsTools`)."""

from __future__ import annotations

import inspect
import warnings
from pathlib import Path

import anyio
import pytest

from fastmcp import FastMCP
from fastmcp.client import Client

from robotmcp.config import ServerConfig
from robotmcp.skills import (
    BUILTIN_SKILLS_ROOT,
    build_server_instructions,
    mount_skills_provider,
    resolve_skill_roots,
)


async def test_skills_provider_mounts_builtin_external_and_project_skills(
    tmp_path: Path,
) -> None:
    # 1. Create a project-local skill in <workspace>/.robotframework/skills/project-conventions/SKILL.md
    proj_skill_dir = (
        tmp_path / ".robotframework" / "skills" / "project-conventions"
    )
    proj_skill_dir.mkdir(parents=True)
    (proj_skill_dir / "SKILL.md").write_text(
        """---
name: project-conventions
description: Custom project locator and login conventions.
---

# Project Conventions
Always use `resources/auth.resource` for login.
""",
        encoding="utf-8",
    )

    # 2. Create an external plugin layout (`plugins/ext-plugin/skills/ext-skill/SKILL.md`)
    ext_repo = tmp_path / "external_plugins"
    ext_skill_dir = ext_repo / "plugins" / "ext-plugin" / "skills" / "ext-skill"
    ext_skill_dir.mkdir(parents=True)
    (ext_skill_dir / "SKILL.md").write_text(
        """---
name: ext-skill
description: External ecosystem plugin skill.
---

# External Skill
External workflow guidance.
""",
        encoding="utf-8",
    )

    cfg = ServerConfig(
        workspace_root=tmp_path,
        skills_dirs=[ext_repo],
    )
    roots = resolve_skill_roots(cfg)
    assert len(roots) == 3

    mcp = FastMCP("skills-test")
    summaries = mount_skills_provider(mcp, cfg)
    skill_names = [s["name"] for s in summaries]
    assert "project-conventions" in skill_names
    assert "ext-skill" in skill_names
    assert "rf-robotcode" in skill_names

    instructions = build_server_instructions(cfg, summaries)
    assert "skill://rf-robotcode/SKILL.md" in instructions
    assert "skill://project-conventions/SKILL.md" in instructions

    async with Client(mcp) as client:
        # Verify native MCP resource listing
        resources = await client.list_resources()
        uris = {str(r.uri) for r in resources}
        assert "skill://rf-robotcode/SKILL.md" in uris
        assert "skill://project-conventions/SKILL.md" in uris
        assert "skill://ext-skill/SKILL.md" in uris

        # Verify ResourcesAsTools (`read_resource` tool) for tools-only coding agents
        tools = await client.list_tools()
        tool_names = {t.name for t in tools}
        assert "list_resources" in tool_names
        assert "read_resource" in tool_names

        read_res = await client.call_tool(
            "read_resource", {"uri": "skill://project-conventions/SKILL.md"}
        )
        assert "resources/auth.resource" in str(read_res.data)


async def test_server_started_in_a_subfolder_serves_the_project_skills(tmp_path: Path, monkeypatch) -> None:
    """project-independent-server 1.3: the startup root is the working directory walked up to robot.toml."""
    from robotmcp.server import create_server

    (tmp_path / "robot.toml").write_text("", encoding="utf-8")
    (tmp_path / ".robotframework" / "hints.toml").parent.mkdir(parents=True)
    skill = tmp_path / ".robotframework" / "skills" / "shop-conventions"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: shop-conventions\ndescription: Shop rules.\n---\n\n# Shop\n", encoding="utf-8")
    (tmp_path / "tests" / "ui").mkdir(parents=True)
    monkeypatch.chdir(tmp_path / "tests" / "ui")

    mcp = create_server(ServerConfig(execution_mode="server"))
    assert "skill://shop-conventions/SKILL.md" in (mcp.instructions or "")
    async with Client(mcp) as client:
        uris = {str(r.uri) for r in await client.list_resources()}
        assert "skill://shop-conventions/SKILL.md" in uris


# project-skills-follow-client-roots: project skills follow the call's root, including the client's MCP roots.

_LEGACY = {"mode": "legacy"} if "mode" in inspect.signature(Client.__init__).parameters else {}


def _skill_project(path: Path, *skills: str, body: str = "") -> Path:
    (path / ".robotframework" / "skills").mkdir(parents=True)
    (path / "robot.toml").write_text("", encoding="utf-8")
    for name in skills:
        folder = path / ".robotframework" / "skills" / name
        folder.mkdir()
        (folder / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: {name} of {path.name}.\n---\n\n# {name}\nFrom {path.name}. {body}\n",
            encoding="utf-8",
        )
    return path


async def _skill_uris(client: Client) -> set[str]:
    return {str(r.uri) for r in await client.list_resources()}


async def _read(client: Client, uri: str) -> str:
    return "".join(getattr(c, "text", "") for c in await client.read_resource(uri))


@pytest.fixture
def outside(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A server started in a folder outside any project, like VS Code's user entries in the home folder."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.chdir(home)
    return tmp_path


def _server(**kwargs):
    from robotmcp.server import create_server

    return create_server(ServerConfig(execution_mode="server", **kwargs))


def test_resolve_skill_roots_takes_the_project_roots_in_order(tmp_path: Path) -> None:
    shop = _skill_project(tmp_path / "shop", "a")
    api = _skill_project(tmp_path / "api", "b")
    cfg = ServerConfig(workspace_root=tmp_path / "pinned")
    assert resolve_skill_roots(cfg)[0] == BUILTIN_SKILLS_ROOT  # pinned root without skills
    roots = resolve_skill_roots(cfg, project_roots=[shop])
    assert roots == [(shop / ".robotframework" / "skills").resolve(), BUILTIN_SKILLS_ROOT]
    roots = resolve_skill_roots(cfg, project_roots=[api, shop, api])
    assert roots[:2] == [(api / ".robotframework" / "skills").resolve(), (shop / ".robotframework" / "skills").resolve()]
    assert len(roots) == 3


async def test_a_client_root_outside_the_startup_folder_serves_its_skills(outside: Path) -> None:
    shop = _skill_project(outside / "shop", "shop-conventions")
    async with Client(_server(), roots=[shop.as_uri()], **_LEGACY) as client:
        assert "skill://shop-conventions/SKILL.md" in await _skill_uris(client)
        assert "From shop." in await _read(client, "skill://shop-conventions/SKILL.md")
        listed = await client.call_tool("list_resources", {})
        assert "skill://shop-conventions/SKILL.md" in str(listed.data)
        read = await client.call_tool("read_resource", {"uri": "skill://shop-conventions/SKILL.md"})
        assert "From shop." in str(read.data)


async def test_several_client_roots_serve_every_projects_skills(outside: Path) -> None:
    shop = _skill_project(outside / "shop", "shared", "shop-only")
    api = _skill_project(outside / "api", "shared", "api-only")
    async with Client(_server(), roots=[shop.as_uri(), api.as_uri()], **_LEGACY) as client:
        uris = await _skill_uris(client)
        assert {"skill://shop-only/SKILL.md", "skill://api-only/SKILL.md", "skill://shared/SKILL.md"} <= uris
        assert "From shop." in await _read(client, "skill://shared/SKILL.md")


async def test_a_pinned_root_ignores_the_client_roots_for_skills(outside: Path) -> None:
    pinned = _skill_project(outside / "pinned", "pinned-skill")
    shop = _skill_project(outside / "shop", "shop-conventions")
    async with Client(_server(workspace_root=pinned), roots=[shop.as_uri()], **_LEGACY) as client:
        uris = await _skill_uris(client)
        assert "skill://pinned-skill/SKILL.md" in uris
        assert "skill://shop-conventions/SKILL.md" not in uris


async def test_a_project_skill_wins_over_the_builtin_of_the_same_name(outside: Path) -> None:
    shop = _skill_project(outside / "shop", "rf-locators", body="Project locators.")
    async with Client(_server(), roots=[shop.as_uri()], **_LEGACY) as client:
        assert "Project locators." in await _read(client, "skill://rf-locators/SKILL.md")


async def test_changed_client_roots_change_the_skills(outside: Path) -> None:
    shop = _skill_project(outside / "shop", "shop-conventions")
    api = _skill_project(outside / "api", "api-conventions")
    current = [shop.as_uri()]  # Client.set_roots only affects later sessions, so answer from a list we change
    async with Client(_server(), roots=lambda _ctx: list(current), **_LEGACY) as client:
        assert "skill://shop-conventions/SKILL.md" in await _skill_uris(client)
        current[:] = [api.as_uri()]
        with warnings.catch_warnings():  # MCPDeprecationWarning: roots are deprecated in the 2026-07-28 protocol
            warnings.simplefilter("ignore")
            await client.send_roots_list_changed()
        await anyio.sleep(0.1)  # the notification is processed asynchronously
        uris = await _skill_uris(client)
        assert "skill://api-conventions/SKILL.md" in uris
        assert "skill://shop-conventions/SKILL.md" not in uris


async def test_concurrent_clients_see_only_their_own_project_skills(outside: Path) -> None:
    shop = _skill_project(outside / "shop", "shop-conventions")
    api = _skill_project(outside / "api", "api-conventions")
    server = _server()
    results: dict[str, set[str]] = {}

    async def session(name: str, root: Path) -> None:
        async with Client(server, roots=[root.as_uri()], **_LEGACY) as client:
            for _ in range(3):
                results[name] = await _skill_uris(client)
                await anyio.sleep(0)

    async with anyio.create_task_group() as tg:
        tg.start_soon(session, "shop", shop)
        tg.start_soon(session, "api", api)
    assert "skill://shop-conventions/SKILL.md" in results["shop"]
    assert "skill://api-conventions/SKILL.md" not in results["shop"]
    assert "skill://api-conventions/SKILL.md" in results["api"]
    assert "skill://shop-conventions/SKILL.md" not in results["api"]


async def test_a_client_without_roots_gets_the_startup_folders_skills(outside: Path, monkeypatch) -> None:
    shop = _skill_project(outside / "shop", "shop-conventions")
    monkeypatch.chdir(shop)
    async with Client(_server(), **_LEGACY) as client:
        assert "skill://shop-conventions/SKILL.md" in await _skill_uris(client)


async def test_client_mode_ignores_the_client_roots_for_skills(outside: Path) -> None:
    from robotmcp.server import create_server

    shop = _skill_project(outside / "shop", "shop-conventions")
    async with Client(create_server(ServerConfig(execution_mode="client")), roots=[shop.as_uri()], **_LEGACY) as client:
        assert "skill://shop-conventions/SKILL.md" not in await _skill_uris(client)


def test_instructions_point_to_project_skills_when_started_outside_a_project(outside: Path, monkeypatch) -> None:
    note = "Project skills: list_resources shows them once your workspace is known."
    mcp = _server()
    assert note in (mcp.instructions or "")
    assert len(mcp.instructions or "") <= 2500
    assert "skill://rf-robotcode/SKILL.md" in (mcp.instructions or "")

    shop = _skill_project(outside / "shop", "shop-conventions")
    monkeypatch.chdir(shop)
    in_project = _server().instructions or ""
    assert note not in in_project
    assert "skill://shop-conventions/SKILL.md" in in_project
    assert note not in (_server(workspace_root=outside / "home").instructions or "")
