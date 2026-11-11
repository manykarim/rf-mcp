"""Unit and integration tests for Skills over MCP (`SkillsDirectoryProvider` + `ResourcesAsTools`)."""

from __future__ import annotations

from pathlib import Path

from fastmcp import FastMCP
from fastmcp.client import Client

from robotmcp.config import ServerConfig
from robotmcp.skills import (
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
