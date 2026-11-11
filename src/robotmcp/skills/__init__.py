"""Skills over MCP package exports."""

from robotmcp.skills.provider import (
    BUILTIN_SKILLS_ROOT,
    build_server_instructions,
    discover_skill_summaries,
    mount_skills_provider,
    resolve_skill_roots,
)

__all__ = [
    "BUILTIN_SKILLS_ROOT",
    "build_server_instructions",
    "discover_skill_summaries",
    "mount_skills_provider",
    "resolve_skill_roots",
]
