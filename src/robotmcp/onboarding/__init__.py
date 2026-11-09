"""Onboarding package exports."""

from robotmcp.onboarding.cli import (
    generate_agent_mcp_configs,
    run_doctor,
    run_onboarding_cli,
    sync_skills_to_directory,
)

__all__ = [
    "generate_agent_mcp_configs",
    "run_doctor",
    "run_onboarding_cli",
    "sync_skills_to_directory",
]
