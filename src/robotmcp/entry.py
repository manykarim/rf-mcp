"""CLI entry point for `rf-mcp` / `robotmcp`."""

from __future__ import annotations

import sys
from typing import Sequence

from robotmcp.config import ServerConfig
from robotmcp.onboarding.cli import run_onboarding_cli
from robotmcp.server import create_server
from robotmcp.skills.official import start_background_refresh
from robotmcp.telemetry import configure_otel

ONBOARDING_SUBCOMMANDS = {"doctor", "init", "sync-skills", "update-skills"}


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for `rf-mcp` server and onboarding subcommands."""
    args = list(sys.argv[1:] if argv is None else argv)

    if args and args[0] in ONBOARDING_SUBCOMMANDS:
        return run_onboarding_cli(args)

    config = ServerConfig.from_env_and_args(argv=args)
    configure_otel(
        endpoint=config.otel_endpoint,
        service_name=config.otel_service_name,
    )
    server = create_server(config)
    # A ref channel is fetched in the background when due; the snapshot is served from the next start.
    start_background_refresh(config.official_skills, config.official_skills_refresh_hours)

    if config.transport == "stdio":
        server.run(transport="stdio", show_banner=False)
    else:
        server.run(
            transport=config.transport,
            host=config.host,
            port=config.port,
            show_banner=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
