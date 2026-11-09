"""Configuration management for rf-mcp server and CLI."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

ExecutionMode = Literal["server", "client", "auto"]
ResolvedMode = Literal["server", "client"]
ShellKind = Literal["powershell", "bash"]
TransportKind = Literal["stdio", "http", "sse"]


def default_shell_for_os() -> ShellKind:
    """Return the default shell syntax for the current platform."""
    return "powershell" if sys.platform == "win32" else "bash"


class ServerConfig(BaseModel):
    """Runtime configuration for the rf-mcp server."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    execution_mode: ExecutionMode = "auto"
    workspace_root: Path = Field(default_factory=Path.cwd)
    profile: str | None = None
    skills_dirs: list[Path] = Field(default_factory=list)
    hints_file: Path | None = None
    otel_endpoint: str | None = None
    otel_service_name: str = "rf-mcp"
    default_shell: ShellKind = Field(default_factory=default_shell_for_os)
    transport: TransportKind = "stdio"
    host: str = "127.0.0.1"
    port: int = 8000
    session_idle_timeout_minutes: int = 15
    session_command_timeout_seconds: int = 60
    max_output_chars: int = 20000

    def resolved_mode(self, override: str | None = None) -> ResolvedMode:
        """Resolve effective execution mode ('server' or 'client')."""
        candidate = (override or self.execution_mode).lower()
        if candidate == "server":
            return "server"
        if candidate == "client":
            return "client"
        return "server" if shutil.which("robotcode") is not None else "client"

    def resolve_root(self, root_override: str | Path | None = None) -> Path:
        """Return resolved workspace root path."""
        if root_override is not None:
            return Path(root_override).expanduser().resolve()
        return self.workspace_root.expanduser().resolve()

    def resolve_hints_file(self, root_override: str | Path | None = None) -> Path | None:
        """Return configured hints.toml path or workspace .robotframework/hints.toml if present."""
        if self.hints_file is not None:
            candidate = self.hints_file.expanduser().resolve()
            return candidate if candidate.is_file() else None
        root = self.resolve_root(root_override)
        default_path = root / ".robotframework" / "hints.toml"
        return default_path if default_path.is_file() else None

    @classmethod
    def build_arg_parser(cls) -> argparse.ArgumentParser:
        """Build argument parser for rf-mcp server flags."""
        from robotmcp import __version__

        parser = argparse.ArgumentParser(
            prog="rf-mcp",
            description="Robot Framework MCP Server powered by RobotCode CLI and Skills over MCP",
        )
        parser.add_argument(
            "-V",
            "--version",
            action="version",
            version=f"rf-mcp {__version__}",
        )
        parser.add_argument(
            "--mode",
            choices=["server", "client", "auto"],
            default=None,
            help="Execution mode: 'server' (run robotcode on MCP host), 'client' (return shell-safe commands for client agent), or 'auto'.",
        )
        parser.add_argument(
            "-r",
            "--root",
            default=None,
            help="Default Robot Framework workspace root directory.",
        )
        parser.add_argument(
            "-p",
            "--profile",
            default=None,
            help="Default robot.toml profile name.",
        )
        parser.add_argument(
            "--skills-dir",
            action="append",
            dest="skills_dirs",
            default=None,
            help="Additional skill directory path (can be specified multiple times).",
        )
        parser.add_argument(
            "--hints-file",
            default=None,
            help="Path to custom hints.toml file (defaults to <root>/.robotframework/hints.toml).",
        )
        parser.add_argument(
            "--otel-endpoint",
            default=None,
            help="OpenTelemetry OTLP collector endpoint URL.",
        )
        parser.add_argument(
            "--otel-service-name",
            default=None,
            help="OpenTelemetry service.name attribute (default: rf-mcp).",
        )
        parser.add_argument(
            "--shell",
            choices=["powershell", "bash"],
            default=None,
            help="Default shell syntax for client mode command generation.",
        )
        parser.add_argument(
            "--transport",
            choices=["stdio", "http", "sse"],
            default=None,
            help="MCP transport protocol (default: stdio).",
        )
        parser.add_argument(
            "--host",
            default=None,
            help="HTTP/SSE bind host (default: 127.0.0.1).",
        )
        parser.add_argument(
            "--port",
            type=int,
            default=None,
            help="HTTP/SSE bind port (default: 8000).",
        )
        parser.add_argument(
            "--idle-timeout",
            type=int,
            default=None,
            help="Default idle timeout in minutes for robotcode session (default: 15).",
        )
        parser.add_argument(
            "--timeout",
            type=int,
            default=None,
            help="Default per-command timeout in seconds for robotcode session (default: 60).",
        )
        return parser

    @classmethod
    def from_env_and_args(
        cls,
        argv: Sequence[str] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> ServerConfig:
        """Load ServerConfig from CLI arguments with environment variable fallbacks."""
        environ = dict(os.environ if env is None else env)
        parser = cls.build_arg_parser()
        ns = parser.parse_args(list(argv) if argv is not None else [])

        mode_raw = ns.mode or environ.get("RF_MCP_MODE", "auto")
        if mode_raw not in ("server", "client", "auto"):
            mode_raw = "auto"

        root_raw = ns.root or environ.get("RF_MCP_ROOT")
        workspace_root = (
            Path(root_raw).expanduser().resolve()
            if root_raw
            else Path.cwd().resolve()
        )

        profile = ns.profile or environ.get("RF_MCP_PROFILE") or None

        skills_dirs: list[Path] = []
        env_skills = environ.get("RF_MCP_SKILLS_DIR", "")
        if env_skills:
            sep = ";" if ";" in env_skills else os.pathsep
            for item in env_skills.split(sep):
                if item.strip():
                    skills_dirs.append(Path(item.strip()).expanduser().resolve())
        if ns.skills_dirs:
            for item in ns.skills_dirs:
                if item:
                    skills_dirs.append(Path(item).expanduser().resolve())

        hints_raw = ns.hints_file or environ.get("RF_MCP_HINTS_FILE")
        hints_file = Path(hints_raw).expanduser().resolve() if hints_raw else None

        otel_endpoint = (
            ns.otel_endpoint
            or environ.get("RF_MCP_OTEL_ENDPOINT")
            or environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
            or None
        )
        otel_service_name = (
            ns.otel_service_name
            or environ.get("OTEL_SERVICE_NAME")
            or "rf-mcp"
        )

        shell_raw = ns.shell or environ.get("RF_MCP_SHELL")
        default_shell: ShellKind = (
            shell_raw  # type: ignore[assignment]
            if shell_raw in ("powershell", "bash")
            else default_shell_for_os()
        )

        transport_raw = ns.transport or environ.get("RF_MCP_TRANSPORT", "stdio")
        transport: TransportKind = (
            transport_raw  # type: ignore[assignment]
            if transport_raw in ("stdio", "http", "sse")
            else "stdio"
        )

        host = ns.host or environ.get("RF_MCP_HOST", "127.0.0.1")
        port_raw = ns.port if ns.port is not None else environ.get("RF_MCP_PORT", "8000")
        port = int(port_raw)

        idle_raw = (
            ns.idle_timeout
            if ns.idle_timeout is not None
            else environ.get("RF_MCP_IDLE_TIMEOUT", "15")
        )
        timeout_raw = (
            ns.timeout
            if ns.timeout is not None
            else environ.get("RF_MCP_TIMEOUT", "60")
        )

        return cls(
            execution_mode=mode_raw,  # type: ignore[arg-type]
            workspace_root=workspace_root,
            profile=profile,
            skills_dirs=skills_dirs,
            hints_file=hints_file,
            otel_endpoint=otel_endpoint,
            otel_service_name=otel_service_name,
            default_shell=default_shell,
            transport=transport,
            host=host,
            port=port,
            session_idle_timeout_minutes=int(idle_raw),
            session_command_timeout_seconds=int(timeout_raw),
        )
