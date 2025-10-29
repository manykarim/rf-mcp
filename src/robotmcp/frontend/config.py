"""Configuration helpers for the optional Django frontend."""

from __future__ import annotations

from dataclasses import dataclass
import os


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def frontend_enabled_from_env(default: bool = False) -> bool:
    """Return whether the frontend should be enabled based on environment variables."""

    return _env_bool("ROBOTMCP_ENABLE_FRONTEND", default)


@dataclass(slots=True)
class FrontendConfig:
    """Runtime configuration for the Django frontend server."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8001
    base_path: str = "/"
    debug: bool = True

    @property
    def url(self) -> str:
        """Return the base URL (without path) for convenience."""

        scheme = "http"
        return f"{scheme}://{self.host}:{self.port}{self.base_path.rstrip('/') or ''}"


def build_frontend_config(
    *,
    enabled: bool,
    host: str | None = None,
    port: int | None = None,
    base_path: str | None = None,
    debug: bool | None = None,
) -> FrontendConfig:
    """Build a `FrontendConfig` combining CLI arguments and environment variables."""

    env_host = os.environ.get("ROBOTMCP_FRONTEND_HOST")
    env_port = os.environ.get("ROBOTMCP_FRONTEND_PORT")
    env_base = os.environ.get("ROBOTMCP_FRONTEND_BASE_PATH")
    env_debug = os.environ.get("ROBOTMCP_FRONTEND_DEBUG")

    resolved_host = host or env_host or "127.0.0.1"
    resolved_port = port
    if resolved_port is None:
        if env_port:
            try:
                resolved_port = int(env_port)
            except ValueError:
                resolved_port = 8001
        else:
            resolved_port = 8001

    resolved_base = base_path or env_base or "/"
    if not resolved_base.startswith("/"):
        resolved_base = f"/{resolved_base}"
    # Ensure trailing slash for predictable URL joining
    if not resolved_base.endswith("/"):
        resolved_base = f"{resolved_base}/"

    resolved_debug = debug
    if resolved_debug is None:
        resolved_debug = _env_bool("ROBOTMCP_FRONTEND_DEBUG", default=True)

    return FrontendConfig(
        enabled=enabled,
        host=resolved_host,
        port=resolved_port,
        base_path=resolved_base,
        debug=resolved_debug,
    )
