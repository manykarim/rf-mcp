"""Fill a server-mode call's missing `root` from the client's MCP roots.

Only clients on the handshake-era protocol can be asked: the 2026-07-28 revision (SEP-2322) removed requests from
the server to the client, so FastMCP 4 dropped `ctx.list_roots()` and only the deprecated session call remains.
New-protocol clients, clients without the `roots` capability and failing requests leave `root` unset, and the call
falls back to the working directory (`ServerConfig.resolve_root`).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

import anyio
from fastmcp.exceptions import ToolError
from fastmcp.server.middleware import Middleware, MiddlewareContext
from mcp import types

from robotmcp.config import ServerConfig

logger = logging.getLogger(__name__)

ROOTS_REQUEST_TIMEOUT_SECONDS = 5.0


def file_uri_to_path(uri: str) -> Path | None:
    """Return the local path of a `file://` URI, or None for any other scheme."""
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    path = url2pathname(unquote(parsed.path))
    if parsed.netloc and parsed.netloc != "localhost":
        path = f"//{parsed.netloc}{path}"
    return Path(path)


def choose_root(roots: list[Path], cwd: Path) -> Path | None:
    """Pick the call's root from the client's roots, or raise ToolError when several leave it open."""
    if not roots:
        return None
    if len(roots) == 1:
        return roots[0]
    here = cwd.resolve()
    containing = [r for r in roots if here == r.resolve() or r.resolve() in here.parents]
    if containing:
        return max(containing, key=lambda r: len(r.resolve().parts))
    listed = ", ".join(str(r) for r in roots)
    raise ToolError(
        f"The client has several workspace folders ({listed}) and none contains the server's working directory. "
        "Pass `root` with the project folder to use."
    )


def _is_modern_protocol(ctx: Any) -> bool:
    check = getattr(ctx, "_is_modern_protocol", None)
    return bool(check()) if callable(check) else False


class ClientRoots(Middleware):
    """Inject `root` into `rf_*` calls from the client's MCP roots (handshake-era clients only)."""

    def __init__(self, config: ServerConfig) -> None:
        self._config = config
        self._cache: dict[int, list[Path]] = {}

    async def on_call_tool(self, context: MiddlewareContext, call_next) -> Any:
        name = getattr(context.message, "name", "")
        args = getattr(context.message, "arguments", None)
        if (
            name.startswith("rf_")
            and isinstance(args, dict)
            and args.get("root") is None
            and self._config.workspace_root is None
            and self._config.resolved_mode(args.get("target")) == "server"
        ):
            root = choose_root(await self._roots(context.fastmcp_context), Path.cwd())
            if root is not None:
                args["root"] = str(root)
        return await call_next(context)

    async def on_notification(self, context: MiddlewareContext, call_next) -> Any:
        if context.method == "notifications/roots/list_changed":
            self._cache.clear()
        return await call_next(context)

    async def _roots(self, ctx: Any) -> list[Path]:
        if ctx is None or _is_modern_protocol(ctx):
            return []
        try:
            session = ctx.session
        except Exception:
            return []
        key = id(session)
        if key in self._cache:
            return self._cache[key]
        roots: list[Path] = []
        if session.check_client_capability(types.ClientCapabilities(roots=types.RootsCapability())):
            try:
                with warnings.catch_warnings():  # MCPDeprecationWarning (a UserWarning) on SDK v2
                    warnings.simplefilter("ignore")
                    with anyio.fail_after(ROOTS_REQUEST_TIMEOUT_SECONDS):
                        result = await session.list_roots()
                roots = [p for p in (file_uri_to_path(str(r.uri)) for r in result.roots) if p is not None]
            except Exception as exc:  # a client that declares roots but cannot answer falls back to the cwd
                logger.info("roots/list failed, using the working directory: %s", exc)
        self._cache[key] = roots
        return roots
