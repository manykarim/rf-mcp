"""Stdio-safe logging setup for the MCP server (change: mcp-stdio-log-safety).

- B1: stderr defaults to WARNING (overridable via ROBOTMCP_LOG_LEVEL), formatted.
- C : logging can never freeze keyword execution. All records go through a
  bounded in-memory queue drained by a background listener thread, so a full or
  blocked stderr pipe blocks only the listener (and overflowing records are
  dropped) — never an RF worker thread.
- B2: an opt-in, non-blocking bridge (ROBOTMCP_MCP_LOG_NOTIFICATIONS) that
  forwards records to the MCP client as structured notifications/message. Off by
  default. Hardened against the feedback loop / stale-session failure modes the
  adversarial review surfaced.
"""
from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os
import queue
import sys
import threading
from typing import Any, Optional

_MCP_THRESHOLDS = (
    (logging.CRITICAL, "critical"),
    (logging.ERROR, "error"),
    (logging.WARNING, "warning"),
    (logging.INFO, "info"),
    (logging.DEBUG, "debug"),
)

# Loggers whose records must never be forwarded over B2 — forwarding a failed
# send re-enters via asyncio's "exception never retrieved" ERROR and loops.
_B2_EXCLUDED_LOGGER_PREFIXES = ("asyncio", "mcp", "fastmcp", __name__)
_B2_MAX_INFLIGHT = 256  # bound B2's pending-send memory; drop beyond this


def _mcp_level(levelno: int) -> str:
    for thr, name in _MCP_THRESHOLDS:
        if levelno >= thr:
            return name
    return "debug"


# ── B2 sink: a single stdio session + its loop, captured by middleware ────────
_sink_lock = threading.Lock()
_sink: dict = {"session": None, "loop": None}
_b2_inflight = 0


def set_mcp_log_sink(session: Any, loop: asyncio.AbstractEventLoop) -> None:
    with _sink_lock:
        _sink["session"] = session
        _sink["loop"] = loop


def clear_mcp_log_sink() -> None:
    with _sink_lock:
        _sink["session"] = None
        _sink["loop"] = None


def _session_alive(session: Any) -> bool:
    # Best-effort liveness so a disconnected session stops receiving sends.
    for attr in ("is_closed", "_closed", "closed"):
        val = getattr(session, attr, None)
        if callable(val):
            try:
                return not bool(val())
            except Exception:
                return True
        if isinstance(val, bool):
            return not val
    return True


class SafeStreamHandler(logging.StreamHandler):
    """Stream handler (drained by the listener thread) that never lets a
    broken/closed stderr spew stdlib's '--- Logging error ---' traceback."""

    def handleError(self, record: logging.LogRecord) -> None:
        # Drop silently on broken/closed/blocked stderr. The queue+listener
        # already guarantees the RF worker thread is never the one blocking.
        pass


class DropQueueHandler(logging.handlers.QueueHandler):
    """QueueHandler that drops (never blocks) when the bounded queue is full —
    so a stuck stderr can never apply backpressure onto execution."""

    def enqueue(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            pass  # under a flood, drop rather than block the caller


class _B2Filter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name or ""
        return not any(
            name == p or name.startswith(p + ".") for p in _B2_EXCLUDED_LOGGER_PREFIXES
        )


_FASTMCP_LOGGER_PREFIXES = ("FastMCP", "fastmcp", "mcp")


class _CollapseFrameworkTracebackFilter(logging.Filter):
    """Drop the multi-frame Python traceback from framework log records (anything
    not originating in robotmcp's own code — e.g. FastMCP's per-failed-tool
    `logger.exception('Error calling tool …')`). The one-line message is kept;
    the full error already reaches the client in the tool result, so the stack is
    pure stderr noise. Version-independent (matches by origin, not logger name).
    robotmcp's own tracebacks are unaffected (and already DEBUG-gated)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info and not (record.name or "").startswith("robotmcp"):
            record.exc_info = None
            record.exc_text = None
        return True


class McpNotificationHandler(logging.Handler):
    """Forward records to the MCP client as structured log notifications (B2),
    non-blocking and loop-safe. No-op without a live captured session."""

    def __init__(self) -> None:
        super().__init__()
        self.addFilter(_B2Filter())

    def emit(self, record: logging.LogRecord) -> None:
        global _b2_inflight
        with _sink_lock:
            session = _sink["session"]
            loop = _sink["loop"]
            inflight = _b2_inflight
        if session is None or loop is None or loop.is_closed():
            return
        if inflight >= _B2_MAX_INFLIGHT or not _session_alive(session):
            return
        try:
            msg = self.format(record)
            level = _mcp_level(record.levelno)
            name = record.name
        except Exception:
            return

        def _dispatch() -> None:
            global _b2_inflight
            try:
                task = asyncio.ensure_future(
                    session.send_log_message(level=level, data=msg, logger=name)
                )
                with _sink_lock:
                    _b2_inflight += 1

                def _done(t: "asyncio.Future") -> None:
                    global _b2_inflight
                    with _sink_lock:
                        _b2_inflight -= 1
                    # Retrieve+swallow so asyncio never logs "exception never
                    # retrieved" (which would re-enter this handler -> loop).
                    try:
                        t.exception()
                    except Exception:
                        pass

                task.add_done_callback(_done)
            except Exception:
                pass  # transport gone / not ready — drop silently

        try:
            loop.call_soon_threadsafe(_dispatch)
        except Exception:
            pass  # loop closed between check and schedule — drop


# ── listener lifecycle ────────────────────────────────────────────────────────
_listener: Optional[logging.handlers.QueueListener] = None
_listener_lock = threading.Lock()


def configure_logging() -> int:
    """Install queue-based, backpressure-safe stderr logging (and, if enabled, the
    MCP bridge). Replaces logging.basicConfig. Returns the effective level."""
    global _listener

    level_name = os.environ.get("ROBOTMCP_LOG_LEVEL", "WARNING").upper()
    level = getattr(logging, level_name, logging.WARNING)
    if not isinstance(level, int):
        level = logging.WARNING

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    stderr_handler = SafeStreamHandler(sys.stderr)
    stderr_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    stderr_handler.setLevel(level)

    real_handlers: list = [stderr_handler]
    if os.environ.get("ROBOTMCP_MCP_LOG_NOTIFICATIONS"):
        mcp_handler = McpNotificationHandler()
        mcp_handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        mcp_handler.setLevel(level)
        real_handlers.append(mcp_handler)

    # Bounded queue: the RF worker thread only ever enqueues (put_nowait, drop on
    # full); the listener thread does the (possibly blocking) real writes.
    log_queue: "queue.Queue" = queue.Queue(maxsize=10000)
    with _listener_lock:
        if _listener is not None:
            try:
                _listener.stop()
            except Exception:
                pass
        _listener = logging.handlers.QueueListener(
            log_queue, *real_handlers, respect_handler_level=True
        )
        _listener.start()

    _tb_filter = _CollapseFrameworkTracebackFilter()
    queue_handler = DropQueueHandler(log_queue)
    # Strip framework tracebacks HERE (before QueueHandler.prepare bakes exc_text
    # into the queued message), keeping the one-line error.
    queue_handler.addFilter(_tb_filter)
    root.addHandler(queue_handler)
    root.setLevel(level)

    # Align FastMCP's own logger so it does not flood stderr on the default path,
    # and attach the traceback-collapse filter at the LOGGER level so it applies
    # regardless of which handler renders the record or whether it propagates
    # (FastMCP's logger name/propagation differs across versions).
    _fastmcp_level = max(level, logging.WARNING)
    for _noisy in ("FastMCP", "fastmcp", "mcp"):
        _lg = logging.getLogger(_noisy)
        _lg.setLevel(_fastmcp_level)
        if not any(isinstance(f, _CollapseFrameworkTracebackFilter) for f in _lg.filters):
            _lg.addFilter(_tb_filter)

    # Disable FastMCP's decorative Rich traceback PANELS on stderr — they are pure
    # rendering (a full box per raising tool call) that the client shows as a wall
    # of warnings; the error itself still reaches the client via the tool result.
    try:
        import fastmcp as _fastmcp
        if hasattr(_fastmcp, "settings") and hasattr(_fastmcp.settings, "enable_rich_tracebacks"):
            _fastmcp.settings.enable_rich_tracebacks = False
        from rich.logging import RichHandler as _RichHandler
        for _lname in ("FastMCP", "fastmcp"):
            _lg = logging.getLogger(_lname)
            for _h in list(_lg.handlers):
                if isinstance(_h, _RichHandler):
                    _lg.removeHandler(_h)
    except Exception:
        pass

    return level


def mcp_log_notifications_enabled() -> bool:
    return bool(os.environ.get("ROBOTMCP_MCP_LOG_NOTIFICATIONS"))


def _effective_level_name() -> str:
    return logging.getLevelName(logging.getLogger().level or logging.WARNING)


def format_ready_banner(transport: str = "stdio") -> str:
    """One deliberate, always-visible startup line: the server is ready + its key
    config. Goes to STDERR (never stdout — that is the JSON-RPC channel)."""
    try:
        from importlib.metadata import version as _v
        ver = _v("rf-mcp")
    except Exception:
        ver = "?"
    libs = "-"
    try:
        from robotmcp.onboarding.diagnostics import library_status
        present = [m for m, ok in library_status().items() if ok]
        libs = ",".join(present) if present else "core-only"
    except Exception:
        pass
    attach = "on" if os.environ.get("ROBOTMCP_ATTACH_HOST") else "off"
    notif = "on" if mcp_log_notifications_enabled() else "off"
    return (
        f"RobotMCP {ver} ready | transport={transport} | log_level={_effective_level_name()} "
        f"| mcp_log_notifications={notif} | attach={attach} | libraries={libs}"
    )


def emit_ready_banner(transport: str = "stdio") -> None:
    """Write the readiness banner to stderr; never raises (backpressure-safe)."""
    try:
        sys.stderr.write(format_ready_banner(transport) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def build_log_sink_middleware():
    """FastMCP middleware capturing the stdio session + loop for the B2 bridge."""
    try:
        from fastmcp.server.middleware import Middleware
    except Exception:
        return None

    class _McpLogSinkMiddleware(Middleware):
        async def on_message(self, context, call_next):
            try:
                fctx = getattr(context, "fastmcp_context", None)
                session = getattr(fctx, "session", None)
                if session is not None:
                    set_mcp_log_sink(session, asyncio.get_running_loop())
            except Exception:
                pass
            return await call_next(context)

    return _McpLogSinkMiddleware()
