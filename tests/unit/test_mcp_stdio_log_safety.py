"""Tests for mcp-stdio-log-safety: fd-1 is never redirected, stderr defaults to
WARNING and is backpressure-safe, and the opt-in MCP log bridge is non-blocking."""
from __future__ import annotations

import asyncio
import logging
import os
import tempfile

import pytest


# ── §5.1 fd-1 safety: _suppress_stdout must not redirect fd 1 ────────────────
def test_suppress_stdout_does_not_touch_fd1(monkeypatch):
    from robotmcp.components.execution import rf_native_context_manager as rf

    calls = []
    real_dup2 = os.dup2
    monkeypatch.setattr(rf._os, "dup2", lambda a, b: calls.append((a, b)))
    with rf._suppress_stdout():
        pass
    assert calls == [], "_suppress_stdout must never os.dup2 the JSON-RPC fd"


def test_response_write_during_suppress_stays_on_fd1():
    from robotmcp.components.execution.rf_native_context_manager import _suppress_stdout

    out1 = tempfile.NamedTemporaryFile(delete=False)
    out2 = tempfile.NamedTemporaryFile(delete=False)
    report = os.dup(2); saved1 = os.dup(1); saved2 = os.dup(2)
    try:
        os.dup2(out1.fileno(), 1)
        os.dup2(out2.fileno(), 2)
        with _suppress_stdout():
            os.write(1, b"RESPONSE_DURING\n")
    finally:
        os.dup2(saved1, 1); os.dup2(saved2, 2)
        os.close(report)
    s1 = open(out1.name, "rb").read(); s2 = open(out2.name, "rb").read()
    assert b"RESPONSE_DURING" in s1 and b"RESPONSE_DURING" not in s2


# ── §5.2 console='none' produces NoOutput (nothing on fd 1) ──────────────────
def test_console_none_uses_nooutput():
    from robot.conf.settings import RobotSettings
    from robot.output.console import ConsoleOutput, NoOutput

    s = RobotSettings(console="none", output=None, log=None, report=None)
    co = ConsoleOutput(type=s.console_type)
    inner = getattr(co, "_output", co)
    assert isinstance(inner, NoOutput)


# ── §5.3 logging setup: default WARNING, env override, formatter ────────────
def test_configure_logging_default_warning(monkeypatch):
    monkeypatch.delenv("ROBOTMCP_LOG_LEVEL", raising=False)
    monkeypatch.delenv("ROBOTMCP_MCP_LOG_NOTIFICATIONS", raising=False)
    from robotmcp.utils.logging_setup import configure_logging, SafeStreamHandler

    from robotmcp.utils.logging_setup import DropQueueHandler

    level = configure_logging()
    assert level == logging.WARNING
    root = logging.getLogger()
    # root gets the non-blocking queue handler; the SafeStreamHandler lives in the
    # listener behind it. Verify the queue handler is installed at WARNING.
    assert any(isinstance(h, DropQueueHandler) for h in root.handlers)
    assert root.level == logging.WARNING


def test_configure_logging_env_override(monkeypatch):
    monkeypatch.setenv("ROBOTMCP_LOG_LEVEL", "INFO")
    from robotmcp.utils.logging_setup import configure_logging

    assert configure_logging() == logging.INFO
    monkeypatch.delenv("ROBOTMCP_LOG_LEVEL", raising=False)
    configure_logging()  # restore default for other tests


# ── §5.4 backpressure: a stuck stderr sink must not block the logging caller ─
def test_safe_stream_handler_swallows_broken_pipe():
    from robotmcp.utils.logging_setup import SafeStreamHandler

    class _Broken:
        def write(self, *_a): raise BrokenPipeError()
        def flush(self): pass

    h = SafeStreamHandler(_Broken())
    h.setFormatter(logging.Formatter("%(message)s"))
    rec = logging.LogRecord("x", logging.WARNING, __file__, 1, "boom", None, None)
    h.emit(rec)  # stdlib routes the write error to handleError → we drop; no raise


def test_logging_never_blocks_caller_when_sink_is_stuck():
    """The real backpressure guarantee: with the listener thread stuck writing to
    a blocked sink and the bounded queue full, the logging CALLER never blocks."""
    import queue as _q
    import logging.handlers
    import threading
    import time
    from robotmcp.utils.logging_setup import SafeStreamHandler, DropQueueHandler

    release = threading.Event()

    class _BlockingStream:
        def write(self, _s): release.wait(5)  # freeze the listener thread
        def flush(self): pass

    sh = SafeStreamHandler(_BlockingStream())
    sh.setFormatter(logging.Formatter("%(message)s"))
    log_q: "_q.Queue" = _q.Queue(maxsize=5)
    listener = logging.handlers.QueueListener(log_q, sh, respect_handler_level=True)
    listener.start()
    try:
        lg = logging.getLogger("bp.test.caller")
        lg.handlers = [DropQueueHandler(log_q)]
        lg.setLevel(logging.WARNING)
        lg.propagate = False
        t0 = time.time()
        for _ in range(1000):
            lg.warning("x")  # must not block even though the sink is frozen + queue fills
        elapsed = time.time() - t0
        assert elapsed < 2.0, f"logging blocked the caller for {elapsed:.2f}s"
    finally:
        release.set()
        time.sleep(0.05)
        try:
            while True:
                log_q.get_nowait()
        except _q.Empty:
            pass
        try:
            listener.stop()
        except Exception:
            pass


def test_drop_queue_handler_drops_when_full():
    import queue as _q
    from robotmcp.utils.logging_setup import DropQueueHandler

    h = DropQueueHandler(_q.Queue(maxsize=1))
    rec = logging.LogRecord("x", logging.WARNING, __file__, 1, "m", None, None)
    h.enqueue(rec)
    h.enqueue(rec)  # queue full → must drop, not raise/block


# ── §5.5 B2 bridge: off by default; on → schedules; no session → no-op ──────
def test_mcp_handler_off_by_default(monkeypatch):
    monkeypatch.delenv("ROBOTMCP_MCP_LOG_NOTIFICATIONS", raising=False)
    from robotmcp.utils.logging_setup import configure_logging, McpNotificationHandler

    configure_logging()
    root = logging.getLogger()
    assert not any(isinstance(h, McpNotificationHandler) for h in root.handlers)


def test_mcp_handler_noop_without_session():
    from robotmcp.utils.logging_setup import McpNotificationHandler, clear_mcp_log_sink

    clear_mcp_log_sink()
    h = McpNotificationHandler(); h.setFormatter(logging.Formatter("%(message)s"))
    h.emit(logging.LogRecord("x", logging.ERROR, __file__, 1, "no sink", None, None))  # no raise


def test_mcp_handler_schedules_send_with_mapped_level():
    from robotmcp.utils import logging_setup as L

    sent = []

    class _Session:
        async def send_log_message(self, level, data, logger=None):
            sent.append((level, data, logger))

    async def _run():
        loop = asyncio.get_running_loop()
        L.set_mcp_log_sink(_Session(), loop)
        h = L.McpNotificationHandler(); h.setFormatter(logging.Formatter("%(message)s"))
        h.emit(logging.LogRecord("robotmcp.x", logging.WARNING, __file__, 1, "hi", None, None))
        await asyncio.sleep(0.05)  # let call_soon_threadsafe + ensure_future run
        L.clear_mcp_log_sink()

    asyncio.new_event_loop().run_until_complete(_run())
    assert sent and sent[0][0] == "warning" and sent[0][1] == "hi" and sent[0][2] == "robotmcp.x"


def test_mcp_handler_excludes_internal_loggers():
    """M1: records from asyncio/mcp/fastmcp must NOT be forwarded, or a failed send
    re-enters via asyncio's error log and loops forever."""
    from robotmcp.utils import logging_setup as L

    sent = []

    class _Session:
        async def send_log_message(self, level, data, logger=None):
            sent.append(logger)

    async def _run():
        L.set_mcp_log_sink(_Session(), asyncio.get_running_loop())
        h = L.McpNotificationHandler(); h.setFormatter(logging.Formatter("%(message)s"))
        h.setLevel(logging.DEBUG)
        for name in ("asyncio", "mcp.server.session", "fastmcp.x",
                     "robotmcp.utils.logging_setup"):
            # handle() applies the handler's filters (emit() does not)
            h.handle(logging.LogRecord(name, logging.ERROR, __file__, 1, "x", None, None))
        await asyncio.sleep(0.05)
        L.clear_mcp_log_sink()

    asyncio.new_event_loop().run_until_complete(_run())
    assert sent == [], f"internal loggers must not be forwarded (got {sent})"


# ── readiness banner ────────────────────────────────────────────────────────
def test_ready_banner_has_config_and_no_stdout(capsys):
    from robotmcp.utils.logging_setup import emit_ready_banner, format_ready_banner

    line = format_ready_banner("stdio")
    assert "RobotMCP" in line and "ready" in line and "transport=stdio" in line
    assert "log_level=" in line and "libraries=" in line
    emit_ready_banner("stdio")
    cap = capsys.readouterr()
    assert cap.out == ""          # nothing on stdout (the JSON-RPC channel)
    assert "RobotMCP" in cap.err  # readiness goes to stderr
