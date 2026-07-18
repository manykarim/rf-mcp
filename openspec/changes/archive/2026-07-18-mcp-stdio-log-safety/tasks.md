## 1. A — eliminate the fd-1 redirect hang

- [x] 1.1 Make `_suppress_stdout()` in `rf_native_context_manager.py` a no-op context manager (no `os.dup2` on fd 1); remove the reference-counting globals/lock. Document why (`console='none'`/`NoOutput` + safe Python-level redirect).
- [x] 1.2 Confirm the 4 wrap sites (241/341/752/792) now run without redirecting fd 1 (either unwrapped or wrapping the no-op).
- [x] 1.3 Keep `_protect_mcp_stdout()` (Python-level `sys.__stdout__`→stderr) — verify it does not touch fd 1.

## 2. B1 — stderr level + format

- [x] 2.1 In `server.py:main()`, replace `basicConfig(level=INFO, stream=stderr)` with a dedicated stderr `StreamHandler` + `Formatter`, at `ROBOTMCP_LOG_LEVEL` (default `WARNING`); set the root logger level to match.
- [x] 2.2 Ensure a level of `DEBUG`/`INFO` via the env var restores verbose logging for troubleshooting.

## 3. C — backpressure-safe stderr handler

- [x] 3.1 Stderr handler `emit()` swallows `BlockingIOError`/`BrokenPipeError`/`ValueError` (drop the record); best-effort set the stderr stream non-blocking so a full pipe cannot freeze a keyword thread.

## 4. B2 — opt-in structured MCP logging

- [x] 4.1 `McpNotificationHandler` (logging.Handler) mapping Python levels → MCP levels, forwarding records via `session.send_log_message` scheduled with `loop.call_soon_threadsafe`; never awaited on the calling thread; drop on any failure/absence.
- [x] 4.2 Capture the MCP session + running loop for the active request (FastMCP hook/contextvar) and clear it after; the handler no-ops when absent.
- [x] 4.3 Gate the whole bridge behind `ROBOTMCP_MCP_LOG_NOTIFICATIONS` (off by default). When on, attach the handler alongside the stderr handler.

## 5. Tests

- [x] 5.1 fd-1 safety: during `_suppress_stdout()` a write to fd 1 now stays on fd 1 (no misroute) — the exact explore reproduction, inverted to assert safety.
- [x] 5.2 `start_suite`/`start_test`/`end_test` under `console='none'` write 0 bytes to fd 1 with no `_suppress_stdout` protection (proves A is safe).
- [x] 5.3 Logging setup: default level is WARNING; `ROBOTMCP_LOG_LEVEL=INFO` raises it; the handler has a formatter with a level field.
- [x] 5.4 Backpressure: the stderr handler drops a record on `BlockingIOError`/`BrokenPipeError` instead of raising.
- [x] 5.5 B2: with the flag off, no MCP notifications are attempted; with it on and a captured session, a log record schedules a `send_log_message` (mock the session/loop) with the correct mapped level; a missing session no-ops.

## 6. Experiment — measure the reduction

- [x] 6.1 Rebuild the wheel; run the deep stderr probe (and an agent scenario) in the clean-room BEFORE vs AFTER; record stderr line count, WARNING/ERROR/INFO counts, and confirm 0 JSON-RPC fragments on stderr (no misroute).
- [x] 6.2 Record the numbers in `experiments/uv-tool-install/` and the archive notes.

## 7. Adversarial-review fixes (FIX-FIRST)

- [x] 7.1 H1/C real backpressure: replace the dead-except handler with a QueueHandler + bounded drop-queue + background listener, so a blocked/full stderr never freezes an RF worker thread (only the listener blocks; overflow drops). SafeStreamHandler.handleError drops silently.
- [x] 7.2 H2: replace the false-green backpressure test with a real one — a stuck listener + full bounded queue must not block the logging caller; DropQueueHandler drops when full.
- [x] 7.3 M1: guard the B2 bridge against the self-sustaining feedback loop — a _B2Filter drops records from asyncio/mcp/fastmcp/self, and a task done-callback retrieves/swallows send failures (no "exception never retrieved" ERROR to re-forward).
- [x] 7.4 M2: bound B2 (session-liveness check + in-flight cap) so a stale/closed session and a slow client cannot spiral or grow unbounded.
- [x] 7.5 M3: fix the stale comments at the 3 no-op _suppress_stdout wrap sites.
- [x] 7.6 Defense-in-depth: add `--console none` to the normal suite-run path (_execute_rf_normal) so RF's VerboseWriter never targets fd 1 there.
- [x] 7.7 L3: document ROBOTMCP_LOG_LEVEL / ROBOTMCP_MCP_LOG_NOTIFICATIONS in the README.

## 8. Startup readiness feedback

- [x] 8.1 Emit one deliberate, always-visible readiness line to stderr before serving (version, transport, log level, mcp-notifications, attach, libraries); never to stdout. Suppress FastMCP's own banner (show_banner=False) and align its logger level.
