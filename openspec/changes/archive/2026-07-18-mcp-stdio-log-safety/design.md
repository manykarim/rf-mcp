## Context

Verified in explore (reproductions kept in `experiments/uv-tool-install/`):
- `_suppress_stdout()` `os.dup2(2,1)` misroutes an fd-1 write to stderr — proven
  in-process (`RESPONSE_DURING_SUPPRESS` landed on stderr).
- The MCP SDK stdio writer wraps `sys.stdout.buffer` (fileno 1), so a response
  written during the redirect window is misrouted → client hang.
- `console='none'` → RF `ConsoleOutput` wraps `NoOutput` (writes nothing to fd 1);
  `console='verbose'` → `VerboseOutput`. So the 4 residual `dup2` wraps guard
  against output that no longer exists.
- stderr level breakdown of a normal run: 141 INFO / 182 no-level / 17 WARNING /
  6 ERROR — the client shows all 346 as "warnings".

## Goals / Non-Goals

**Goals:** fd 1 is never redirected while the MCP transport owns it; stderr is
quiet and level-correct by default; a blocked stderr never freezes execution; an
opt-in structured MCP log channel exists.

**Non-Goals:** not changing what RF executes; not converting the whole codebase's
`logging` calls to context calls; not enabling structured notifications by default
(risk containment).

## Decisions

1. **Neutralize the fd-level redirect, don't patch around it.** `_suppress_stdout()`
   becomes a no-op context manager (`contextlib.nullcontext`) with a comment
   explaining that `console='none'` (`NoOutput`) + the Python-level
   `sys.__stdout__`→stderr redirect already keep fd 1 clean, and that `os.dup2` on
   fd 1 is unsafe while the MCP transport owns it. The 4 call sites keep their
   `with _suppress_stdout():` (now a no-op) to minimize churn, or are unwrapped —
   either way no `dup2` runs. The reference-counting globals/lock are removed.

2. **Keep `_protect_mcp_stdout()` (Python-level) — it is safe.** Replacing
   `sys.__stdout__` with a stderr-backed wrapper redirects library `print()` /
   `Log To Console` without touching fd 1, so it cannot misroute JSON-RPC. Retain
   it as defence-in-depth for C-level RF console writes that `console='none'`
   already suppresses.

3. **Dedicated stderr handler, default WARNING.** In `main()`, instead of
   `basicConfig(level=INFO)`, attach one `StreamHandler(sys.stderr)` with a
   `Formatter` (`%(levelname)s %(name)s: %(message)s`) at level
   `ROBOTMCP_LOG_LEVEL` (default `WARNING`). Root logger level set accordingly.
   This cuts ~90% of stderr volume and makes what remains genuinely warning/error.

4. **Backpressure-safe handler.** Subclass the stderr `StreamHandler` so `emit()`
   swallows `BlockingIOError`/`BrokenPipeError`/`ValueError` (drop the record)
   instead of propagating — a full/broken stderr pipe can never freeze a keyword
   thread. Also set the stderr stream non-blocking best-effort.

5. **Opt-in structured MCP logging (B2).** A `logging.Handler`
   (`McpNotificationHandler`) that, when `ROBOTMCP_MCP_LOG_NOTIFICATIONS` is set
   and an MCP session+loop have been captured for the current request, schedules
   `session.send_log_message(level, data, logger)` via
   `loop.call_soon_threadsafe` — never awaited on the calling (RF) thread, never
   blocking. Python level → MCP level map (debug/info/warning/error→…). A tiny
   session/loop capture is installed around request handling (FastMCP
   middleware/hook or a lightweight contextvar set in the tool entry). If no
   session/loop is available (background work, no client interest), it silently
   no-ops. Bounded: drop when the scheduling queue is unavailable; do not retain
   records.

## Risks / Trade-offs

- **Removing `dup2` lets a stray C-level fd-1 write through.** Mitigated: proven
  that `console='none'` yields `NoOutput`, and all Python-level writes go through
  the safe `sys.__stdout__` redirect. A regression test asserts `start_suite`/
  `start_test`/`end_test` under `console='none'` write 0 bytes to fd 1.
- **B2 cross-thread sends could flood or block.** Mitigated: off by default,
  `call_soon_threadsafe` only (never awaited on the RF thread), drop-on-failure,
  and it competes with responses only when explicitly enabled.
- **Changing default level to WARNING hides INFO diagnostics.** Mitigated by
  `ROBOTMCP_LOG_LEVEL=INFO/DEBUG` for troubleshooting; the just-shipped
  diagnostics-hygiene change already moved the useful signals to appropriate
  levels.
- **Existing tests may assert INFO logging is visible.** Adjust any that rely on
  the old default level.

## Measured result & residual (post-implementation)

Deep-probe stderr (a scenario that deliberately fails 3 steps), same tool install:
baseline **346** lines / **149** robotmcp-own log lines → **81** / **7** after this
change (robotmcp's own logging **−95%**, total **−77%**), with **0** JSON-RPC
fragments misrouted to stderr (the hang is gone) across every probe. The readiness
banner is emitted. The `_CollapseFrameworkTracebackFilter` collapses framework
tracebacks that reach robotmcp's root handler.

**Documented residual:** on FastMCP **3.4.4** (what a fresh `uv tool install`
pulls; local dev is pinned to 2.13.3), FastMCP re-configures its OWN logging inside
`mcp.run()` — after `configure_logging()` — and logs each *raising* tool call via
`logger.exception("Error calling tool …")` on a handler our filter no longer owns,
so a failed `execute_step` still prints one plain traceback to stderr. This is
FastMCP-internal and version-dependent; it only appears on step FAILURES (normal
passing runs are quiet). The real source is that `execute_step` **raises** to
signal step failure — collapsing these belongs in a separate change (either return
a structured step-failure result instead of raising, or pin/patch FastMCP), out of
scope for stdio log *safety*.
