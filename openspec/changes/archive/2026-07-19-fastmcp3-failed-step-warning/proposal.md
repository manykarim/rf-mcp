## Why

A failed `execute_step` currently `raise Exception(detailed_error)`, and FastMCP
funnels any *bare* exception through `logger.exception(...)` — an ERROR-level entry
with a full Python traceback on stderr. It is fine for a failed step to appear in
the log, but a scary ERROR + stack for an *expected* failure is noise and reads as
a server fault.

FastMCP 3.x provides the sanctioned mechanism for expected, user-facing tool
errors: `FastMCPError`/`ToolError` carry a `log_level`, and the server logs them
via `logger.log(e.log_level, "Error calling tool …", exc_info=False)` — a clean,
level-tagged line with no traceback — while reserving `logger.exception`
(ERROR+traceback) for genuinely unexpected exceptions. Verified in 3.4.4:
`ToolError(msg, log_level=logging.WARNING).log_level == 30`.

This mechanism exists **only on FastMCP 3.x**. The repo is locked to **2.13.3**
(the `>=2.8.0` pin resolves there), where `ToolError(msg, log_level=…)` raises
`TypeError` and both ToolError and Exception log via `logger.exception`. So the fix
is a coupled fastmcp 2.x→3.x upgrade that MUST ship atomically with the code and
tests, or a failed step regresses to `isError=True` carrying
`"…ToolError() takes no keyword arguments"` — destroying the RF error,
suggested_keyword, hints, and step_id that agents (especially weak models) rely on.

The upgrade is viable and bounded: the product already runs on 3.4.4 (every docker
experiment this session used it), Python requirement is unchanged (fastmcp 3.4.4 /
starlette 1.3.1 / mcp all `>=3.10`, matching rf-mcp), and the ~26 unit failures on
3.x are almost entirely a mechanical test-harness change (the `@mcp.tool` `.fn`
unwrap, ~9 files) plus the version-specific `test_fastmcp_compat` assertions.

## What Changes

- **Upgrade to FastMCP 3.x**: pin `fastmcp>=3.0` in pyproject, regenerate `uv.lock`,
  `uv sync`; verify `fastmcp.__version__` is 3.x and `ToolError(log_level=…)`
  constructs. Python requirement stays `>=3.10` and the CI matrix
  (3.10/3.11/3.12) is unchanged — verified against the new deps.
- **Test-harness migration** for 3.x: a shared `.fn`-unwrap helper applied across
  the affected test files, and updates to the `test_fastmcp_compat` assertions.
- **Expected step failures log as WARNING**: convert the two user-facing raise
  sites — `execute_step` step failure (server.py:4572) and the attach-bridge
  connectivity error (server.py:4448) — to `ToolError(detailed_error,
  log_level=logging.WARNING)` via a small `robotmcp.compat` helper that falls back
  to a plain `ToolError` (still `isError=True`, payload preserved) if `log_level`
  is unsupported — insurance against an accidental 2.x runtime.
- **Regression test**: a failing `execute_step` returns `isError=True` AND its text
  content still contains the RF error, the suggested-keyword/hint, and the step_id.
- Keep `_CollapseFrameworkTracebackFilter` (from mcp-stdio-log-safety) for this
  release as a defensive net; genuinely-unexpected exceptions still get
  ERROR+traceback.

Non-goals: not converting return-a-dict tools (execute_flow/run_test_suite already
return `{success:False}`), not touching non-tool raises (CLI SystemExit,
locally-caught arg validation), not removing the traceback filter yet.

## Capabilities

### New Capabilities
- `failed-step-error-signaling`: how rf-mcp signals an expected tool/step failure —
  as a FastMCP `ToolError` logged at WARNING (no traceback), with the failure
  detail (error, hint, step_id) preserved in the `isError` result for the agent.

### Modified Capabilities
<!-- none -->
