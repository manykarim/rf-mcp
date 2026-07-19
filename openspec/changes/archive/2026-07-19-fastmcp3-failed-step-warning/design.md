## Context

Established by direct inspection + a 4-agent adversarial verification:
- Repo LOCKED at fastmcp **2.13.3** (`uv.lock`); `>=2.8.0` resolves there. On 2.13.3
  `ToolError(msg, log_level=…)` → `TypeError`, and ToolError logs via
  `logger.exception` (ERROR+traceback), same as a bare Exception.
- FastMCP **3.x** (3.4.4 latest): `FastMCPError(*args, log_level=logging.ERROR)`;
  `ToolError` subclasses it; `server.py:1327 except FastMCPError as e:
  logger.log(e.log_level, "Error calling tool …", exc_info=False)`; generic
  `except Exception → logger.exception`. Empirically `ToolError(msg,
  log_level=WARNING).log_level == 30`.
- The MCP FAILED signal (`isError=True`) is produced by the result layer regardless
  of exception type — verified. `mask_error_details` is False and never enabled, so
  the detailed_error already reaches the client today (prefixed). So the wins are:
  no traceback (host-stderr only) + a cleaner/mask-safe payload.
- 3.x unit failures: ~26, dominated by the `@mcp.tool` `.fn` unwrap
  (`AttributeError: 'function' object has no attribute 'fn'`, ~9 test files) +
  `test_fastmcp_compat` version assertions. Product runs on 3.4.4 (docker-proven).
- Python: fastmcp 3.4.4 / starlette 1.3.1 / mcp all `requires-python >=3.10`, equal
  to rf-mcp — no `requires-python` or CI-matrix change.

## Goals / Non-Goals

**Goals:** expected step failures log as a clean WARNING (FastMCP-3.x-standard, no
traceback) with the failure payload preserved for the agent; the 2.x→3.x upgrade
lands atomically and regression-tested.

**Non-Goals:** not changing agent-visible FAILED semantics; not converting tools
that already return `{success:False}`; not removing the traceback filter this
release; not changing the supported Python range.

## Decisions

1. **Raise ToolError(log_level=WARNING) via a single compat seam.**
   `robotmcp/compat/fastmcp_errors.py` (or add to the existing fastmcp_compat):
   ```python
   def tool_error(message: str, level: int = logging.WARNING) -> "ToolError":
       from fastmcp.exceptions import ToolError
       try:
           return ToolError(message, log_level=level)   # FastMCP 3.x → clean level-tagged log
       except TypeError:
           return ToolError(message)                      # FastMCP 2.x fallback: isError, payload kept
   ```
   Call sites `raise tool_error(detailed_error)`. This is the single point that
   guards the version gap; on any accidental 2.x runtime it degrades to a plain
   ToolError instead of a TypeError that nukes the payload.

2. **Convert exactly two raise sites** (both user-facing, inside execute_step):
   4572 (failed RF step) and 4448 (attach connectivity, strict/force). Convert them
   *together* — 4448 must not be left as a bare Exception, or it resurfaces as
   ERROR+traceback the moment the filter is removed. Leave return-a-dict tools and
   non-tool raises (CLI SystemExit, locally-caught ValueError) untouched.

3. **`.fn` unwrap migration via a shared test helper.** Add
   `tests/unit/helpers/mcp_tools.py::tool_fn(tool)` returning the underlying
   callable across 2.x/3.x (`getattr(tool, "fn", None) or tool`), and apply it in
   the ~9 failing test files. Update `test_fastmcp_compat` assertions for 3.x.

4. **Pin as its own validated step.** Bump `fastmcp>=3.0`, `uv lock`, `uv sync`;
   assert `fastmcp.__version__` starts with "3." and `tool_error(...)` yields a
   ToolError with `log_level == WARNING`. Keep `requires-python = ">=3.10"` and the
   CI matrix unchanged (verified). Re-run the full suite + handshake as the upgrade
   gate.

5. **Keep the traceback filter this release.** `_CollapseFrameworkTracebackFilter`
   is a global root-handler exc_info stripper (affects selenium/urllib3/asyncio too)
   and the only suppressant on a 2.x fallback; removing it belongs in a follow-up
   once 3.x is proven. Rich-panel suppression stays intact.

## Risks / Trade-offs

- **Payload loss if `log_level` lands on a 2.x runtime.** Mitigated by the fallback
  helper (degrades to plain ToolError) + the payload-preservation regression test
  (fails loudly on 2.x or on a masking-default flip).
- **The upgrade itself** (starlette 0.50→1.3.1 major, mcp 1.22→1.28.1, keyring
  chain). Mitigated: product already runs on 3.4.4; the `fastmcp_compat` layer
  anticipates 3.x; the upgrade is gated on the full suite + handshake passing.
- **`.fn` migration churn** across ~9 files. Mechanical; the shared helper keeps it
  one-line-per-callsite and 2.x/3.x-agnostic.
- **Cosmetic message change on 3.x** (drops the `"Error calling tool 'execute_step':`
  prefix). Harmless-to-beneficial; the regression test asserts on content, not the
  prefix.

## Measured result (post-implementation, FastMCP 3.4.4)

Deep probe (3 deliberately-failed steps), clean-room tool install:
- Full unit suite on 3.x: **7031 passed, 4 skipped** (all 26 pre-upgrade failures resolved).
- Failed steps now log as one-line `Error calling tool 'execute_step'` at WARNING, **0 tracebacks** (was 3 full tracebacks). Total stderr **346 → 12 lines** across the log-safety + this change.
- Payload preserved: the tool result carries the RF error + hint + step_id (verified by test_failed_step_warning + tests/test_mcp_error_scenarios). 0 misrouted JSON-RPC.
