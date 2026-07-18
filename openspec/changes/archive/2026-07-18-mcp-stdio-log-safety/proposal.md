## Why

Two verified defects in how robotmcp handles output over MCP stdio make scenario
execution noisy and can hang the server (explore reproductions below).

**Hang — `_suppress_stdout()` misroutes JSON-RPC responses.** To keep RF console
output off fd 1 (the JSON-RPC channel), `_suppress_stdout()` does
`os.dup2(2, 1)` — it points fd 1 at stderr during RF `start_suite`/`start_test`/
`end_test`/context-creation (4 live sites). But the MCP SDK writes responses via
`anyio.wrap_file(TextIOWrapper(sys.stdout.buffer))`, whose `fileno()` is 1. If a
response write overlaps the redirect window (concurrent requests / the async
writer flushing), the response lands on **stderr** and the client hangs waiting.
Reproduced deterministically: a write to fd 1 during `_suppress_stdout()` went to
stderr. The maintainer already removed this from the main `runner.run()` path
(the comment at rf_native_context_manager.py:592 describes this exact race) but
left 4 residual `dup2` sites. It is safe to remove them: with `console='none'` RF
uses `NoOutput`, which writes nothing to fd 1 (verified).

**Noise — everything shows as a warning.** robotmcp routes INFO-level logging and
raw RF/traceback text to stderr (`logging.basicConfig(level=INFO,
stream=stderr)`), and MCP clients surface the entire server stderr stream as
undifferentiated warnings. Of 346 stderr lines in a normal run, only 17 WARNING +
6 ERROR were real — 141 INFO + 182 unprefixed raw lines (91%) were not. robotmcp
also uses no MCP structured logging, so the client cannot render levels. Unbounded
synchronous stderr on a pipe the client may not drain is also a secondary hang
vector (64 KB pipe-full → blocked `write()` → frozen server).

## What Changes

- **A (hang fix):** neutralize the `os.dup2(2,1)` fd-1 redirect in
  `_suppress_stdout()` — it no longer touches fd 1. Protection against RF console
  output relies on `console='none'` (`NoOutput`) plus the existing safe,
  Python-level `sys.__stdout__`→stderr redirect (which never touches fd 1).
- **B1 (log level + format):** install a dedicated stderr logging handler with a
  proper formatter, default level **WARNING**, overridable via
  `ROBOTMCP_LOG_LEVEL`. INFO operational chatter no longer floods the client;
  what remains on stderr is genuinely WARNING/ERROR.
- **B2 (structured MCP logging, opt-in):** an optional, thread-safe, non-blocking
  bridge that forwards Python log records to the client as MCP
  `notifications/message` with a mapped level, so multimodal clients render real
  levels. Gated by `ROBOTMCP_MCP_LOG_NOTIFICATIONS` (off by default so the
  default path takes zero new risk); never blocks the event loop.
- **C (backpressure safety):** the stderr handler is resilient — a blocked/broken
  stderr pipe drops the record instead of freezing keyword execution.

## Capabilities

### New Capabilities
- `mcp-stdio-log-safety`: the contract for robotmcp's stdio output safety — fd 1
  is never redirected out from under the JSON-RPC transport, stderr defaults to
  WARNING with a configurable level, stderr writes never block execution, and an
  opt-in structured MCP logging channel is available.

### Modified Capabilities
<!-- none -->
