# Proposal: fast-mcp-handshake-lazy-init

## Why

SPIKE D1 (`experiments/SPIKE_D1_mcp_handshake.md`, 2026-07-10) measured the MCP
stdio handshake at **3.0s warm / 8.4s cold** — 100% process-startup/module-import
time. `tools/list` is answered from static decorator metadata **+7ms** after
`initialize`; none of the heavy init is needed for the handshake, yet all of it
runs before it. Clients (Claude Code) show the server as "pending" for the whole
window. Two costs dominate the warm path: the `fastmcp` import (~1.45–1.6s,
third-party floor, not actionable) and the **eager libdoc preload of 17 RF
libraries** triggered by the module-level `execution_engine =
ExecutionCoordinator()` at `src/robotmcp/server.py:1042` →
`KeywordExecutor.__init__` (`keyword_executor.py:258-266`) →
`get_rf_doc_storage()` (`utils/rf_libdoc_integration.py:579`) →
`RobotFrameworkDocStorage.__init__` calling `_initialize_libraries()`
unconditionally (`rf_libdoc_integration.py:47-61`), which imports Browser/grpc,
SeleniumLibrary/selenium, bs4, requests, … just to extract keyword signatures.

Proposal-time re-measurement on this branch (same machine, noisier/loaded than
the spike run; scratch script `experiments/spike_lazy_init_measure.py`):

| Measurement | Result |
|---|---|
| Warm `import robotmcp.server` (3 runs) | 4.54 / 4.81 / 5.92 s |
| Warm import with libdoc preload no-op'd (R2 prototype, 3 runs) | 2.66 / 3.09 / 3.30 s — **−1.7s median (−36%)** |
| Cold (`PYTHONPYCACHEPREFIX`=empty) baseline | 8.20 s (spike: 8.40 s) |
| Cold with preload no-op'd (2 runs) | 5.86 / 7.24 s — **−1.0 to −2.3 s** (Browser/grpc/selenium/bs4/requests never compiled pre-handshake) |
| `import fastmcp` alone | 1.61 s (floor) |
| `get_rf_doc_storage()` isolated | 1.66 s (13 libs loaded, 4 failed) |
| Failing preload attempts (`DatabaseLibrary`, `SSHLibrary`, `FTPLibrary`, `Telnet`) | **1–2 ms each** warm (fail fast on missing package) |
| `importlib.util.find_spec` gate | 0.1 ms/module; correctly finds `PlatynUI` where distribution-name gating wrongly reports it missing |

Two honest corrections to the spike this produced: (1) R3's warm-latency upside
is ~6ms total, not 0.1–0.2s — its value is startup log hygiene and a smaller
cold compile/attempt set, and it must gate by **module** (`find_spec`), not
distribution name, or `PlatynUI.BareMetal` (dist `robotframework-PlatynUI`)
would be wrongly skipped; (2) the cold 8.4s "pending" symptom is fully removed
only by R1 (bytecode precompiled at install), with R2 shrinking what remains.

## What Changes

- **R2 — lazy libdoc storage.** `RobotFrameworkDocStorage.__init__` stops
  calling `_initialize_libraries()`; population happens on first access via a
  lock-guarded, idempotent `_ensure_initialized()`. `libraries`,
  `keyword_index_by_name`, and `failed_imports` become properties that trigger
  it, so existing direct-attribute consumers (e.g.
  `dynamic_keyword_orchestrator.py`) keep working unchanged.
- **R2 — lazy `execution_engine` singleton.** The module-level
  `ExecutionCoordinator()` at `server.py:1042` is replaced by a thread-safe lazy
  holder that constructs the coordinator on first real use and applies
  `initialize_enhanced_serialization()` to it before publishing (today applied
  eagerly at `server.py:1048`). The module attribute `execution_engine` keeps
  existing import/patch semantics (see design.md) so the ~85 in-module call
  sites and tests that `mock.patch("robotmcp.server.execution_engine", …)` are
  untouched.
- **R2 — post-startup warm-up.** `main()` starts a daemon warm-up thread right
  before `mcp.run()` that builds the coordinator + libdoc storage in the
  background, so the first `execute_step`/`find_keywords` does not pay the
  ~1.1–1.7s. `ROBOTMCP_LAZY_INIT=0` opts back into today's eager init.
- **R3 — gate the preload list by importability.** `_initialize_libraries`
  skips libraries whose top-level module has no spec
  (`importlib.util.find_spec("DatabaseLibrary")` etc.), recording them in
  `failed_imports` as "not installed" without an import attempt or a per-run
  warning. `PlatynUI.BareMetal` gates on `find_spec("PlatynUI")`. `Telnet`
  (RF-stdlib module, import fails on Python 3.13) stays with the existing
  try/except — its attempt costs 2ms.
- **R1 — document `UV_COMPILE_BYTECODE=1`.** README client-config examples
  (`README.md:37-70,226-240,319-330`), `docs/INSTRUCTION_TEMPLATES_GUIDE.md`
  mcpServers samples, and the Docker images gain
  `"env": {"UV_COMPILE_BYTECODE": "1"}` / `uv sync --compile-bytecode` so the
  first launch after install/upgrade doesn't pay the 8.4s cold-pyc path.

Out of scope: the `fastmcp` import floor (third-party); deferring the eager
`robotmcp.components.execution` package import itself (~0.37s cumulative,
would require type-only import restructuring — possible follow-up).

## Capabilities

### New Capabilities

- `fast-mcp-handshake-lazy-init`: the MCP handshake is no longer blocked by RF
  library documentation preload or coordinator construction; heavy init is
  deferred to first use and pre-built by a background warm-up thread; the
  libdoc preload skips uninstalled libraries without import attempts; client
  configs document bytecode precompilation.

## Impact

- `src/robotmcp/utils/rf_libdoc_integration.py:47-61` — constructor no longer
  eager-loads; `_ensure_initialized()` + property accessors; `find_spec` gate
  in `_initialize_libraries`/`_load_library_documentation` (`:63-117`).
- `src/robotmcp/server.py:1038-1048` — lazy `execution_engine` holder;
  `initialize_enhanced_serialization` moves into the factory; `TestBuilder`
  (`:1044`) receives the lazy handle (it only stores the reference,
  `components/test_builder.py:116-117`).
- `src/robotmcp/server.py` `main()` (`:8072+`) — warm-up thread +
  `ROBOTMCP_LAZY_INIT` opt-out.
- Unaffected by design: `tools/list` (static decorator metadata), ADR-006/007/008
  bootstrap (already lazy — called from handlers at `server.py:3207,3791,7353`),
  attach-mode paths (env-probe only at startup, `server.py:388-416,544-559`;
  local fallback touches the engine only inside handlers at request time).
- Tests: `tests/unit/test_fast_handshake_lazy_init.py` (new) — no libdoc load at
  import, thread-safe single construction, serialization applied before first
  response, find_spec gating incl. PlatynUI, opt-out env var;
  `tests/frontend/test_frontend_api.py:10` and
  `tests/unit/test_obs_33_strict_library_mode.py:178-295` guard the
  import/patch compatibility; full unit suite green.
- Acceptance: warm `initialize` < 2.0s and cold < 3.5s (spike handshake-timer
  methodology, idle machine — warm floor is the 1.45–1.6s fastmcp import);
  machine-variance guard: warm `import robotmcp.server` ≥ 30% faster than a
  same-session baseline; first tool call after handshake regresses ≤ 1.5s
  without warm-up and ~0 with it; `tools/list` still returns the identical
  tool set.
