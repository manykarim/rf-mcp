# Design: fast-mcp-handshake-lazy-init

## Context

The handshake is blocked purely by module import (spike: `initialize` answered
at import-time + ~0.02s; `tools/list` +7ms later from static decorator
metadata). The import cost has one actionable driver: the module-level
`execution_engine = ExecutionCoordinator()` (`server.py:1042`) transitively runs
the libdoc preload of 17 libraries, importing Browser/grpc, selenium, bs4,
requests along the way. Everything else heavy is either already lazy (ADR
bootstrap, memory hooks, attach probe) or a third-party floor (`fastmcp`).

## Goals / Non-Goals

- Goal: answer `initialize` without constructing the coordinator or loading any
  libdoc; keep first-tool-call latency effectively unchanged via warm-up.
- Goal: zero behavioural change to tool semantics, test patching, and the
  attach bridge.
- Non-goal: shrinking the `fastmcp` import; deferring the
  `robotmcp.components.execution` package import (server.py imports the classes
  at top level; restructuring to type-only imports is a separate change).

## Decisions

### D1 — Two independent lazy levers, both required

1. **Storage-level**: `RobotFrameworkDocStorage` defers `_initialize_libraries()`
   to a lock-guarded `_ensure_initialized()`. This protects *every* construction
   path (tests construct coordinators directly; `argument_processor.py:174`,
   `libdoc_argument_parser.py:17`, `rf_native_type_converter.py:76` all call
   `get_rf_doc_storage()` from their constructors).
2. **Server-level**: the `execution_engine` singleton itself becomes lazy, so
   the handshake also skips `SessionManager`/`KeywordExecutor`/plugin-manager
   construction (~0.1–0.3s beyond libdoc).

Lever 1 alone recovers most of the measured −1.7s; lever 2 removes the rest and
keeps the invariant "no side effects at import" honest.

### D2 — Lazy attribute via module `__getattr__`, not a proxy object and not an 85-site rewrite

Three options considered for `server.py:1042`:

- **(a) `get_execution_engine()` accessor + rewrite ~85 call sites.** Explicit,
  but a huge diff on the hottest file in the repo, and it breaks
  `from robotmcp.server import execution_engine`
  (`tests/frontend/test_frontend_api.py:10`) and every
  `mock.patch("robotmcp.server.execution_engine", …)`
  (`tests/unit/test_obs_33_strict_library_mode.py:178-295`).
- **(b) Lazy proxy object bound at import.** Minimal diff, but
  `initialize_enhanced_serialization` (`server.py:1048`) patches
  `execution_engine.keyword_executor`
  (`utils/enhanced_serialization_integration.py:184-195`) — any module-import-time
  attribute access on the proxy would defeat the deferral, and a proxy is not an
  `ExecutionCoordinator` for identity/isinstance purposes.
- **(c) Chosen: PEP 562 module `__getattr__` + in-function global publication.**
  Remove the eager assignment; add a module-level factory
  `_get_execution_engine()` with a `threading.Lock` (double-checked) that
  constructs `ExecutionCoordinator()`, runs
  `initialize_enhanced_serialization(engine)`, then **publishes it as the real
  module global** `execution_engine` (and `test_builder.execution_engine`).
  A module `__getattr__` delegates `execution_engine` to the factory for
  external importers. Internal call sites read the global at call time; the
  ~85 usages inside tool handlers are reached only at request time, so the
  factory is invoked from a tiny set of entry points (tool decorator wrapper or
  a one-line `_ensure_engine()` at the top of handlers that use it — see tasks).
  `mock.patch` on the module attribute keeps working because after first init
  (or warm-up) it is a plain attribute again; tests that patch before init are
  covered by publishing a real attribute in `__getattr__`'s first resolution.

Trade-off accepted: internal handlers need the `_ensure_engine()` touch-point
(mechanical, one line per handler that uses the global before any other
materializing call). The alternative (b) hides this but risks accidental
materialization and identity surprises. If implementation shows the touch-point
count is impractical, fallback is (b) with `initialize_enhanced_serialization`
moved inside the factory — the spec requirements are written to allow either.

`TestBuilder(execution_engine)` at `server.py:1044` only stores the reference
(`components/test_builder.py:116-117`); it is constructed inside the factory
alongside the engine so it always holds the real instance.

### D3 — Warm-up thread in `main()`, not in the event loop

`main()` starts `threading.Thread(target=_get_execution_engine, daemon=True)`
immediately before `mcp.run()`. Rationale: the anyio loop must come up
instantly to answer `initialize`; a background *thread* (not a loop task) keeps
the GIL-released import work off the loop entirely. A tool call arriving during
warm-up blocks on the factory lock — bounded by one full init (~1.5–2s), which
equals today's behaviour, never worse. `ROBOTMCP_LAZY_INIT=0` restores eager
construction at import (debug escape hatch; also the safety net if an unknown
consumer assumes import-time existence).

### D4 — Lazy storage via properties, not call-site auditing

`libraries`, `keyword_index_by_name`, `failed_imports` are read directly by
other modules (e.g. `core/dynamic_keyword_orchestrator.py`). Making them
`@property`-guarded by `_ensure_initialized()` (backing fields `_libraries`
etc.) means no consumer audit and no missed path. `_ensure_initialized()` uses
its own `threading.Lock`; it is a no-op when `HAS_LIBDOC` is false (existing
fallback branch at `rf_libdoc_integration.py:57-59` preserved).

### D5 — R3 gates by `find_spec` on the top-level module, never by distribution name

Measured: `importlib.metadata.distribution("platynui")` reports missing while
`PlatynUI.BareMetal` actually loads (dist is `robotframework-PlatynUI`);
`importlib.util.find_spec("PlatynUI")` is correct and costs 0.1ms. Gate rule:
`find_spec(library_name.split(".")[0])`; on miss, record
`failed_imports[name] = "not installed (skipped)"` at DEBUG level and skip the
`LibraryDocumentation` call. Libraries whose module exists but whose import
fails (e.g. `Telnet` on Python 3.13, telnetlib removed) keep the existing
try/except — measured cost 2ms, not worth special-casing.

### D6 — R1 stays documentation-only

`UV_COMPILE_BYTECODE=1` goes in the client-config `env` blocks and Docker
builds; no runtime code checks it. Rationale: precompilation is an install-time
concern; the server cannot compile its own tree faster than the interpreter
already does on the cold path.

## Risks / Trade-offs

- **First-tool-call latency without warm-up** (e.g. instant scripted client):
  bounded at one full init, same as today's pre-handshake cost; acceptance
  criterion caps it at +1.5s.
- **Thread-safety**: two locks (storage, engine factory). Both idempotent,
  double-checked, no lock ordering between them (factory acquires storage's
  lock only transitively after releasing nothing — engine factory holds its
  lock while storage initializes; no reverse path exists).
- **Log-order change**: "Initialized N libraries…" now logs after the handshake
  (from the warm-up thread). Grep found no tests asserting startup log order,
  but attach-banner output in `main()` is unaffected either way.
- **Hidden import-time consumers**: audit found none — module-level users of
  `execution_engine` are only `TestBuilder` (stores ref) and
  `initialize_enhanced_serialization` (moves into factory); ADR bootstrap and
  attach paths are handler-time. `ROBOTMCP_LAZY_INIT=0` is the escape hatch if
  a downstream embedder assumed import-time existence.
- **Machine variance in acceptance numbers**: absolute targets (<2.0s warm,
  <3.5s cold) are defined under the spike's handshake-timer methodology on an
  idle machine; the relative criterion (≥30% warm import reduction, same
  session) guards CI/loaded-machine noise.

## Open Questions

- Whether the `_ensure_engine()` touch-points (option c) end up cleaner than a
  proxy (option b) is an implementation-time call; spec is agnostic.
- Should the warm-up thread also prime `get_keyword_discovery()`'s inspection
  caches beyond libdoc? Default: yes, it constructs the full coordinator, which
  does this transitively.
