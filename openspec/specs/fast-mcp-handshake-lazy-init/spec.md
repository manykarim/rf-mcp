# fast-mcp-handshake-lazy-init Specification

## Purpose
TBD - created by archiving change fast-mcp-handshake-lazy-init. Update Purpose after archive.
## Requirements
### Requirement: The MCP handshake is not blocked by heavy initialization
Importing `robotmcp.server` SHALL NOT construct the `ExecutionCoordinator` and
SHALL NOT populate Robot Framework library documentation (libdoc preload), so
that the stdio `initialize`/`tools/list` handshake is served without importing
automation libraries (Browser/grpc, SeleniumLibrary, bs4, requests, …). The
`tools/list` response SHALL remain byte-for-byte equivalent in tool names and
schemas, since it is served from static decorator metadata.

#### Scenario: server import performs no libdoc preload
- **WHEN** `robotmcp.server` is imported (default configuration)
- **THEN** no `ExecutionCoordinator` is constructed and `RobotFrameworkDocStorage._initialize_libraries` is not invoked

#### Scenario: handshake latency targets
- **WHEN** the handshake is measured with the SPIKE D1 timer (spawn `uv run --no-sync python -m robotmcp.server`, send `initialize` then `tools/list`) on an idle machine
- **THEN** warm `initialize` responds in < 2.0s, cold (empty `PYTHONPYCACHEPREFIX`) in < 3.5s, and `tools/list` returns the same tool set as before the change

#### Scenario: relative import-time guard on any machine
- **WHEN** warm `import robotmcp.server` is timed pre- and post-change in the same session on the same machine
- **THEN** the post-change import is at least 30% faster

### Requirement: Lazy initialization is thread-safe and applied-once
The lazy coordinator factory and the lazy libdoc storage SHALL each be guarded
by a lock and be idempotent: concurrent first accesses SHALL yield exactly one
`ExecutionCoordinator` instance and one library-documentation population.
Enhanced serialization (`initialize_enhanced_serialization`) SHALL be applied
to the coordinator before it is published to any caller, preserving today's
invariant that every response passes through the enhanced serializer.

#### Scenario: concurrent first use constructs one coordinator
- **WHEN** multiple threads request the execution engine simultaneously on a fresh process
- **THEN** exactly one `ExecutionCoordinator` is constructed and all threads receive the same instance

#### Scenario: serialization is applied before first use
- **WHEN** the first tool call obtains the lazily constructed coordinator
- **THEN** its `keyword_executor` already carries the enhanced-serialization patch

### Requirement: First-tool-call latency is bounded and mitigated by warm-up
The server SHALL start a background warm-up (daemon thread, started in `main()`
before `mcp.run()`) that pre-builds the coordinator and libdoc storage so the
first tool call after the handshake does not normally pay initialization. A
tool call arriving during warm-up SHALL block on the same factory (no duplicate
init) and SHALL NOT wait longer than one full initialization.

#### Scenario: first tool call after warm-up pays no init
- **WHEN** a `find_keywords` call arrives after the warm-up thread has finished
- **THEN** its latency is within noise of the pre-change first-call latency

#### Scenario: tool call during warm-up is bounded
- **WHEN** a tool call arrives while warm-up is still running
- **THEN** it waits for the single in-flight initialization (at most one full init, ≤ +1.5s vs pre-change) and no second initialization is started

### Requirement: Lazy behavior is opt-out for compatibility
Setting `ROBOTMCP_LAZY_INIT=0` SHALL restore today's eager behavior: the
coordinator is constructed and the libdoc preload runs at module import.
Existing access patterns SHALL keep working under the default lazy mode:
`from robotmcp.server import execution_engine` and
`mock.patch("robotmcp.server.execution_engine", …)` SHALL continue to resolve
and substitute the instance that tool handlers use.

#### Scenario: eager opt-out
- **WHEN** the server module is imported with `ROBOTMCP_LAZY_INIT=0`
- **THEN** the coordinator and libdoc preload are initialized at import, as before the change

#### Scenario: test patching keeps working
- **WHEN** a test patches `robotmcp.server.execution_engine` with a stub engine
- **THEN** tool handlers observe the stub for the duration of the patch

### Requirement: The libdoc preload skips libraries that are not installed
`_initialize_libraries` SHALL skip a library when
`importlib.util.find_spec(<top-level module of the library name>)` returns
None, recording it in `failed_imports` (e.g. "not installed (skipped)") at
DEBUG level without attempting the import. Gating SHALL be by module name, not
distribution name. Libraries whose module exists but whose import fails (e.g.
`Telnet` on Python 3.13) SHALL keep the existing try/except handling.

#### Scenario: uninstalled library is skipped without an import attempt
- **WHEN** the preload list contains `DatabaseLibrary` and its module is not installed
- **THEN** `LibraryDocumentation` is not called for it, it appears in `failed_imports`, and no warning-level log noise is emitted

#### Scenario: dotted PlatynUI name still loads
- **WHEN** `PlatynUI.BareMetal` is in the preload list and the `PlatynUI` module is importable
- **THEN** it is not skipped by the gate and loads exactly as before

### Requirement: Client configuration examples document bytecode precompilation
The `uv`-based mcpServers examples in the README and instruction templates SHALL
include `UV_COMPILE_BYTECODE=1` (via the env block, or equivalently
`uv sync --compile-bytecode`) with a brief rationale, and the Docker images
SHALL ship with precompiled bytecode, so first launch after install/upgrade
avoids the cold ~8s bytecode-compilation path.

#### Scenario: README example carries the env var
- **WHEN** a user copies a `uv`-based mcpServers example from the README
- **THEN** it sets `UV_COMPILE_BYTECODE=1` and explains in one line why

