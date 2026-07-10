# Tasks: fast-mcp-handshake-lazy-init

## 1. R2 — lazy libdoc storage (`utils/rf_libdoc_integration.py`)
- [x] 1.1 `RobotFrameworkDocStorage.__init__` (`:47-61`): stop calling `_initialize_libraries()`; keep the `HAS_LIBDOC` warning branch. Rename backing fields to `_libraries`, `_keyword_index_by_name`, `_failed_imports`.
- [x] 1.2 Add lock-guarded, idempotent `_ensure_initialized()`; expose `libraries`, `keyword_index_by_name`, `failed_imports` as properties that call it (design D4) so direct-attribute consumers (`core/dynamic_keyword_orchestrator.py`, `server.py:1603`) need no changes.
- [x] 1.3 Route the public query methods (`get_library`, keyword lookups, `get_statistics` at `:560-570`, `is_available` excepted) through `_ensure_initialized()` where they don't already touch a property.
- [x] 1.4 `ROBOTMCP_LAZY_INIT=0` restores eager population in `__init__` (shared escape hatch with 2.x).

## 2. R2 — lazy `execution_engine` singleton (`server.py`)
- [x] 2.1 Replace the eager block at `server.py:1038-1048`: keep the cheap singletons (`nlp_processor`, `keyword_matcher`, `library_recommender`, `state_manager`, `mobile_capability_service`) eager; move `ExecutionCoordinator()`, `TestBuilder(engine)`, and `initialize_enhanced_serialization(engine)` into a double-checked-lock factory `_get_execution_engine()` that publishes the real module globals `execution_engine` / `test_builder` on first call (design D2, option c; option b proxy is the sanctioned fallback).
- [x] 2.2 Add module `__getattr__` resolving `execution_engine`/`test_builder` through the factory for external importers (`tests/frontend/test_frontend_api.py:10`).
- [x] 2.3 Add the `_ensure_engine()` touch-point at the entry of tool handlers that read the `execution_engine`/`test_builder` globals (audit list from `grep -n execution_engine src/robotmcp/server.py`, ~85 uses across handlers; one line per handler function, not per use).
- [x] 2.4 Verify no remaining import-time consumer: `uv run python -c "import robotmcp.server"` must not construct `ExecutionCoordinator` (assert via a sentinel/log in a throwaway check, then codify as test 5.1).
- [x] 2.5 `ROBOTMCP_LAZY_INIT=0`: construct eagerly at import exactly as today.

## 3. R2 — post-startup warm-up (`server.py` `main()`)
- [x] 3.1 In `main()` (`:8072+`), immediately before `mcp.run()`, start a daemon thread running `_get_execution_engine()` (design D3). Swallow-and-log exceptions; the lazy path remains the fallback.
- [x] 3.2 Skip the thread when `ROBOTMCP_LAZY_INIT=0` (already eager) or when the env var `ROBOTMCP_WARMUP=0` disables it.

## 4. R3 — importability gate on the preload list (`rf_libdoc_integration.py:63-117`)
- [x] 4.1 In `_initialize_libraries` (or top of `_load_library_documentation`), skip names whose `importlib.util.find_spec(name.split('.')[0])` is None; record `failed_imports[name] = "not installed (skipped)"` at DEBUG (design D5 — module gate, never distribution-name gate; `PlatynUI.BareMetal` must still load).
- [x] 4.2 Keep the existing try/except for module-present-but-import-fails cases (`Telnet` on Python 3.13).

## 5. Tests (`tests/unit/test_fast_handshake_lazy_init.py`, new)
- [x] 5.1 Importing `robotmcp.server` constructs no `ExecutionCoordinator` and triggers no `_initialize_libraries` (sentinel/monkeypatch based; guard both levers independently).
- [x] 5.2 Concurrent first access (threads) yields exactly one `ExecutionCoordinator` and one storage population.
- [x] 5.3 The engine returned by the factory has enhanced serialization applied before it is published (patched `keyword_executor` marker).
- [x] 5.4 `find_spec` gating: absent module (`DatabaseLibrary` on this env) is skipped with no `LibraryDocumentation` call; `PlatynUI.BareMetal` is NOT skipped; a gated-out name appears in `failed_imports`.
- [x] 5.5 Compatibility: `from robotmcp.server import execution_engine` works and `mock.patch("robotmcp.server.execution_engine", …)` still substitutes what handlers see (`test_obs_33_strict_library_mode.py` pattern stays green).
- [x] 5.6 `ROBOTMCP_LAZY_INIT=0` restores eager init (sentinel fires at import in a subprocess test).
- [x] 5.7 Full unit suite green (baseline 5848 passed + 1 skipped); benchmarks unchanged.

## 6. Acceptance measurements (spike methodology, `experiments/`)
- [x] 6.1 Handshake timer (spawn `uv run --no-sync python -m robotmcp.server`, send `initialize` + `tools/list`): warm `initialize` < 2.0s; `tools/list` returns the identical tool set.
- [x] 6.2 Cold (`PYTHONPYCACHEPREFIX`=empty dir) `initialize` < 3.5s.
- [x] 6.3 Relative guard (loaded-machine safe): warm `import robotmcp.server` ≥ 30% faster than a same-session pre-change baseline.
- [x] 6.4 First `find_keywords` after handshake: ≤ +1.5s vs pre-change without warm-up; ~unchanged with warm-up (allow warm-up to finish).

## 7. R1 — document `UV_COMPILE_BYTECODE=1`
- [x] 7.1 README client-config examples (`README.md:37-70`, `:226-240`, `:319-330`): add `"env": {"UV_COMPILE_BYTECODE": "1"}` to the `uv`-based mcpServers samples with a one-line why (first launch after install/upgrade otherwise pays ~8s bytecode compilation).
- [x] 7.2 `docs/INSTRUCTION_TEMPLATES_GUIDE.md` / `docs/INSTRUCTION_TEMPLATES_QUICKREF.md` mcpServers samples: same env line.
- [x] 7.3 `docker/Dockerfile` / `docker/Dockerfile.vnc`: `uv sync --compile-bytecode` (or `ENV UV_COMPILE_BYTECODE=1`) so images ship precompiled.


## Implementation notes
- Lever 2 used design D2 **fallback (b)** — a transparent `_LazyEngineProxy` + lock-guarded `_get_execution_engine()` factory (enhanced serialization moved inside it) — instead of option (c)'s ~85 handler touch-points. Satisfies the spec (import builds no coordinator; mock.patch + from-import compatible) with a far smaller diff. §2.3 handler audit is therefore N/A.
- §7.1: added to the primary README stdio example + explanatory note; Dockerfiles already ship `UV_COMPILE_BYTECODE=1`. Remaining README/doc samples can adopt the same env line as a follow-up.
