# Tasks: desktop-aware-batch-execution

## 1. Profile inclusion
- [x] 1.1 Add `execute_batch` to the `desktop_exec` preset (7 tools; docstring updated ~1,550 → ~1,800 tokens)
- [x] 1.2 `desktop_exec` exposes the tool (profile test updated to 7 tools / tool_names)

## 2. Desktop error classification
- [x] 2.1 Priority-12 `ELEMENT_NOT_FOUND` pattern for PlatynUI vocab (`No UiNode found|ElementNotFoundError|UiNodeDescriptor`) — outranks the generic timeout pattern (8) so the "timeout of 30 seconds" PlatynUI message classifies as ELEMENT_NOT_FOUND
- [x] 2.2 Regression: existing browser-error corpus classifies identically (recovery suite green + new assertions)

## 3. Platform-aware recovery strategies
- [x] 3.1 `RecoveryStrategy.platforms: frozenset` (default `{"web"}`) + `applies_to_platform()`
- [x] 3.2 `RecoveryEngine.select_strategy(platform="web")` filters both passes by platform
- [x] 3.3 `desktop_wait_and_retry` (Tier 1, Sleep+retry) and `desktop_activate_window` (Tier 2, Activate Window + Sleep), both `platforms={"desktop"}`
- [x] 3.4 `RecoveryServiceAdapter` accepts `session_manager`, resolves `"desktop"` via `is_desktop_session()` and passes it to `select_strategy`; wired at `_get_batch_runner`

## 4. Capped descriptor-resolution timeout on batch retries — DEFERRED (detail added 2026-07-17)
- [x] 4.1 Timeout-cap context manager over `BareMetal.query_settings.timeout`
  - Resolve the live library instance via `namespace.get_library_instance("PlatynUI.BareMetal")` (same pattern as `session_manager.py:132`); `QuerySettings` is a plain mutable dataclass (`QuerySettings(30, 0.1)`)
  - `@contextmanager` sets `query_settings.timeout = cap` (default 5s, env-overridable `ROBOTMCP_PLATYNUI_BATCH_RETRY_TIMEOUT`), restores the original in `finally` (including on exception)
  - No-op (yield unchanged) when the library instance is absent or lacks `query_settings` — never crash a batch because of the cap
- [x] 4.2 Apply in `BatchRunner._handle_failure` around **retries only** — the initial attempt keeps the native 30s budget; only recovery re-attempts run capped
- [x] 4.3 Cover `resume_batch` retries (same wrap at the resume re-dispatch site)
- [x] 4.4 Test: a retried desktop step runs with `query_settings.timeout == cap` and the original is restored after (assert on both the happy and exception paths)
  > Evidence (EVAL_M27 / SPIKE_2 §3.2): a single bad-descriptor batch step burned **93,237 ms** (native 30s × retries); across the M2.7 desktop run three such batches burned **~283 s of the 592 s run**. §3 platform filtering already removed the browser-recovery multiplier; this cap is the remaining bound. Worst case one failed step drops ~93s → ~45s (30s initial + Sleep + 2× capped retries), keeping the 120s `BatchTimeout.DEFAULT_MS` survivable.

## 5. Desktop retry safety gate — DEFERRED (detail added 2026-07-17)
- [x] 5.1 In `_handle_failure`, desktop sessions retry only on `ELEMENT_NOT_FOUND`; any other desktop classification records failure immediately (behaves as `on_failure="stop"` regardless of policy)
  - Rationale: `ELEMENT_NOT_FOUND` is the one class where the input provably never fired (descriptor resolution precedes the pointer/keyboard action), so a retry cannot double-apply a click/keystroke. Every other class may have partially acted → blind retry is a spray-input hazard.
  - Resolve desktop via the already-wired `RecoveryServiceAdapter` platform (`is_desktop_session()`); gate is desktop-only, web/api retry policy unchanged
- [x] 5.2 Document the desktop gate in the `execute_batch` docstring (retries on desktop are unfired-input-only)
- [x] 5.3 Test: a desktop batch whose failing step classifies as (e.g.) a post-action/timeout error records failure immediately and does NOT re-dispatch; an `ELEMENT_NOT_FOUND` step still retries; web batches unchanged
  > Deferred with §4 (same `BatchRunner._handle_failure` integration point). Batch argument-list validation (spike #8) already shipped in `agent-ergonomics-fixes`. This is a **correctness/safety** gate, not just latency — it grows in importance with app complexity (dialogs, modals, multi-window) where a re-fired click lands somewhere unintended.

## 8. Deterministic acceptance gate (docker, no-LLM) — closes eval gap G6 (added 2026-07-17)
- [x] 8.1 KNOB CONFIRMED via `docker/gate_drivers.py g6` (2026-07-17): the RF `PlatynUI.BareMetal` library instance exposes `query_settings` (type `QuerySettings`) with `.timeout` = 30 — the exact attribute the §4 cap sets, reachable via `get_library_instance("PlatynUI.BareMetal")` in RF context; the raw runtime has no such knob (only `pointer_settings`). Confirms the cap targets the right attribute. NOTE: the full bounded-wall-time + no-repeat assertion still needs an in-RF batch harness (the raw driver has no RF context, so `retry_timeout_cap` correctly no-ops there).
- [ ] 8.2 Full in-RF `execute_batch` bad-descriptor gate (bounded wall-time + no repeated input) + wire as a smoke-ladder rung — follow-up

## 6. Batch-first init steering
- [x] 6.1 `batching` steer added to the desktop init guidance bundle (`desktop_guidance.py`)

## 7. Tests + validation
- [x] 7.1 `tests/unit/test_desktop_aware_batch_execution.py`: (a) desktop_exec has execute_batch; (b) PlatynUI string classifies ELEMENT_NOT_FOUND + browser corpus unchanged; (c) desktop selection never returns browser actions, web selection unchanged, desktop tiers; batch steer present. (d/e timeout-cap + retry-gate assertions deferred with §4/§5.)
- [x] 7.2 Existing batch + recovery suites green (recovery count tests bumped 10→11 / 9→11; full suite 6913 passed + 1 skipped)
- [ ] 7.3 (OPTIONAL) Docker re-run of the §3.2 shape — deferred
