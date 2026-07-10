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

## 4. Capped descriptor-resolution timeout on batch retries — DEFERRED
- [ ] 4.1 Timeout-cap context manager over `BareMetal.query_settings.timeout`
- [ ] 4.2 Apply in `BatchRunner._handle_failure` around retries only
- [ ] 4.3 Cover `resume_batch` retries
  > Deferred: touches PlatynUI `query_settings` internals + the BatchRunner retry loop; follow-up. The platform-filtered desktop strategies already stop browser recovery firing on desktop (the largest waste); the per-retry native-timeout cap is an additional bound.

## 5. Desktop retry safety gate — DEFERRED
- [ ] 5.1 In `_handle_failure`, desktop sessions retry only on `ELEMENT_NOT_FOUND`; any other desktop failure records failure immediately
- [ ] 5.2 Document the desktop gate in the `execute_batch` docstring
  > Deferred with §4 (same BatchRunner integration point). Note: batch argument-list validation (spike #8) already shipped in `agent-ergonomics-fixes`.

## 6. Batch-first init steering
- [x] 6.1 `batching` steer added to the desktop init guidance bundle (`desktop_guidance.py`)

## 7. Tests + validation
- [x] 7.1 `tests/unit/test_desktop_aware_batch_execution.py`: (a) desktop_exec has execute_batch; (b) PlatynUI string classifies ELEMENT_NOT_FOUND + browser corpus unchanged; (c) desktop selection never returns browser actions, web selection unchanged, desktop tiers; batch steer present. (d/e timeout-cap + retry-gate assertions deferred with §4/§5.)
- [x] 7.2 Existing batch + recovery suites green (recovery count tests bumped 10→11 / 9→11; full suite 6913 passed + 1 skipped)
- [ ] 7.3 (OPTIONAL) Docker re-run of the §3.2 shape — deferred
