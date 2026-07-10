# Tasks: desktop-aware-batch-execution

## 1. Profile inclusion
- [ ] 1.1 Add `execute_batch` to the `desktop_exec` preset
      (`tool_profile/aggregates.py:294-300`); update the docstring tool count
      and token estimate (~1,550 → ~1,800).
- [ ] 1.2 Verify `activate_profile("desktop_exec")` exposes the tool and the
      8192-window budget still validates (MINIMAL fallback not triggered, or
      acceptable if it is — assert which).

## 2. Desktop error classification
- [ ] 2.1 Register a priority-12 `ELEMENT_NOT_FOUND` pattern for PlatynUI
      vocabulary (`No UiNode found|ElementNotFoundError|UiNodeDescriptor`) in
      `RecoveryEngine._register_default_patterns`
      (`recovery/aggregates.py:131-187`) — must outrank the generic timeout
      pattern (priority 8) because the PlatynUI message contains
      "timeout of 30 seconds".
- [ ] 2.2 Regression: the full existing classifier corpus (browser error
      strings) still classifies identically.

## 3. Platform-aware recovery strategies
- [ ] 3.1 `RecoveryStrategy` (`recovery/value_objects.py:91-119`): add
      `platforms: frozenset[str]` defaulting to `frozenset({"web"})`.
- [ ] 3.2 `RecoveryEngine.select_strategy` (`recovery/aggregates.py:61-100`):
      add `platform: str = "web"` parameter; filter both the preferred-tier
      pass and the fallback pass by `platform in strategy.platforms`.
- [ ] 3.3 Register desktop strategies in `_register_default_strategies`:
      `desktop_wait_and_retry` (Tier 1, ELEMENT_NOT_FOUND, `Sleep 2s`) and
      `desktop_activate_window` (Tier 2, ELEMENT_NOT_FOUND,
      `Activate Window` on the session's current root + `Sleep 1s`; skipped
      when no root is set — never unscoped).
- [ ] 3.4 `RecoveryServiceAdapter` (`adapters/recovery_adapter.py`): accept a
      session-manager reference (wire at `_get_batch_runner`,
      `server.py:5057-5089`); resolve `"desktop"` vs `"web"` via
      `session.is_desktop_session()` (`session_models.py:274`) and pass it to
      `select_strategy`.

## 4. Capped descriptor-resolution timeout on batch retries
- [ ] 4.1 Add a context manager (helper beside `keyword_executor.py` or in
      `desktop_execution_signals.py`) that caps
      `BareMetal.query_settings.timeout` via
      `namespace.get_library_instance("PlatynUI.BareMetal")` and restores the
      prior value in `finally`; soft no-op when the library is not loaded.
      Cap default 5 s, overridable via
      `ROBOTMCP_BATCH_DESKTOP_RETRY_TIMEOUT`.
- [ ] 4.2 Apply it in `BatchRunner._handle_failure`
      (`batch_execution/services.py:209-252`) around retry executions only —
      the initial attempt keeps the native 30 s budget.
- [ ] 4.3 Confirm the same capped path covers `resume_batch` retries (same
      `BatchRunner`; add an assertion-level test).

## 5. Desktop retry safety gate
- [ ] 5.1 In `_handle_failure`, for desktop sessions enter the retry loop
      only when the failure classifies as `ELEMENT_NOT_FOUND` (input provably
      never fired); any other desktop failure records failure immediately
      (behaves as `on_failure="stop"`), for both `retry` and `recover`
      policies.
- [ ] 5.2 Document the desktop gate in the `execute_batch` docstring
      (`server.py:5100-5137`) so agents know post-action desktop failures are
      not blind-retried.

## 6. Batch-first init steering
- [ ] 6.1 Add a one-line `execute_batch` steer to the desktop session init
      guidance (coordinate with `desktop-turn-economy-guidance`, which owns
      the desktop init cheat-sheet — do not duplicate its content; land this
      line in the same payload surface it establishes, or in the current
      desktop hint path if that change is not yet applied).

## 7. Tests + validation
- [ ] 7.1 `tests/unit/test_desktop_aware_batch_execution.py`:
      (a) `desktop_exec` contains `execute_batch`;
      (b) the verbatim PlatynUI error string classifies ELEMENT_NOT_FOUND;
      (c) `select_strategy(..., platform="desktop")` never returns a strategy
      whose actions include `Execute Javascript`/`Reload Page`/`Go Back`/
      `Handle Alert`, and `platform="web"` selection is unchanged;
      (d) timeout cap applied during retry and restored afterward, including
      when the retry raises;
      (e) desktop post-action failure is not retried; ELEMENT_NOT_FOUND is;
      (f) docstring/guidance text assertions.
- [ ] 7.2 Existing batch + recovery suites green
      (`tests/unit/domains/batch_execution/`, recovery tests) — no behavior
      change for web sessions.
- [ ] 7.3 Docker re-run of the §3.2 shape (optional, post-implementation): a
      desktop batch with one deliberately-missing descriptor completes the
      failed step in ≤ ~50 s and reports desktop recovery strategies in
      `recovery_log`.
