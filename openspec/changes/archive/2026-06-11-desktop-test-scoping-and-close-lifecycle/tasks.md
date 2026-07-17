# Tasks: desktop-test-scoping-and-close-lifecycle

## 1. Test scoping integrity (D1-D4)

- [x] 1.1 start_test handler (`server.py` ~3505-3567): when `get_session_context_info(session_id)["context_exists"]` is false, create the RF context via `create_context_for_session(session_id, libraries=session.search_order or list(session.loaded_libraries))` before `start_test_in_context`
- [x] 1.2 Make start_test atomic: context-creation or `start_test_in_context` failure → `{success: False, error: …}` returned BEFORE `TestRegistry.start_test` runs (registry activation is last); update the handler's docstring for the loud-failure behavior
- [x] 1.3 end_test handler: registry-first — when the registry has an active test, end it and return success; attach the `end_test_in_context` result and downgrade a context-layer "No active test to end" to a `warning` field; keep failing when the registry also has no active test
- [x] 1.4 `build_test_suite` (test_builder.py): add `suite_level_step_count` to the response; emit the orphaned-steps top-level warning when multi-test mode and suite-level steps > in-test steps (the more specific scoping warning takes precedence over the I-1 empty-suite warning when both fire)
- [x] 1.5 Audit existing tests that pinned the old soft-success start_test shape (grep "No context for session" / start_test tests) and update them deliberately
- [x] 1.6 Regression tests (`tests/unit/test_test_scoping_integrity.py`): start_test-before-any-step succeeds; atomic failure leaves registry untouched (force context failure via monkeypatch); the run-3 interleaving (start_test → steps → build → steps → build → steps → end_test) lands every step in the named test; registry-first end_test with context-layer miss; orphaned-steps warning fires on the 43/3 shape and stays silent on healthy sessions

## 2. Close liveness (D5)

- [x] 2.1 `desktop_execution_signals.py`: add `is_close_keyword()` (`close window` basename) and `close_liveness_hint(process_alive)` (pure; hint only when True, naming residual frames + `Terminate Process`)
- [x] 2.2 Wire into the executor's post-success desktop block next to the launch-liveness check: close keyword + `session.desktop_aut_pid` set → `os.kill(pid, 0)` liveness (`ProcessLookupError` → dead; `PermissionError` → alive) → append hint
- [x] 2.3 Unit tests (`tests/unit/test_close_liveness.py`): hint on alive, silent on dead, skipped without aut_pid; pure-function contract

## 3. Empty-display diagnostic (D6)

- [x] 3.1 `_build_exposure_diagnostic` (ui_tree_service.py): on `presence == "unknown"` with no app filters, consult the cached display-PID probe — `frozenset()` → `{type: "display_empty", window_present: False, …}`; non-empty / `None` → existing behavior
- [x] 3.2 Unit tests (`tests/unit/test_empty_display_diagnostic.py`): empty set → display_empty; None → unchanged "(X11 probe unavailable)"; non-empty set → existing undetermined path; app-filters-given path untouched

## 4. Validation

- [x] 4.1 Full unit suite green: `uv run pytest tests/unit/ -q` (baseline 6774 passed + 1 skipped; no regressions) — **6790 passed + 1 skipped, +16 net**
- [x] 4.2 Benchmarks: no new failures (the 2 token-overhead failures + latency collection error are pre-existing, documented in prior changes) — **285 passed, only the 2 pre-existing failures**
