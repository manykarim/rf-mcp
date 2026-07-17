# Tasks: desktop-evidence-and-display-scoping

## 1. Display backend confinement (already implemented — ratify)

- [x] 1.1 Verify `_pin_gtk_x11_backend` in `platynui_plugin.py` matches the spec scenarios (pin on wayland-socket-reachable + DISPLAY; respect pre-set values; keep-Wayland opt-out; pure-X11 no-op) and the 5 tests in `tests/unit/test_platynui_newcore_plugin.py` cover all four scenarios; add any missing scenario test
- [x] 1.2 Document the wayland-0 fallback mechanism + pin in the desktop locator/driveability guidance (`rf_native_type_converter.py` PlatynUI guidance) and the isolation recipe note

## 2. Screenshot evidence integrity (D2/D3/D8)

- [x] 2.1 Add `evidence_missing_hint(keyword, arguments, result_value)` to `desktop_execution_signals.py`: detect screenshot keywords by basename, extract the path from the returned value (preferred) or first path-like argument, return a hint dict when the file is absent; ≤5ms, never raises
- [x] 2.2 Wire the hint into the executor success branch for desktop sessions (next to the D2a block in `keyword_executor.py`)
- [x] 2.3 Implement save-then-copy for `PlatynUI.BareMetal.Take Screenshot` with out-of-outputdir paths: allowed roots `/tmp` + `ROBOTMCP_SCREENSHOT_DIR`; no overwrite of existing files; refusal hint names allowed roots (locate the current path rejection and adapt at that point)
- [x] 2.4 Extend `TestBuilder._LAUNCH_SETUP_KEYWORDS` with `take screenshot`, `create directory`, `is process running`, `get process id`, `terminate process`, `wait for process`
- [x] 2.5 Unit tests: ghost screenshot warns / real file silent; copy-out path works + refusal hint; Run-3 suite shape (launch + evidence + 41 failed) now triggers the empty-suite warning (`tests/unit/test_evidence_integrity.py`)

## 3. ui_tree display scoping (D4)

- [x] 3.1 Extend the subprocess EWMH probe (`platynui_focus.py` `_X11_WINDOW_PROBE_SRC` family) with a mode that prints ALL `_NET_WM_PID`s on the bound display (one shot, batched)
- [x] 3.2 In `_collect_ui_tree_sync` / `get_ui_tree`: when `classify_bound_display_detailed()` reports marker isolation, read each app's ProcessId, filter apps whose PID is absent from the probe set, keep PID-less apps annotated `display_scoped: false`, add `host_apps_filtered` count; probe failure → unfiltered + `display_scoping: "unavailable"`
- [x] 3.3 Cache the probe result until `desktop_tree_dirty` invalidates it
- [x] 3.4 Unit tests with stubbed probe + fake app nodes: filtering, fail-open annotation, active-display bypass, probe-failure degradation (`tests/unit/test_tree_display_scoping.py`)

## 4. Batch resume argument fidelity (D5)

- [x] 4.1 Fix `server.py` resume_batch fix_steps construction (~line 5178) to resolve arguments with the same dual-key semantics as `BatchExecution._resolve_step_args` (share the helper rather than duplicating)
- [x] 4.2 Regression tests: failed `BuiltIn.Sleep  2s` resumes with `["2s"]`; fix_steps with `arguments=` and with `args=` both execute with their values (`tests/unit/domains/batch_execution/` or `tests/unit/test_batch_resume_args.py`)

## 5. Test lifecycle preservation (D6)

- [x] 5.1 In `rf_native_context_manager.create_context_for_session`: seed `_initial_run_test`/`_initial_res_test` from the existing session-context entry in the reuse branch before the unconditional write (~lines 394-395)
- [x] 5.2 Guard `execution_coordinator.run_suite_dry_run` (~line 661): skip `create_context_for_session` when the session context already exists — **deviation: the refresh is KEPT (the dry run tears down EXECUTION_CONTEXTS and the call re-establishes it); 5.1 makes it state-preserving, which is the actual requirement. Documented in a code comment.**
- [x] 5.3 Regression test: start_test → build_test_suite (dry-run path) → end_test succeeds; multi-test session suite stays green (`tests/unit/test_lifecycle_preservation.py`)

## 6. Unfocused typing warning (D7)

- [x] 6.1 Track verified focus on `PlatynUIFocusManager` (set on focused with strategy ≠ `x11_raise` or `input_ready is True`); expose `has_verified_focus`
- [x] 6.2 In `ensure_focused`: keyboard keyword + no descriptor + no verified focus → emit the unfocused type-at-focus warning via `outcome.warnings`; once per session (one-shot flag read/written via the executor session, mirroring `desktop_wayland_warned`)
- [x] 6.3 Unit tests: blind typing warns once; typing after verified focus silent; descriptor-targeted typing unaffected (`tests/unit/test_unfocused_typing_warning.py`)

## 7. Validation

- [x] 7.1 Full unit suite green: `uv run pytest tests/unit/ -q` (baseline 6731 passed + 1 skipped; no regressions) — **6774 passed + 1 skipped, +43 net**
- [x] 7.2 Benchmarks (excluding the pre-broken `test_robustness_latency.py` collection error documented in platynui-visible-safe-targeting): no new failures; ui_tree probe latency within budget — **285 passed; only the 2 pre-existing test_robustness_token_overhead failures (verified failing at HEAD)**
- [x] 7.3 Stub-level Run-3 regression: ghost-screenshot hint, host-app filtering, Sleep-args resume, start/end_test bracket, and blind-typing warning all fire on the recorded transcript shapes — **covered across the five new test files (each pins its Run-3 transcript shape verbatim)**
