# Tasks: platynui-visible-safe-targeting

## 1. Stepwise suite feedback (I-1)

- [x] 1.1 Add `executed_step_count` / `failed_step_count` fields to `ExecutionSession` (`src/robotmcp/models/session_models.py`) with a `recorded_step_count` helper that sums legacy steps, registry tests, and suite-level steps
- [x] 1.2 Increment counters in `keyword_executor.py` at the success branch (~line 1965) and failure branch (~line 2059); add `steps_executed` / `steps_recorded` to the `execute_step` response next to the existing `recorded` flag
- [x] 1.3 Add `steps_executed` / `steps_recorded` / `steps_failed` to `build_test_suite` statistics in `test_builder.py` (~lines 508-602)
- [x] 1.4 Implement the empty/near-empty suite `warning` in `test_builder.py` (fires when executed ≥ 3, failed ≥ 1, and suite body has only launch/setup steps), mirroring the ADR-021 P12 top-level warning pattern
- [x] 1.5 Unit tests: counters across success/failure/inspection-only paths; warning fires on the LibreOffice regression shape (20 executed / 1 recorded) and stays silent on healthy sessions (`tests/unit/test_stepwise_suite_feedback.py`)

## 2. Upstream-verified focus + unverifiable warning (I-2)

- [x] 2.1 Add pattern introspection to `PlatynUIFocusManager` (`platynui_focus.py`): hasattr-guarded `supported_patterns()` on the resolved window; store names on `FocusOutcome.patterns`
- [x] 2.2 Rework `focus_window()` tiers: Tier 1 upstream `runtime.bring_to_front(window, wait_ms≤1500)` with typed `BringToFrontError` handling and `accepts_user_input()` → `FocusOutcome.input_ready`; Tier 2 `runtime.focus(node)` when `Focusable` supported; Tier 3 ctypes `_x11_raise` last-resort always paired with the unverifiable warning
- [x] 2.3 Emit the I-2 warning ("input focus could not be verified for this target — keystrokes may not land (no WindowSurface/Focusable pattern)") when introspection shows neither pattern; verify it flows through the existing `platynui_focus_warning` hint channel in `keyword_executor.py:1956-1963`
- [x] 2.4 Add AUT PID scope check: compare session-launched process PID with the target's `app:Application` ancestor `ProcessId` attribute; warning on mismatch
- [x] 2.5 Cache the bring-to-front outcome per (session, window runtime_id); invalidate via the ADR-031 `desktop_tree_dirty` flag
- [x] 2.6 Unit tests with stub runtimes: verified activation, input_ready false/null, missing-pattern warning, x11_raise flagging, PID mismatch, graceful degradation on a runtime lacking `supported_patterns`/`wait_ms` (`tests/unit/test_focus_verifiability.py`)

## 3. Visible-and-confined execution (highlight + session state + recipe)

- [x] 3.1 Implement target highlighting in the focus-before-act gate: `runtime.highlight(bounds_rect, duration_ms=600)` from the already-resolved node's `Bounds`, soft-fail, session config `platynui_highlight` (default True) + `ROBOTMCP_PLATYNUI_HIGHLIGHT=0` kill switch
- [x] 3.2 Call `clear_highlight()` before screenshot keywords to avoid overlay contamination
- [x] 3.3 Add `desktop_environment` section to `get_session_state` desktop output (`ui_tree_service.py` / `server.py`): bound display, isolation classification + source (reuse `desktop_display_safety.classify()`), upstream `runtime.desktop_info()` (technology, bounds, monitors, os)
- [x] 3.4 Rework `build_isolation_recipe()` (`desktop_display_safety.py`): visible Xephyr mode first ("recommended — app visible on your screen, input confined to the nested display"), Xvfb second, append `platynui-cli-rs` verification commands (`info`, `window --list`, `highlight`, `snapshot`); point the safety-guard refusal message at it
- [x] 3.5 Update `scripts/platynui_desktop_bootstrap.sh --mode visible` to start a minimal EWMH WM (`openbox` when available) inside Xephyr, with a recipe note that focus verification degrades to a warning without one
- [x] 3.6 Unit tests: highlight call ordering + kill switch + soft-fail; `desktop_environment` content for marker-isolated display; recipe ordering and CLI command presence (`tests/unit/test_visible_confined_execution.py`)

## 4. Desktop assignment persistence (I-3)

- [x] 4.1 Persist `assign_to` variables into `session.variables` on the non-context path (`keyword_executor.py` ~2065-2069), reusing the context path's `${name}` normalization — **verified already-correct: the executor is context-only (`if True:` at ~1940), assignment persists on both flag values; no production change needed, behavior pinned by 4.2 tests**
- [x] 4.2 Unit tests: non-context Query assignment resolves in a later step; context path unchanged; multi-assign (`tests/unit/test_assignment_persistence.py`)

## 5. Process `=`-argument detect-and-hint (I-4)

- [x] 5.1 Add the misparse heuristic (dash-prefixed, contains `=`, left side not an identifier, not a known Process config prefix `env:`/`shell`/`cwd`/`alias`/`stdout`/`stderr`) as a shared helper; emit proactive hint from `_maybe_sanitize_desktop_launch()` (`keyword_executor.py` ~313-432) — detection only, never rewrite
- [x] 5.2 Add reactive checker `_check_process_named_arg_misparse` to `utils/hints.py` for failed Process launches
- [x] 5.3 Unit tests: `-env:UserInstallation=…` flagged proactively and reactively; `env:HOME=/x` and `shell=True` not flagged (`tests/unit/test_process_arg_hints.py`)

## 6. Validation

- [x] 6.1 Full unit suite green: `uv run pytest tests/unit/ -q` (baseline 6665 passed + 1 skipped; no regressions) — **6731 passed + 1 skipped, +66 net**
- [x] 6.2 Benchmarks green; confirm focus-gate hot-path overhead stays within existing budgets (`uv run pytest tests/ -k benchmark`) — **285 passed; 2 failures in test_robustness_token_overhead.py + 1 collection error in test_robustness_latency.py are PRE-EXISTING on this branch (verified failing at HEAD before this change; `_extract_force_flag` no longer exists)**
- [x] 6.3 E2E re-validation of the LibreOffice scenario shape (stub-level): empty-suite warning, focus-unverifiable warning, and Process arg hint all fire on the Run-2 transcript pattern
