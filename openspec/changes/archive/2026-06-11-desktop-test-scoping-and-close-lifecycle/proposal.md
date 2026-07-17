# Proposal: desktop-test-scoping-and-close-lifecycle

## Why

Run 3 of the LibreOffice validation (2026-06-11, `/tmp/dsef-libre-run3/agent_report.md`) was the first end-to-end success — visible launch, typing, bold formatting, save, reopen, content assertion all verified — but it exposed the next layer of defects. The biggest: **the generated suite contained only 3 of 46 recorded steps.** The agent's early `start_test` failed with "No context for session" (the handler never creates the RF context; `execute_step` does), the failure was swallowed as a soft warning, and the registry/RF-context layers diverged — 43 steps landed in `suite_level_steps`, which `build_test_suite` silently drops, and the final `end_test` failed with "No active test to end" despite an active registry test. Two smaller gaps: closing the document window left LibreOffice running as a start-center frame with **no signal** that the AUT process survived the close; and the pre-launch exposure diagnostic on a genuinely empty display claimed "X11 probe unavailable" when the display was simply empty.

## What Changes

- **Atomic, layer-consistent test scoping (#1)**: `manage_session(action="start_test")` creates the RF context when missing (same path `execute_step` uses) before `start_test_in_context`; if the context layer still fails, the WHOLE start_test fails — the registry is never activated into an inconsistent half-state. `end_test` treats the TestRegistry as the source of truth: an active registry test always ends successfully, with a context-layer miss downgraded to a soft warning. `build_test_suite` stops silently dropping `suite_level_steps`: the response reports `suite_level_step_count` and warns when recorded-but-outside-test steps outnumber in-test steps, naming the cause and remedy.
- **Close liveness signal (#3)**: after a successful desktop `Close Window` (or close-shaped keyboard step is out of scope — keyword-level only), robotmcp checks whether the launched AUT process (`session.desktop_aut_pid`) is still alive and hints when it is ("document window closed but the application process is still running — a residual frame such as a start center may remain; close it or use `Terminate Process` for a hard stop"). Mirrors the launch-liveness hint (ADR-029).
- **Empty-display diagnostic (#5)**: `_build_exposure_diagnostic` consults the batched `x11_display_pids()` probe when the name-based presence probe returns "unknown" with no app filters: an empty PID set produces a clear "display is reachable but has no application windows — launch the AUT first" diagnostic instead of "(X11 probe unavailable)"; a real probe failure keeps the current wording.

No breaking changes: one behavior fix (start_test now fails loudly instead of half-succeeding — previously it returned success while broken), new warnings/counters, and a re-worded diagnostic.

## Capabilities

### New Capabilities

- `desktop-test-scoping-integrity`: start_test auto-creates the RF context and is atomic across the registry/context layers; end_test is registry-first; suite-level step accumulation is visible in build_test_suite.
- `desktop-close-liveness`: post-close still-running hint for desktop AUT processes.
- `desktop-empty-display-diagnostic`: empty-but-reachable display distinguished from probe failure.

### Modified Capabilities

(none — `openspec/specs/` has no archived capabilities yet)

## Impact

- `src/robotmcp/server.py` (~3505-3602) — start_test/end_test handlers: context auto-creation, atomic failure, registry-first end.
- `src/robotmcp/components/execution/rf_native_context_manager.py` (start_test_in_context/end_test_in_context ~699-787) — unchanged API; consumed differently.
- `src/robotmcp/models/execution_models.py` — TestRegistry untouched or minimally extended (no half-activation).
- `src/robotmcp/components/test_builder.py` — suite-level step count + warning in build response.
- `src/robotmcp/components/execution/desktop_execution_signals.py` + `keyword_executor.py` (post-success desktop block ~2196) — `is_close_keyword` / `close_liveness_hint`.
- `src/robotmcp/components/execution/ui_tree_service.py` (`_build_exposure_diagnostic` ~178-259) — empty-display branch via `x11_display_pids()`.
- Tests: new unit files per capability; baseline 6774 passed + 1 skipped stays green.
