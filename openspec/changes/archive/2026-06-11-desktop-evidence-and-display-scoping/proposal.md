# Proposal: desktop-evidence-and-display-scoping

## Why

The 2026-06-11 external-agent re-validation (Codex/gpt-5.4 driving robotmcp against LibreOffice Writer on an isolated Xephyr `:100`; full report `/tmp/dsef-libre-run2/agent_report.md`) confirmed the platynui-visible-safe-targeting diagnostics work (the I-4 `\=` hint was used verbatim to fix the launch; `desktop_environment` proved isolation; counters surfaced 100 executed / 59 recorded / 41 failed) — but exposed six further defects. The dominant one (GTK's `wayland-0` fallback letting the AUT escape to the host compositor) was root-caused and hot-fixed during the run (`_pin_gtk_x11_backend`, verified: Writer maps, renders, and registers in AT-SPI on `:100`); it needs a spec to be archivable. The remaining five are unfixed: screenshot keywords report success while writing **nothing** to disk; the AT-SPI tree shows **all host desktop apps** (Chrome, keepassxc, …) inside an "isolated" session — a gnome-shell dash icon labeled "LibreOffice Writer" actively misled the agent; PlatynUI `Take Screenshot` rejects user-requested output paths; `resume_batch` retried `BuiltIn.Sleep` with zero arguments; and `end_test` failed with "No active test to end" after a context recreation silently cleared the active-test state.

## What Changes

- **Display backend confinement (already implemented, needs spec)**: `ensure_x11_session_env()` pins `GDK_BACKEND=x11` / `QT_QPA_PLATFORM=xcb` when a Wayland compositor socket is reachable and an X display is bound, because `wl_display_connect(NULL)` falls back to the literal `wayland-0` socket even with `WAYLAND_DISPLAY` unset — without the pin, GTK AUTs launched via Process render on the user's active desktop while input goes to the isolated display.
- **Screenshot evidence integrity**: (a) post-action file verification — after a screenshot/save keyword succeeds with a path argument, robotmcp checks the file exists and emits a warning when it does not (the run produced 5 ghost .jpgs that exist nowhere); (b) user-requested screenshot paths work — PlatynUI `Take Screenshot` no longer hard-fails with `… is not in the subpath of /tmp/rf_mcp_*` for paths under the operator's working area (resolve into the RF outputdir then copy to the requested path, or validate against a session-configurable allowlist); (c) evidence/scaffolding keywords (`take screenshot`, `create directory`, `is process running`, `get process id`, `terminate process`, `wait for process`) count as scaffolding for the I-1 empty-suite warning, so a suite of launch + screenshots with 41 failed interactions warns instead of staying silent.
- **AT-SPI tree display scoping**: for isolation-marked desktop sessions, `ui_tree` filters applications whose process has no X client window on the bound display (one batched EWMH `_NET_WM_PID` subprocess probe), reports `host_apps_filtered` count, and never lists host-desktop applications as automation targets.
- **Batch resume argument fidelity**: `resume_batch` resolves step arguments with the same `_resolve_step_args` semantics as `execute_batch` (accepts both `args` and `arguments` keys) and preserves the failed step's original arguments on retry.
- **Test lifecycle state preservation**: recreating/reusing the RF native context (e.g. the dry-run path in `build_test_suite`) MUST NOT clear `current_run_test` / `current_res_test`, so `end_test` after `start_test` always finds the active test.
- **Unfocused typing warning**: a desktop `Keyboard Type`/`Keyboard Press` with no descriptor (type-at-focus) emits a `platynui_focus_warning` when no AUT window has been verifiably focused in the session — blind typing into an empty display was robot-level "success" 20+ times in the run.

No breaking changes; one already-shipped env mutation (backend pinning, opt-out via existing `KEEP_WAYLAND` env), the rest are new warnings, fixed argument handling, and a discovery filter with an explicit count.

## Capabilities

### New Capabilities

- `desktop-display-backend-confinement`: GTK/Qt backends pinned to the bound X display when a Wayland compositor socket is reachable (implemented; spec + archival).
- `desktop-evidence-integrity`: screenshot/save post-action file verification, user-reachable screenshot output paths, evidence keywords classified as scaffolding for the empty-suite warning.
- `desktop-tree-display-scoping`: ui_tree limited to applications present on the bound display for isolation-marked sessions, with filtered-count reporting.
- `batch-resume-argument-fidelity`: resume_batch argument resolution identical to execute_batch; retry preserves original arguments.
- `desktop-test-lifecycle-preservation`: active-test state survives RF context recreation/reuse.
- `desktop-unfocused-typing-warning`: type-at-focus without a verified focused AUT window warns.

### Modified Capabilities

(none — `openspec/specs/` has no archived capabilities yet)

## Impact

- `src/robotmcp/plugins/builtin/platynui_plugin.py` — `_pin_gtk_x11_backend` (already implemented + 5 tests in `tests/unit/test_platynui_newcore_plugin.py`).
- `src/robotmcp/components/execution/keyword_executor.py` — post-screenshot file verification hook; unfocused-typing warning wiring; screenshot path handling.
- `src/robotmcp/domains/instruction/security.py:416` area / RF outputdir policy (`rf_native_context_manager.py:190`) — screenshot path resolution.
- `src/robotmcp/components/test_builder.py` — `_LAUNCH_SETUP_KEYWORDS` extension.
- `src/robotmcp/components/execution/ui_tree_service.py` + `platynui_focus.py` (`x11_window_present` probe reuse) — display scoping.
- `src/robotmcp/server.py:5178` + `domains/batch_execution/aggregates.py:132` — resume_batch args.
- `src/robotmcp/components/execution/rf_native_context_manager.py:394-395` + `execution_coordinator.py:661` — lifecycle preservation.
- Tests: new unit files per capability; baseline 6731 passed + 1 skipped must stay green.
