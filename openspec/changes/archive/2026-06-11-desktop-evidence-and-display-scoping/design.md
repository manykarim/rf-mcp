# Design: desktop-evidence-and-display-scoping

## Context

Findings from the 2026-06-11 Codex re-validation run (`/tmp/dsef-libre-run2/agent_report.md`), root-caused in-session:

| # | Finding | Root cause (file:line) |
|---|---------|------------------------|
| 1 | AUT escaped to host compositor | GTK `wayland-0` fallback; **fixed** by `_pin_gtk_x11_backend` (platynui_plugin.py) |
| 2 | `Screenshot.Take Screenshot` success, file nowhere | RF outputdir is `tempfile.mkdtemp("rf_mcp_")` (`rf_native_context_manager.py:190`), `console='none'` + fd-1 suppression hide backend errors; **no post-execution file check exists** |
| 3 | PlatynUI `Take Screenshot` rejects `/tmp/dsef-libre-run2/shots/...` | `Path.relative_to` check anchored to the RF outputdir (`SecurePathValidator.validate`, domains/instruction/security.py:416 pattern) |
| 4 | Host apps (Chrome, keepassxc…) in isolated session's ui_tree | AT-SPI bus is session-global, not display-scoped; `/app:*` lists every registered app |
| 5 | `resume_batch` retried `Sleep` with 0 args | server.py:5178 reads only `s.get("args")`; canonical key is `arguments` (handled by `BatchExecution._resolve_step_args`, batch_execution/aggregates.py:132) |
| 6 | `end_test` → "No active test to end" | `run_suite_dry_run` (execution_coordinator.py:661) recreates context; `create_context_for_session` overwrites `current_run_test`/`current_res_test` with None on reuse (rf_native_context_manager.py:394-395) |
| 7 | 20+ blind `Keyboard Type` "successes" into an empty display | type-at-focus has no descriptor → `ensure_focused` returns early, no warning |
| 8 | Empty-suite warning stayed `null` despite 41 failed steps | suite body contained Create Directory / Take Screenshot — not in `_LAUNCH_SETUP_KEYWORDS`, so `_suite_body_is_launch_only` returned False |

## Goals / Non-Goals

**Goals:** every "success" robotmcp reports for evidence-producing keywords is backed by a file on disk or carries a warning; isolated sessions never present host-desktop applications as targets; batch retries re-run the exact failed call; `start_test`/`end_test` is a reliable bracket; blind typing is never silently unverified; the backend-pin fix is specced for archive.

**Non-Goals:** fixing RF's `Screenshot` library itself; per-window input isolation; display-scoping arbitrary `Query` keywords executed via execute_step (discovery surface `ui_tree` only — Query is the user's explicit instruction); reworking the artifact/externalization policy (ADR-015) beyond screenshot paths; Wayland-native support.

## Decisions

### D1 — Backend confinement is already implemented; spec ratifies it
`_pin_gtk_x11_backend(env)` runs inside `ensure_x11_session_env()` at all three exit paths; pins only when (a) no explicit `GDK_BACKEND`/`QT_QPA_PLATFORM` set, (b) a Wayland socket is reachable (env or `$XDG_RUNTIME_DIR/wayland-0`), (c) `DISPLAY` is bound; `KEEP_WAYLAND` opt-out short-circuits earlier. Children inherit via Process. 5 unit tests exist. No further code work — tasks only verify and document.

### D2 — Screenshot evidence verification: file-exists check in the executor, soft warning
After a successful desktop-session step whose keyword is a screenshot/save producer (`take screenshot`, `screenshot`, suffix-matched like other desktop signal helpers) with a recognizable path argument or path-bearing return value, the executor checks `os.path.isfile(path)` (≤5ms) and appends a hint `{type: "evidence_missing", message: "keyword reported success but '<path>' does not exist on disk"}`. Implemented as a small helper in `desktop_execution_signals.py` (same pattern as `launch_liveness_hint`), wired in the success branch next to the D2a desktop block. The keyword's return value is preferred over the argument when it is a string path (RF `Screenshot` returns the saved path).
*Alternative rejected*: resurrecting the full ADR-021 P5 PostActionVerifier surface — the .pyc-only ghost module shows that scope stalled; a single-purpose file check ships now and composes later.

### D3 — PlatynUI screenshot paths: save inside outputdir, then copy out
Keep the security anchor (writes happen under the RF outputdir) but make the user's requested absolute path work: when the requested path falls outside the outputdir, the keyword wrapper saves to `<outputdir>/<basename>` and then copies to the requested path, returning the requested path; the copy targets only paths under `/tmp` or the session's configured artifact directory (`ROBOTMCP_SCREENSHOT_DIR` env or session attr) — refused otherwise with the existing error plus a hint naming the allowed roots.
*Alternative rejected*: relaxing `SecurePathValidator` itself — it guards instruction-file traversal (CVE-INST-002); widening it is a security regression.

### D4 — ui_tree display scoping via one batched EWMH PID probe
For sessions whose bound display is isolation-marked (`classify_bound_display_detailed()["isolation_source"] == "marker"`), `_collect_ui_tree_sync` collects the AT-SPI app list, reads each app's `ProcessId` attribute, and runs ONE subprocess EWMH probe (extension of the existing `_X11_WINDOW_PROBE_SRC` in `platynui_focus.py`, already subprocess-isolated for Xlib safety) that returns the set of `_NET_WM_PID`s present on the bound display. Apps with a PID not in that set are dropped from `applications` and counted in `host_apps_filtered`; apps with no readable PID are KEPT (fail-open for discovery, the count + flag make it visible). Active/unknown displays: no filtering (the whole desktop IS the session's scope).
*Alternative rejected*: bounds-based heuristics (host and nested coordinates overlap); per-app probes (N subprocesses); filtering at the runtime/Query layer (would silently change explicit user queries).

### D5 — resume_batch uses execute_batch's argument resolution
server.py resume_batch's fix_step construction switches from `s.get("args", [])` to the same dual-key resolution as `BatchExecution._resolve_step_args` (extract a small shared helper or call the aggregate's logic); the failed-step retry keeps `list(failed.args)` and a regression test pins that a paused `BuiltIn.Sleep  2s` resumes with `["2s"]`, plus fix_steps given with `arguments=` keep their values.

### D6 — Context reuse preserves active-test state
`create_context_for_session` (rf_native_context_manager.py): in the reuse branch, seed `_initial_run_test`/`_initial_res_test` from the EXISTING session context entry instead of None before the unconditional write at :394-395. Belt-and-braces: `run_suite_dry_run` (execution_coordinator.py:661) skips `create_context_for_session` when `get_session_context_info(session_id)["context_exists"]` is true. Regression test: start_test → build_test_suite (dry-run path) → end_test succeeds.

### D7 — Unfocused-typing warning
Track `last_verified_focus` on the focus manager (set when `focus_window` returns focused with strategy ≠ `x11_raise`, or `input_ready is True`). In `ensure_focused`, when the keyword is a keyboard interaction AND `extract_descriptor` returns None (type-at-focus): if no verified focus has been established this session, return an outcome with `attempted=False` but a warning *"type-at-focus with no previously verified AUT window focus — keystrokes may land nowhere"* via the existing `platynui_focus_warning` channel. One warning per session per the de-dup conventions (one-shot flag like `desktop_wayland_warned`).

### D8 — Evidence keywords are scaffolding for the empty-suite warning
Extend `TestBuilder._LAUNCH_SETUP_KEYWORDS` with `take screenshot`, `create directory`, `is process running`, `get process id`, `terminate process`, `wait for process`. The Run-2 regression test gains a variant: suite of launch + screenshots + Create Directory with 41 failed steps → warning fires.

## Risks / Trade-offs

- [PID probe adds latency to ui_tree] → one subprocess (~50-150ms) per ui_tree call, only for marker-isolated sessions; reuses the proven isolation-safe probe; result cached until `desktop_tree_dirty`.
- [Fail-open for PID-less apps weakens scoping] → deliberate: discovery must not hide the AUT; `host_apps_filtered` + per-app `display_scoped: false` annotation keep it honest.
- [Screenshot copy-out could overwrite user files] → copy refuses to overwrite existing files unless the path is under the RF outputdir lineage; hint explains.
- [D6 touches the RF context bootstrap (high blast radius)] → change is additive (seed-from-existing instead of None); full 6731-test suite + the multi-test session suite gate it.
- [D7 may warn on legitimate type-at-focus flows right after a successful click] → a successful verified focus or pointer interaction with verified focus clears the flag; warning is once per session.

## Migration Plan

Additive; no API removals. D1 already shipped. Rollback = revert. Baseline 6731 passed + 1 skipped stays green.

## Open Questions

- Should display scoping also annotate (not filter) in `active`-display sessions? Deferred — active sessions see the desktop they automate.
- Should the evidence check extend to `Save File`-style keywords beyond screenshots (e.g. the report.odt save flow)? The save was keyboard-driven (no path argument visible to robotmcp), so out of detection reach; revisit if a Save-As intent verb lands.
