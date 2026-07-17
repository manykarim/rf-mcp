## 1. AUT window resolution + focus core

- [x] 1.1 Add `resolve_aut_window(node)` helper that walks a resolved PlatynUI node to its nearest WindowSurface-capable top-level ancestor (`top_level_or_self` / window-surface pattern); return None for unresolved descriptors
- [x] 1.2 Add a tiered focus strategy interface: (a) PlatynUI WindowSurface activate/raise, (b) portable raise/focus fallback (X11 XRaiseWindow-class), (c) no-op when already active; select by environment and catch WindowSurface PatternError to fall through
- [x] 1.3 Add an "is this window currently active/topmost" check used to skip redundant raises (focus-thrash mitigation, design D-risk)
- [x] 1.4 Cache the resolved AUT window per (session, app) and invalidate on app switch
- [x] 1.5 Unit tests for window resolution (nested descriptor, unresolved descriptor) and focus-strategy selection/fallback (mock `platynui_native` at the module boundary, call-shape style)

## 2. Focus-before-act default at the desktop chokepoint

- [x] 2.1 In `keyword_executor`, at the existing desktop chokepoint, resolve the AUT window and ensure raise+focus immediately before dispatching any PlatynUI pointer/keyboard keyword (default ON for DESKTOP_TESTING)
- [x] 2.2 Add escape hatch: per-call `focus=False` arg and `ROBOTMCP_PLATYNUI_NO_FOCUS=1` env; record `focus_bypassed=True` in the step result
- [x] 2.3 Ensure the focus default also covers the `intent_action` and batch execution paths (verify single chokepoint coverage)
- [x] 2.4 Unit tests: focus invoked before dispatch by default; bypassed under escape hatch; not invoked for non-desktop sessions (strict-bool `is_desktop_session()` guard)

## 3. Window-scoped action safety

- [x] 3.1 After target resolution, verify the resolved element's `top_level_or_self()` equals the AUT window (treat same-app child dialogs as in-scope); warn by default, error under strict mode
- [x] 3.2 Add strict-mode toggle (per-call + per-session) for the scope check
- [x] 3.3 Surface guidance that flags unscoped `//` desktop descriptors and recommends app-scoped form
- [x] 3.4 Unit tests: in-scope target dispatches; cross-window target warns/errors; child-dialog target treated in-scope

## 4. Visibility detection + precondition

- [x] 4.1 Compute AUT-window visibility from `IsVisible`/`IsInView`/`Bounds` intersected with `desktop_info` bounds; expose visible/on-screen/mapped state in the `ui_tree` section (`ui_tree_service.py` + `get_session_state`)
- [x] 4.2 Execution-time precondition: warn by default when an operation targets a non-visible AUT window; `fail_on_hidden=True` (per-call + per-session) fails fast with a clear error
- [x] 4.3 Init-time visibility guidance: indicate when an app process started but its window is not yet mapped; steer agents to wait for the window
- [x] 4.4 Ensure visibility warnings are carried through stepwise execution and available to the agent before/during `build_test_suite`
- [x] 4.5 Unit tests: visible vs unmapped/off-screen classification; warn vs fail-fast; warning surfaced in step result

## 5. Focus intent + agent guidance

- [x] 5.1 Add an "ensure focused / activate window" intent verb and map it to the PlatynUI focus path for desktop sessions
- [x] 5.2 Update the `platynui` locator-guidance topic: focus-before-act default, explicit focus keyword/intent, visibility verification, app-scoped descriptors, and the launch-app-visibly + no-PlatynUI-keyword-before-launch ordering rule
- [x] 5.3 Update MCP instructions to direct agents to ensure focus at test start and after any app switch, and to verify visibility before interacting
- [x] 5.4 Unit tests for the intent mapping + guidance topic content

## 6. Live e2e: operations only affect the AUT window

- [x] 6.1 Extend the GNOME e2e suite with a two-window scenario (Calculator + Text Editor overlapping at the same coordinates) asserting that an operation targeting one app does NOT mutate the other (read-back the non-target window is unchanged)
- [x] 6.2 Add an assertion that focus-before-act recovers the AUT when another window is raised between steps
- [x] 6.3 Add a non-visible-window scenario asserting the warning (default) and fail-fast (opt-in) behavior
- [x] 6.4 Verify the focus/visibility guarantees hold across the stepwise → `build_test_suite` flow (generated suite reflects real targeted actions, no silent pass)

## 7. Multi-CLI experiment harness

- [x] 7.1 Build the scripted harness: stage isolated Xvfb, launch the AUT visibly via `systemd-run`, normalize MCP-server env (DISPLAY, XDG_SESSION_TYPE, GDK_BACKEND, GSK_RENDERER, DBUS_SESSION_BUS_ADDRESS, XDG_RUNTIME_DIR, HOME), capture transcript + verdict, tear down
- [x] 7.2 Per-CLI adapters encoding MCP-config + approval/sandbox flags: Claude CLI (`--mcp-config`, `--strict-mcp-config`), Codex CLI (`-c mcp_servers.*`, sandbox bypass, `< /dev/null`), OpenCode CLI, Kilo CLI; record skip+reason when a CLI is unavailable
- [x] 7.3 Encode the canonical scenarios: Calculator (launch → calculate → assert each entered value + result → build suite) and Text Editor (type → out-of-band read-back → build suite), with focus/visibility verification steps
- [x] 7.4 Run the matrix across all four CLIs and aggregate a findings report (CLI, scenario, symptom, root cause, mitigation/status)
- [x] 7.5 Characterize OpenCode and Kilo MCP-config/approval mechanics (first-run discovery) and document any additional env normalization they require

## 8. Validation, docs, and wrap-up

- [x] 8.1 Run full unit suite + new tests; confirm no regressions (desktop guarantees do not affect web/mobile/API sessions)
- [x] 8.2 Run the extended live e2e suite green on isolated Xvfb
- [x] 8.3 Author an ADR (proposed) capturing the focus/visibility/window-scope model and the cross-agent findings; update ADR-025 implementation notes
- [x] 8.4 Update CHANGELOG/release notes for the behavioral default change + escape hatch
- [x] 8.5 (DEFERRED at archive time 2026-06-11: external action — upstream issues are documented in the harness findings and memory; superseded in part by the new-core findings of desktop-evidence-and-display-scoping) File the upstream PlatynUI issues collected (WM-less coordinate fallback, GTK4 text-content exposure, AT-SPI bus-registration env requirements) referenced by the harness findings
