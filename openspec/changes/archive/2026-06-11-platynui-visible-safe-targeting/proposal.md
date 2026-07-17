# Proposal: platynui-visible-safe-targeting

## Why

The LibreOffice Writer validation run (docs/DESKTOP_PLATYNUI_LIBREOFFICE_VALIDATION_REPORT.md, 2026-06-11) proved that the read/locate/launch half of the PlatynUI desktop workflow works end-to-end against a real complex application, but the interaction half failed **silently**: 20 executed steps produced an empty generated suite with no warning (I-1), keystrokes were sent to a frame whose focus could not be verified — without any `platynui_focus_warning` firing (I-2), read-back `Query` assignments quietly dropped variables unless `use_context=True` (I-3), and a Robot Framework `=`-in-argument quirk silently corrupted the LibreOffice launch command (I-4). Separately, the run only succeeded at being *visible and safe* because a human hand-rolled a nested Xephyr display and env wiring — the "visible on the tester's screen, input confined to the test application" path exists (`scripts/platynui_desktop_bootstrap.sh --mode visible` + ADR-027 isolation marker) but is neither surfaced nor verifiable through the MCP tools, so agents cannot discover or prove it.

**Guiding principle (user directive)**: prefer the CLI tools and APIs the PlatynUI new-core already ships (`platynui_native` Runtime API, pattern introspection, `bring_to_front`/`accepts_user_input`, `highlight`, `screenshot`, `desktop_info`, `platynui-cli`) over rf-mcp-custom mechanisms. Existing custom fallbacks (ctypes `XRaiseWindow`-by-PID) become last-resort only.

## What Changes

- **Stepwise suite feedback (I-1)**: `ExecutionSession` tracks executed-vs-recorded step counters; `execute_step` responses surface `recorded` plus running counts; `build_test_suite` returns `steps_executed` / `steps_recorded` / `steps_failed` and emits an explicit warning when a session with N>0 executed steps yields an empty or launch-only suite body.
- **Upstream-verified focus + unverifiable warning (I-2)**: focus-before-act is rebuilt on upstream primitives — `node.supported_patterns()` / `has_pattern(WindowSurface|Focusable)` for explicit capability detection, `runtime.bring_to_front(node, wait_ms=…)` (which restores + activates + polls `WindowSurface.accepts_user_input()`) for **verified** activation, and `runtime.focus(node)` for element focus. When the resolved AUT window supports neither pattern, emit a `platynui_focus_warning` hint ("input focus could not be verified for this target — keystrokes may not land") instead of silently falling through; report which upstream mechanism succeeded (`bring_to_front` / `focus` / fallback) and the `accepts_user_input` verdict in `platynui_focus`.
- **Visible-and-confined execution surfacing (user requirement)**: before desktop interaction steps, optionally mark the target on the tester's screen with upstream `runtime.highlight()` (configurable per session; default on for interactive/visible displays) so the human can SEE exactly which element receives input; `get_session_state` desktop sections report the bound display, its isolation classification (`isolated` / `active` / `unknown`), and upstream `desktop_info()` (technology, bounds, monitors) so an agent can prove "app is visible AND input is confined to display :N"; the isolation recipe (`build_isolation_recipe()`) and safety-guard refusal message present the **visible Xephyr mode as the recommended interactive path** and reference `platynui-cli` (`window --list`, `highlight`, `screenshot`) for operator-side verification.
- **Desktop Query variable persistence (I-3)**: variables assigned via `assign_to` persist into `session.variables` on the non-context execution path too (today only the `use_context=True` path merges them back), removing the confusing default for desktop read-back.
- **Process `=`-argument detect-and-hint (I-4)**: when a `Start Process` / `Run Process` argument contains `=` with a non-identifier left side that RF would swallow as a named argument (e.g. `-env:UserInstallation=file:///…`), emit a hint suggesting the `\=` escape; desktop launch sanitization applies the same detection.

No breaking changes; all additions are new response fields, new warnings/hints, and one default-behavior fix (I-3) that makes assignment strictly more persistent.

## Capabilities

### New Capabilities

- `desktop-stepwise-suite-feedback`: executed/recorded/failed step accounting on the session, surfaced in `execute_step` and `build_test_suite` responses, with an explicit empty-suite warning for stepwise sessions.
- `desktop-focus-verifiability`: focus-before-act uses upstream pattern introspection (`supported_patterns`) and verified activation (`bring_to_front` + `accepts_user_input`); emits a focus-unverifiable warning when no upstream focus path exists on the resolved AUT window, and reports which mechanism was used.
- `desktop-visible-confined-execution`: upstream `highlight()` target marking before interaction; visible (Xephyr) isolated display surfaced as the recommended interactive mode in recipes and guard messages; display identity + isolation classification + upstream `desktop_info()` reported in session state so input confinement to the test application is provable.
- `desktop-assignment-persistence`: `assign_to` variables persist to session scope regardless of execution path (context or non-context).
- `desktop-process-arg-hints`: detect-and-hint for Process library `=`-containing positional arguments that RF would misparse as named arguments.

### Modified Capabilities

(none — `openspec/specs/` has no archived capabilities yet; prior desktop capabilities live in unarchived changes)

## Impact

- `src/robotmcp/models/session_models.py` — executed/recorded/failed counters on `ExecutionSession`.
- `src/robotmcp/components/execution/keyword_executor.py` — counter increments at success/failure recording points (~lines 1965-2062); non-context `assign_to` persistence (~lines 2065-2069); Process `=`-arg detection in desktop launch sanitization (~lines 313-432).
- `src/robotmcp/components/execution/platynui_focus.py` — focus tiers rebuilt on upstream `bring_to_front(node, wait_ms)` + `accepts_user_input()` verification and `supported_patterns()` introspection; ctypes `_x11_raise` demoted to last-resort; focus-unverifiable warning when no upstream focus path exists (lines 339-394, 512-518); optional `runtime.highlight()` target marking.
- `src/robotmcp/components/execution/desktop_display_safety.py` — `build_isolation_recipe()` gains visible-mode (Xephyr) recipe as recommended; classification surfaced for session-state reporting.
- `src/robotmcp/components/test_builder.py` — suite statistics (`steps_executed`/`steps_recorded`/`steps_failed`) + empty-suite warning (~lines 363-432, 508-602).
- `src/robotmcp/server.py` / `ui_tree_service.py` — desktop session-state section reporting display + isolation classification.
- `src/robotmcp/utils/hints.py` — Process `=`-argument hint checker.
- Tests: new unit tests per capability under `tests/unit/`; existing desktop suites (6665 passed baseline) must stay green.
