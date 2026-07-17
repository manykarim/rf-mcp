## Why

A live end-to-end run (external Codex agent driving the robotmcp MCP server
against GNOME Calculator on an isolated display) completed the full stepwise
scenario — three calculations, per-entry and result assertions, a clean
generated suite — but surfaced two concrete, reproduced issues that made the
agent work harder than necessary:

1. **The documented display-state read path is unreliable for this app.** The
   PlatynUI driveability guidance (added by `desktop-stepwise-execution-fidelity`)
   tells agents to read the calculator display via
   `native:Text.CharacterCount` on the entry `Text` node. In the live run that
   attribute returned **0 across all six Text nodes** even while the display
   visibly changed, so the agent could not assert entered values that way and
   fell back to OCR on screenshots. The agent independently found that the
   calculator's **history Labels DO work** (e.g. a `54÷6 = 9` history row),
   which is the reliable path the guidance should lead with.

2. **Unqualified `Process` keywords do not resolve in desktop sessions.** The
   agent's first `Start Process    gnome-calculator` failed with
   "No keyword with name 'Start Process' found"; only `Process.Start Process`
   worked. Reproduced in code: `plugin_manager.get_library_for_keyword(
   "Start Process")` returns `None` because core RF libraries registered via
   static definitions (Process, OperatingSystem, …) do not expose a
   keyword→library map (only plugins like PlatynUI/Requests do). With no
   mapping, `_ensure_library_registration` never registers Process into the
   live RF namespace, so the unqualified keyword is unresolved. It manifests in
   desktop sessions because launching the app requires `Process`, which is not a
   default-loaded library. Qualifying the name works because the dotted-name
   branch short-circuits the lookup.

## What Changes

- **Display-state guidance prefers history Labels.** Rework the PlatynUI
  `display_state_reading` guidance so the PRIMARY, reliable assertion path is
  reading the calculator's history/result `Label` nodes (named with the
  equation and the result value). `native:Text.CharacterCount` is demoted to a
  secondary length proxy explicitly flagged as "may report 0 on some GTK
  builds — do not rely on it alone", and screenshot/OCR is named as the last
  resort. The guidance reflects what actually works.
- **Unqualified Process keywords resolve in desktop sessions.** A desktop
  session can launch and manage the app under test with unqualified `Process`
  keywords (`Start Process`, `Run Process`, `Terminate Process`,
  `Process Should Be Running`, …). Achieved by giving the `Process` library a
  keyword→library mapping so `get_library_for_keyword` resolves its keywords,
  AND/OR eagerly registering `Process` into the desktop RF namespace at session
  setup, so `_ensure_library_registration` imports it on first use. Qualified
  names keep working; non-desktop sessions are unaffected.

## Capabilities

### New Capabilities
- `desktop-display-assertion-guidance`: PlatynUI display-state guidance leads
  with the reliable history-Label path; `native:Text.CharacterCount` demoted to
  a flagged secondary proxy; OCR named as last resort.
- `desktop-process-keyword-resolution`: unqualified `Process` keywords resolve
  in desktop sessions (keyword→library mapping and/or eager Process
  registration), so launching the AUT does not require dotted names.

### Modified Capabilities
<!-- The PlatynUI guidance and desktop routing live in not-yet-archived changes
     (desktop-stepwise-execution-fidelity, desktop-mcp-workflow-correctness), so
     these are new specs that compose with that work. -->

## Impact

- **Code**: `utils/rf_native_type_converter.py` (`get_platynui_locator_guidance`
  display-state section); `plugins/builtin/` (a `Process` keyword→library map,
  e.g. a small Process plugin or definition entry) and/or
  `components/execution/keyword_executor.py` /
  `models/session_models.py` (eager Process registration for desktop sessions).
- **Behavior**: agents reading desktop display state are pointed at the working
  history-Label path; unqualified `Process.Start Process`-style launches work in
  desktop sessions. No change to web/api/mobile sessions or to qualified calls.
- **Tests**: unit tests asserting the guidance now leads with history Labels and
  flags CharacterCount; a test that `get_library_for_keyword("Start Process")`
  resolves to `Process` and that a desktop session can register Process for an
  unqualified Process keyword; a regression check that non-desktop sessions are
  unchanged.
- **Dependencies/env**: builds on `desktop-stepwise-execution-fidelity`
  (ADR-029) and `desktop-mcp-workflow-correctness` (ADR-028). No new dependency.
- **Docs**: an ADR noting both fixes and the live-run evidence that motivated
  them (the GNOME Calculator agent run + `tests/e2e/...` artifacts).
