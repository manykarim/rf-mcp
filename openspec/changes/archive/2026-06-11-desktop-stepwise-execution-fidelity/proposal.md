## Why

A re-run of the GNOME Calculator stepwise scenario (after the
`desktop-mcp-workflow-correctness` fixes) produced a new maintainer report
(`docs/gnome-calculator-mcp-maintainer-report.md`) and a captured trace
(`tests/e2e/gnome_calculator_mcp_stepwise_trace.robot`). Routing is now correct
— the scenario classifies as `desktop_testing` and PlatynUI is recommended —
but the agent **still never completes a real calculation**. It is forced into
raw `Evaluate` Python introspection of `UiNode` objects, hits a launch/runtime
mismatch, gets blocked by the active-desktop guard, can't tell which of three
duplicate controls to click, observes input keywords "succeed" without changing
the app, and finally builds a suite that is a debugging trace rather than a
validated test. Two of the new findings are reproducible directly in code:

- **OBS-11 false substitution (finding #9):** the recorded
  `Keyboard Type ${None} 1` becomes `Keyboard Type ${None} ${active_desktop_override}`
  because the override step captured the string `"1"` and
  `_propagate_assigned_variables_to_literal_args` rewrites the coincidentally
  equal single-character literal. Reproduced.
- **Dead snap-sanitization (finding #2):** `_maybe_sanitize_desktop_launch`
  (which builds a snap-decontaminated child env for a GUI `Start Process`)
  exists but is never wired into the execution path, so a snap-confined launch
  can exit `127` while PlatynUI still sees stale/other accessibility nodes.

## What Changes

- **Desktop launch alignment (findings #2, #4).** Wire the existing
  snap-sanitized launch env into desktop `Start Process` of a GUI binary, and
  detect/surface a Process-vs-discovery disagreement (a launched process that
  exits non-zero quickly, or input keywords that return success without any
  accessible-state change) so "OK" stops being mistaken for "the app reacted".
- **Actionable desktop isolation guidance (finding #3).** The active-desktop
  refuse path returns a concrete, actionable isolation recipe (and a discovery
  pointer), instead of only naming the bypass env var, so a desktop scenario in
  a normal session has a guided path to an isolated display rather than manual
  recovery.
- **Desktop driveability guidance (findings #5, #6, #7, #8).** Expand the
  PlatynUI guidance so agents stop guessing through `Evaluate`: the correct
  `UiNode` attribute API (`node.attribute(...)`, `node.attributes()` +
  `attr.value()`, `node.name`, `node.role` — NOT `get_attribute`), duplicate
  application-root / duplicate-control disambiguation (multiple live roots on
  Wayland/AT-SPI; how to pick the interactable node), control naming (operator
  keys are symbol `Name`s like `+`, while labels read as words like `plus`),
  and how to read the calculator's display state (which `Text` node, and the
  `native:Text.CharacterCount` binding).
- **Suite recording fidelity (finding #9).** Constrain the OBS-11
  literal→variable propagation so it does not rewrite coincidental short/ambiguous
  literals (e.g. a single digit/character) nor values captured by an
  introspection/`Evaluate` probe, so the generated suite faithfully reflects the
  executed intent.
- **Stepwise-suite hygiene (finding #10).** Keep exploratory desktop
  introspection (`Query`/`Evaluate` probes used only to inspect the tree) out of
  the generated suite body by default, so `build_test_suite` serializes
  validated intent — real interactions and assertions — not investigation
  history. Surface what was filtered.
- **Desktop classification determinism (finding #1).** Make an explicit
  `context="desktop"` deterministic regardless of prompt phrasing, and document
  the determinism guarantee, so the same scenario class does not flip between
  desktop and mobile across attempts.

## Capabilities

### New Capabilities
- `desktop-launch-alignment`: snap-sanitized GUI launch + Process-vs-PlatynUI
  discovery disagreement detection + input-effect (success≠mutation) signalling.
- `desktop-isolation-guidance`: actionable isolation recipe on the
  active-desktop refuse path instead of only naming the bypass env var.
- `desktop-driveability-guidance`: PlatynUI guidance for the UiNode attribute
  API, duplicate-root/control disambiguation, control naming, and display-state
  reading so agents stop reverse-engineering via `Evaluate`.
- `suite-recording-fidelity`: OBS-11 literal→variable propagation guarded
  against coincidental short/ambiguous and introspection-sourced captures.
- `stepwise-suite-hygiene`: exploratory desktop introspection excluded from the
  generated suite body by default, with a report of what was filtered.
- `desktop-classification-determinism`: explicit `context="desktop"` is
  deterministic and documented, independent of prompt phrasing.

### Modified Capabilities
<!-- The desktop classification/routing (desktop-mcp-workflow-correctness) and
     the safety guard (platynui-desktop-safety-isolation) are not yet archived
     openspec capabilities, so these are new specs that compose with — and
     depend on — that work. -->

## Impact

- **Code**: `keyword_executor.py` (wire `_maybe_sanitize_desktop_launch`;
  Process/discovery disagreement + input-effect signalling); `test_builder.py`
  (OBS-11 guard; exploratory-introspection suite hygiene);
  `desktop_display_safety.py` (actionable refuse message);
  `rf_native_type_converter.py` (PlatynUI driveability guidance);
  `nlp_processor.py`/`session_models.py` (deterministic explicit desktop
  context — verification + docs).
- **Behavior**: desktop GUI launches are snap-decontaminated; an input keyword
  that does not change the app is flagged; the active-desktop refusal guides the
  user to isolation; generated desktop suites contain validated interactions,
  not a debugging trace; recorded args reflect what was executed. No change to
  web/mobile/API flows.
- **Tests**: unit tests for the OBS-11 guard (single-char literal not
  substituted; introspection-source not propagated), launch-env wiring,
  refuse-message content, suite-hygiene filtering, and the guidance payloads;
  plus an e2e assertion that the generated suite from the report flow is a clean
  validated suite rather than an exploratory trace.
- **Dependencies/env**: builds on `desktop-mcp-workflow-correctness` (ADR-028),
  `platynui-desktop-safety-isolation` (ADR-027), ADR-025/026. No new dependency.
- **Docs**: an ADR mapping each new finding (1–10) to its fix; the maintainer
  report + trace referenced as the source of record.
