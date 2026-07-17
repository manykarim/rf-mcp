## 1. Reproduction (actionable findings)

- [x] 1.1 Red unit test: OBS-11 `_propagate_assigned_variables_to_literal_args`
  rewrites a later literal `"1"` into `${VAR}` when an earlier step captured the
  single character `"1"` (the `Keyboard Type ${None} 1` corruption, finding #9)
- [x] 1.2 Test: `_maybe_sanitize_desktop_launch` is NOT currently invoked on the
  execution path (finding #2 — dead sanitizer), then assert it IS after D1
- [x] 1.3 Reproduce: exploratory `Query`/`Evaluate` `assign_to` probes are
  serialized into the generated suite body (finding #10)

## 2. Suite recording fidelity (D5)

- [x] 2.1 Guard OBS-11: skip captures sourced from an introspection probe
  (`Evaluate`/`Query`) — the SOURCE keyword is the discriminator, not value
  length (refined during apply: a min-length guard wrongly broke legitimate
  single-char `Get Element Count → 5` dependencies)
- [x] 2.2 Preserve OBS-11 for genuine data dependencies (any length) from
  non-introspection keywords
- [x] 2.3 Unit tests: Evaluate/Query-captured value does not propagate; legit
  single-char dependency still propagates; multi-char dependency still propagates

## 3. Desktop launch alignment (D1, D2)

- [x] 3.1 Wire `_maybe_sanitize_desktop_launch` into the desktop branch of
  `_execute_keyword` for `Start Process`/`Run Process` of a known GUI binary
  (gate on `is_desktop_session()`; honor `platynui_no_sanitize`)
- [x] 3.2 Surface a Process-vs-discovery disagreement warning when a launched
  desktop process is not running shortly after success
- [x] 3.3 Provide a PURE input-effect decision helper (soft-warn when a
  successful interaction left state unchanged). NOTE (apply refinement):
  auto-probing under the execution lock risks deadlock and can't be CI-validated
  live, so the helper is exposed/tested; D2a launch-liveness IS wired
  non-reentrantly via the handle's `poll()`
- [x] 3.4 Unit tests: GUI launch gets sanitized env; non-GUI/non-desktop
  unchanged; launch-liveness hint (dead handle); input-effect helper
  (no-change/changed/missing)

## 4. Stepwise-suite hygiene (D6)

- [x] 4.1 Classify exploratory desktop introspection (`Query`, inspection-only
  `Evaluate`) as non-recorded for suite generation by default
- [x] 4.2 Dependency-aware retention: keep an introspection step whose assigned
  variable a retained later step consumes; report the filtered count
- [x] 4.3 Unit tests: introspection probes filtered; load-bearing captures
  retained; filtered count reported; generated suite compiles

## 5. Desktop isolation guidance (D3)

- [x] 5.1 Extend the active-desktop refuse payload with an actionable isolation
  recipe (Xvfb/`systemd-run` + `ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY`) and reframe
  the bypass env var as an escape hatch
- [x] 5.2 Unit tests: refusal includes the isolation recipe; bypass framed as
  escape hatch

## 6. Desktop driveability guidance (D4)

- [x] 6.1 PlatynUI guidance: `UiNode` attribute API (no `get_attribute`; use
  `attribute()`/`attributes()`+`value()`, `name`, `role`, `Get Attribute`)
- [x] 6.2 PlatynUI guidance: duplicate application-root / duplicate-control
  disambiguation (`Set Root`, prefer interactable node)
- [x] 6.3 PlatynUI guidance: control naming (symbol `Button[@Name='+']` vs word
  `Label[@Name='plus']`); discover names via ui_tree
- [x] 6.4 PlatynUI guidance: display-state reading via
  `native:Text.CharacterCount` length proxy on the entry `Text` node
- [x] 6.5 Unit tests: each guidance section present and correct

## 7. Desktop classification determinism (D7)

- [x] 7.1 Determinism tests: multiple phrasings of the GNOME scenario with
  `context="desktop"` all classify `desktop_testing`
- [x] 7.2 Document the explicit-desktop-context determinism guarantee in the
  `analyze_scenario` guidance/docstring

## 8. e2e + validation + docs

- [x] 8.1 e2e under the isolation bootstrap: the suite generated from the report
  flow is a clean validated calculator test (interactions + assertions, no
  exploratory `Query`/`Evaluate` probes, faithful recorded args)
- [x] 8.2 Full unit suite + new tests green; confirm web/mobile/API + existing
  desktop flows unaffected
- [x] 8.3 ADR mapping each new finding (1–10) → fix/status; cross-reference the
  maintainer report + trace
- [x] 8.4 Release notes for the stepwise-execution-fidelity fixes
- [x] 8.5 Acceptance finding-matrix (findings 1–10 → decision D1–D8 → status →
  covering test) in the ADR and appended to the maintainer report
