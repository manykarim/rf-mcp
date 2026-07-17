## 1. Reproduction (actionable findings)

- [x] 1.1 Red unit test: `analyze_scenario(..., context="desktop")` currently
  does not force desktop (the NLP processor lacks a `desktop` branch)
- [x] 1.2 Red unit test: `execute_batch` drops step `arguments` (only `args`
  is read) → "expected N arguments, got 0"
- [x] 1.3 Reproduce: `recommend_libraries` ranks Appium ahead of PlatynUI for a
  GNOME desktop scenario; `get_session_state` page_source uses the mobile
  source path for a desktop session; `find_keywords(PlatynUI.BareMetal)` returns
  zero matches (capture the cause for D4)

## 2. Desktop context routing + recommender

- [x] 2.0 (D0) Add `PlatynUI.BareMetal` to the library registry/recommender
  (desktop/gui categories + use-cases) and the `PlatynUI` → `PlatynUI.BareMetal`
  alias; prerequisite for 2.2 and group 5
- [x] 2.1 Force desktop routing at ALL three sites for `context="desktop"`:
  add a `desktop` branch in `nlp_processor.analyze_scenario`; a desktop override
  in `detect_platform_from_scenario` (so generic "app" does not win); thread
  context into `configure_from_scenario` (desktop profile + Process allow-list).
  Mirror `docs/issues/gnome_calculator_fix_plan.md`
- [x] 2.2 Rank `PlatynUI.BareMetal` ahead of `AppiumLibrary` in the recommender
  for desktop context/signals (gate strictly on desktop); search order
  PlatynUI-first
- [x] 2.3 Tests on the FULL `analyze_scenario` tool result: context=desktop →
  desktop session_type + Process allowed + PlatynUI-led search order;
  context=web unchanged (regression guard); recommender order correct

## 3. execute_batch argument compatibility

- [x] 3.1 Parse batch-step positional arguments from `args` OR `arguments`
  (`arguments` canonical); both-present-and-equal accepted; both-present-and-
  DIFFERENT → validation error (no silent shadowing)
- [x] 3.2 Update the `execute_batch` tool docstring to state both keys are
  accepted, `arguments` canonical, and conflicts are rejected
- [x] 3.3 Unit tests: `arguments`-only runs; `args`-only runs; equal-both runs;
  conflicting-both → validation error

## 4. Desktop state inspection routing

- [x] 4.1 Route `get_session_state` state/page-source for desktop sessions
  through the PlatynUI `ui_tree` path; never use the mobile-source lookup for a
  desktop session
- [x] 4.2 Return a clear desktop message when no app resolves (not "Failed to
  get mobile source: No application is open")
- [x] 4.3 Unit tests: desktop session → ui_tree path (no mobile lookup);
  web/mobile sessions unchanged; no-app message is desktop-appropriate

## 5. PlatynUI keyword discovery

- [x] 5.1 Make `find_keywords(library_name="PlatynUI.BareMetal")` (and the
  `PlatynUI` alias) list the library's keywords (catalog mode); keep `catalog`
  a literal substring filter but add a documented fallback/guidance so a
  natural-language query does not silently return 0
- [x] 5.2 Ensure desktop-session discovery guidance prioritizes PlatynUI
  desktop keywords over web/mobile
- [x] 5.3 Unit tests: library listing returns the interaction keywords; alias
  works; single-term intent query surfaces a desktop entry point; multi-word
  literal query returns guidance not a silent zero

## 6. Execution-environment consistency + Evaluate guidance

- [x] 6.1 Resolve a desktop launch/recovery executable to an absolute path via
  `shutil.which` against the SERVER-process PATH (after `desktop_launch_env`
  sanitization) before dispatching Process/Evaluate; do NOT inherit interactive
  shell startup state; surface the effective PATH on resolution failure
- [x] 6.2 Document `BuiltIn.Evaluate` expression-only behavior and point to a
  statement-capable alternative (`Run Process`) in guidance/instructions
- [x] 6.3 Tests/checks for executable resolution and the documented guidance

## 7. Stepwise-suite isolation

- [x] 7.1 Change the build model so pre-`start_test` exploratory steps are not
  silently adopted into the generated test body (exclude by default; surface
  count + an opt-in `include_pre_start`)
- [x] 7.2 Ensure `build_test_suite` emits real recorded desktop interactions +
  assertions, not `Log`-only placeholders, when interactions were recorded
- [x] 7.3 Make the start_test message explain handling (kept/isolated) instead
  of only warning about adoption
- [x] 7.4 Unit tests: pre-start steps excluded by default; opt-in includes
  them; generated suite contains interactions when recorded

## 8. Live e2e + validation + docs

- [x] 8.1 Live desktop e2e under the isolation bootstrap reproducing the report
  flow end-to-end: analyze(context=desktop) → init → discover PlatynUI keywords
  → inspect state → stepwise calculator interactions with per-entry + result
  assertions → build_test_suite (real interactions) → run
- [x] 8.2 Full unit suite + new tests green; confirm web/mobile/API flows
  unaffected
- [x] 8.3 Author an ADR mapping each report finding (1–10) to its fix/status;
  cross-reference the maintainer report
- [x] 8.4 Release notes for the workflow-correctness fixes (context routing,
  recommender, state inspection, discovery, batch args, exec env, suite
  isolation)
- [x] 8.5 REQUIRED deliverable: acceptance finding-matrix mapping each report
  finding (1–10) → fix decision (D0–D8) → status → covering test, in the ADR and
  as a resolution table appended to the maintainer report
