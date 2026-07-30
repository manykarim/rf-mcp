## 1. Gate split: adr009 (schema keywords)

- [x] 1.1 Audited `test_adr009_schema_validation`: 100% MCP-observable (23 schema-constraint tests read the
  live advertised `inputSchema`; 3 runtime tests call the tool). Ported all 26 to
  `tests/agenteval/integration/test_adr009_schema_validation.robot`. Note: `MCP.Get Tool Schema`/`Validate
  Tool Schema` read a *statically-declared* `.mcp.json:tools` map, NOT the live FastMCP schema — so the
  faithful port reads the live `MCP.List Tools` output via new schema-inspection helpers in `mcp_result.py`.
- [x] 1.2 Port green (26/26). Fully ported → pytest original deleted (494 integration tests collect, was 520).
  Runtime-rejection tests mirror the pytest's try/except leniency: an invalid enum is rejected by the server
  *raising* `Invalid arguments` at the schema boundary (the observable rejection).

## 2. Split the remaining high-ROI partials (each: audit -> port MCP subset -> green -> trim pytest -> green)

- [x] 2.1 `test_real_browser_prevalidation`: ported 13 MCP-observable tests (KeywordExecution ×3,
  PageSourceService ×2, intent_action(extract) OBS-06 ×7, Drag-And-Drop OBS-10 ×1) to
  `test_real_browser_prevalidation.robot` — all green locally (headless Chromium). Trimmed pytest keeps 11:
  `TestBrowserPreValidation` (5, internal `_pre_validate_element`) + OBS-01 (6, straddles: drives a tool
  THEN asserts internal verdicts). Suite skips cleanly when Chromium isn't provisioned. Unused `typing`
  import dropped.
- [x] 2.2 `test_real_selenium_prevalidation`: ported 6 MCP-observable tests (KeywordExecution ×4 incl. the
  P0 no-timeout-injection Click Element, PageSourceService ×2) to `test_real_selenium_prevalidation.robot` —
  all green locally (headless Chrome). Trimmed pytest keeps 4 (`TestSeleniumPreValidation`, internal). Each
  ported suite spawns its own rf-mcp subprocess, so SeleniumLibrary/Browser never coexist (the exclusion
  group that forced separate pytest invocations is moot here).
- [~] 2.3 `test_platynui_focus_e2e` — DEFERRED (desktop-verification-blocked). Audited: 4 payload-asserting
  tests (`test_operation_targets_aut_not_other_window`, `test_focus_recovers_after_other_window_raised`,
  `test_focus_outcome_surfaced_on_step`, `test_ui_tree_reports_window_visibility`) are MCP-observable but
  require a live overlapping-two-GNOME-apps desktop launched via `Xvfb`+`systemd-run --user`;
  `test_escape_hatch_bypasses_focus` reaches into `execution_engine.session_manager` (internal, stays). A
  port can only run desktop-gated, and this headless environment has no such desktop — so a port cannot be
  verified green here. Per D4 (pytest originals stay until desktop-verified) and the Risk rule (don't ship
  an unverifiable port), left whole in pytest; recorded for a desktop-verified follow-up.
- [~] 2.4 `test_platynui_newcore_e2e` — DEFERRED (desktop-verification-blocked). `test_locator_guidance_dispatch`
  runs headless but returns a *different shape* there (`element_not_found_suggestions` absent) — so it is not
  faithfully portable without the desktop context it asserts; `test_full_desktop_workflow` also asserts an
  in-process `XDG_SESSION_TYPE` env side-effect (not an MCP-surface fact, straddles); `test_ui_tree_expansion_with_filter`
  needs a live desktop `ui_tree`. Left whole in pytest, recorded for the same desktop-verified follow-up.

## 3. Record the internal-dominant exclusions

- [x] 3.1 Documented in `tests/agenteval/README.md` (new "Phase-2b split" section): the split table
  (adr009/browser/selenium), the 5 internal-dominant files left whole (`test_architecture_improvements`,
  `test_nlp_improvements`, `test_library_loading_improvements`, `test_sampling_feature_flag`,
  `test_real_page_source_routing`), and the 2 deferred platynui files (desktop-verification-blocked).

## 4. CI + verify

- [x] 4.1 The keyless ports (adr009 26, browser 13, selenium 6) join the always-on tier. Added `rfbrowser init`
  to the `agenteval-harness` CI job so the browser port runs for real (selenium uses the runner's Chrome).
  Verified locally: full keyless harness green — 179 tests, 156 passed, 0 failed, 23 skipped (agentic suites
  skip without a key; the desktop gnome-apps port skips cleanly headless).
- [x] 4.2 Trimmed files collect + pass: browser 11/11 (5 internal + 6 OBS-01), selenium 4/4 (internal); 494
  integration tests collect after the adr009 deletion. Coverage preserved: adr009 26=26; browser 13 ported +
  11 kept = 24 = original; selenium 6 ported + 4 kept = 10 = original.

## 5. Wrap-up

- [x] 5.1 `openspec validate agenteval-port-partial-integration --strict` passes.
- [x] 5.2 Report:
  - **Ported (verified green here):** adr009 26 (all; original deleted), browser 13, selenium 6 = **45 tests**
    now in `integration/*.robot`, driving the live rf-mcp over MCP.
  - **Retained in pytest (internal/straddle, from split files):** browser 11 (`_pre_validate_element` + OBS-01
    straddle), selenium 4 (`_pre_validate_element`).
  - **Left whole (internal-dominant, MCP slice too small):** `test_architecture_improvements`,
    `test_nlp_improvements`, `test_library_loading_improvements`, `test_sampling_feature_flag`,
    `test_real_page_source_routing`.
  - **Deferred (desktop-verification-blocked):** `test_platynui_focus_e2e`, `test_platynui_newcore_e2e` — MCP
    subsets need a live GNOME desktop no headless env provides; not shipped unverified (see design Open
    Questions + README).
