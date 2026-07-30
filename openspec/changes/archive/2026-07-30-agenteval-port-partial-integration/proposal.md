## Why

Phase 2a ported the 10 *fully* black-box integration files. The audit also found 10 **partial** files
(224 tests): each mixes MCP-observable tests (drive rf-mcp, assert on tool results/traces) with
internal-state tests (assert on `ExecutionSession`/components/mock call-args). Phase 2b splits the ones
worth splitting — porting the MCP-observable subset to the harness and leaving the internal remainder in
pytest — so more of rf-mcp's black-box surface is covered by the harness without weakening any assertion.

Not every partial is worth splitting: five are **internal-dominant** (a small MCP slice inside a large
internal-logic suite), where a split yields little and just fragments the file. Phase 2b targets the
**high-ROI** partials only.

## What Changes

- **Split the high-ROI partials** — port their MCP-observable tests to `tests/agenteval/integration/` and
  delete only those tests from the pytest original (trim, not delete the file):
  - `test_adr009_schema_validation` (26) — tool-schema/enum assertions map to agenteval's
    `MCP.Get Tool Schema` / `MCP.Validate Tool Schema` (Tier-1 config keywords); likely near-fully portable.
  - `test_real_browser_prevalidation` (20) and `test_real_selenium_prevalidation` (10) — the
    `call_tool`/`intent_action` tests port; the `executor._pre_validate_element` internal-tuple tests stay.
  - `test_platynui_focus_e2e` (5) and `test_platynui_newcore_e2e` (3) — the payload-asserting MCP tests
    port (desktop-gated like the gnome-apps port); the session/env-side-effect assertions stay.
- **Leave the internal-dominant partials whole in pytest** (documented, not silently dropped):
  `test_architecture_improvements` (57), `test_nlp_improvements` (46), `test_library_loading_improvements`
  (21), `test_sampling_feature_flag` (22), `test_real_page_source_routing` (14) — their MCP slice is too
  small to justify splitting.
- **Preserve coverage + retire-when-green**, per the `agenteval-test-harness` migration-integrity
  requirement: each ported test asserts the same observable facts; a pytest test is removed only after its
  port passes; anything not expressible over the MCP surface stays in pytest.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `agenteval-test-harness`: add the split rule — a partial file's MCP-observable tests migrate while its
  internal-state tests remain in a trimmed pytest file.

## Impact

- **New/updated**: agenteval `.robot` ports for the high-ROI partial subsets; the corresponding pytest
  files trimmed to their internal-state remainder.
- **CI**: keyless ports join the always-on tier; desktop-needing ports (`platynui_*`) are `AGENTEVAL_DESKTOP`-gated.
- **Unchanged**: internal-dominant partials, all internal-state tests, MCP tools/params/returns.
- **Honest note**: splitting is per-test judgment and the payoff is uneven — this change deliberately
  stops at the high-ROI partials rather than force-splitting every file.
