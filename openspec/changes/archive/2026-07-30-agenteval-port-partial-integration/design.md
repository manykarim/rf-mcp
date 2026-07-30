## Context

10 `partial` integration files (224 tests) each mix MCP-observable tests with internal-state tests. The
Phase-2a mechanics (support lib `mcp_result.py`, `rfmcp.resource`, `Rf Tool`, `\${VAR}` escaping,
per-suite server) apply unchanged. The new wrinkle is **splitting**: within a file, some tests port and
some stay.

## Goals / Non-Goals

**Goals:** port the MCP-observable subset of the high-ROI partials; trim the pytest originals to their
internal remainder; preserve coverage on both sides.

**Non-Goals:** the 5 internal-dominant partials (left whole); any internal-state test; changing what a
test asserts; desktop-in-CI (the platynui ports are gated).

## Decisions

**D1 — Per-file audit before porting.** For each high-ROI file, classify each test method as
MCP-observable (drives tools, asserts on results/traces) or internal-state (touches
`ExecutionSession`/components/mocks), reusing the Phase-2a audit criteria. Only the first group ports.
*Why:* the split boundary is per-test; guessing risks weakening an assertion. A parallel-draft agent does
the classification + port together, and a verify pass runs both the new `.robot` and the trimmed pytest.

**D2 — Trim, don't delete.** The pytest file loses only its ported tests; its internal-state tests (and
their fixtures/imports) stay. Remove now-unused imports/helpers only if nothing else references them.

**D3 — `adr009_schema_validation` first (likely near-full).** Its bulk asserts tool input-schema/enums,
which agenteval reads directly via `MCP.Get Tool Schema` / `MCP.Validate Tool Schema` (Tier-1, no server
spawn). It is the cleanest, highest-value split and validates the schema-keyword path — do it as the gate.

**D4 — Desktop-needing ports are gated.** `platynui_focus` / `platynui_newcore` MCP tests port but sit
behind `AGENTEVAL_DESKTOP` (like the gnome-apps port) — they need a real display to run; they skip
headless. Their pytest originals stay until desktop-verified.

## Risks / Trade-offs

- **[A test straddles the boundary]** (drives a tool AND asserts internals) → keep it in pytest; do not
  port a half-observable test.
- **[Trimming breaks a shared fixture]** → run the trimmed pytest file after each split; restore anything
  the remaining tests still need.
- **[Uneven payoff]** → accepted; the internal-dominant partials are explicitly excluded and recorded.

## Migration Plan

Per high-ROI file: audit → port the MCP subset → run the `.robot` green → trim the pytest file → run the
trimmed pytest green. Start with `adr009` (D3). Record the internal-dominant exclusions. Rollback per file
= restore the pytest tests until the port is green.

## Open Questions

- Does `adr009` split cleanly to near-100%, or does a residue of runtime-error tests stay pytest?
  **Resolved:** 100% — all 26 tests ported (23 live-schema + 3 runtime), pytest original deleted. Note the
  faithful path reads the *live* `MCP.List Tools` schema, not the config-declared `MCP.Get Tool Schema`.
- Are the `platynui_*` MCP subsets worth porting at all given they only run desktop-gated, or better left
  with the desktop suite? **Resolved: left whole (deferred).** Their MCP subsets need a live GNOME desktop
  (`Xvfb`+`systemd-run`, overlapping windows, live-app `ui_tree`) that this environment can't provide, so a
  port can't be verified green — and even the one seemingly-static test (`test_locator_guidance_dispatch`)
  returns a desktop-dependent shape headless. Shipping an unverifiable desktop suite risks a silent no-op
  (an earlier adversarial review caught exactly that), so per D4 they stay in pytest until a desktop-verified
  follow-up can port + verify them together. The three browser/selenium/adr009 splits (all verifiable here)
  are done and green.
