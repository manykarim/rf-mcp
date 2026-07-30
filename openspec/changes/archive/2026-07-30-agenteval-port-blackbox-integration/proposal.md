## Why

Phase 1 (`adopt-agenteval-harness`, archived) stood up the agenteval harness and ported the agentic e2e
layer. A verified, adversarially-audited scoping of `tests/integration/` then found **12 files (129
tests) that are pure black-box MCP-surface** — they drive rf-mcp through an in-memory `Client(mcp)` and
assert *only* on tool result payloads and the recorded tool-call trace, never on internal Python. Those
are the clean Phase-2a candidates.

Porting them to the agenteval harness continues the consolidation the harness adoption committed to:
dogfood rf-mcp with Robot Framework, unify the black-box test surface under one harness, and raise
fidelity — the ported suites drive rf-mcp as a **real subprocess over stdio**, the way an agent actually
uses it, rather than an in-process client object. It also shrinks pytest's integration layer toward what
genuinely needs internal access.

## What Changes

- **Port the 12 clean black-box candidates** to agenteval `.robot` suites under `tests/agenteval/integration/`:
  `test_adr010_e2e` (24), `test_mcp_e2e_builtin_only` (21), `test_adr_integration` (20),
  `test_platynui_gnome_apps_e2e` (20), `test_variable_file_loading` (14), `test_variable_handling_e2e` (9),
  `test_library_preferences` (6), `test_intent_fallback_e2e` (5), `test_keyword_routing_e2e` (5),
  `test_fastmcp_argument_resolution` (3), `test_fastmcp_context_keywords` (1),
  `test_recommend_libraries_keywords` (1).
- **Preserve coverage exactly.** Each ported test asserts the same observable behavior (the same tool
  results / trace facts) as its pytest original — a translation, not a re-specification.
- **Retire the pytest original only when its port is green.** No file is deleted from `tests/integration/`
  until its agenteval suite passes, so there is never a coverage gap.
- **Incremental, ergonomics-first.** Port one or two files first to confirm the `.robot` payload-assertion
  ergonomics and the subprocess speed cost, then proceed; a file whose assertions do not translate cleanly
  is left in pytest and reported, not force-fit.
- **Scope boundary (NON-goals).** Only the 12 clean candidates. The 10 `partial` files (224 tests — an MCP
  subset plus an internal remainder) are a later **Phase 2b** (split per file); the 14 `stays-pytest` files
  (257 tests — batch/instruction domains, attach-bridge harnesses, session/component internals) are the
  permanent pytest boundary and are out of scope.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `agenteval-test-harness`: add the migration-integrity contract — a black-box integration test moved to
  the harness preserves its coverage and its pytest original is retired only once the agenteval port is
  green.

## Impact

- **New**: `tests/agenteval/integration/` `.robot` suites (the 12 ports), run by the existing runner/CI job.
- **Removed**: 12 files / 129 tests from `tests/integration/` (once each port is green).
- **CI**: the ports are keyless black-box, so they join the **deterministic, always-on** tier of the
  `agenteval-harness` job (no model key). Slower than the in-memory pytest originals (subprocess spawn).
- **No production code changes. No MCP tool names, parameters, or return shapes change.**
- **Honest trade-off**: `.robot` assertions on deeply-nested tool-result payloads are more verbose than
  Python `dict`/`Should Contain` assertions, and subprocess-per-suite is slower than the in-memory client
  — the win is harness unification + real-protocol fidelity + dogfooding, not fixing broken tests.
