## Context

Phase 1 built the agenteval harness (`tests/agenteval/`, isolated env spawning rf-mcp as a subprocess).
A verified + audited scoping of `tests/integration/` (36 files, 610 tests) found **12 files / 129 tests**
that are pure black-box: they drive rf-mcp via an in-memory `Client(mcp)` and assert only on tool result
payloads and the recorded tool-call trace. This change ports those 12 to the harness.

Two realities shape the port:
- **Transport differs.** The pytest originals use an *in-process* `Client(mcp)` (same interpreter as the
  server object). The harness runs rf-mcp as a *subprocess over stdio* — higher fidelity (real protocol)
  but slower, and each `MCP.Start Server` is a fresh server process.
- **rf-mcp tool results are JSON-in-text.** A tool returns `MCPToolResult(content=[{type:"text",
  text:"<json>"}])`; the pytest tests parse that and assert on the dict. `.robot` must do the same.

## Goals / Non-Goals

**Goals:**
- Port the 12 clean candidates to `tests/agenteval/integration/*.robot`, preserving each test's observable
  assertions exactly.
- Retire each pytest original only once its port is green (no coverage gap, no long-lived duplication).
- Keep the ports in the keyless, always-on deterministic tier (they need no model).

**Non-Goals:**
- The 10 `partial` files (Phase 2b) and the 14 `stays-pytest` files (permanent boundary).
- Any production/runtime change, or re-specifying what a test checks.
- Forcing a candidate across if it turns out to need internal access.

## Decisions

**D1 — Subprocess transport (inherent, not a choice).** The isolated harness cannot hold rf-mcp's `mcp`
object in-process, so it spawns rf-mcp over stdio. Upside: tests exercise the real protocol exactly as an
agent does. Cost: process-spawn latency; accepted.

**D2 — A small result-assertion support library.** Add `tests/agenteval/integration/mcp_result.py` with
keywords like `Parse Tool Result` (→ dict from the JSON-in-text), `Result Field Should Be`,
`Result Should Contain Field`. Without it, every ported assertion is an inline `Evaluate json.loads(...)`
— unreadable. *Alternative rejected:* inline everywhere.

**D3 — One server per test, session_id threaded.** rf-mcp's keyword cache / execution context is
process-global, so a shared server across tests risks cross-test contamination (a Phase-0/demoshop trap).
Start a fresh server per test (Suite/Test setup) and thread the returned `session_id` through the call
sequence, mirroring the pytest per-test isolation. *Alternative considered:* per-suite server reuse for
speed — revisit only if the ergonomics gate (D5) shows the spawn cost dominates.

**D4 — Retire-when-green, in this change.** Port a file, confirm its `.robot` suite passes, then delete
the pytest original in the same change. Brief duplication during the port is fine; permanent duplication
is not, and a separate deletion follow-up (the Phase-1 2.4 shape) is avoided here because these ports are
1:1 and self-contained.

**D5 — Incremental with an ergonomics gate.** Port a representative pair FIRST — one trivial
(`test_fastmcp_context_keywords`, 1 test) and one multi-step with session continuity
(`test_mcp_e2e_builtin_only`, 21) — measure `.robot` assertion ergonomics and per-suite wall-clock, then
proceed. If a candidate reveals a hidden in-process assumption the audit missed, leave it in pytest and
report it (per the spec's non-translatable-stays-pytest requirement).

## Risks / Trade-offs

- **[Verbose payload assertions in `.robot`]** → D2 support library; deep-nested checks may still read
  worse than Python — the honest cost of black-box `.robot`.
- **[Subprocess-per-test CI time]** → 12 suites × per-test spawn adds real seconds; mitigate with D3-vs-
  reuse decision from D5, and these run in the always-on tier so they must stay reasonably fast.
- **[`test_platynui_gnome_apps_e2e` (20) needs a desktop]** → it is black-box over MCP but *running* it
  needs a display + GNOME apps; it cannot join the headless always-on tier. Port it, but gate it like the
  demoshop web scenario (opt-in / desktop-available), not always-on.
- **[A "clean" candidate hides an in-process assumption]** → D5's read-the-original + gate catches it;
  fall back to pytest.

## Migration Plan

Per file: (1) read the pytest original's tool calls + assertions; (2) write the `.robot` port
(server-per-test, threaded session_id, D2 helpers) asserting the same facts; (3) run it green; (4) delete
the pytest original. Do the D5 pair first. Rollback per file = keep the pytest original until green.

## Open Questions

- Per-test vs per-suite server reuse — decide from the D5 measurement.
- Where do desktop/display-needing ports (`platynui_gnome_apps`) run — a gated tier, or left in pytest if
  the display setup is not worth it in CI?
- How much support-keyword vocabulary to build in D2 before it is over-engineering.
