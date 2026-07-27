## Context

rf-mcp's test suite is ~4,635 test functions: unit (3,639, 79%), integration (611, 13%), e2e (39, 1%),
benchmarks (303, 7%). The e2e layer is thin in test count but heavy in **bespoke harness code**
(~4,790 lines under `tests/e2e/`): a Copilot CLI runner, a quality gate, model comparison, an agent
loop, MiniMax wiring, a metrics collector, a tracked MCP client. `robotframework-agenteval` (v0.3.0,
PyPI, same author) provides that machinery as maintained keyword libraries — `MCPLibrary`,
`MetricsLibrary`, `StatLibrary`, the in-process pydantic-ai adapter, and vendor CLI adapters — built on
the same pydantic-ai + MiniMax stack rf-mcp already depends on.

Two feasibility spikes (this exploration) drove rf-mcp 0.35.0 from an isolated env and passed:
deterministic (spawn → handshake → 19 tools → `analyze_scenario`) and agentic (MiniMax-M3 → 8 real tool
calls + token metrics). No dependency conflict; the MiniMax `service_tier` quirk did not surface.

## Goals / Non-Goals

**Goals:**
- Make agenteval the default harness for two layers: black-box MCP-server-surface tests and agentic e2e.
- Retire the bespoke `tests/e2e/` harness in favor of maintained agenteval keywords.
- Test rf-mcp as an agent actually uses it: spawned subprocess, real MCP protocol.
- Keep agenteval fully isolated from rf-mcp's own dependency graph.

**Non-Goals:**
- Replacing unit tests, internal-state integration tests, or benchmarks (they stay pytest).
- Any production/runtime change to rf-mcp, its tools, or their schemas.
- Claiming a specific coding agent's fidelity from the in-process proxy (see Risks).
- A big-bang cutover — adoption is phased and reversible per phase.

## Decisions

**D1 — Isolation over integration.** agenteval runs in its OWN environment (`uv run --no-project --with
'robotframework-agenteval[all]'` or a dedicated venv) and spawns rf-mcp as a subprocess via its launcher
(`.venv/bin/robotmcp`, or a `.mcp.json` descriptor). *Why:* agenteval exact-pins `robotframework==7.4.2`,
`mcp==1.27.1`, `pydantic-ai==2.12.0`; merging those into rf-mcp's env risks conflicts and couples two
release cadences. The spike proved the isolated model (33 packages in a throwaway env, rf-mcp from its
own venv). *Alternative rejected:* add agenteval to rf-mcp's dev group — simpler imports, but drags the
pins in and defeats the black-box fidelity.

**D2 — Phase by value, retire glue first.** Phase 1 = the agentic e2e layer (biggest win: it replaces
~4.8k lines of glue with keyword calls, and the assertions are already "did the agent call the right
tools / stay in budget", which map 1:1 to agenteval readers). Phase 2 = the black-box MCP-surface subset
of integration tests (a genuine rewrite from `fastmcp.Client` pytest to `.robot`). *Why phase:* limits
blast radius, lets each phase prove out before the next, and keeps a working suite throughout.
*Alternative rejected:* migrate MCP-surface integration first — more rewrite for less consolidation gain.

**D3 — Black-box boundary is the migration filter.** A test moves to agenteval only if its assertions are
expressible over the MCP surface (tool results, tool-call traces, run metrics). Any test that asserts on
internal Python state stays pytest. *Why:* agenteval has no hook into rf-mcp internals by design; forcing
those tests across would mean weakening assertions.

**D4 — Pin exact, upgrade deliberately.** agenteval's libraries are `provisional`. Pin an exact version;
upgrades are reviewed against its CHANGELOG. *Why:* provisional means minor-version breaks are allowed;
an unpinned test harness would flake on upstream releases.

**D5 — Model config reuses rf-mcp's proven MiniMax setup.** `AGENTEVAL_MODEL=MiniMax-M3`,
`AGENTEVAL_BASE_URL=https://api.minimax.io/v1`, key from the environment — the same endpoint rf-mcp's
harness already validates. *Why:* known-good, and keys never enter Robot Framework variables (agenteval
reads them from the process env, keeping `log.html` clean).

## Risks / Trade-offs

- **[agenteval is v0.3.0 / provisional]** → Pin exact; adopt incrementally; the deterministic tier (most
  of the value) depends on the most stable keywords.
- **[`.robot` is less expressive than pytest for deep assertions]** → That's the boundary in D3 —
  black-box tests only; complex Python assertions stay pytest.
- **[In-process adapter is a proxy, not a specific agent's runtime]** → rf-mcp's current harness already
  uses the same pydantic-ai proxy, so there is no fidelity regression; vendor-specific claims use the CLI
  adapters (Copilot etc.), which rf-mcp already exercises.
- **[Rewrite effort for Phase 2]** → Bounded by D3 (only the black-box subset), and deferred behind the
  Phase 1 payoff; a scenario-by-scenario translation, not a rewrite-everything.
- **[Two CIs / cadences]** → The isolated env is created on demand in CI (`uv run --with`), so there is no
  second lockfile to maintain; the cost is a per-job install (~ms in the spike).

## Migration Plan

Phase 1 (this change's concrete deliverable): stand up the agenteval harness, port a representative set
of agentic e2e scenarios, delete the superseded bespoke modules, wire the CI job. Phases 2+ are roadmap,
each its own follow-up change. Rollback per phase = keep the pytest originals until the agenteval port is
green, then remove.

## Phase-1 outcome (recorded 2026-07-27)

Foundation stood up under `tests/agenteval/` (isolated pinned harness, launch descriptor,
deterministic + agentic suites, runner, CI job) — deterministic tier **3/3 green**, pytest collection
unaffected. Port set = demoshop (web) + restful-booker (API), driven live by MiniMax-M3:

- **demoshop (web): parity reached** — 22 tool calls, hit-rate 0.75 ≥ gate 0.70.
- **restful-booker (API): blocked upstream** — exceeds agenteval 0.3.0's in-process
  `request_limit=50` (no override in that release); handled as a documented skip. Fix path: an
  upstream usage-limit knob, or a coding-agent CLI adapter (no pydantic-ai default cap).

Two findings recorded in `tests/agenteval/README.md`: the in-process request cap, and that the Tier-3
hit-rate gate measures tool *usage*, not automation success (web scenario needs a real display).

**Deletion (task 2.4) deliberately NOT done** — those bespoke modules serve more scenarios than the two
ported here, so removal waits on a fuller port. **Phase 2** (black-box MCP-surface integration
migration) remains a separate follow-up change, not implemented here.

## Open Questions

- Which agentic e2e scenarios are the Phase-1 port set (demoshop? restful-booker? gnome-calculator)?
- Does the CI agentic tier keep MiniMax, or move to a pinnable self-hostable reference model for
  determinism?
- Do the black-box integration tests migrate as one Phase-2 change, or per-subsystem?
