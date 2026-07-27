# agenteval harness

Black-box tests for the **rf-mcp MCP server**, authored with
[`robotframework-agenteval`](https://github.com/manykarim/robotframework-agenteval).
The suites spawn `robotmcp` as a subprocess and speak the real MCP protocol to it —
so rf-mcp is exercised exactly as an agent uses it.

Change: `adopt-agenteval-harness` (Phase 1). This is the harness foundation; scenario
porting from the bespoke `tests/e2e/` machinery is a follow-up (see that change's tasks).

## Scope — what this harness owns, and what it does NOT

This harness owns the **MCP-server-surface** and **agentic e2e** layers only:

- ✅ deterministic MCP-surface checks (spawn → handshake → tool surface → tool results)
- ✅ agentic e2e (a real agent drives rf-mcp; tool-call trace + token/cost/latency metrics)

It does **not** replace, and this directory must not grow to include:

- ✗ unit tests (`tests/unit/`) — internal Python (session models, converters, discovery)
- ✗ integration tests that assert on rf-mcp's internal objects (`tests/integration/`)
- ✗ performance benchmarks (`tests/benchmarks/`)

Those stay in **pytest**. A test belongs here only if its assertions are expressible over
the MCP surface: tool results, recorded tool-call traces, and run metrics.

## Isolation model

```
uv run --no-project --with-requirements requirements.txt robot ...   <-- agenteval's OWN
   │  (ephemeral env: RF 7.4.2, mcp SDK, pydantic-ai — all agenteval's exact pins)         env
   │
   └── spawns  .venv/bin/robotmcp   (subprocess, stdio MCP)  ---------> rf-mcp's OWN env
```

agenteval's pinned dependencies never enter rf-mcp's runtime or dev environment; rf-mcp
runs from its own `.venv`. That is why agenteval's exact pins can't conflict with rf-mcp's.

## Tiers

| Suite | Tier | Needs a model key? |
|---|---|---|
| `deterministic_mcp_surface.robot` | 1 — deterministic | no — always-on gate |
| `agentic_e2e.robot` | 3 — real in-process agent | yes; skips cleanly without one |

## Running

```bash
# Deterministic tier — no key needed:
tests/agenteval/run.sh deterministic_mcp_surface.robot

# Agentic tier — set a model credential (read from the env, never a RF variable):
export AGENTEVAL_API_KEY=...                      # e.g. your MiniMax key
export AGENTEVAL_BASE_URL=https://api.minimax.io/v1
export AGENTEVAL_MODEL=MiniMax-M3
tests/agenteval/run.sh agentic_e2e.robot

# Whole harness (agentic self-skips without a key):
tests/agenteval/run.sh
```

Results land in `tests/agenteval/results/` (git-ignored).

## Findings from the Phase-1 port (evaluation notes)

Porting the two scenarios (`scenarios/*.yaml`, driven by MiniMax-M3) surfaced two concrete
constraints worth knowing before leaning on the harness:

1. **In-process request cap (agenteval 0.3.0).** The `in-process` adapter runs
   `agent.run(prompt)` with pydantic-ai's default `request_limit=50` and exposes **no override**.
   Short scenarios pass (demoshop: 22 tool calls, hit-rate 0.75); a long one hits it —
   restful-booker's read+create+auth+delete workflow raises `UsageLimitExceeded` past 50 requests.
   Because *running* it burns ~50 live requests only to fail on the cap, the restful-booker test is
   **skipped by default** and opted in with `AGENTEVAL_ALLOW_LONG=1` (a `UsageLimitExceeded` at
   runtime is still caught and downgraded to a skip). Real fix: an upstream usage-limit knob, or
   drive long scenarios through a coding-agent **CLI adapter** (no pydantic-ai default cap).
2. **Tier-3 measures tool usage, not automation success.** The hit-rate gate (like the bespoke
   quality gate) asserts the agent *selected and called* the right tools — it is robust to page
   state. It does **not** assert the web automation succeeded. The demoshop prompt uses
   `headless=False`, so on a headless runner Chromium can't launch (Missing X server); the agent
   still met the tool gate. For real browser automation the web scenario needs a display (Xvfb) or
   a headless variant. That is why the web suite is opt-in (`AGENTEVAL_WEB=1`).

## Upgrading agenteval

The pin lives in `requirements.txt`. agenteval's libraries are `provisional` (v0.3.x) —
bumping the version is a deliberate change; check its CHANGELOG for breaking changes first.
