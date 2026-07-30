## 1. Harness scaffolding (isolated env)

- [x] 1.1 Add a pinned `robotframework-agenteval[all]==<ver>` declaration in an ISOLATED test group (its own dependency group / env — NOT rf-mcp runtime or the main dev set).
- [x] 1.2 Add a launch descriptor for the harness: a `.mcp.json` (or a documented `command`/`args`) that spawns rf-mcp from its own venv (`.venv/bin/robotmcp`) over stdio.
- [x] 1.3 Land the two proven feasibility spikes as the first suites: a deterministic MCP-surface suite (list tools + call `analyze_scenario`) and an agentic suite (in-process adapter drives rf-mcp), under a new `tests/agenteval/` directory.
- [x] 1.4 Add a runner entry point (`uv run --no-project --with 'robotframework-agenteval[all]' robot tests/agenteval/`) and document how to run keyless vs with a model key.

## 2. Phase 1 — port the agentic e2e layer

- [x] 2.1 Choose the Phase-1 port set of agentic scenarios (resolve the design Open Question) and record it.
- [x] 2.2 Rebuild each chosen scenario as an agenteval `.robot` suite: spawn rf-mcp, `MCP.As Agent Toolset`, drive via the in-process (and/or CLI) adapter, assert tool-call trace + `Metric.*` budgets + `Stat.*` pass@k.
- [x] 2.3 Confirm each ported suite reaches parity with the bespoke original (same behavior asserted, same or better fidelity), running against the MiniMax endpoint.
- [ ] 2.4 Remove the superseded bespoke modules under `tests/e2e/` (agent_integration, metrics_collector, quality_gate, model_comparison, minimax_support, tracked_client, and the CLI runner) only after their scenarios are green on agenteval; keep anything still referenced by un-ported tests.

## 3. CI wiring

- [x] 3.1 Add a CI job that installs agenteval in an isolated env and runs `tests/agenteval/` deterministic suites on every change (no key required).
- [x] 3.2 Gate the agentic suites behind a model-credential check so they run when a key is present and skip cleanly otherwise (mirror the current e2e credential gate).

## 4. Scope boundary + docs

- [x] 4.1 Document the boundary: unit tests, internal-state integration tests, and benchmarks stay in pytest; agenteval owns the MCP-surface + agentic layers.
- [x] 4.2 Add a short "how the harness works" doc (isolation model, launch descriptor, tiers, running locally) and link it from the test README.

## 5. Wrap-up

- [x] 5.1 `openspec validate adopt-agenteval-harness --strict` passes.
- [ ] 5.2 Full pytest suite still green (nothing removed that a pytest test depended on); the agenteval deterministic tier green in CI.
- [x] 5.3 Record the outcome (lines of bespoke harness removed, scenarios ported) and note Phase 2 (black-box MCP-surface integration migration) as a follow-up change — NOT implemented here.
