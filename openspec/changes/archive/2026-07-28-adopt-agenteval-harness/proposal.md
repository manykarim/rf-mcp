## Why

rf-mcp's `tests/e2e/` carries **~4,790 lines of bespoke agentic-evaluation machinery** — a Copilot CLI
runner (616), a quality gate (587), model comparison (330), an agent-integration loop (263), MiniMax
wiring (250), a metrics collector (195), a tracked MCP client (101), and more. `robotframework-agenteval`
(v0.3.0, same author, built on the *same* pydantic-ai + MiniMax stack rf-mcp already uses) now provides
every one of those as a maintained, independently-tested Robot Framework keyword library. rf-mcp is
re-implementing, and re-patching, a harness that already exists as a package.

Adopting agenteval as the default harness for rf-mcp's **MCP-server-surface** and **agentic e2e** tests
(1) sheds that bespoke glue, (2) tests rf-mcp exactly as a real agent uses it — spawned as a subprocess
over the real MCP protocol, higher fidelity than the current in-process `fastmcp.Client` tests, and
(3) dogfoods rf-mcp's own value proposition (driving MCP servers from Robot Framework).

Feasibility is **proven**, not assumed: spikes drove rf-mcp 0.35.0 from an isolated env — deterministic
(spawn → handshake → list 19 tools → `analyze_scenario`, 2/2 pass) and agentic (MiniMax-M3 → 8 real tool
calls `analyze_scenario, manage_session, execute_batch×3, execute_step×3`, token metrics off the trace,
1/1 pass). The MiniMax `service_tier` quirk rf-mcp's harness patches did not affect agenteval.

## What Changes

- **Adopt agenteval as the default harness for two test layers**: black-box MCP-server-surface tests
  (spawn rf-mcp, call tools, assert on results and tool-call traces) and agentic e2e scenarios (a real
  in-process or CLI agent drives rf-mcp; deterministic readers project tool calls, metrics, pass@k).
- **Isolation model.** agenteval runs in its own environment and spawns rf-mcp as a subprocess via its
  launcher. agenteval's exact pins (`robotframework==7.4.2`, `mcp==1.27.1`, `pydantic-ai==2.12.0`) never
  enter rf-mcp's own runtime dependencies.
- **Phased, not big-bang.** Phase 1: rebuild the agentic e2e layer on agenteval (the biggest win, lowest
  rewrite — it replaces glue, not assertions), retiring the bespoke harness. Phase 2: migrate the
  black-box MCP-surface subset of the integration tests to agenteval `.robot` suites.
- **Explicit scope boundary (NON-goal).** Unit tests, internal-state integration tests, and performance
  benchmarks REMAIN in pytest. agenteval is a black-box MCP/agentic library; it does not — and this
  change will not attempt to — replace tests of internal Python.
- **Pin the dependency.** agenteval's four libraries are `provisional` (v0.3.x); the change pins an exact
  version and gates upgrades on its CHANGELOG.

## Capabilities

### New Capabilities
- `agenteval-test-harness`: the contract for how rf-mcp's MCP-server-surface and agentic e2e tests are
  authored and executed — via agenteval, in an isolated env, spawning rf-mcp as a subprocess — and the
  explicit boundary of what stays in pytest.

### Modified Capabilities
<!-- none — no existing capability's requirements change; unit/integration/benchmark testing is unaffected -->

## Impact

- **Dependencies**: adds `robotframework-agenteval[all]` as a **test-only, isolated** dependency (its own
  env; not in rf-mcp's runtime or the main dev env), version-pinned.
- **Code removed over the phases**: the bespoke `tests/e2e/` harness modules (~4.8k lines) shrink as
  scenarios move to agenteval keywords.
- **New test assets**: agenteval `.robot` suites + an rf-mcp `.mcp.json` launch descriptor for the harness.
- **CI**: a job that installs agenteval in an isolated env and runs the `.robot` suites against a spawned
  rf-mcp (deterministic tier gated on every PR; agentic tier gated behind a model key, like today).
- **No production code changes. No MCP tool names, parameters, or return shapes change.**
