# E2E AI Agent Testing

This directory contains end-to-end tests for validating AI agent tool discovery and usage patterns.

## Quick Start

```bash
# Run all E2E tests
uv run pytest tests/e2e/ -v

# Run with real LLM (requires OPENAI_API_KEY)
USE_REAL_LLM=true uv run pytest tests/e2e/ -v

# Run specific test
uv run pytest tests/e2e/test_agent_tool_discovery.py::TestAgentToolDiscovery::test_mcp_tools_discoverable -v

# Run the autonomous agent suite driven by MiniMax (requires MINIMAX_API_KEY)
MINIMAX_API_KEY=... MINIMAX_MODELS=MiniMax-M3 \
    uv run pytest tests/e2e/test_minimax_autonomous.py -v -s
```

## Autonomous agents with MiniMax

The in-process pydantic-ai harness (`agent_integration.py`) can drive **MiniMax**
models — `MiniMax-M2`, `MiniMax-M2.5`, `MiniMax-M2.7`, `MiniMax-M3` — over MiniMax's
OpenAI-compatible endpoint (`https://api.minimax.io/v1`). Wiring lives in
`minimax_support.py`; set `MINIMAX_API_KEY` and the harness routes MiniMax model IDs
there automatically (everything else keeps the default OpenAI provider).

- `test_minimax_autonomous.py` — runs each MiniMax tier through a deterministic
  workflow and enforces the **instruction-quality gate** (`quality_gate.py`).
- The shared `test_autonomous_agents.py` also runs under MiniMax when `MINIMAX_API_KEY`
  is set (it prefers MiniMax; override with `E2E_AGENT_MODEL`).

Env knobs: `MINIMAX_MODELS` (comma-separated subset, default all four),
`E2E_AGENT_MODEL` (force one model for the shared suite), `E2E_RUNS` (runs per model
for aggregation, default 3).

### Harness fidelity — the agent depends on rf-mcp's own instructions

The whole point is to measure rf-mcp's **instruction quality** — whether its tool
descriptions and MCP instructions let an agent discover and drive the tools. So the
harness must NOT hand the agent a cheat-sheet:

- The system prompt is **neutral** (`agent_integration.NEUTRAL_SYSTEM_PROMPT`) — it does
  not restate which tools to call or how. The agent relies on the tools' real
  descriptions.
- The server's **MCP instructions** (the WORKFLOW GUIDE, `mcp.instructions`) are injected
  into the agent context — `FastMCPToolset` does not forward them otherwise, so without
  this the MCP instructions would be untested. Degrading them (e.g.
  `ROBOTMCP_INSTRUCTIONS=off`) now changes agent behaviour and the metrics.

### Instruction-quality gate — "ensure no decrease"

Wrong / failed / absent tool calls ARE the signal: for a **pinned** model, only rf-mcp's
instruction surface can have changed, so a drop in tool-call quality means rf-mcp got
worse. `quality_gate.py`:

- Measures per run: `task_completion` and `first_try_ok` (**primary**, robust — a
  floundering agent cannot inflate them), `tool_success_rate`, plus `tool_hit_rate`,
  `unexpected_tool_rate`, `discovery:execute` ratio and `artifact_executes`
  (`run_test_suite` passed). `tool_hit_rate` is reported and gates only on **validated**
  scenarios (it is non-monotonic — a floundering agent inflates it).
- Aggregates **N runs** (median for rates) and compares against that model's committed
  **baseline** (`baselines/instruction_quality_baselines.json`). A drop below
  `baseline − tolerance` (tolerance = `max(0.10, IQR@capture)`) is a **regression → HARD
  fail** (the "no decrease" ratchet — lowering a baseline is a reviewed commit).
- **Reference tier** also enforces absolute floors (success ≥ 0.70, first-try ≥ 0.50,
  completes in ≥ 1 of N). **Inform tiers** only warn on regression. **Infra faults**
  (recursion/registration/handshake) HARD-fail any tier on any run. **Excluded models**
  (broken tool-calling, e.g. `llama-3.1-8b`) are not run.

### Scenario validation protocol (the keystone)

A scenario hard-gates only when a **validation canary** proves it is:
- **calibrated** — the reference model completes it on good rf-mcp (headroom to drop); and
- **sensitive** — degrading the relevant instruction surface lowers the metric
  monotonically.

Scenarios that fail this (e.g. `custom_library_keyword_discovery`: reference can't
complete it, and degradation made hit-rate *rise*) are `_validated: false` and demoted to
inform — they cannot hard-fail, because an invalid probe gives misleading signal. New
scenarios (`desktop_discovery`, `data_driven_generic`, `locator_ergonomics`) are tagged
`needs-validation` until the canary admits them (`validate_scenario` in `quality_gate.py`).

### Model roster & the reference model

Tiers live in the baseline file (`reference_models` / `hard_gate_models` /
`inform_models` / `excluded_models`). The **active reference is `MiniMax-M3`** (proven
reliable 3/3). The **goal** is a pinnable **self-hostable** reference
(`qwen/qwen3-coder-30b-a3b`, Apache-2.0) to eliminate baseline drift — but OpenRouter's
*default* routing for it is **not reproducible** (it intermittently returns prose with
zero tool calls: N=3 → success `[0,0,1]`). Realizing the self-hostable reference needs
OpenRouter **provider pinning** or actual self-hosting; until then it runs inform-only.
See `reference_pin.note` in the baseline file.

Update baselines deliberately (a reviewed diff) after an intended instruction change
(capture refuses degenerate/infra results):

```bash
MINIMAX_API_KEY=... E2E_CAPTURE_BASELINE=1 E2E_RUNS=5 E2E_MODELS=MiniMax-M3 \
    uv run pytest tests/e2e/test_minimax_autonomous.py
```

Per-model metrics + gate verdicts (with provenance: `captured_at`, `captured_pin`,
`rf_mcp_git_sha`) are written to `metrics/minimax/` and the baseline file.

## CI

| Workflow / job | Gate | What runs |
|---|---|---|
| `ci.yml` → `e2e-no-llm` | none (always) | `test_agent_tool_discovery` (gates tool registration); `test_mcp_stdout_leak` runs informationally |
| `ci.yml` → `e2e-minimax-smoke` | `MINIMAX_API_KEY` | MiniMax-M3 autonomous smoke + instruction-quality gate vs baseline (per commit) |
| `ci.yml` → `e2e-tests` | `PERSONAL_ACCESS_TOKEN` | Copilot CLI autonomous / workflow suites |
| `e2e-weekly.yml` → `minimax-autonomous` | `MINIMAX_API_KEY` | Full MiniMax matrix (M2/M2.5/M2.7/M3) + metrics |
| `e2e-weekly.yml` → `opencode-e2e` | `OPENROUTER_API_KEY` | opencode small-LLM intent-action + realistic e2e |

## Directory Structure

```
tests/e2e/
├── README.md                    # This file
├── models.py                    # Pydantic models
├── metrics_collector.py         # Metrics tracking
├── fixtures.py                  # Pytest fixtures
├── test_agent_tool_discovery.py # Tests
├── scenarios/                   # YAML scenario definitions
│   ├── todomvc_browser.yaml
│   ├── todomvc_selenium.yaml
│   ├── restful_booker_api.yaml
│   └── xml_testing.yaml
└── metrics/                     # Generated JSON metrics
```

## What's Tested

- ✅ MCP tool discoverability
- ✅ Tool call tracking and metrics
- ✅ Tool hit rate calculation
- ✅ Realistic scenario execution (4 scenarios)

## Key Metrics

**Tool Hit Rate**: Percentage of expected tools called correctly

```
Tool Hit Rate = (Expected Tools Met) / (Total Expected Tools)
```

## Adding Scenarios

Create a new YAML file in `scenarios/`:

```yaml
id: my_scenario
name: My Test Scenario
description: What this tests
context: web
prompt: |
  The full prompt given to the AI agent

  For web UI scenarios, ALWAYS include headless mode:
  - Browser Library: Please use headless=True (New Browser with headless=True)
  - Selenium Library: Please use headlesschrome or headlessfirefox
expected_tools:
  - tool_name: analyze_scenario
    min_calls: 1
    max_calls: 1
expected_outcome: What should happen
min_tool_hit_rate: 0.8
tags: [web, my-feature]
```

**Important**: For web UI scenarios, always specify headless mode in the prompt to ensure tests can run in CI/CD environments without a display server.

## Documentation

See [docs/e2e_testing_implementation.md](../../docs/e2e_testing_implementation.md) for complete documentation.
