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
  workflow and enforces the **quality gate** (`quality_gate.py`).
- The shared `test_autonomous_agents.py` also runs under MiniMax when `MINIMAX_API_KEY`
  is set (it prefers MiniMax; override with `E2E_AGENT_MODEL`).

Env knobs: `MINIMAX_MODELS` (comma-separated subset, default all four),
`E2E_AGENT_MODEL` (force one model for the shared suite).

### Quality gate — "no rf-mcp quality decrease"

`quality_gate.py` separates **model-agnostic rf-mcp health** (HARD, fails the build)
from **model-choice noise** (SOFT, warns only), by attributing each failed tool call:

- **rf-mcp fault** (HARD): tool not registered, server exception/traceback, recursion,
  handshake/libdoc failure — a real regression.
- **model-framing fault** (SOFT): graceful RF keyword/argument hints, malformed args
  rejected at the pydantic boundary, or the agent going off-script — weak-tier
  flakiness that must not red the build.

So a flaky MiniMax run stays green (warn), while a genuine rf-mcp regression (e.g. the
FastMCP tool-registration recursion) trips a HARD failure. Per-model metrics + gate
verdicts are written to `metrics/minimax/`.

## CI

| Workflow / job | Gate | What runs |
|---|---|---|
| `ci.yml` → `e2e-no-llm` | none (always) | `test_agent_tool_discovery` (gates tool registration); `test_mcp_stdout_leak` runs informationally |
| `ci.yml` → `e2e-minimax-smoke` | `MINIMAX_API_KEY` | MiniMax-M3 autonomous smoke + quality gate (per commit) |
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
