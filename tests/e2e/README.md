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

### Experiment findings — self-hostable model calibration (2026-07-20)

Capable self-hostable (24–32 GB) models were run across four technologies
(data-driven / XML / suite-exec / API), OpenRouter provider-pinned to ≥98%-uptime
fp8/bf16 backends. Tool-call reliability ranking (s=success, c=completed):

| Model (pinned) | data-driven | xml | suite-exec | API | tier |
|---|---|---|---|---|---|
| `MiniMax-M3` (ref) | s0.77 c✓ | s1.00 | s1.00 c✓ | s0.98 | reference |
| `glm-4.7-flash` (DeepInfra) | s1.00 c✓ | s0.93 | s0.89 c✓ | s0.54 | inform *(top promotion candidate)* |
| `qwen3-coder-30b` (Novita) | s0.91 c✓ | s0.86 c✓ | s1.00 c✓ | s0.00 ⚠ | inform *(aspirational ref; API no-call flake)* |
| `mistral-small-24b` (DeepInfra) | s0.46 | s0.38 *(78 calls)* | s0.83 c✓ | s0.00 | inform *(weak/flails)* |
| `gemma-3-27b` (DeepInfra) | s0.00 | s0.00 | s0.00 | s0.00 | **excluded** *(zero tool calls)* |

Notes: `gemma-3-27b` emits **zero tool calls** everywhere (broken, like `llama-3.1-8b`) →
excluded. `qwen3-coder-30b`'s API no-call reproduces the routing flake even under a
provider pin. No model is promotable to hard-gate on one run each — promotion needs
N≥5 reliable captures, ideally self-hosted. Technology difficulty: suite-exec < data-driven
< xml < **API** (the instruction-sensitive surface).

### Experiment findings — scenario validation & the degradation lever (2026-07-20)

The canary **rejected all 3 new scenarios** (`desktop_discovery`, `data_driven_generic`,
`locator_ergonomics`) as INVALID probes — the validation protocol working correctly. Root
cause: **blanking tool *descriptions* is too weak a degradation when the tool *names* are
self-explanatory** (`find_keywords`, `build_test_suite`…) — a capable model infers usage
from the name, so degradation is insensitive (one even *inverted*). A scenario is a valid
probe only when correct usage depends on **non-obvious** instructions.

Stronger degradation levers (for future scenario validation):
1. Blank only the **`session_id` arg prose** (session-threading is the most
   name-underdetermined contract — the same surface that makes `basic_list` sensitive).
2. Stub the **runtime output** of `get_locator_guidance` / requests-guidance (the
   non-obvious knowledge is in the returned cookbook, not the tool name).
3. Remove **one specific fact** (e.g. `EVALUATE_VAR_RULE`, the `[Template]` recipe).
4. Inject a **plausible-*wrong*** instruction (adversarial > absent — absent lets
   name-obvious tools self-recover).

Best valid-probe candidate: **`restful_booker_api`** — RequestsLibrary response access
(`${resp.json()}`, `json=` not `data=`, `…On Session`) is genuinely non-obvious. Its
completion definition was also fixed (`build_test_suite` **OR** `run_test_suite` OR an
asserting flow — see `compute_run_metrics` OR-groups) so a drive-and-assert run isn't
scored incomplete for not building a suite.

**Open problem — sensitive degradation is hard.** A re-canary of `suite_validation` with a
plausible-*wrong* instruction *also* inverted (wrong: comp 1.00 vs good: comp 0.50) because
the wrong instruction accidentally *simplified* the flow. A degradation only yields the
required monotonic drop when it makes the agent emit *failing* calls (e.g. a false
`session_id` contract on a scenario that truly threads it) — not merely absent or simpler
guidance. So far only `minimax_basic_list` is a validated probe; admitting a non-toy one
needs a lever tied to a genuinely failing contract (the requests-guidance stub on the API
scenario is the leading candidate). The validation protocol has correctly rejected 4
candidate probes to date — that refusal is the feature.

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
