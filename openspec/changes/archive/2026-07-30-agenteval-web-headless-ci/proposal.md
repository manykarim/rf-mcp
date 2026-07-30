## Why

The agenteval harness has a demoshop **web** agentic scenario, but it can't run in CI: its prompt
hardcodes `New Browser chromium headless=False`, and a headed browser needs a display the headless CI
runner doesn't have. So the web surface has no CI coverage — it only runs locally with `AGENTEVAL_WEB=1`.
Playwright's true headless mode needs **no display**, so the scenario can run in CI headless — the only
blockers are the hardcoded `headless=False` and the missing browser install.

## What Changes

- **Parameterize the browser mode.** `scenario_lib.load_agentic_scenario` rewrites `headless=False` →
  `headless=True` in a loaded scenario's prompt when `AGENTEVAL_BROWSER_HEADLESS` is truthy — a
  deterministic text substitution, so the agent launches a headless browser. The scenario YAML is
  unchanged (headless stays the default for local observability); the override only applies when opted in.
- **Add a gated CI web job.** A new job in the scheduled `e2e-weekly.yml` installs the browser
  (`rfbrowser init`) and runs the web scenario with `AGENTEVAL_WEB=1` + `AGENTEVAL_BROWSER_HEADLESS=true`
  + the MiniMax credential. It is **gated/scheduled, not on every push** — the scenario is agentic (a
  live multi-step model-driven browser run: minutes, cost, occasional provider transient), so it does not
  belong in the always-on per-push tier.
- **Self-skips without a key or browser**, exactly like the existing agentic tier.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `agenteval-test-harness`: add that the browser-driving scenario is runnable headless (no display) via an
  env override, and that CI runs it in a gated context rather than on every push.

## Impact

- **Code**: `tests/agenteval/scenario_lib.py` (headless prompt override) + a unit test for the override.
- **CI**: a new gated job in `.github/workflows/e2e-weekly.yml` (scheduled real-LLM workflow).
- **Unchanged**: the demoshop scenario YAML (headless=False default), the always-on per-push tier, all MCP
  tools/params/returns. Desktop stays gated (out of scope — see the separate desktop evaluation).
