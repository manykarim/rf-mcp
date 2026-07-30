## 1. Headless browser-mode override

- [x] 1.1 In `scenario_lib.load_agentic_scenario`, when `AGENTEVAL_BROWSER_HEADLESS` is truthy, rewrite
  `headless=False` -> `headless=True` in the returned prompt (deterministic substitution).
- [x] 1.2 Add a unit test: override off -> prompt unchanged; override on -> prompt says `headless=True`.

## 2. Local validation

- [x] 2.1 Confirmed headless launch locally. A deterministic keyless probe drives rf-mcp over MCP
  (`manage_session` Browser -> `New Browser chromium headless=True` -> `New Page` demoshop) and both steps
  return `success=True` with no display present — proving the headless path launches Chromium without an X
  server (the earlier headed run failed *instantly* on "Missing X server"). The tool-hit gate itself was
  already established by the prior demoshop agentic run (hit_rate 0.75); the only thing headed CI could not
  prove — that the browser launches headless — is now proven. The full agentic scenario is inherently long
  (data-driven cart, ~15+ min headless) so it stays the gated weekly job rather than a local gate.

## 3. Gated CI job

- [x] 3.1 Add a job to `.github/workflows/e2e-weekly.yml` (scheduled): checkout -> uv -> `uv sync` ->
  `rfbrowser init` -> run the web scenario with `AGENTEVAL_WEB=1`, `AGENTEVAL_BROWSER_HEADLESS=true`,
  `AGENTEVAL_API_KEY=secrets.MINIMAX_API_KEY` (+ base_url/model). Self-skips without the key.

## 4. Wrap-up

- [x] 4.1 `openspec validate agenteval-web-headless-ci --strict` passes; web-headless unit test green (7 passed).
