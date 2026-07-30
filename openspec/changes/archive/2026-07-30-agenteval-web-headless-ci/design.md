## Context

The demoshop scenario (`tests/agenteval/scenarios/demoshop_dd_cart.yaml`) instructs the agent: `New Browser
chromium headless=False`. On a headless CI runner a headed Chromium can't launch ("Missing X server"). The
existing `build-and-test` CI job already proves browsers install fine (`rfbrowser init` + `playwright
install --with-deps`). Playwright headless needs no display, so the only real changes are: make the agent
launch headless, install the browser in the (gated) job, and set `AGENTEVAL_WEB=1`.

## Goals / Non-Goals

**Goals:** run the demoshop web scenario in CI headless; keep it gated (scheduled), keyless-self-skipping;
leave the scenario definition unchanged for local runs.

**Non-Goals:** desktop-in-CI (separately evaluated; stays gated); making the web scenario always-on
per-push; changing what the scenario asserts.

## Decisions

**D1 — Override at load, not in the YAML.** `load_agentic_scenario` substitutes `headless=False` ->
`headless=True` in the returned prompt when `AGENTEVAL_BROWSER_HEADLESS` is truthy. *Why:* the scenario
prompt is fixed text the model obeys; a deterministic substitution reliably flips the browser mode without
duplicating the scenario or relying on the model to infer "run headless". The YAML stays authored for
local observability (headless=False). *Alternative rejected:* a second headless YAML (duplication); an
Xvfb wrapper (heavier, and unnecessary when true-headless works).

**D2 — Gated job in `e2e-weekly.yml`, not the per-push job.** The web scenario is a live, multi-step,
model-driven browser run — minutes, cost, and the occasional provider transient (which self-skips). That
matches the *scheduled real-LLM* workflow, not the fast always-on `agenteval-harness` tier. *Alternative
rejected:* a conditional step in the per-push job (would add a schedule trigger to `ci.yml` and risk
per-push cost).

**D3 — Same credential gate.** The job wires `AGENTEVAL_API_KEY=secrets.MINIMAX_API_KEY` (+ base_url/model)
and self-skips without it, mirroring the existing agentic tier; it also `rfbrowser init`s the browser.

## Risks / Trade-offs

- **[Substitution misses a differently-worded prompt]** → it targets the exact `headless=False` token the
  scenario uses; a unit test locks it, and any web scenario added later must use the same token or set the
  mode explicitly.
- **[Agentic flakiness in the gated job]** → it is scheduled, not blocking per-push CI; the transient-error
  skip already added in the scenario suite keeps a provider hiccup from red-ing it.
- **[Browser install time in the gated job]** → acceptable off the per-push path.

## Migration Plan

Add the override + its unit test; add the gated job; validate a real headless demoshop run locally
(browser actually launches headless and the scenario reaches its tool-hit gate). Rollback = revert the two
files.
