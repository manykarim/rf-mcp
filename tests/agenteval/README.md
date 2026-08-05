# agenteval harness

Black-box tests for the **rf-mcp MCP server**, authored with
[`robotframework-agenteval`](https://github.com/manykarim/robotframework-agenteval).
The suites spawn `robotmcp` as a subprocess and speak the real MCP protocol to it —
so rf-mcp is exercised exactly as an agent uses it.

Change: `adopt-agenteval-harness` (Phase 1) — the harness foundation plus the first two
scenario ports (demoshop web + restful-booker API), whose data is preserved from the
bespoke `tests/e2e/` machinery. Requires `robotframework-agenteval>=0.4.0` (see Findings).

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
| `agentic_e2e.robot` | 3 — real in-process agent (smoke) | yes; skips cleanly without one |
| `agentic_scenarios.robot` | 3 — ported demoshop + restful-booker | yes; each scenario also opt-in (below) |

The agentic suites inject rf-mcp's own MCP `instructions` (the WORKFLOW GUIDE, via
`MCP.Get Server Instructions`) into the agent and raise the in-process `request_limit`, so the
agent is steered like a compliant MCP client and long scenarios complete — both require
agenteval **>= 0.4.0**.

## Running

```bash
# Deterministic tier — no key needed:
tests/agenteval/run.sh --suite 'Deterministic Mcp Surface'

# Agentic tier — set a model credential (read from the env, never a RF variable):
export AGENTEVAL_API_KEY=...                      # e.g. your MiniMax key
export AGENTEVAL_BASE_URL=https://api.minimax.io/v1
export AGENTEVAL_MODEL=MiniMax-M3
tests/agenteval/run.sh --suite 'Agentic E2E'

# Ported scenarios — long/costly, so off by default:
export AGENTEVAL_ALLOW_LONG=1                      # enable the restful-booker API scenario
export AGENTEVAL_WEB=1                             # enable the demoshop web scenario (needs a browser)
tests/agenteval/run.sh --suite 'Agentic Scenarios'

# Whole harness (agentic self-skips without a key; scenarios skip without their opt-ins):
tests/agenteval/run.sh
```

Results land in `tests/agenteval/results/` (git-ignored).

## Findings from the Phase-1 port (evaluation notes)

Porting the two scenarios (`scenarios/*.yaml`, driven by MiniMax-M3) surfaced two things worth
knowing before leaning on the harness:

1. **In-process steering + request limit — a v0.3.0 gap, fixed in v0.4.0.** In 0.3.0 the in-process
   adapter ran `agent.run(prompt)` with pydantic-ai's default `request_limit=50` and **no override**,
   and never surfaced the MCP server's own `instructions`. So the agent flew blind and long scenarios
   died on the cap: restful-booker (read + create + auth + delete + build, ~50–100 model requests)
   raised `UsageLimitExceeded`. Injecting the instructions alone did **not** bring it under 50 — the
   scenario is genuinely long — so the *binding* cause was the un-raisable limit, with the dropped
   instructions a separate real fidelity gap (the agent never saw *"call `get_locator_guidance`
   before you interact"*, the exact steering rf-mcp relies on). **agenteval v0.4.0** adds
   `request_limit`/`usage_limits` overrides, an `instructions` argument, and the
   `MCP.Get Server Instructions` reader. The harness now injects rf-mcp's WORKFLOW GUIDE and sets
   `request_limit=120`, so the agent is steered and long scenarios complete. restful-booker stays
   off by default (`AGENTEVAL_ALLOW_LONG=1` to run) purely to keep per-push CI cheap.
2. **Tier-3 measures tool usage, not automation success.** The hit-rate gate (like the bespoke
   quality gate) asserts the agent *selected and called* the right tools — it is robust to page
   state. It does **not** assert the web automation succeeded. The demoshop prompt uses
   `headless=False`, so on a headless runner Chromium can't launch (Missing X server); the agent
   still met the tool gate. For real browser automation the web scenario needs a display (Xvfb) or
   a headless variant. That is why the web suite is opt-in (`AGENTEVAL_WEB=1`).

## Desktop scenarios (gated)

Desktop suites (`integration/test_platynui_gnome_apps_e2e.robot`, and future `platynui_*` ports) drive
real GTK apps via PlatynUI/AT-SPI. They are **gated behind `AGENTEVAL_DESKTOP=1`** and **skip cleanly on
the stock headless CI runner** — hosted GitHub runners have no `systemd --user` session, which those
suites need to launch the apps, so they can't run there (change: `agenteval-desktop-ci-gating`).

The first slice of desktop CI coverage **is** shipped: the gated, keyless **`desktop-smoke`** workflow
(`.github/workflows/desktop-smoke.yml`, change `desktop-smoke-ci`) builds `docker/Dockerfile.desktop`
(Xvfb + fluxbox EWMH WM + `at-spi2-core` + `GTK_A11Y=atspi` + gnome apps) and runs the deterministic
smoke inside it (headful, systemd-free — the container launches apps via a direct `Popen`, so no
`systemd-run` is involved). See `docs/desktop_docker_harness.md`.

Running the *agenteval desktop `.robot` ports* (and the platynui pytest tests) in that image is a
sequenced follow-on: the `test_platynui_newcore_e2e` workflow needs only an app-launch fixture (proven
to pass in-image), while `test_platynui_focus_e2e`'s overlapping-window scenario needs the
`systemd-run`→direct-launch seam. Until those land, the `AGENTEVAL_DESKTOP` suites still skip on the
stock runner; verify them locally or via the Docker harness.

## Phase-2b split (partial integration files)

Some `tests/integration/` files mix MCP-observable tests (drive a tool, assert on the result/trace)
with internal-state tests (reach into `ExecutionSession`/executor/components). Those are **split**: the
MCP-observable subset ports to an `integration/*.robot` suite here; the internal-state tests stay in a
trimmed pytest file. Coverage is preserved on both sides — no assertion is dropped or weakened
(change: `agenteval-port-partial-integration`).

**Split so far:**

| pytest file | ported here | kept in pytest |
|---|---|---|
| `test_adr009_schema_validation.py` | `integration/test_adr009_schema_validation.robot` (26 — all, original deleted) | — |
| `test_real_browser_prevalidation.py` | `integration/test_real_browser_prevalidation.robot` (13: keyword-exec, page-source, `intent_action(extract)`, drag-drop pre-scroll) | 11: `_pre_validate_element` (internal) + OBS-01 verdict-equivalence (drives a tool **then** asserts internal verdicts — straddles) |
| `test_real_selenium_prevalidation.py` | `integration/test_real_selenium_prevalidation.robot` (6: keyword-exec incl. no-timeout-injection, page-source) | 4: `_pre_validate_element` (internal) |

The browser/selenium ports are keyless but **browser-gated**: each spawns its own rf-mcp subprocess and
skips cleanly if the browser can't launch (mirroring the pytest `skipif`). CI provisions Chromium
(`rfbrowser init`) so they run for real; the selenium port uses the runner's preinstalled Chrome.

**Left whole in pytest (MCP-observable slice too small to justify a split):**
`test_architecture_improvements`, `test_nlp_improvements`, `test_library_loading_improvements`,
`test_sampling_feature_flag`, `test_real_page_source_routing`.

**Deferred (desktop-verification-blocked):** `test_platynui_focus_e2e`, `test_platynui_newcore_e2e`.
Their MCP-observable subsets are real but need a live GNOME desktop (`Xvfb` + `systemd-run`, overlapping
windows, live-app `ui_tree`) that no headless environment provides — so a port can't be verified green
here, and even the one seemingly-static test (`test_locator_guidance_dispatch`) returns a
desktop-dependent shape headless. Rather than ship an unverifiable desktop suite (which risks a silent
no-op), they stay whole in pytest until a desktop-verified follow-up can port + verify them together with
the gnome-apps desktop suite.

## Upgrading agenteval

The pin lives in `requirements.txt` (currently `==0.4.0`). agenteval's libraries are `provisional`
(0.x) — bumping is a deliberate change; check its CHANGELOG for breaking changes first. The agentic
suites need the in-process overrides added in **0.4.0** (`request_limit` + `instructions`), so do not
pin below it.
