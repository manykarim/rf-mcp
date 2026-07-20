## Context

The instruction-quality gate (shipped as commit `6fe7a9e`) drives an autonomous agent
against rf-mcp via `FastMCPToolset` with a neutral system prompt + injected server
instructions, and gates a fixed model's N-run aggregate against a committed baseline. It
currently runs one inline scenario (`minimax_basic_list`) with a baseline for one model
(MiniMax-M3). This design broadens it correctly, anchored in experiments run during
exploration (all live, MiniMax + OpenRouter keys already in `.env`):

- **Scenario validity experiment** — `basic_list`: M3 good 1.00 → degraded 0.67
  (monotonic, VALID probe). `custom_library_keyword_discovery`: M3 good 0.17 / not
  completed, degraded 0.33 (metric INVERTED — a floundering agent brute-forces more
  tools). Conclusion: uncalibrated scenarios produce misleading signal.
- **Self-hostable reference experiment** (toy scenario, 1 run each): `qwen3-coder-30b-a3b`
  1.00/1.00/completed (25s), `mistral-small-3.2-24b` 1.00/1.00 (24s), `glm-4.7-flash`
  0.94/1.00 (68s) — all match/near-match M3; `qwen-2.5-7b` 0.52 (flails), `llama-3.1-8b`
  0.33 / malformed responses (broken tool-calling). Conclusion: pinnable open-weight
  models can be the reference; tool-call capability tiers cleanly.

## Goals / Non-Goals

**Goals:**
- Only *validated* scenarios hard-gate; uncalibrated/insensitive ones are inform-only.
- Gate on robust signals (completion, first-try selection) not gameable hit-rate.
- Make the reference model a pinnable open-weight model → reproducible baselines.
- A capability-tiered model roster (reference / hard_gate / inform / excluded).
- Real surface coverage (API, XML, suite-exec, discovery, desktop-discovery, DD,
  ergonomics) via outcome-focused scenarios.
- Tiered CI: fast per-commit reference smoke; scheduled full roster matrix.

**Non-Goals:**
- Unifying the CLI lanes (Copilot/opencode) into the *hard* gate — they may feed the
  report as inform-only trend only.
- Adding a frontier proprietary reference model.
- Self-hosting infrastructure in CI — CI uses OpenRouter/MiniMax APIs; local self-hosting
  is the authoritative baseline-regeneration path, not a CI dependency.
- Deep rewrites of rf-mcp instruction surfaces beyond narrowly-scoped fixes for gaps a
  validated scenario surfaces.

## Decisions

1. **Validation protocol as a first-class step, stored per scenario.** A scenario's
   baseline entry records a `validated` flag set by a canary that runs the reference model
   on good rf-mcp (must complete) and on a targeted degradation (metric must drop). Only
   `validated` scenarios enter the hard-gate set; others are `inform`. *Alternative
   considered:* trust `min_tool_hit_rate` from the YAML — rejected, because the discovery
   experiment shows the declared threshold does not predict validity, and hit-rate is
   non-monotonic.

2. **Completion + first-try selection are primary; hit-rate is demoted to reporting.**
   `task_completion` (start + artifact tools succeeded) and first-try tool-selection
   correctness resist the brute-force inflation that makes hit-rate rise under
   degradation. Hit-rate is still recorded for trend. *Alternative:* keep hit-rate
   primary — rejected by the inversion observed in the experiment.

3. **Reference = pinnable open-weight model via a generalized provider.** Generalize
   `minimax_support.resolve_model` into a provider-routing helper: MiniMax and OpenRouter
   both use the OpenAI-compatible path (`api.minimax.io/v1`, `openrouter.ai/api/v1`) with
   the existing `service_tier` sanitizing transport. Default reference =
   `qwen/qwen3-coder-30b-a3b-instruct` (Apache-2.0); MiniMax-M3 becomes a secondary
   cross-check. Dense `qwen3-32b` documented for byte-stable determinism (no MoE routing
   variance). Baseline stores the reference identifier + pin descriptor. *Alternative:*
   keep MiniMax-M3 as reference — rejected due to silent-update baseline drift.

4. **Roster tiers live in the baseline JSON** (`reference_models` / `hard_gate_models` /
   `inform_models` / `excluded_models`), so promotion/demotion is a reviewed config diff,
   not code. Excluded = models with demonstrated broken tool-calling (e.g.
   `llama-3.1-8b`).

5. **Coverage via validated YAML scenarios + three new display-free scenarios.** Wire the
   YAMLs that validate cleanly (candidates: `restful_booker_api`, `xml_testing`,
   `suite_validation_execution`, discovery scenarios *if* fixed). Add: desktop-discovery
   (recommend_libraries→PlatynUI + find_keywords + get_locator_guidance, no execution),
   generic data-driven ([Template] + add_data_row + build), ergonomics
   (get_locator_guidance + intent_action). All prompts outcome-focused.

6. **Tiered CI.** Per-commit `e2e-minimax-smoke` → reference model over the fast validated
   subset (N=3). Weekly → full roster × full validated set (N=5). OpenRouter used for
   zero-infra CI; document that OpenRouter provider routing can serve different
   quant/backends for the same slug, and add a periodic local-vs-OpenRouter diff to catch
   provider drift (tolerance absorbs minor variance).

## Risks / Trade-offs

- **A validated scenario later drifts (model or provider update) → false regression.**
  → Staleness metadata + the local-vs-OpenRouter diff surface drift; tolerance derived
  from captured IQR absorbs minor variance; baselines only lowered via reviewed PR.
- **Capable models infer tools from names, so subtle single-tool docstring regressions
  are caught weakly.** → Breadth of validated scenarios (each stressing specific tools) is
  the mitigation; the validation canary confirms each scenario is sensitive to *its*
  surface.
- **OpenRouter MoE routing nondeterminism for the reference.** → Prefer the pin recipe
  (temperature 0); offer dense `qwen3-32b` when byte-stable determinism is required.
- **Cost/wall-clock of the full matrix.** → Tiering keeps per-commit fast; weekly absorbs
  breadth; models are cheap ($0.08–0.40/M) and MiniMax/OpenRouter keys already exist.
- **A validated scenario surfaces a real rf-mcp instruction gap (e.g. custom-library
  import).** → Treat as a scoped follow-up fix, not a blocker; until fixed the scenario
  stays inform-only so it cannot red the build on a known gap.

## Migration Plan

1. Land the metric + validation-protocol changes and the generalized provider (no
   behavior change for the existing single-scenario M3 gate until baselines are
   recaptured).
2. Capture reference (`qwen3-coder-30b-a3b`) baselines for the validated scenario set;
   commit with provenance. Keep MiniMax-M3 as secondary.
3. Add new scenarios one at a time, each passing the validation canary before entering the
   hard-gate set.
4. Flip CI: per-commit smoke → reference subset; weekly → full roster.
5. Rollback = revert to the single-scenario M3 baseline (the current committed state).

## Apply-time findings (empirical)

- **OpenRouter default routing is NOT reproducible for the open-weight reference.** A
  single-run experiment showed `qwen3-coder-30b` scoring 1.00/1.00; but at N=3 it scored
  `[0,0,1]` — 2 runs returned prose with ZERO tool calls. Root cause: OpenRouter routes
  the same slug across 5 providers at different quantizations/uptimes (Novita 99.6%,
  others 0%). So the self-hostable-reference *goal* is sound but OpenRouter *default*
  routing defeats reproducibility. **Mitigations implemented:** (a) the ACTIVE reference
  reverts to MiniMax-M3 (proven 3/3 reliable); (b) `capture_entry` now refuses a
  degenerate baseline (zero completion/success) so a flaky capture can't be blessed;
  (c) `resolve_model` supports `OPENROUTER_PROVIDER` to PIN a single provider+quant via
  `extra_body={"provider":{"order":[...],"allow_fallbacks":false}}`. Realizing the
  self-hostable reference = pin a high-tool-success provider (or self-host) + recapture.
  **VERIFIED:** `qwen3-coder-30b` pinned to Novita (`OPENROUTER_PROVIDER=Novita`) went
  from default-routing `[0,0,1]` (success 0.0) to **success 1.0 / hit 1.0 / 3-of-3 valid
  runs** (completion 2/3) at N=3 — pinning restores reproducibility, so the open-weight
  self-hostable reference IS viable once its provider is pinned. Recommended next step:
  capture its pinned baseline and promote it from inform to hard_gate/reference.
- **Selecting reliable tool-callers.** OpenRouter exposes a per-model/provider
  tool-success stat (UI filter `?min_tool_success_rate=0.9`) and a per-provider endpoints
  API (`/api/v1/models/<slug>/endpoints`: uptime/latency/quant per provider). Use these
  to choose the pinned provider and to admit models to the roster.

## Open Questions

- Which of `restful_booker_api` / `xml_testing` / `suite_validation_execution` pass the
  validation canary as-is vs need prompt/expectation recalibration? (Resolve during apply
  by running the canary per candidate.)
- Promote `mistral-small-3.2-24b` or `glm-4.7-flash` to a second hard-gate reference for
  cross-model corroboration, or keep single reference + MiniMax cross-check?
- Should first-try tool-selection scoring require a canonical WORKFLOW-GUIDE order, or
  just "the right tool before any error"? (Latter is lower-noise; lean that way.)
