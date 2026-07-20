## Why

The agentic e2e instruction-quality gate we just shipped works but is dangerously
narrow: it enforces quality against **one toy scenario** (`minimax_basic_list`) with a
baseline for **one proprietary model** (MiniMax-M3). Experiments run during exploration
proved two things this narrowness gets wrong:

1. **Scenario validity — not count — is the binding constraint.** The toy scenario is a
   *valid* probe (M3 scored 1.00 on good rf-mcp, 0.67 when instructions were degraded — a
   monotonic drop). But a richer scenario (`custom_library_keyword_discovery`) is an
   *invalid* probe: M3 could not even complete it on good rf-mcp (hit 0.17, not
   completed), and degrading the relevant instructions made the metric go *up* (0.17 →
   0.33) because a floundering agent brute-forces more tools. Adding uncalibrated
   scenarios would inject **misleading** signal — worse than no coverage.

2. **A proprietary reference model makes baselines un-reproducible.** A slug like
   `MiniMax-M3` points to vendor-controlled weights that can change silently (checkpoint
   swap, hidden-prompt change, decoder retune) with no version bump — corrupting golden
   baselines into false failures or masked regressions, and making old CI runs
   irreproducible. Live experiments showed three **self-hostable, Apache-2.0** models
   (`qwen3-coder-30b-a3b` 1.00/1.00, `mistral-small-3.2-24b` 1.00/1.00, `glm-4.7-flash`
   0.94/1.00) match MiniMax-M3 on the calibrated scenario — so the reference can be
   pinnable open weights instead.

Now, because the gate is new and single-scenario: broadening it correctly (validated
scenarios, robust metrics, a pinnable reference, real coverage) is the difference between
a gate that catches instruction regressions across rf-mcp's surface and one that only
watches a toy list workflow.

## What Changes

- **Scenario validation protocol (keystone)**: a scenario may hard-gate only after a
  canary run proves it is *calibrated* (the reference model completes it on good rf-mcp)
  AND *sensitive* (a targeted degradation of the relevant instruction surface lowers the
  metric monotonically). Uncalibrated/insensitive scenarios are demoted to
  inform/diagnostic, never hard-gate.
- **Metric hardening**: make `task_completion` and first-try tool-selection the primary
  gate signals; add `unexpected_tool_rate`, `discovery:execute` ratio, and
  `artifact_executes` (build → `run_test_suite` dry/full passes). Demote raw
  `tool_hit_rate` to reporting — it is non-monotonic (a floundering agent inflates it).
- **Pinnable self-hostable reference model**: adopt `qwen/qwen3-coder-30b-a3b-instruct`
  (Apache-2.0) as the reference, pinned by HF revision + quant SHA256 + chat-template
  hash at temperature 0; keep MiniMax-M3 as a secondary cross-check. Dense `qwen3-32b`
  documented as the byte-stable-determinism option.
- **Capability-tiered model roster**: reference (tool-tuned ≥24B), hard_gate, inform
  (weak/7B), and an *excluded* set for models with broken tool-calling
  (`llama-3.1-8b` emitted malformed responses). Membership lives in the baseline JSON.
- **Coverage expansion**: wire the *validated* CI-safe scenarios (API, XML, suite-exec)
  into the gate, and add three new display-free scenarios — desktop discovery, a generic
  data-driven scenario, and an ergonomics scenario (`get_locator_guidance`,
  `intent_action`). All gate scenarios must use **outcome-focused prompts** (describe the
  goal, never prescribe the tool calls — prescribing defeats the discoverability signal).
- **Baseline capture + staleness**: capture baselines for the roster × validated set;
  record `captured_at`, model pin, and rf-mcp git SHA, and warn when a baseline is stale
  or its model pin changed.
- **Tiered CI**: per-commit smoke = reference × fast validated subset; weekly = full
  roster × full validated set × N=5. Document the OpenRouter provider-routing caveat and
  a periodic local-vs-OpenRouter diff.
- **Fix rf-mcp instruction gaps the validated scenarios surface** (e.g. the
  custom-library import→discover→execute→build flow a neutral agent could not follow).

## Capabilities

### New Capabilities
- `agentic-e2e-instruction-quality`: how the agentic e2e gate measures rf-mcp instruction
  quality — the scenario-validation protocol, the gate metrics and their robustness
  requirements, per-model baseline regression with a capability-tiered roster, a pinnable
  self-hostable reference model, coverage requirements for gate scenarios, and the tiered
  CI execution model.

### Modified Capabilities
<!-- none — the current gate was shipped without an OpenSpec capability; this introduces it -->

## Impact

- **Tests/harness**: `tests/e2e/quality_gate.py` (metrics + validation protocol),
  `tests/e2e/test_minimax_autonomous.py` (multi-scenario + roster), `minimax_support.py`
  (generalize provider routing to OpenRouter), new scenario YAMLs under
  `tests/e2e/scenarios/`, `tests/e2e/baselines/instruction_quality_baselines.json`
  (roster + validated-scenario baselines + pin metadata), `tests/unit/` gate-logic tests.
- **CI**: `.github/workflows/ci.yml` (`e2e-minimax-smoke` → reference-model smoke over the
  fast validated subset) and `.github/workflows/e2e-weekly.yml` (full roster matrix).
- **Secrets/keys**: reuses `MINIMAX_API_KEY` and `OPENROUTER_API_KEY` (both already
  present); no new proprietary dependency.
- **Docs**: `tests/e2e/README.md`.
- **Possible rf-mcp source touch-ups** for instruction gaps surfaced by validated
  scenarios (scoped narrowly; e.g. custom-library discoverability), decided per finding.
- **Non-goals**: unifying the CLI lanes (Copilot/opencode) into the hard gate (they may
  feed the report as inform-only trend); adding a frontier proprietary reference.
