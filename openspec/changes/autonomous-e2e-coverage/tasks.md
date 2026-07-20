## 1. Generalized provider + pinnable reference

- [x] 1.1 Generalize `resolve_model` to route MiniMax (`api.minimax.io/v1`) AND OpenRouter (`openrouter.ai/api/v1`) via the OpenAI-compatible path + shared service_tier sanitizer; provider chosen by slug (`provider_for`).
- [x] 1.2 Reference-model config (`REFERENCE_MODEL` + `REFERENCE_PIN`) and `reference_pin` in the baseline JSON; `OPENROUTER_PROVIDER` pins a single provider+quant via `extra_body` (the reproducibility fix for OpenRouter routing variance).
- [x] 1.3 Unit tests (no network): provider routing per slug; OpenRouter base_url + key-required; provider pinning sets `extra_body.provider.order`.

## 2. Metric hardening (quality_gate.py)

- [x] 2.1 `task_completion` + first-try selection are PRIMARY; `tool_hit_rate` demoted to reporting/validated-only (never the sole/absolute-floor hard-fail).
- [x] 2.2 Added `unexpected_tool_rate`, discovery:execute ratio, and `artifact_executes` (`run_test_suite` passed) to `compute_run_metrics`/`AggMetrics`.
- [x] 2.3 First-try scoring = first tool call is an expected tool AND succeeds (un-inflatable by flailing).
- [x] 2.4 Unit tests: first-try not inflated; artifact metric; hit-rate DROP hard-fails a validated scenario; completion regression fails.

## 3. Scenario validation protocol (keystone)

- [x] 3.1 `is_scenario_validated` + gate demotion: a scenario marked `_validated:false` (or unvalidated) is inform-only (regressions warn, never hard-fail).
- [x] 3.2 `validate_scenario` (calibrated: reference completes on good; sensitive: degradation drops a metric monotonically). Demonstrated live: basic_list VALID (1.00→0.67 monotonic); custom_library_discovery INVALID (uncalibrated + inverted).
- [x] 3.3 Unit tests: uncalibrated rejected; insensitive rejected; calibrated+sensitive admitted; unvalidated demotes to warn.

## 4. Capability-tiered roster

- [x] 4.1 `excluded_models` in the baseline JSON; gate skips them (`llama-3.1-8b` — broken tool-calling, proven live).
- [x] 4.2 Tier semantics: reference (regression + absolute floors), hard_gate (regression), inform (warn), excluded (skip); infra faults hard-fail any non-excluded tier.
- [x] 4.3 Unit tests: excluded not gated; tier fail/warn matrix.

## 5. Scenario coverage (outcome-focused, validated)

- [ ] 5.1 Run the validation canary against candidate CI-safe YAMLs (`restful_booker_api`, `xml_testing`, `suite_validation_execution`) + the 3 new scenarios; admit those that pass. (Mechanism ready; live canary capture per candidate is the remaining operational step — new scenarios ship `needs-validation` = inform until then.)
- [x] 5.2 New gate scenarios use outcome-focused prompts (describe the goal, not the tool calls).
- [x] 5.3 New display-free scenario: `desktop_discovery` (recommend_libraries→PlatynUI + find_keywords + get_locator_guidance, no execution).
- [x] 5.4 New display-free scenario: `data_driven_generic` ([Template] over BuiltIn/String, no display).
- [x] 5.5 New scenario: `locator_ergonomics` (get_locator_guidance + build skeleton, no live browser).

## 6. Baselines + provenance

- [x] 6.1 Baseline schema v2: provenance (`captured_at`, `captured_pin`, `rf_mcp_git_sha`) + `staleness_warning` on pin mismatch; tolerance widened by measured IQR.
- [x] 6.2 Reference baseline committed (MiniMax-M3, active reference; `minimax_basic_list` validated). Full-roster / new-metric recapture (N=5) is the operational `E2E_CAPTURE_BASELINE=1` step.
- [x] 6.3 No-decrease ratchet: a normal (non-capture) run never rewrites the baseline; `capture_entry` refuses infra/degenerate captures (caught qwen's flaky route).

## 7. Tiered CI

- [x] 7.1 `ci.yml` → `e2e-instruction-smoke`: active reference (MiniMax-M3) over the fast validated subset; gated on the key, skips (not fails) when absent.
- [x] 7.2 `e2e-weekly.yml` → `instruction-quality-matrix`: full roster (MiniMax + OpenRouter open-weight), N=5.
- [x] 7.3 Documented the OpenRouter provider-routing caveat + provider pinning + local-vs-OpenRouter diff guidance (README + design).

## 8. Surfaced rf-mcp gaps + docs

- [x] 8.1 Triaged the custom-library import→discover→execute→build discoverability gap (neutral agent gives up); scenario kept inform-only until fixed (documented as a follow-up, not a blocker).
- [x] 8.2 Updated `tests/e2e/README.md`: validation protocol, robust metrics, roster + reference (+ OpenRouter routing finding), tiered CI, capture command.

## 9. Verification

- [x] 9.1 Full unit suite green (42 gate/provider tests; 7076 total).
- [ ] 9.2 Live: reference (MiniMax-M3) PASSES the validated set on good rf-mcp (proven — success/hit/completion/first-try 1.0). Degradation FAILS proven earlier on basic_list. The ≥2-validated-scenario degradation proof completes once §5.1 admits the rich scenarios.
- [x] 9.3 `openspec validate autonomous-e2e-coverage --strict` passes.
