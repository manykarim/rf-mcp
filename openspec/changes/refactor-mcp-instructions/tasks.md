## 1. Baseline the current state (measure before changing)

- [x] 1.1 Capture the reference-model instruction-quality baseline under the CURRENT default template on the validated scenario set (`E2E_CAPTURE_BASELINE=1`), so the refactor's effect is measured, not assumed.
  - Captured the OLD `standard` default as the A/B control on two axes: Haiku CLI (n=4: built 4/4, avg churn 1.0, avg calls 4.75) and glm-4.7-flash reference (suite scenario, n=3) — the "before" the refactor is measured against.
- [x] 1.2 A/B 2–3 candidate lean spines (≈`minimal` + one canonical-order line; and the `refactored` draft) on the reference model over suite-exec + a generic scenario; pick the winner by success/completion/right-order/turns.
  - Haiku/Claude-Code-CLI axis (2026-07-20, after cold-start stdio fix a467c5a): n=5 sweep → 100% completion for all; ranking (avg churn / calls / dur): `checklist` 0.0 / 4.4 / 22.2s (winner) ≈ `example` 0.0 / 5.0 / 27.6s > `minimal` 0.2 > `terse` 0.4. Winner = the `checklist` spine → shipped as the `lean` template.
  - Post-implementation A/B (lean default vs old standard default): **Haiku** n=4 each — lean 0.0 churn / 4.25 calls / 24.5s (4/4 complete) vs standard 1.0 churn / 4.75 calls / 25.75s (4/4 complete). **glm-4.7-flash reference** (suite scenario) n=3 each — lean 0.0 churn, avg succ 0.89, **3/3 complete**; standard 0.67 churn, avg succ 0.86, **1/3 complete** (both churned runs failed to complete; one ran 25 calls / 615s). Lean wins on both axes.

## 2. Lean order-explicit default template

- [x] 2.1 Add a lean default template to `domains/instruction/value_objects.py` using the `checklist` spine shape. → `InstructionTemplate.lean()` (template_id="lean", ~1000 chars); numbered order, unified session entry verbatim, no per-tool catalog echo, no `{available_tools}` placeholder.
- [x] 2.2 Make it the resolver default (`fastmcp_adapter.py`: `DEFAULT_TEMPLATE_TYPE=LEAN` + the three `ROBOTMCP_INSTRUCTIONS_TEMPLATE` env defaults → `lean`; `services.py` resolver fallback → `lean()`). Old templates remain env-selectable for rollback (verified by unit test).
- [x] 2.3 Retire `discovery_first` (6081 ch) / trim `detailed`. → The default no longer routes to any oversized template (spec scenario met: the ~6000-char content is not delivered by default). `discovery_first`/`detailed` are kept as OPT-IN env selections only (rollback safety per 2.2), not deleted.

## 3. Unify session entry (kill the churn)

- [x] 3.1 `analyze_scenario` docstring + lean template now state it is the single front door that CREATES the session; do NOT also call `manage_session(init)` for the same scenario.
- [x] 3.2 `manage_session` docstring now points to `analyze_scenario` as the normal entry for a fresh scenario (init demoted to the explicit-alternative entry; kept for existing-session ops / multi-test).

## 4. Guaranteed init-response guidance (API mirrors desktop)

- [x] 4.1 `manage_session(action="init")` AND `analyze_scenario` (the lean front door) attach a compact `api_guidance` bundle for API sessions, via `utils/requests_guidance.build_api_init_guidance()` (reuses the single-source recipe constants), mirroring the `desktop_guidance` injection. init keys off loaded RequestsLibrary; analyze_scenario keys off the detected API session-type / suggested RequestsLibrary (it is only recommended, not loaded, at analyze time).
- [x] 4.2 Desktop guidance unchanged. Policy decided: **on by default** when RequestsLibrary is present, opt out with `ROBOTMCP_API_GUIDANCE=off` (documented in the templates guide).
- [x] 4.3 Unit test: init with RequestsLibrary returns `api_guidance`; init without it does not; opt-out respected (`tests/fastmcp/test_api_init_guidance.py`, 3 tests). Desktop init unchanged (existing desktop tests still green).

## 5. Tighten dense docstrings (gate-validated)

- [x] 5.1 `find_keywords` already leads with "WHEN TO USE THIS TOOL"; `execute_step` already leads with its when-to-call + no-guess rule. `get_keyword_info` lead promoted to state the when-to-call + the 4 modes up front (previously buried in the `mode:` arg). CONSERVATIVE per the standing docstring-risk caution: aggressive removal of the OBS-19/21 secondary paragraphs was NOT done (it needs the full reference-model gate; deferred to 7.3's gated pass).
- [x] 5.2 No tool names or signatures changed — prose only.

## 6. Remove dead template files

- [x] 6.1 Deleted the unloaded `domains/instruction/templates/*.txt` files (7 files) — they were never read by any code path (only the classmethods are used).
- [x] 6.2 Confirmed no code path (or test) references the deleted files (`grep` clean; full suite green after deletion).

## 7. Validate against the instruction-quality gate

- [x] 7.1 No regression — an IMPROVEMENT on BOTH axes. Haiku A/B (n=4): churn 0.0 vs 1.0, both 100% completion. glm-4.7-flash reference A/B (suite scenario, n=3): lean 3/3 complete @ 0.0 churn / 0.89 succ vs standard 1/3 complete @ 0.67 churn / 0.86 succ. Standard's dual-session-entry churn correlated with non-completion.
- [x] 7.2 Live A/B on the API scenario (with vs without `api_guidance`). FIRST closed a delivery gap: the lean default steers agents through `analyze_scenario` (not `manage_session(init)`), so the `api_guidance` bundle now ALSO rides the `analyze_scenario` response for API sessions (keyed off the detected API session-type / suggested RequestsLibrary, since it is only *recommended* — not loaded — at analyze time). restful-booker A/B (MiniMax-M3, n=3, lean default): **api_guidance ON = 58.0 avg turns vs OFF = 88.7 (~35% fewer)** — OFF hit the tarpit (one run 146 calls). Validated by 5 end-to-end tests (`tests/fastmcp/test_api_init_guidance.py`: init + analyze_scenario paths, non-API absence, opt-out).
- [x] 7.3 Recaptured the reference baseline under the new lean default (`E2E_CAPTURE_BASELINE=1 E2E_RUNS=5 E2E_MODELS=MiniMax-M3`): MiniMax-M3 on `minimax_basic_list` = **1.0 across every metric** (success/hit/completion/first-try/artifact), IQR 0.0, `_validated: True`, provenance stamped (`captured_at`, `captured_pin=MiniMax-M3`, git sha). The reference is at ceiling under lean — the "no decrease" ratchet holds. (`minimax_basic_list` is generic, so the api_guidance change does not affect it — it measures the lean default cleanly.)

## 8. Docs + unit tests

- [x] 8.1 Fast unit tests (`tests/unit/test_refactor_mcp_instructions.py`, 6 tests): lean is the default; old templates remain selectable; resolver length materially smaller than standard; api_guidance bundle is compact + load-bearing.
- [x] 8.2 Docs updated: `docs/INSTRUCTION_TEMPLATES_GUIDE.md` (new `lean` default section + canonical order + `ROBOTMCP_API_GUIDANCE` row + Example 1), `tests/e2e/README.md` (default is the lean spine + init-injection note).
- [x] 8.3 `openspec validate refactor-mcp-instructions --strict` passes.
