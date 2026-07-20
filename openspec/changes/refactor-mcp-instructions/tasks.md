## 1. Baseline the current state (measure before changing)

- [ ] 1.1 Capture the reference-model instruction-quality baseline under the CURRENT default template on the validated scenario set (`E2E_CAPTURE_BASELINE=1`), so the refactor's effect is measured, not assumed.
- [ ] 1.2 A/B 2–3 candidate lean spines (≈`minimal` + one canonical-order line; and the `refactored` draft) on the reference model over suite-exec + a generic scenario; pick the winner by success/completion/right-order/turns.
  - Haiku/Claude-Code-CLI axis DONE (2026-07-20, after the cold-start stdio fix a467c5a made CLI measurement possible): n=5/candidate sweep, candidates differ only in server `instructions`, neutral prompt → 100% completion for all (5/5 built + success). Firmed ranking (avg churn / avg calls / avg dur): `checklist` 0.0 / 4.4 / 22.2s (winner) ≈ `example` 0.0 / 5.0 / 27.6s > `minimal` 0.2 / 5.0 / 30.8s > `terse` 0.4 / 4.4 / 26.0s. Winner draft captured in design.md Decision 1. Reference-model (glm/qwen) A/B still to run at apply.

## 2. Lean order-explicit default template

- [ ] 2.1 Add a lean default template to `domains/instruction/value_objects.py` using the `checklist` spine shape (design.md Decision 1 concrete draft; ~500–800 chars — the Haiku sweep showed order/entry EXPLICITNESS beats raw brevity for CLI clients): numbered order — `analyze_scenario` first (it creates the session; NEVER also call `manage_session(init)`) → discover-if-unknown (`find_keywords`) → one keyword per `execute_step` (arguments=list of strings) → retry-once-on-failure → `get_locator_guidance` for locators/API + `get_session_state` for web DOM → FINAL `build_test_suite`. No per-tool docstring-echo. Keep the explicit unified-session-entry line verbatim (it is the measured churn-killer).
- [ ] 2.2 Make it the resolver default (`server.py::_resolve_server_instructions` / the instruction service default), keeping the old templates selectable via env for rollback.
- [ ] 2.3 Retire `discovery_first` (6081 ch) as an option (or collapse to the spine + recovery-ladder pointer); trim `detailed` toward the spine shape.

## 3. Unify session entry (kill the churn)

- [ ] 3.1 In the lean template + `analyze_scenario` docstring: state `analyze_scenario` is the single front door that creates the session; do NOT also call `manage_session(init)` for a new scenario.
- [ ] 3.2 Update `manage_session` docstring to point to `analyze_scenario` as the normal entry for a fresh scenario (keep manage_session for explicit session ops / multi-test).

## 4. Guaranteed init-response guidance (API mirrors desktop)

- [ ] 4.1 In `manage_session(action="init")`, when RequestsLibrary is among the libraries, attach an `api_guidance` bundle (compact RequestsLibrary rules + pointer), reusing `utils/requests_guidance.py`, mirroring the existing `desktop_guidance` injection.
- [ ] 4.2 Keep the desktop guidance behaviour unchanged; decide the env-flag vs on-by-default policy (per design open question) and document it.
- [ ] 4.3 Unit test: init with RequestsLibrary returns `api_guidance`; init without it does not; desktop init unchanged.

## 5. Tighten dense docstrings (gate-validated)

- [ ] 5.1 Tighten `get_keyword_info` (promote the 4 modes; compress OBS-19/21 externalisation to one line), `execute_step` (compress `record`/`pre_validate_timeout_ms` paragraphs), `find_keywords` (lead with when-to-use) — lead with load-bearing guidance.
- [ ] 5.2 Do NOT change tool names or signatures. Keep changes to prose only.

## 6. Remove dead template files

- [ ] 6.1 Delete the unloaded `domains/instruction/templates/*.txt` files (they diverge from the live classmethods) — or wire the loader to read them as the single source of truth; pick one and make source-of-truth unambiguous.
- [ ] 6.2 Confirm no code path references the deleted files.

## 7. Validate against the instruction-quality gate

- [ ] 7.1 Re-run the `agentic-e2e-instruction-quality` gate (reference model, validated scenario set) after the template + docstring + init-injection changes; confirm NO regression vs the pre-change baseline (or recapture as a deliberate, reviewed improvement).
- [ ] 7.2 Live A/B on the API scenario: with vs without the init `api_guidance` — confirm fewer turns / higher first-try response-access correctness.
- [ ] 7.3 Recapture and commit the reference baseline under the new default (the reviewed "no decrease" ratchet); update `reference_pin`/provenance.

## 8. Docs + unit tests

- [ ] 8.1 Fast unit tests (no network): the lean default is selected by default; the old templates remain selectable; the resolver length is materially smaller; the api_guidance injection fires correctly.
- [ ] 8.2 Update docs (`tests/e2e/README.md` instruction section + any instruction docs) with the new default, the canonical order, and the init-injection behaviour.
- [ ] 8.3 `openspec validate refactor-mcp-instructions --strict` passes.
