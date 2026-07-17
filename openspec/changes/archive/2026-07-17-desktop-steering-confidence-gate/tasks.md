# Tasks: desktop-steering-confidence-gate

## 1. Verdict model
- [x] 1.1 Define a `SteeringConfidence` verdict enum: `confirmed` / `unconfirmed` / `contradicted`
- [x] 1.2 `steering_confidence(...)` composer in `desktop_execution_signals.py` taking: keyword, success, verified_focus (bool), focus/visibility warnings, state_before/after snapshot, wayland_risk (bool)
- [x] 1.3 Rules: `contradicted` when success AND not verified_focus AND input-effect absent (state unchanged), OR wayland drop-risk on an unverified target; `confirmed` when verified_focus OR observed state change; else `unconfirmed`

## 2. Enforcement + opt-out
- [x] 2.1 `ROBOTMCP_PLATYNUI_STEERING_CONFIDENCE` reader (`enforce` default / `warn` opt-out), mirroring `desktop_display_safety.warn_mode`
- [x] 2.2 On `contradicted` + enforce: raise a structured steering failure whose hint says the input did not land and to re-verify focus / refocus and retry
- [x] 2.3 On `warn`: attach the verdict as a warning, do not fail

## 3. Wire into the interaction result boundary
- [x] 3.1 Reuse the existing before/after snapshot path that already feeds `input_effect_hint` (no second AT-SPI read)
- [x] 3.2 Pull `has_verified_focus()` + `FocusOutcome.warnings` + visibility warnings from the focus manager into the composer
- [x] 3.3 Attach `steering_confidence` to every interaction step result; interaction keywords only (`is_interaction_keyword`)

## 4. Tests
- [x] 4.1 `contradicted` (unverified focus + unchanged CharacterCount) fails by default; passes under `=warn` with the verdict attached
- [x] 4.2 `confirmed` via verified focus; `confirmed` via observed state change — both pass
- [x] 4.3 `unconfirmed` (no readable state, focus not verified) passes carrying the verdict (no false failure)
- [x] 4.4 Non-interaction keywords (Query / Get Attribute / Take Screenshot) carry no verdict and are unaffected
- [x] 4.5 Verdict shape is stable/machine-parseable (regression on the field name + enum values)

## 5. Deterministic validation (docker, no-LLM) — closes eval gaps G3/G4
- [x] 5.1 `docker/gate_drivers.py g3` PASS 2026-07-17: keystroke sent while a DIFFERENT window is focused leaves mousepad A's CharacterCount unchanged (5→5) → verdict `contradicted` (real AT-SPI, no LLM)
- [x] 5.2 Positive control: focused keystroke `0→5` (typed "hello") → verdict `confirmed`
  > The G3 gate surfaced a latent product bug: `node.attribute("native:Text.CharacterCount")` returns None on the new-core runtime (the bare colon-string accessor doesn't resolve). The working read enumerates `node.attributes()` + `.value()`. Fixed in `_desktop_text_count_before` via new helper `_read_native_character_count` (keyword_executor.py) — this also repairs the pre-existing `input_effect_hint`, which had been silently dead. Unit tests added.
