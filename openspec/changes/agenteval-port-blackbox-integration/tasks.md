## 1. Support library + ergonomics gate

- [x] 1.1 Add `tests/agenteval/integration/mcp_result.py` support keywords for asserting on rf-mcp's
  JSON-in-text tool results (`Parse Tool Result` → dict, `Result Field Should Be`, `Result Should Contain Field`).
- [x] 1.2 Port the gate pair to `tests/agenteval/integration/`: `test_fastmcp_context_keywords` (1) and
  `test_mcp_e2e_builtin_only` (21). Run green; measure per-suite wall-clock + assertion ergonomics and
  decide per-test vs per-suite server reuse (design D3).
- [x] 1.3 Retire the two pytest originals once their ports are green.

## 2. Port the remaining clean candidates (each: read original -> port -> green -> delete original)

- [x] 2.1 `test_adr010_e2e` (24)
- [x] 2.2 `test_adr_integration` (20)
- [x] 2.3 `test_variable_file_loading` (14)
- [x] 2.4 `test_variable_handling_e2e` (9)
- [x] 2.5 `test_library_preferences` (6)
- [x] 2.6 `test_intent_fallback_e2e` (5)
- [x] 2.7 `test_keyword_routing_e2e` (5)
- [x] 2.8 `test_fastmcp_argument_resolution` (3)
- [x] 2.9 `test_recommend_libraries_keywords` (1)

## 3. Desktop-gated candidate

- [ ] 3.1 Port `test_platynui_gnome_apps_e2e` (20) with a display/opt-in gate (needs a desktop + GNOME apps;
  NOT part of the headless always-on tier); retire the pytest original once green in a desktop environment.

## 4. CI wiring

- [ ] 4.1 The keyless ports join the deterministic always-on tier of the `agenteval-harness` CI job; confirm green.
- [x] 4.2 Confirm the desktop-gated port (3.1) skips cleanly on the headless runner.

## 5. Verify + wrap-up

- [x] 5.1 Coverage check: each ported test asserts the same observable facts as its origin; run the full pytest
  suite and confirm nothing else depended on the 12 removed files.
- [x] 5.2 Report ported files/tests, any candidate that fell back to pytest (per the spec's non-translatable
  rule), and the before/after `tests/integration/` count. Note Phase 2b (the 10 `partial` files) as a follow-up.
- [x] 5.3 `openspec validate agenteval-port-blackbox-integration --strict` passes.
