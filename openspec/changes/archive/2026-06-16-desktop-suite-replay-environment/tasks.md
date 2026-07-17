# Tasks: desktop-suite-replay-environment

## 1. Generator preamble

- [x] 1.1 Add `_inject_desktop_replay_environment(suite, session)` to `test_builder.py`: build the `Prepare Desktop Display Environment` BddKeyword (DISPLAY pin from `classify_bound_display_detailed()["display"]` when known; XDG_SESSION_TYPE/GDK_BACKEND/QT_QPA_PLATFORM pins; Remove WAYLAND_DISPLAY; [Documentation] stating the wayland-0/Wayland-stub rationale), append to `suite.bdd_keywords`
- [x] 1.2 Wire in `build_suite` for `session.is_desktop_session() is True` (after BDD transformation): set `suite.setup` only when no session/user setup exists, else add the response hint; ensure `OperatingSystem` lands in the imports
- [x] 1.3 Audit existing desktop suite-shape tests for new Settings/Keywords lines and update deliberately — **zero fallout: full suite green unchanged**

## 2. Tests

- [x] 2.1 `tests/unit/test_suite_replay_environment.py`: desktop suite contains preamble keyword + OperatingSystem import + Suite Setup wiring; unknown display omits DISPLAY line only; web/API suites untouched; user suite_setup preserved with hint; composes with `bdd_style=True`
- [x] 2.2 Pin RF OperatingSystem `Remove Environment Variable` ignore-missing semantics (execute via engine: remove a non-existent var succeeds)

## 3. Validation

- [x] 3.1 Full unit suite green (baseline 6795 passed + 1 skipped; no regressions) — **6805 passed + 1 skipped, +10 net**
- [x] 3.2 Standalone replay smoke: regenerate a desktop suite shape with the preamble and run it via plain `robot` from a Wayland-shaped env on `:100` (no Wayland-provider error) — **PASS (results in /tmp/robot-smoke/results4); smoke also exposed + fixed two render defects: dash-arg `=` escaping (D4) and last-dot prefix removal (D5)**
