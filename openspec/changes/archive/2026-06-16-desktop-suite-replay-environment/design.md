# Design: desktop-suite-replay-environment

## Context

Validated facts from the 2026-06-12 standalone replay session:
- Plain `robot` on a Wayland host fails before any UI work: upstream session detection (`XDG_SESSION_TYPE` authoritative) selects the Wayland backend whose screenshot/window providers are stubs; GTK AUTs additionally face the `wayland-0` fallback escape.
- The fix is timing-safe: upstream `PlatynUI.BareMetal.runtime` is a lazy `@cached_property`, so env pinned in `Suite Setup` lands before the first keyword creates the runtime. Hand-patched suite passed with the user's exact invocation.
- Renderer mechanics: `_generate_rf_text` emits `*** Keywords ***` from `suite.bdd_keywords` whenever non-empty, regardless of `bdd_style` (test_builder.py ~2888); `BddKeyword(name, steps)` is the carrier; `suite.setup` renders as `Suite Setup` in Settings.

## Goals / Non-Goals

**Goals:** generated desktop suites replay with bare `robot <suite>` on the recording host (display still provisioned); never clobber a user-defined suite setup; zero effect on web/API/mobile suites.

**Non-Goals:** provisioning the isolated display itself (Xephyr/Xvfb stay the bootstrap script's job — a suite must not spawn X servers); Windows/macOS-specific preambles (the pins are inert there); retrofitting previously generated suites.

## Decisions

### D1 — Preamble as a rendered keyword, reusing the bdd_keywords channel
Build a `BddKeyword(name="Prepare Desktop Display Environment", steps=[TestCaseStep…])` and append to `suite.bdd_keywords` for desktop sessions in `build_suite` (after BDD transformation so it composes with bdd_style suites). Steps: `Set Environment Variable XDG_SESSION_TYPE x11`, `… GDK_BACKEND x11`, `… QT_QPA_PLATFORM xcb`, `Remove Environment Variable WAYLAND_DISPLAY` (RF OperatingSystem semantics: silently ignores a missing variable — verify in test), and `Set Environment Variable DISPLAY <bound>` first when the session's display is known (`classify_bound_display_detailed()["display"]`). A `[Documentation]` line states why (wayland-0 fallback + Wayland-stub providers).
*Alternative rejected*: a new `suite.custom_keywords` field — the renderer already has exactly one Keywords-section channel; a second one doubles rendering paths for no benefit.

### D2 — Suite Setup wiring respects the user's setup
`suite.setup` is set to the preamble keyword ONLY when neither `session.suite_setup` nor an existing `suite.setup` is present. When a user setup exists, the keyword is still emitted (manually callable) and the build response gains a hint naming it.
*Alternative rejected*: composing via `Run Keywords … AND <user setup>` — silently re-ordering a user's setup is the kind of magic this codebase keeps clawing back.

### D3 — Desktop gate and import
Gate: `session.is_desktop_session() is True` (same strict-callable check as the executor). `OperatingSystem` is added to `all_imports` alongside the injection so the Settings section carries the library.

## Risks / Trade-offs

- [Suites recorded on `:100` hardcode that DISPLAY] → matches the already-hardcoded `env:DISPLAY=:100` in the recorded `Start Process` steps; the suite is internally consistent. Replaying elsewhere requires editing both, as today.
- [Existing tests pin desktop rf_text shape] → audit task; expected fallout is line-count/Settings assertions in desktop suite tests.
- [`Remove Environment Variable` behavior on absent var] → pinned by a test against RF's OperatingSystem (documented as ignore-missing).

## Migration Plan

Additive; revert to roll back. Baseline 6795 passed + 1 skipped stays green.

## Open Questions

(none — mechanics were proven by the hand-patched suite)

### D4 — Dash-arg `=` escaping at render time (discovered by the replay smoke)
The renderer emitted recorded args verbatim; an unescaped `-env:UserInstallation=…` regenerated unescaped and misparsed as a named argument on replay (run 4 only survived because the agent recorded the pre-escaped form). `_escape_robot_argument` now escapes the first `=` in dash-prefixed args — universally safe because Python kwarg names can never start with `-`, and RF unescapes `\=` back to the literal at runtime.

### D5 — Prefix removal uses the last dot (discovered by the replay smoke)
`_remove_library_prefix` split on the FIRST dot, mangling `PlatynUI.BareMetal.Take Screenshot` into the unresolvable `BareMetal.Take Screenshot`. Now `rsplit(".", 1)[1]` — third member of the dotted-library-name bug family (after the import derivation and context-prefix fixes of 2026-06-12).
