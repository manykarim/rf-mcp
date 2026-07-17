# Spec: desktop-suite-replay-environment

## ADDED Requirements

### Requirement: Desktop suites carry a replay-environment preamble
`build_test_suite` for a desktop (PlatynUI) session SHALL emit a `Prepare Desktop Display Environment` keyword (Keywords section) that pins `XDG_SESSION_TYPE=x11`, `GDK_BACKEND=x11`, `QT_QPA_PLATFORM=xcb`, removes `WAYLAND_DISPLAY`, and sets `DISPLAY` to the session's bound display when known — and SHALL include `OperatingSystem` in the suite imports.

#### Scenario: Generated desktop suite is self-sufficient
- **WHEN** `build_test_suite` runs for a desktop session bound to display `:100`
- **THEN** the rf_text contains `Library         OperatingSystem`, a `Prepare Desktop Display Environment` keyword with the four pins and the `DISPLAY    :100` line, and `Suite Setup     Prepare Desktop Display Environment`

#### Scenario: Unknown display omits only the DISPLAY pin
- **WHEN** the session has no bound display at build time
- **THEN** the preamble keyword is emitted without a `DISPLAY` line, retaining the backend pins

#### Scenario: Non-desktop suites unchanged
- **WHEN** `build_test_suite` runs for a web or API session
- **THEN** no preamble keyword, no `OperatingSystem` import, and no Suite Setup are added by this feature

### Requirement: User-defined suite setup takes precedence
When the session defines its own suite setup, the preamble SHALL NOT replace it as `Suite Setup`; the keyword is still emitted in the Keywords section and the build response notes it.

#### Scenario: Existing setup preserved
- **WHEN** a desktop session has `suite_setup` defined via manage_session
- **THEN** the generated `Suite Setup` is the user's keyword, the preamble keyword still appears in the Keywords section, and the response hints that it can be invoked manually

### Requirement: Preamble executes before the PlatynUI runtime exists
The preamble SHALL be wired so it runs before any PlatynUI keyword (Suite Setup ordering), relying on the upstream lazy runtime; replaying the generated suite with plain `robot` on a Wayland host SHALL NOT fail with the Wayland-provider error.

#### Scenario: Standalone replay on a Wayland host
- **WHEN** a generated desktop suite runs via plain `robot` from a Wayland shell with the isolated display provisioned
- **THEN** the PlatynUI runtime initializes on the X11 backend (no "Wayland screenshot provider" error)
