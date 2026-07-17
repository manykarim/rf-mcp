# Spec: desktop-suite-replay-environment

## MODIFIED Requirements

### Requirement: Desktop suites carry a replay-environment preamble
`build_test_suite` for a desktop (PlatynUI) session SHALL emit a `Prepare Desktop Display Environment` keyword (Keywords section) that pins `XDG_SESSION_TYPE=x11`, `GDK_BACKEND=x11`, `QT_QPA_PLATFORM=xcb`, pins `GTK_A11Y=atspi` (the AT-SPI backend by name — not `1`, which modern GTK rejects, leaving a freshly launched GTK AUT with no accessibility tree), removes `WAYLAND_DISPLAY`, and sets `DISPLAY` to the session's bound display when known — and SHALL include `OperatingSystem` in the suite imports.

#### Scenario: Generated desktop suite is self-sufficient
- **WHEN** `build_test_suite` runs for a desktop session bound to display `:100`
- **THEN** the rf_text contains `Library         OperatingSystem`, a `Prepare Desktop Display Environment` keyword with the backend pins (`XDG_SESSION_TYPE`, `GDK_BACKEND`, `QT_QPA_PLATFORM`), the `GTK_A11Y    atspi` pin, and the `DISPLAY    :100` line, and `Suite Setup     Prepare Desktop Display Environment`

#### Scenario: Accessibility bridge pinned to the backend name, not 1
- **WHEN** the preamble is emitted for a desktop session
- **THEN** it contains `Set Environment Variable    GTK_A11Y    atspi` and does not contain `Set Environment Variable    GTK_A11Y    1`

#### Scenario: Unknown display omits only the DISPLAY pin
- **WHEN** the session has no bound display at build time
- **THEN** the preamble keyword is emitted without a `DISPLAY` line, retaining the backend pins

#### Scenario: Non-desktop suites unchanged
- **WHEN** `build_test_suite` runs for a web or API session
- **THEN** no preamble keyword, no `OperatingSystem` import, and no Suite Setup are added by this feature
