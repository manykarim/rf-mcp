# platynui-visibility-guarantees Specification

## Purpose
TBD - created by archiving change platynui-focused-execution. Update Purpose after archive.
## Requirements
### Requirement: Report AUT window visibility state

The system SHALL determine and expose whether the AUT top-level window is
mapped, on-screen (within the desktop bounds), and visible, using the
PlatynUI accessibility attributes (e.g. `IsVisible`, `IsInView`, `Bounds`)
together with the desktop bounds. This state SHALL be available through
`get_session_state` for desktop sessions.

#### Scenario: Visible mapped window
- **WHEN** the AUT window is mapped within the desktop bounds with non-zero
  size
- **THEN** `get_session_state` (ui_tree section) reports the window as
  visible/on-screen for that application

#### Scenario: Unmapped or off-screen window
- **WHEN** the AUT window is not mapped, has zero size, or lies entirely
  outside the desktop bounds
- **THEN** `get_session_state` reports the window as not visible/off-screen

### Requirement: Init-time visibility guidance for launched apps

When a desktop session launches or attaches to an AUT, the system SHALL
surface guidance and state indicating whether the app is shown, steering
agents to launch apps visibly rather than relying on a tree that exists
without a shown window.

#### Scenario: App started but no window shown yet
- **WHEN** an AUT process has started but its top-level window is not yet
  mapped
- **THEN** the system indicates the window is not yet visible and guides the
  agent to wait for the window before interacting

### Requirement: Execution-time visibility precondition

The system SHALL check, before dispatching a pointer or keyboard operation in
a DESKTOP_TESTING session, that the resolved AUT window is visible. If it is
not visible, the system SHALL emit a warning by default, and SHALL fail-fast
when fail-fast is enabled (per-call or per-session), instead of silently
reporting success on an operation that could not have visibly acted.

#### Scenario: Operation against a non-visible window warns by default
- **WHEN** a desktop operation targets an AUT whose window is not visible
- **THEN** the step result includes a visibility warning identifying the
  non-visible window

#### Scenario: Fail-fast on non-visible window
- **WHEN** fail-fast is enabled and a desktop operation targets a
  non-visible AUT window
- **THEN** the operation fails with a clear visibility error rather than
  reporting success

### Requirement: Visibility reflected in generated suites and reports

The system SHALL ensure visibility warnings/failures discovered during
stepwise execution are surfaced to the agent so they are not lost when
`build_test_suite` produces the final suite (i.e. a silently-non-visible run
does not yield a clean-looking passing suite).

#### Scenario: Stepwise visibility warning is reported before suite build
- **WHEN** an operation produced a visibility warning during stepwise
  execution
- **THEN** that warning is available to the agent prior to and during
  `build_test_suite`

