# desktop-process-keyword-resolution Specification

## Purpose
TBD - created by archiving change desktop-stepwise-followups. Update Purpose after archive.
## Requirements
### Requirement: Unqualified Process keywords resolve to the Process library

The system SHALL resolve the core `Process` library's keywords by their
unqualified names, so `get_library_for_keyword` returns `Process` for keywords
such as `Start Process`, `Run Process`, `Terminate Process`,
`Process Should Be Running`, and `Wait For Process`. This closes the gap where
core RF libraries registered via static definitions exposed no keyword→library
map, leaving their unqualified keywords unresolvable.

#### Scenario: Start Process resolves to Process
- **WHEN** the plugin manager is asked for the library of `Start Process`
- **THEN** it returns `Process` (case-insensitively, same for `Run Process`,
  `Terminate Process`, `Process Should Be Running`, `Wait For Process`)

#### Scenario: qualified names still work
- **WHEN** a caller uses the dotted form `Process.Start Process`
- **THEN** resolution is unchanged (the explicit library prefix wins)

### Requirement: A desktop session can launch its app with unqualified Process keywords

The system SHALL allow a desktop (PlatynUI) session to register and execute
unqualified `Process` keywords so launching the application under test does not
require dotted names. On first use of an unqualified `Process` keyword, the
`Process` library MUST be loaded and registered into the live RF namespace.

#### Scenario: unqualified Start Process is registered for a desktop session
- **WHEN** a desktop session executes an unqualified `Start Process` keyword for
  the first time
- **THEN** the pre-execution library registration resolves it to `Process`,
  loads/registers `Process` into the RF context, and the keyword is found (no
  "No keyword with name 'Start Process' found")

#### Scenario: non-desktop sessions are unaffected
- **WHEN** a web/api/mobile session is configured
- **THEN** its library set and keyword resolution are unchanged by this change

