# desktop-process-arg-hints Specification

## Purpose
TBD - created by archiving change platynui-visible-safe-targeting. Update Purpose after archive.
## Requirements
### Requirement: Detect Process arguments RF would misparse as named arguments
When `Start Process` / `Run Process` receives a positional argument that contains `=` and would be consumed by Robot Framework as a named argument (dash-prefixed left side that is not a valid identifier and not a known Process configuration prefix such as `env:`, `shell`, `cwd`, `alias`, `stdout`, `stderr`), the system SHALL emit a hint recommending the `\=` escape. The argument itself SHALL NOT be modified.

#### Scenario: LibreOffice UserInstallation argument
- **WHEN** a desktop session executes `Start Process` with the argument `-env:UserInstallation=file:///tmp/profile`
- **THEN** the response contains a hint recommending `-env:UserInstallation\=file:///tmp/profile` and the executed command is unchanged

#### Scenario: Legitimate Process configuration not flagged
- **WHEN** `Start Process` receives `env:HOME=/tmp/home` or `shell=True`
- **THEN** no misparse hint is emitted

### Requirement: Reactive hint on Process launch failure
When a `Start Process` / `Run Process` execution fails and a dash-prefixed `=`-containing argument is present, the failure response SHALL include the same escape hint.

#### Scenario: Failed launch gets the hint
- **WHEN** a `Start Process` call fails and its arguments include `-env:UserInstallation=file:///tmp/profile`
- **THEN** the error response hints include the `\=` escape recommendation

