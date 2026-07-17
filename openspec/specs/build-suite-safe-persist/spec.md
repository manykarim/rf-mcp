# build-suite-safe-persist Specification

## Purpose
TBD - created by archiving change build-suite-safe-persist. Update Purpose after archive.
## Requirements
### Requirement: build_test_suite can persist the generated suite to disk safely
`build_test_suite` (and the underlying `TestBuilder.build_suite`) SHALL accept an
optional `output_path`. When provided and the build succeeds, the system SHALL
write the generated `.robot` text to that path using plain UTF-8 file I/O
(creating parent directories as needed), preserving the text byte-for-byte with
**no** Robot Framework variable resolution and **no** escape-sequence expansion.
The response SHALL include `output_path` and `output_bytes` on success, or
`output_error` if the write fails — and a write failure SHALL NOT fail the build.
When `output_path` is empty/omitted, behaviour is unchanged (text is returned
only).

#### Scenario: persisted suite is byte-for-byte identical to rf_text
- **WHEN** `build_test_suite(output_path="/…/suite.robot")` succeeds for a session
  whose steps include a multi-line argument and a `${var}`-assigned step
- **THEN** the file at that path equals the response `rf_text` exactly — escaped
  newlines remain `\n` (not raw line breaks) and `${var}` references remain
  literal (not resolved to their runtime values) — and the response reports
  `output_path` and `output_bytes`

#### Scenario: persisted suite parses as Robot Framework
- **WHEN** the persisted suite is loaded with `robot.api.TestSuiteBuilder`
- **THEN** it parses without error and yields at least one test case

#### Scenario: write failure is a soft error
- **WHEN** `output_path` cannot be written (e.g. an unwritable location)
- **THEN** the response still reports `success: true` with the generated `rf_text`,
  plus an `output_error` describing the failure, and no `output_path`

#### Scenario: omitting output_path is unchanged
- **WHEN** `build_test_suite` is called without `output_path`
- **THEN** no file is written and the response contains no `output_path`/`output_bytes`

### Requirement: guidance steers persistence away from the Create File keyword
The `build_test_suite` tool documentation SHALL instruct callers to persist a
generated suite via `output_path` and SHALL warn against writing `rf_text` through
the Robot Framework `Create File` keyword, because RF resolves `${variables}` and
expands `\n`/`\t` escapes inside the argument and thereby corrupts the suite.

#### Scenario: docstring documents the safe path and the trap
- **WHEN** an agent reads the `build_test_suite` tool description
- **THEN** it states that `output_path` persists the suite safely and that writing
  `rf_text` via `Create File` corrupts it (variables resolved, escapes expanded)

