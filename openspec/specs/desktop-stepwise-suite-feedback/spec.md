# desktop-stepwise-suite-feedback Specification

## Purpose
TBD - created by archiving change platynui-visible-safe-targeting. Update Purpose after archive.
## Requirements
### Requirement: Session step accounting
The execution session SHALL count every `execute_step` keyword execution (`executed_step_count`) and every failed execution (`failed_step_count`), independent of whether the step is recorded into the session's step list.

#### Scenario: Counters track success and failure
- **WHEN** a session executes 5 steps of which 2 fail
- **THEN** `executed_step_count` is 5, `failed_step_count` is 2, and the recorded step count is at most 3

### Requirement: execute_step surfaces running counts
The `execute_step` response SHALL include `steps_executed` and `steps_recorded` fields alongside the existing `recorded` flag, so an agent can detect divergence while still executing.

#### Scenario: Failed step response shows divergence
- **WHEN** the 4th executed step fails in a session where only 1 step was recorded
- **THEN** the failure response contains `steps_executed: 4` and `steps_recorded: 1`

### Requirement: build_test_suite reports executed/recorded/failed statistics
The `build_test_suite` response SHALL include `steps_executed`, `steps_recorded`, and `steps_failed` in its statistics.

#### Scenario: Statistics present on successful build
- **WHEN** `build_test_suite` succeeds for a session with 20 executed, 1 recorded, 19 failed steps
- **THEN** the response statistics contain `steps_executed: 20`, `steps_recorded: 1`, `steps_failed: 19`

### Requirement: Empty or near-empty suite warning
`build_test_suite` SHALL emit a top-level `warning` when the session has at least 3 executed steps, at least 1 failed step, and the generated suite body contains no steps beyond launch/setup steps. The warning MUST state that failed steps are never recorded and direct the operator to `execute_step` results.

#### Scenario: LibreOffice regression case warns
- **WHEN** a session executed 20 steps, 19 failed, and the suite body contains only a `Start Process` step
- **THEN** the `build_test_suite` response includes a warning naming the executed and recorded counts and stating that failed steps are not recorded

#### Scenario: Healthy session does not warn
- **WHEN** a session executed 10 steps, all succeeded, and 8 were recorded (2 inspection-only)
- **THEN** no empty-suite warning is emitted

