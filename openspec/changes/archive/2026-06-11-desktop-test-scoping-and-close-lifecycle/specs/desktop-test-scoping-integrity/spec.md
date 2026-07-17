# Spec: desktop-test-scoping-integrity

## ADDED Requirements

### Requirement: start_test works regardless of call order
`manage_session(action="start_test")` SHALL create the session's RF native context when it does not exist yet (using the session's library configuration), so starting a test before the first `execute_step` succeeds.

#### Scenario: start_test immediately after init
- **WHEN** an agent initializes a session and calls `start_test` before any `execute_step`
- **THEN** start_test succeeds and does not report "No context for session"

### Requirement: start_test is atomic across layers
When the RF-context layer of `start_test` cannot be established, the call SHALL fail with the underlying error and SHALL NOT activate the registry's multi-test mode — a successful start_test always leaves both layers consistent.

#### Scenario: Context failure fails the whole call
- **WHEN** RF context creation fails during start_test
- **THEN** the response has `success: false` with the cause, and the session's registry has no active test and is not in multi-test mode

### Requirement: Steps recorded during an active test land in that test
Steps executed between a successful `start_test` and `end_test` SHALL be recorded into the named test, including when `build_test_suite` is called repeatedly in between (stepwise suite building).

#### Scenario: Run-3 interleaving
- **WHEN** an agent runs start_test → steps → build_test_suite → steps → build_test_suite → steps → end_test
- **THEN** all recorded steps appear in the named test and in the generated suite body, and none accumulate in suite-level storage

### Requirement: end_test is registry-first
`manage_session(action="end_test")` SHALL succeed whenever the registry has an active test, ending it; a context-layer "No active test to end" SHALL be reported as a warning on the successful response, not as a failure. When the registry also has no active test, the call fails as before.

#### Scenario: Registry test active, context layer out of sync
- **WHEN** end_test runs while the registry has an active test but the RF context layer has none
- **THEN** the response is successful, the registry test is ended, and a warning notes the context-layer miss

### Requirement: Suite-level step accumulation is visible
`build_test_suite` SHALL report `suite_level_step_count` and SHALL emit a top-level warning when the session is in multi-test mode and more recorded steps sit outside any named test than inside, stating that suite-level steps are not rendered in test bodies and how to avoid it.

#### Scenario: Orphaned steps warned
- **WHEN** a multi-test session has 43 suite-level steps and 3 in-test steps
- **THEN** the build response includes `suite_level_step_count: 43` and a warning naming the imbalance

#### Scenario: Healthy session silent
- **WHEN** all recorded steps are inside named tests
- **THEN** no suite-level warning is emitted
