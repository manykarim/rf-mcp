# desktop-test-lifecycle-preservation Specification

## Purpose
TBD - created by archiving change desktop-evidence-and-display-scoping. Update Purpose after archive.
## Requirements
### Requirement: Active-test state survives RF context recreation
Recreating or reusing the RF native context for a session (including the dry-run path used by `build_test_suite`) SHALL preserve the session's `current_run_test` / `current_res_test` state, so `manage_session(action="end_test")` after a successful `start_test` always finds the active test.

#### Scenario: start_test → build_test_suite → end_test
- **WHEN** an agent calls `start_test`, then `build_test_suite` (triggering the dry-run context path), then `end_test`
- **THEN** `end_test` succeeds and does not report "No active test to end"

#### Scenario: Context reuse seeds from existing state
- **WHEN** `create_context_for_session` runs for a session that already has a context with an active test
- **THEN** the stored context entry retains the existing active-test references instead of being reset to None

