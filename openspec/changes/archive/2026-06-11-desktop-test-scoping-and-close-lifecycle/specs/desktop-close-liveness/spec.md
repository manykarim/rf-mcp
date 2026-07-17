# Spec: desktop-close-liveness

## ADDED Requirements

### Requirement: Post-close still-running hint
After a successful desktop `Close Window` step in a session whose AUT process id is known, the system SHALL check whether that process is still alive and SHALL append a hint when it is, stating that a residual frame (e.g. a start center) may remain and naming `Terminate Process` as the hard stop. The check never fails the step and is skipped when no AUT pid is known.

#### Scenario: LibreOffice survives the document close
- **WHEN** `Close Window` succeeds and the launched AUT process is still running
- **THEN** the step response contains a hint that the application process is still running and a residual frame may remain

#### Scenario: Clean exit stays silent
- **WHEN** `Close Window` succeeds and the AUT process has exited
- **THEN** no close-liveness hint is emitted

#### Scenario: Unknown AUT pid skips the check
- **WHEN** the session never launched a process via the desktop launch path
- **THEN** no liveness check runs and no hint is emitted
