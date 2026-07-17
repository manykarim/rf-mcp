# Spec: desktop-evidence-integrity

## ADDED Requirements

### Requirement: Screenshot success is verified against the filesystem
After a successful desktop-session screenshot keyword whose requested path or returned value names a file, the system SHALL verify the file exists on disk and SHALL append an `evidence_missing` hint to the step response when it does not. The check never fails the step.

#### Scenario: Ghost screenshot warned
- **WHEN** `Screenshot.Take Screenshot  /tmp/run/shots/after-typing.jpg` returns success but no file exists at that path (or the keyword's returned path)
- **THEN** the response contains a hint of type `evidence_missing` naming the path

#### Scenario: Real screenshot stays clean
- **WHEN** the screenshot file exists after the keyword succeeds
- **THEN** no `evidence_missing` hint is emitted

### Requirement: User-requested screenshot paths are reachable
`PlatynUI.BareMetal.Take Screenshot` with an absolute path outside the RF output directory SHALL succeed by saving inside the output directory and copying to the requested path, when the requested path is under an allowed root (`/tmp` or the configured screenshot directory). Paths outside allowed roots SHALL fail with a hint naming the allowed roots. Existing files SHALL NOT be overwritten by the copy.

#### Scenario: Operator shots directory works
- **WHEN** `Take Screenshot  /tmp/dsef-libre-run2/shots/after-launch.png` executes in a desktop session
- **THEN** the keyword succeeds and the file exists at the requested path

#### Scenario: Disallowed path refused with guidance
- **WHEN** the requested path is `/etc/shot.png`
- **THEN** the step fails and the response hints at the allowed roots

### Requirement: Evidence keywords count as scaffolding for the empty-suite warning
The empty/near-empty suite classifier SHALL treat `take screenshot`, `create directory`, `is process running`, `get process id`, `terminate process`, and `wait for process` as launch/setup scaffolding, so a stepwise session whose only recorded steps are launch + evidence still triggers the warning when failures occurred.

#### Scenario: Run-3 shape now warns
- **WHEN** a session executed 100 steps with 41 failures and the suite body contains only Create Directory, Start Process, Sleep, process probes, and Take Screenshot steps
- **THEN** `build_test_suite` emits the top-level empty-suite warning
