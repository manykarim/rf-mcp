# Spec: batch-resume-argument-fidelity

## ADDED Requirements

### Requirement: Resume re-executes the failed step with its original arguments
`resume_batch` retrying a failed step SHALL pass the step's original argument list unchanged.

#### Scenario: Sleep retains its duration
- **WHEN** a batch step `BuiltIn.Sleep  2s` fails and `resume_batch` retries it
- **THEN** the retried call receives arguments `["2s"]`, never an empty list

### Requirement: fix_steps accept both argument keys
`resume_batch` SHALL resolve `fix_steps` arguments with the same dual-key semantics as `execute_batch` (`arguments` canonical, `args` alias).

#### Scenario: Canonical key honored
- **WHEN** `resume_batch` receives `fix_steps=[{"keyword": "Log", "arguments": ["hello"]}]`
- **THEN** the fix step executes with `["hello"]`

#### Scenario: Alias key still honored
- **WHEN** a fix step uses `args=["hello"]`
- **THEN** behavior is identical
