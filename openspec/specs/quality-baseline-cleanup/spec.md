# quality-baseline-cleanup Specification

## Purpose

Governs the correctness of code that only runs when something has already gone wrong, and
the honesty of the static-analysis baseline that is supposed to surface it.

The `Quality` job reported 17 undefined-name findings. Three were real `NameError`s on
ERROR or DEGRADED paths - an import-failure handler that logged before its logger existed,
a snapshot field referencing the name being bound by its own statement, and a helper
nested inside the wrong function - which is exactly why 7,300 passing tests never reached
them. Writing a regression test for the second uncovered three further `AttributeError`s
behind it, in a method with no callers that had never once executed.

The lesson this capability encodes: a finding on an error path is not noise because the
happy path is green, and a linter's output is only useful when mechanical debris is not
hiding the findings that matter.

## Requirements

### Requirement: Error handling paths must not raise on their own
Code that runs when something has already gone wrong - exception handlers, fallback
branches, degraded-mode paths - SHALL be executable. A handler SHALL NOT reference a name
that is unbound at the point it runs, so that a recoverable condition is reported rather
than converted into a second, more confusing failure.

#### Scenario: an import-failure handler logs instead of crashing
- **WHEN** an optional dependency cannot be imported and the module's import-failure handler runs
- **THEN** the handler emits its warning and the module continues in degraded mode, rather than raising NameError

#### Scenario: a fallback execution path is reachable
- **WHEN** the primary keyword-execution path fails and the fallback path runs
- **THEN** the fallback executes, rather than raising NameError for a helper that is not in scope there

### Requirement: No name is used before it is bound
No shipped code path SHALL reference a name before it is assigned, including within the
expression that performs the assignment.

#### Scenario: a value is not referenced inside its own constructor call
- **WHEN** a snapshot is constructed with a computed field
- **THEN** the computation does not reference the not-yet-bound result, and the path completes without NameError

#### Scenario: undefined names are absent from the analysed source
- **WHEN** the configured linter analyses the project source
- **THEN** it reports no undefined-name findings

### Requirement: Fixed defects are covered by tests that execute the defective path
Each defect corrected by this change SHALL have a test that exercises the specific path
which was failing. A correction without such a test SHALL NOT be considered complete,
since none of these defects were detected by the existing suite.

#### Scenario: the regression test fails against the unfixed code
- **WHEN** a regression test added for one of these defects is run against the original code
- **THEN** it fails, demonstrating that it exercises the defective path rather than passing vacuously

### Requirement: The reported baseline reflects decisions, not debris
Findings that are mechanically removable - unused imports, unused variables, placeholder
f-strings, semicolon-joined statements - SHALL be cleared, so that the remaining count
represents findings someone has chosen to accept rather than noise obscuring them. The
recorded baseline SHALL be updated to the post-cleanup figure.

#### Scenario: mechanical findings are absent
- **WHEN** the linter analyses the project source after this change
- **THEN** it reports no unused imports, unused variables, placeholder f-strings or semicolon-joined statements

#### Scenario: behaviour is unchanged
- **WHEN** the full test suite runs after the cleanup
- **THEN** it passes with no fewer tests than before, confirming the cleanup removed only dead code
