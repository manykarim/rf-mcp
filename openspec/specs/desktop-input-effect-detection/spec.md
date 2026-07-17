# desktop-input-effect-detection Specification

## Purpose
TBD - created by archiving change desktop-input-and-runtime-diagnostics. Update Purpose after archive.
## Requirements
### Requirement: A successful desktop interaction with no state change is flagged

The system SHALL, for a desktop keyboard/typing interaction whose target text
node is resolvable, snapshot the target's accessible display state (e.g.
`native:Text.CharacterCount`) before and after the keyword using the shared
native runtime DIRECTLY (no re-entrant Robot Framework keyword execution under
the execution lock), and when the keyword reports success but the state did not
change, attach a `desktop_input_no_effect` warning to the response. The check
MUST be best-effort and soft — it never fails or blocks the step, and is skipped
when no usable before/after snapshot can be obtained.

#### Scenario: success with unchanged state warns
- **WHEN** a desktop keyboard interaction succeeds and the target text node's
  character count is unchanged before vs after
- **THEN** the response includes a `desktop_input_no_effect` warning

#### Scenario: state changed → no warning
- **WHEN** the before/after snapshot differs (the input landed)
- **THEN** no `desktop_input_no_effect` warning is added

#### Scenario: no snapshot available → no warning, no failure
- **WHEN** the target text node / character count cannot be snapshotted
- **THEN** the step proceeds normally with no warning (best-effort, never raises)

#### Scenario: non-reentrant — no deadlock under the execution lock
- **WHEN** the before/after snapshot runs during keyword execution
- **THEN** it reads state via the native runtime directly, NOT by re-entering
  Robot Framework keyword execution (which holds the lock)

