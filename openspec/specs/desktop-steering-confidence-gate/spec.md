# desktop-steering-confidence-gate Specification

## Purpose
TBD - created by archiving change desktop-steering-confidence-gate. Update Purpose after archive.
## Requirements
### Requirement: Interaction keywords carry a structured steering-confidence verdict
Every desktop interaction keyword result SHALL carry a structured
`steering_confidence` verdict — one of `confirmed`, `unconfirmed`, or
`contradicted` — composed from the already-computed verified-focus state,
window-visibility warnings, before/after input-effect snapshot, and
Wayland-input-drop risk. The verdict is the single field an agent consults to
decide whether an input actually reached the application, replacing the scatter
of advisory warning strings for the landing question.

#### Scenario: verified focus or observed effect yields confirmed
- **WHEN** an interaction keyword succeeds and either window focus was verified or the target's accessible state changed (e.g. `native:Text.CharacterCount` increased)
- **THEN** the step result carries `steering_confidence = confirmed`

#### Scenario: no positive evidence yields unconfirmed
- **WHEN** an interaction keyword succeeds against a target that exposes no readable state and focus could not be positively verified
- **THEN** the step result carries `steering_confidence = unconfirmed` and the step still succeeds

#### Scenario: non-interaction keywords carry no verdict
- **WHEN** a `Query`, `Get Attribute`, or `Take Screenshot` keyword runs
- **THEN** no `steering_confidence` verdict is attached and behavior is unchanged

### Requirement: A contradicted verdict fails the step by default
A desktop interaction that reports success while its input demonstrably did not
land SHALL be reported as a failure by default. "Demonstrably did not land"
means success was returned AND focus was not verified AND the target's
accessible state did not change (input-effect absent), or a Wayland
input-drop risk applies to an unverified target. The failure hint states that
the input did not reach the application and to re-verify focus and retry.

#### Scenario: success with unverified focus and no effect fails
- **WHEN** an interaction returns success but focus was never verified and the `native:Text.CharacterCount` snapshot is unchanged before and after
- **THEN** the step is reported as a failure with `steering_confidence = contradicted` and a hint to refocus and retry — not a silent PASS

#### Scenario: the enforcement can be downgraded to a warning
- **WHEN** `ROBOTMCP_PLATYNUI_STEERING_CONFIDENCE` is set to `warn` and a `contradicted` verdict occurs
- **THEN** the step proceeds as success with the `contradicted` verdict attached as a warning, mirroring the existing safety-guard `warn` opt-out

#### Scenario: a genuinely landed input is not failed
- **WHEN** an interaction is `confirmed` (verified focus or observed state change)
- **THEN** the gate does not fail the step
