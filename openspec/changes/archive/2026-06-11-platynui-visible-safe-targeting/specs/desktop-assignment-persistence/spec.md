# Spec: desktop-assignment-persistence

## ADDED Requirements

### Requirement: assign_to persists to session scope on every execution path
Variables assigned via `assign_to` in `execute_step` SHALL be stored into the session's variable store regardless of whether the step executed through the RF native context (`use_context=True`) or the non-context path, using the same `${name}` normalization in both paths.

#### Scenario: Desktop Query assignment persists without context
- **WHEN** a desktop session executes `Query` with `assign_to="nodes"` and `use_context` unset
- **THEN** `${nodes}` resolves in a subsequent step and appears in the session's variables

#### Scenario: Context path behavior unchanged
- **WHEN** the same assignment runs with `use_context=True`
- **THEN** the resulting session variable is identical to the non-context path
