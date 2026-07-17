## ADDED Requirements

### Requirement: Captures from introspection probes do not drive propagation

The system SHALL NOT use a value captured by an introspection/`Evaluate`/`Query`
probe (a step whose purpose is to inspect the tree or apply a side effect, not
to produce load-bearing test data) as a source for the OBS-11 literal→variable
substitution in `build_test_suite`, so investigative expressions do not distort
later recorded arguments. The discriminator is the SOURCE keyword, not the
captured value's length: a recorded `Keyboard Type ${None} 1` MUST NOT become
`Keyboard Type ${None} ${VAR}` because an `Evaluate` env-override step captured
the string `"1"`.

#### Scenario: an Evaluate-captured value does not rewrite later literals
- **WHEN** an `Evaluate` step captures a value (e.g. `"1"`) and a later
  interaction step has a literal equal to it
- **THEN** the later literal is kept (not rewritten to the Evaluate step's
  variable)

#### Scenario: a Query-captured value does not rewrite later literals
- **WHEN** a `Query` step captures a value and a later step has a literal equal
  to it
- **THEN** the later literal is kept

### Requirement: Genuine data dependencies still propagate

The system SHALL preserve the OBS-11 benefit for real data dependencies captured
by non-introspection keywords, regardless of the captured value's length, so a
legitimate capture-then-reuse still renders as a `${VAR}` reference.

#### Scenario: legitimate single-character data dependency still propagates
- **WHEN** a `Get Element Count` step captures `"5"` and a later `Fill Text`
  passes the literal `"5"`
- **THEN** the substitution to `${VAR}` still occurs

#### Scenario: genuine multi-character data dependency still propagates
- **WHEN** a non-introspection step captures a distinctive multi-character value
  and a later step passes that exact value as a literal
- **THEN** the substitution to `${VAR}` still occurs
