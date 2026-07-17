# stepwise-suite-hygiene Specification

## Purpose
TBD - created by archiving change desktop-stepwise-execution-fidelity. Update Purpose after archive.
## Requirements
### Requirement: Exploratory desktop introspection is excluded from the generated suite

The system SHALL exclude exploratory desktop introspection steps — tree probes
(`Query`) and raw `Evaluate` expressions that CAPTURED a variable (an `assign_to`
data probe) which nothing downstream consumes — from the generated
`build_test_suite` body by default, so the suite serializes validated intent
(real interactions + assertions) rather than investigation history. Steps that
are load-bearing (the captured variable is referenced by a retained later step)
MUST be retained, and a STANDALONE `Query`/`Evaluate` with no captured variable
MUST be retained (it may be a side-effect/existence assertion). Variable
comparison MUST normalize the `${VAR}` storage form against bare reference names.

#### Scenario: investigative Query/Evaluate probes are filtered
- **WHEN** a desktop session recorded exploratory `Query`/`Evaluate`
  introspection steps alongside real `Pointer Click`/`Keyboard Type` and
  assertion steps, and `build_test_suite` is called
- **THEN** the generated `.robot` body contains the interactions and assertions
  but not the exploratory introspection probes, and the response reports how
  many introspection steps were filtered

#### Scenario: load-bearing captures are retained
- **WHEN** a `Query`/`Get Attribute` step assigns a variable (stored as
  `${VAR}`) that a later retained step (e.g. an assertion referencing `${VAR}`)
  consumes
- **THEN** that step is retained (not filtered) so the generated suite compiles

#### Scenario: standalone unassigned probe is retained
- **WHEN** a `Query`/`Evaluate` step has no captured variable (no `assign_to`)
- **THEN** it is retained — it may be a side-effect/existence assertion and is
  never silently erased

#### Scenario: a clean validated suite is produced for the report flow
- **WHEN** the GNOME Calculator stepwise flow is recorded and built
- **THEN** the generated suite reads as a maintainable calculator test (real
  interactions + result assertions), not a long single body of exploratory
  queries, debug snapshots, and safety-override plumbing

