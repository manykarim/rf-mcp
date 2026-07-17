# stepwise-suite-isolation Specification

## Purpose
TBD - created by archiving change desktop-mcp-workflow-correctness. Update Purpose after archive.
## Requirements
### Requirement: build_test_suite isolates pre-start steps via an explicit opt-in

The system SHALL add an `include_pre_start: bool = False` parameter to
`build_test_suite`. By default (`False`), exploratory steps executed before
`start_test` are EXCLUDED from the generated test body, and the response
reports `excluded_pre_start_count` and a short summary. `include_pre_start=
True` preserves the prior adoption behavior (with an INFO-level deprecation
log). This makes the previously-silent adoption an explicit, reversible
choice.

#### Scenario: pre-start steps excluded by default with reporting
- **WHEN** steps run before `start_test`, then a test's real steps run, then
  `build_test_suite()` is called (default `include_pre_start=False`)
- **THEN** the generated test body contains only the in-test steps, and the
  response reports the excluded pre-start count + summary

#### Scenario: opt-in preserves prior adoption
- **WHEN** `build_test_suite(include_pre_start=True)` is called with pre-start
  steps present
- **THEN** the prior adoption behavior is preserved (pre-start steps included)

#### Scenario: start_test message explains handling
- **WHEN** pre-start steps exist at `start_test` time
- **THEN** the message explains they are excluded by default (with the opt-in)
  rather than only warning that they "will be adopted"

### Requirement: Generated desktop suite reflects real interactions

The system SHALL produce a runnable desktop suite from executed desktop
interaction steps (clicks, typing, assertions), not a placeholder of `Log`-only
steps, when real interaction steps were recorded.

#### Scenario: recorded PlatynUI interactions appear in the .robot
- **WHEN** a desktop session records at least one real PlatynUI interaction
  step (e.g. a Pointer Click) and an assertion, and `build_test_suite` is
  called
- **THEN** the generated `.robot` text contains the PlatynUI interaction
  keyword line(s) and the assertion — not only `Log` placeholders

