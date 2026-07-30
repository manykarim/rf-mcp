## ADDED Requirements

### Requirement: Migrating a black-box integration test preserves coverage and retires the pytest origin

When a black-box MCP-surface test is moved from pytest to the agenteval harness, the migration SHALL
preserve the original's observable coverage — the agenteval suite asserts the same tool results and
tool-call-trace facts the pytest test asserted — and the pytest original SHALL be removed only after its
agenteval port passes, so there is never a window of missing or long-lived duplicated coverage. A
candidate whose assertions cannot be expressed over the MCP surface SHALL be left in pytest, not
force-fit with weakened assertions.

#### Scenario: a ported test preserves its assertions
- **WHEN** a black-box integration test is ported to an agenteval `.robot` suite
- **THEN** the suite drives rf-mcp over the MCP protocol and asserts the same observable facts (tool result payloads and/or the recorded tool-call trace) that the pytest original asserted

#### Scenario: the pytest original is retired only when green
- **WHEN** an agenteval port of a pytest integration test is added
- **THEN** the pytest original is removed only after the port passes, and never before — so coverage is neither dropped nor left permanently duplicated

#### Scenario: a non-translatable candidate stays in pytest
- **WHEN** a candidate turns out to assert on internal Python state with no MCP-surface equivalent
- **THEN** it remains a pytest test and is reported as out of scope, rather than migrated with weakened assertions
