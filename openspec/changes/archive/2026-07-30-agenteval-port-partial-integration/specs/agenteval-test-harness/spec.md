## ADDED Requirements

### Requirement: A partial integration file is split, not migrated wholesale

When an integration file mixes MCP-observable tests with internal-state tests, the migration SHALL port
ONLY the MCP-observable tests to the agenteval harness and SHALL leave the internal-state tests in a
trimmed pytest file — never deleting the whole file and never porting an internal-state test with
weakened assertions. A file whose MCP-observable subset is too small to justify a split SHALL be left
whole in pytest and recorded as such, rather than fragmented for little gain.

#### Scenario: only the MCP-observable tests move
- **WHEN** a partial file is split
- **THEN** its MCP-observable tests become an agenteval suite and are removed from the pytest file, while its internal-state tests remain in the now-trimmed pytest file

#### Scenario: the split preserves coverage on both sides
- **WHEN** a partial file is split
- **THEN** every ported test asserts the same observable facts it did in pytest, and every retained pytest test is unchanged — no assertion is dropped or weakened on either side

#### Scenario: a low-value partial is left whole
- **WHEN** a partial file's MCP-observable subset is too small to justify a split
- **THEN** the whole file stays in pytest and the decision is recorded
