# agent-ergonomics-fixes Specification

## Purpose
TBD - created by archiving change agent-ergonomics-fixes. Update Purpose after archive.
## Requirements
### Requirement: Malformed execute_batch steps produce an actionable error
`execute_batch` SHALL validate each step before execution. A step that is not a
mapping, or that lacks a non-empty `keyword`, or whose `arguments`/`args` is not
a list, SHALL produce an actionable error that names the offending step index and
the problem (the required `keyword` field, or that `arguments` must be a list) —
NOT a bare `KeyError`/`'keyword'` message and NOT silent coercion (e.g.
`list(dict)` yielding the dict's keys). The error SHALL be surfaced as a
structured failure result, not an unhandled exception.

#### Scenario: step missing keyword is rejected with guidance
- **WHEN** `execute_batch` is called with a step dict that has no `keyword` (or an empty one)
- **THEN** the result is a structured failure whose message names the step index and the required `keyword` field, and no `KeyError`/bare `'keyword'` text is surfaced

#### Scenario: non-list arguments are rejected, not coerced to dict keys
- **WHEN** a step supplies `arguments` (or `args`) as a dict or string rather than a list
- **THEN** the call fails with a message naming the step index and stating that arguments must be a list — the value is NOT silently converted to the dict's keys or the string's characters

#### Scenario: well-formed batch still builds
- **WHEN** every step has a non-empty `keyword` and a list `arguments` (or the legacy `args`)
- **THEN** the batch is constructed and executed as before

### Requirement: Standard utility libraries are allowed in every session type
The per-session-type library allowlist SHALL always include the domain-agnostic
Robot Framework standard libraries — `BuiltIn`, `OperatingSystem`, `Collections`,
`String`, `DateTime`, `Process` — regardless of session type, so that keywords for
file I/O, data manipulation, dates, and processes are never blocked. Domain
libraries (`Browser`, `SeleniumLibrary`, `AppiumLibrary`) SHALL remain governed by
the session profile.

#### Scenario: api_testing session may use OperatingSystem
- **WHEN** an `api_testing` session validates a keyword from `OperatingSystem` (e.g. writing a result file)
- **THEN** the library is allowed without an explicit `import_library` workaround

#### Scenario: web libraries still profile-governed
- **WHEN** a non-web session type is checked for excluded libraries
- **THEN** `Browser`/`SeleniumLibrary`/`AppiumLibrary` remain excluded unless the profile or explicit preference allows them

### Requirement: Agent-facing text documents batch BDD limits and safe suite persistence
The `execute_batch` tool documentation SHALL state that batch steps do not support
`bdd_group`/`bdd_intent` and that per-step `execute_step` is required for BDD
grouping. The server WORKFLOW GUIDE SHALL instruct agents to persist a generated
suite via `build_test_suite(output_path=…)` and NOT by writing `rf_text` through
the `Create File` keyword.

#### Scenario: batch docstring notes the BDD limitation
- **WHEN** an agent reads the `execute_batch` tool description
- **THEN** it states that `bdd_group`/`bdd_intent` are not supported in batch steps and to use `execute_step` for BDD grouping

#### Scenario: workflow guide notes safe suite persistence
- **WHEN** an agent reads the rf-mcp WORKFLOW GUIDE
- **THEN** it says to persist suites via `build_test_suite(output_path=…)` and not via the `Create File` keyword

