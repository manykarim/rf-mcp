# batch-step-argument-compat Specification

## Purpose
TBD - created by archiving change desktop-mcp-workflow-correctness. Update Purpose after archive.
## Requirements
### Requirement: execute_batch accepts arguments as well as args

The system SHALL accept a batch step's positional arguments under either the
`args` key or the `arguments` key (the same field `execute_step` uses), so
batch steps do not silently lose their arguments and fail with "expected N
arguments, got 0".

#### Scenario: step uses the arguments key
- **WHEN** `execute_batch` receives a step `{"keyword": "Log", "arguments":
  ["hello"]}`
- **THEN** the keyword runs with `["hello"]` (the arguments are not dropped)

#### Scenario: step uses the args key
- **WHEN** `execute_batch` receives a step `{"keyword": "Log", "args":
  ["hello"]}`
- **THEN** the keyword runs with `["hello"]`

#### Scenario: both keys present and equal are accepted
- **WHEN** a step provides both `args` and `arguments` with equal values
- **THEN** the keyword runs with those arguments

#### Scenario: conflicting dual specification is rejected
- **WHEN** a step provides both `args` and `arguments` with DIFFERENT values
- **THEN** the system returns a validation error (rather than silently
  shadowing one), so the ambiguity is surfaced to the caller

#### Scenario: arguments is the canonical key
- **WHEN** documentation or precedence must name a canonical key
- **THEN** `arguments` is canonical (for `execute_step` parity)

### Requirement: Batch argument compatibility is documented

The system SHALL document in the `execute_batch` tool that step positional
arguments may be given as `args` or `arguments`.

#### Scenario: docstring mentions both keys
- **WHEN** a caller reads the `execute_batch` tool documentation
- **THEN** it states that `args` or `arguments` are accepted for step arguments

