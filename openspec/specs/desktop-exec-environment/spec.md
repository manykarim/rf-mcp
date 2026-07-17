# desktop-exec-environment Specification

## Purpose
TBD - created by archiving change desktop-mcp-workflow-correctness. Update Purpose after archive.
## Requirements
### Requirement: Resolve executables via a hook against the server PATH

The system SHALL resolve a desktop launch/recovery executable to an absolute
path using `shutil.which` against the **server process** `PATH` (after the
existing desktop-launch sanitization) before dispatching `Process`/`Evaluate`,
rather than inheriting an interactive shell's startup environment. A tool
resolvable for the server process MUST then be found from step execution; an
optional config MAY add explicit desktop-tool paths.

#### Scenario: server-resolvable tool is found
- **WHEN** a desktop step launches or probes a tool resolvable via
  `shutil.which` on the server `PATH` (e.g. `bash`, `which`)
- **THEN** the tool is dispatched by its resolved absolute path (no
  `FileNotFoundError` for a tool the server can resolve)

#### Scenario: no broad shell-env inheritance
- **WHEN** executable resolution runs
- **THEN** it uses the server-process PATH (post-sanitization) and does not
  import arbitrary interactive shell startup state

#### Scenario: missing tool reports clearly with effective PATH
- **WHEN** a tool is genuinely unresolvable
- **THEN** the failure message names the missing executable and surfaces the
  effective PATH used, rather than implying an opaque environment mismatch

### Requirement: Document Evaluate expression-only behavior with an alternative

The system SHALL document that `BuiltIn.Evaluate` accepts a single Python
expression (not statements), and SHALL point callers to a statement-capable
path (e.g. `Run Process` / a Process keyword) for recovery logic that needs
imports or multiple statements.

#### Scenario: guidance explains the limitation
- **WHEN** a caller needs multi-statement recovery logic
- **THEN** the guidance explains Evaluate is expression-only and names a
  statement-capable alternative

