# robotcode-error-hinting Specification

## Purpose
Defines the `robotcode`-native, library-aware, and project-customizable error hinting engine that provides actionable recovery feedback to coding agents when keywords, REPL statements, or test runs fail.

## Requirements
### Requirement: Built-in library and REPL recovery hints
The system SHALL evaluate failed keyword calls, REPL statements, and test outputs against a ranked catalog of built-in error rules covering `Browser` (Playwright strict mode, `Evaluate JavaScript` argument shape, click interception, `Get Aria Snapshot`), `SeleniumLibrary`, `AppiumLibrary`, `RequestsLibrary` (named arguments, dict bracket access `${d}[key]`), `PlatynUI`, `BuiltIn`/`Process`, and `robotcode` REPL syntax (`*** Settings ***` in REPL, missing keywords).

#### Scenario: Hint on Browser Evaluate JavaScript single-argument misuse
- **WHEN** an agent executes `Evaluate JavaScript    () => document.title` with a single argument and it fails with a CSS selector parse error
- **THEN** the hinting engine returns a top-ranked hint explaining that `Browser.Evaluate JavaScript` expects `(selector, expression)` and provides the exact fix `Evaluate JavaScript    ${None}    () => document.title`

#### Scenario: Hint on Playwright strict mode violation
- **WHEN** a `Browser` keyword fails with `strict mode violation: ... resolved to N elements`
- **THEN** the hinting engine returns a hint recommending `>> nth=0`, `>> visible=true`, or inspecting the page structure via `Get Aria Snapshot    css=body`

#### Scenario: Hint on section headers used inside robotcode repl
- **WHEN** an agent sends `*** Settings ***` into `robotcode repl` and receives `No keyword with name '*** Settings ***' found`
- **THEN** the hinting engine returns a hint explaining that `robotcode repl` accepts test-body statements directly and shows `Import Library`, `Import Resource`, and `Import Variables` examples

### Requirement: Automatic hinting in server mode and on-demand diagnosis in client mode
The system SHALL automatically attach matching hints to `rf_repl`, `rf_debug`, and `rf_run` responses when a failure occurs in `server` mode, and SHALL expose an `rf_diagnose` tool so agents running `robotcode` locally in `client` mode can submit error output and receive identical hints.

#### Scenario: Agent diagnoses local robotcode error via remote MCP in client mode
- **WHEN** an agent running `robotcode` locally in `client` mode encounters a failure and passes the error output and keyword name to `rf_diagnose`
- **THEN** `rf_diagnose` returns up to 3 top-ranked `Hint` objects (`title`, `message`, `fix_examples`, `relevance`) formatted in native Robot Framework / `robotcode` syntax

### Requirement: Project-customizable hint rules via hints.toml
The system SHALL load custom hint rules from `<workspace>/.robotframework/hints.toml` (or paths configured via `--hints-file` / `RF_MCP_HINTS_FILE`), allowing projects to define pattern-based hints (`error_pattern`, `keyword_pattern`, `title`, `message`, `fix_examples`, `relevance`) that merge with and can override built-in rules.

#### Scenario: Custom project hint matches internal library failure
- **WHEN** `.robotframework/hints.toml` defines a rule matching `error_pattern = "CorpAuthExpiredError"` with `relevance = 99` and a keyword fails with that error
- **THEN** the hinting engine returns the custom project hint as the highest-ranked recovery suggestion

#### Scenario: Malformed hints.toml does not take the server down
- **WHEN** `.robotframework/hints.toml` is not valid TOML (for example `error_pattern = "(?i)!=\s*180"`, an invalid escape in a basic string) or a rule holds an invalid regex
- **THEN** the server starts with the built-in rules plus every valid custom rule, logs a warning naming the file, line, and rule id, `rf-mcp doctor` reports the error, and the server `instructions` mention that custom hints failed to load

### Requirement: Hints are gated by the kind of failure
The system SHALL attach hints only when they match the kind of output they are evaluated against: runtime failures from `rf_run`/`rf_repl`/`rf_debug` (recognising both `[ FAIL ]` lines and `| FAIL |` console status lines), robotcode errors (exit `252`), and static diagnostics from `rf_analyze`, which SHALL only receive rules tagged for static analysis. Runtime-only rules (such as shell-expansion advice for `Variable not found`) SHALL NOT be attached to static diagnostics.

#### Scenario: Static VariableNotFound diagnostic
- **WHEN** `rf_analyze` reports `VariableNotFound` for `${RUN_ID}`
- **THEN** no "expanded by shell / use single quotes" hint is attached

#### Scenario: Run selects no tests
- **WHEN** `rf_debug` or `rf_run` fails with `contains no tests after model modifiers`
- **THEN** the top hint explains that the longname must come from `rf_discover` and that a single-file `paths` changes the top suite name
