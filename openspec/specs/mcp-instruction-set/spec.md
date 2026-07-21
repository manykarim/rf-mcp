# mcp-instruction-set Specification

## Purpose
TBD - created by archiving change refactor-mcp-instructions. Update Purpose after archive.
## Requirements
### Requirement: The default MCP instruction set is a lean, order-explicit spine

The default server instructions SHALL be a lean spine (target ~250–400 characters) that
states ONE canonical tool order and nothing that merely repeats the tool schemas, because
experiments show shorter, order-explicit instructions yield better tool-calling than the
verbose default while the docstrings carry the rest.

The canonical order is: `analyze_scenario` first → discover-if-unknown
(`find_keywords`/`recommend_libraries`) → `get_locator_guidance` for locators/API →
`execute_step` → `build_test_suite`, plus a "never guess — discover" rule.

#### Scenario: default instructions are lean and order-explicit
- **WHEN** the server resolves its default instructions
- **THEN** the resulting text names `analyze_scenario` as the first call and presents the canonical order, and is substantially shorter than the previous ~2800-character default

#### Scenario: docstring-echo is not duplicated in the template
- **WHEN** the default instructions are reviewed against the tool schemas
- **THEN** they do not restate per-tool "use X to do Y" catalog lines that the tool descriptions already provide

### Requirement: A single canonical first tool and unified session entry

The instructions and the tool docstrings SHALL agree that `analyze_scenario` is the single
front door that creates the session, and MUST NOT present `manage_session(action="init")`
as a competing entry point for a fresh scenario, because the dual entry causes redundant
session churn (observed: small models call `manage_session` repeatedly after
`analyze_scenario` already created the session).

#### Scenario: no dual session-entry ambiguity
- **WHEN** the default instructions or the `analyze_scenario` / `manage_session` docstrings describe how to start
- **THEN** `analyze_scenario` is named as the session-creating first call and `manage_session(init)` is not presented as an alternative starting point for a new scenario

### Requirement: Verbose and dead instruction templates are retired

The redundant verbose templates SHALL be removed or shrunk to the lean spine's shape, and
the unloaded `templates/*.txt` files MUST be deleted or wired as the single source of
truth, because the verbose templates produce worse completion and more turns and the dead
`.txt` files have already drifted from the live classmethods.

#### Scenario: the oversized template is no longer the default path
- **WHEN** instructions are resolved with default settings
- **THEN** the ~6000-character `discovery_first`-style content is not delivered; only the lean spine (or a domain template) is

#### Scenario: no dead template files diverge from production
- **WHEN** the instruction template source is inspected
- **THEN** there are no unloaded template files that disagree with the live templates on tool ordering

### Requirement: Guaranteed init-response guidance for instruction-sensitive libraries

When a session is initialized with an instruction-sensitive library, the
`manage_session(action="init")` response SHALL include a guidance bundle for that library
delivered in the init response itself, because the non-obvious usage rules (e.g.
RequestsLibrary response access) are otherwise unreachable first-try and cause a turn-sink.

RequestsLibrary sessions SHALL receive an `api_guidance` bundle mirroring the existing
`desktop_guidance` injection for desktop sessions.

#### Scenario: RequestsLibrary init returns API guidance
- **WHEN** `manage_session(action="init")` is called with RequestsLibrary among the libraries
- **THEN** the response includes an api_guidance bundle carrying the RequestsLibrary response-access rules (or a pointer to the requests guidance), independent of whether the agent later calls `get_locator_guidance`

#### Scenario: desktop guidance behaviour is preserved
- **WHEN** a desktop session is initialized
- **THEN** the existing desktop guidance in the init response is unchanged

### Requirement: Tool docstrings are concise with the load-bearing guidance first

Tool docstrings SHALL lead with their load-bearing guidance (when-to-call, the primary
modes/params) and MUST NOT bury it under long secondary paragraphs, because the docstrings
are the primary driver of tool-calling and also consume small-model prompt budget.

Any docstring or template change that affects agent behaviour SHALL be validated against
the `agentic-e2e-instruction-quality` gate (reference model over the validated scenario
set) and MUST NOT regress it before landing.

#### Scenario: dense docstrings are tightened
- **WHEN** the `get_keyword_info`, `execute_step`, and `find_keywords` docstrings are reviewed
- **THEN** their primary when-to-call/mode guidance appears before secondary mechanics, and the overall docstring is shorter

#### Scenario: instruction changes are gate-validated
- **WHEN** an instruction/docstring change is prepared for landing
- **THEN** the instruction-quality gate is run on the reference model and shows no regression versus the committed baseline (or the baseline is deliberately recaptured as a reviewed change)

