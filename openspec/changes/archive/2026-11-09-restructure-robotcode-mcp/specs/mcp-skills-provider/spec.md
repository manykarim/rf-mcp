# Spec Delta

## Purpose

Defines the Skills-over-MCP capability (FastMCP `SkillsDirectoryProvider` + MCP SEP-2640 alignment + `ResourcesAsTools` fallback), coexistence with external skill repositories, project-specific skill customization, and multi-agent CLI setup.

## ADDED Requirements

### Requirement: Serve layered skills over MCP resources and tool bridge
The system SHALL expose `SKILL.md` directories over MCP using FastMCP's non-deprecated `fastmcp.skills.provider.SkillsDirectoryProvider` (`skill://{name}/SKILL.md`, `skill://{name}/_manifest`, `skill://{name}/{path*}`, plus the MCP skills extension), inject a compact skill catalog signpost into the MCP server `instructions`, and enable `ResourcesAsTools` (`list_resources` / `read_resource`) so both resource-capable and tools-only coding agents can read skills.

#### Scenario: Tools-only coding agent reads skill over MCP
- **WHEN** a coding agent connects to `rf-mcp` and invokes `read_resource` with `uri="skill://rf-robotcode/SKILL.md"`
- **THEN** `rf-mcp` returns the full `SKILL.md` content for the `rf-robotcode` skill

#### Scenario: Server starts without deprecation warnings
- **WHEN** `rf-mcp` starts with FastMCP 3.4.4 or later
- **THEN** no `FastMCPDeprecationWarning` about `SkillsDirectoryProvider` is emitted

### Requirement: Server instructions as an intent routing table
The MCP server `instructions` SHALL contain a compact intent-to-tool routing table mirroring the upstream robotcode skill's "Pick the mode first" section, plus hard rules: never read or grep `output.xml`, select tests by longname from `rf_discover`, run `rf_analyze` before running new or edited suites, and stop sessions or announce open ones. The instructions SHALL stay under 2,500 characters.

#### Scenario: Instructions carry the routing rules
- **WHEN** a client reads the server `instructions` after `initialize`
- **THEN** they contain the routing table and the four hard rules, and are at most 2,500 characters long

#### Scenario: Instructions state the authoring loop and when it is done
- **WHEN** a client reads the server `instructions`
- **THEN** they give one ordered authoring loop (start one `rf_repl` session and reuse it; look keywords up instead of guessing; send several known steps per `send`; after a failure fix and retry once, never repeat the same failing call; write files, `rf_analyze` until clean, `rf_run` by longname until passing) and state that the task is done only when `rf_analyze` is clean and `rf_run` passes, carried over from the legacy `lean` template whose fixed order and finish condition gave zero session churn in a five-run Haiku sweep

#### Scenario: Instructions route project orientation to one multi-file read
- **WHEN** a client reads the server `instructions` in server mode
- **THEN** the routing table sends reading specs, docs, robot.toml, resources and tests to `rf_discover action="read"` with all paths or globs in one call, instead of one `Read` per file
- **AND** the first step of the authoring loop is that one read

#### Scenario: Instructions route UI work to the locator skill
- **WHEN** the `rf-locators` skill is in the catalog and a client reads the server `instructions`
- **THEN** the routing table sends writing or fixing UI steps (Browser, SeleniumLibrary, AppiumLibrary) to reading `skill://rf-locators/SKILL.md` first

### Requirement: Verified library cookbooks as skill references
The built-in skills SHALL carry verified library cookbooks as `references/` files or companion skills: locators (including shadow DOM and iframes) for Browser, SeleniumLibrary and AppiumLibrary; RequestsLibrary for REST, SOAP/XML and GraphQL; and an intent-to-keyword and library-chooser table replacing `get_locator_guidance`, `intent_action` and `recommend_libraries`. An automated test SHALL execute every snippet.

#### Scenario: Shadow DOM and iframe cookbook
- **WHEN** an agent reads the locator cookbook
- **THEN** it learns, per library: Browser CSS/text/role pierce open shadow roots but XPath does not; iframes need `>>>` (nested, cross-origin, iframe inside a shadow root, shadow root inside an iframe); `Get Aria Snapshot` shows each iframe only as `- iframe`, so snapshot `<frame> >>> body`; SeleniumLibrary needs `shadow_root` or a deep-query JavaScript helper and `Select Frame` / `Unselect Frame`; and closed shadow roots need a workaround: force `attachShadow` open (Browser: recreate the element, which resets its state; Selenium/Chrome: CDP script before load) or, in Selenium/Chrome, read them via CDP `DOM.getDocument(pierce=true)` and `DOM.performSearch` while clicking by offset inside the host

#### Scenario: RequestsLibrary cookbook
- **WHEN** an agent reads the RequestsLibrary cookbook
- **THEN** it learns to import the library in the REPL first, to pass dictionaries (`${{ {...} }}` or `Create Dictionary` with `${7}` numbers) instead of JSON text for `json=` and `headers=`, to build nested payloads from inner dictionaries, to use `expected_status` for negative tests, to do arithmetic inside `${{ }}` with `precision=2` for money, and that `Run Keywords` cannot assign
- **AND** it learns to explore an API without a schema by sending with `expected_status=any` and logging the body (an `HTTPError` message hides it), and to send forms with `data=`, files with `files=` and query strings with `params=`
- **AND** for SOAP it learns to build the envelope in one cell with `Catenate SEPARATOR=`, that faults arrive as HTTP 500, and that `Parse XML` strips namespaces so XPaths use plain element names below `Envelope`
- **AND** for GraphQL it learns to keep the query in a variable, to pass `variables` as a dictionary, and that errors arrive with status 200 so a test asserts that `errors` is absent

#### Scenario: Generated API tests pass when run alone
- **WHEN** an agent follows the cookbook and the authoring loop to write a suite of API tests
- **THEN** each test creates the records it needs through a keyword instead of receiving ids from an earlier test via `Set Suite Variable`, and every test passes when run alone by longname or as a rerun of failed tests

#### Scenario: Snippets are verified against a fixture
- **WHEN** the test suite runs
- **THEN** every cookbook snippet runs against a local fixture page (open, nested and closed shadow roots; same-origin, nested and cross-origin iframes; an iframe inside a shadow root; a shadow root inside an iframe) or a local API fixture (REST, SOAP and GraphQL), and a snippet that stops working fails the build

### Requirement: Built-in skill covers the upstream robotcode practices
The built-in `rf-robotcode` skill SHALL translate every applicable practice of `robotframework-agent-plugins@next` `plugins/robotcode/skills/robotcode` (SKILL.md and `references/*.md`, practices P1-P46) into `rf_*` tool terms, or list it as intentionally not applicable. Detailed material SHALL live in `references/` files of the skill, served via `skill://rf-robotcode/{path*}`.

#### Scenario: Practice checklist is covered
- **WHEN** the practice checklist P1–P46 derived from the upstream skill is compared against `skill://rf-robotcode/SKILL.md` and its references
- **THEN** every practice is either covered in rf-mcp terms or listed in the skill as intentionally not applicable (for example CLI-only install advice), and a unit test asserts the presence of the key rules

### Requirement: Layered skill roots for ecosystem plugins and project customizations
The system SHALL load skill roots in precedence order from (1) built-in `rf-mcp` skills, (2) external plugin directories configured via `--skills-dir` or `RF_MCP_SKILLS_DIR` (such as `robotcodedev/robotframework-agent-plugins` and `manykarim/robotframework-agentskills`), and (3) project-local `<workspace>/.robotframework/skills/`, with `reload=True` for live updates.

#### Scenario: Project-local skill overrides or extends built-in skills without code changes
- **WHEN** a team places `.robotframework/skills/project-conventions/SKILL.md` in the workspace
- **THEN** `rf-mcp` automatically discovers and serves `skill://project-conventions/SKILL.md` alongside built-in and external skills

### Requirement: Multi-agent CLI onboarding and skill synchronization
The system SHALL provide CLI subcommands (`rf-mcp init`, `rf-mcp sync-skills`, `rf-mcp doctor`) that configure MCP server entries and optionally materialize skills into local agent directories for Claude Code, OpenAI Codex, Google Antigravity (`agy`), GitHub Copilot, and OpenCode.

#### Scenario: Synchronize MCP skills into local agent skill directory
- **WHEN** a user runs `rf-mcp sync-skills --agent claude` (or `--target .agents/skills`)
- **THEN** the CLI exports the active skill directories (or downloads skills from a configured remote `rf-mcp` URL) into the target agent skill directory

#### Scenario: sync-skills does not silently overwrite existing skills
- **WHEN** the target directory already contains a skill folder with the same name whose content differs
- **THEN** the CLI skips it and reports the conflict unless `--force` is passed

#### Scenario: init runs rf-mcp from the project environment
- **WHEN** a user runs `rf-mcp init --agent claude` in a project where `rf-mcp` is installed into the project venv (including an editable install from a source checkout)
- **THEN** the generated server command runs `rf-mcp` from the project environment (`uv run --directory <project root> rf-mcp ...`, or the project venv's `rf-mcp` executable), never from the `rf-mcp` source checkout, so `robotcode` sees the project's libraries

#### Scenario: doctor verifies session support and custom hints
- **WHEN** a user runs `rf-mcp doctor`
- **THEN** it reports the `robotcode` version, whether `session`, `repl`, `analyze`, and `doc` subcommands are available, whether the project `hints.toml` parses, and the active skill roots

#### Scenario: Skill covers the key upstream workflows
- **WHEN** the built-in skill and its references are read
- **THEN** they cover mode routing, results-first triage without parsing `output.xml`, longname selection, analyze-before-run with the exit bitmask, one complete input per `send`, session reuse and announcing open sessions, sandbox honesty, `rf_repl save` with `keep_failed`, the pinned full-run / `rerunfailed` / merge workflow, `profile_show`, and RPA naming
