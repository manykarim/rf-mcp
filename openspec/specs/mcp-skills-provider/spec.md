# mcp-skills-provider Specification

## Purpose
Defines the Skills-over-MCP capability (FastMCP `SkillsDirectoryProvider` + MCP SEP-2640 alignment + `ResourcesAsTools` fallback), coexistence with external skill repositories, project-specific skill customization, and multi-agent CLI setup.

## Requirements
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
The system SHALL serve skills from, in precedence order, (1) the project-local `<root>/.robotframework/skills/`, (2) external plugin directories configured via `--skills-dir` or `RF_MCP_SKILLS_DIR`, in the order given, (3) the skills of the installed official `robotframework-plugins` package, and (4) the built-in `rf-mcp` skills. When two sources define a skill of the same name, the first one SHALL win. Skills SHALL be re-read on each request, so edits appear without a restart.

#### Scenario: Project-local skill overrides or extends built-in skills without code changes
- **WHEN** a team places `.robotframework/skills/project-conventions/SKILL.md` in the workspace
- **THEN** `rf-mcp` automatically discovers and serves `skill://project-conventions/SKILL.md` alongside built-in and external skills

#### Scenario: Project skill wins over a built-in skill of the same name
- **WHEN** a project defines `.robotframework/skills/rf-locators/SKILL.md`
- **THEN** reading `skill://rf-locators/SKILL.md` returns the project's file, not the built-in one

#### Scenario: Official skills are served when the package is installed
- **WHEN** `robotframework-plugins` with a supported catalogue format is installed next to `rf-mcp`
- **THEN** `list_resources` includes its skills, for example `skill://rf-browser/SKILL.md`, without any `--skills-dir` configuration

#### Scenario: Official skill wins over a built-in skill of the same name
- **WHEN** the official package bundles a skill named `rf-requests` and `rf-mcp` has a built-in skill of the same name
- **THEN** reading `skill://rf-requests/SKILL.md` returns the official skill

#### Scenario: Company skills folder wins over an official skill
- **WHEN** `--skills-dir /opt/acme-skills` holds `rf-browser/SKILL.md` and the official package is installed
- **THEN** reading `skill://rf-browser/SKILL.md` returns the company's file

### Requirement: Project skills follow the request's project root
The `<root>` of project skills SHALL be resolved per request like a tool call's root in server mode: the pinned root (`--root`, `RF_MCP_ROOT`); else the client's workspace folders (MCP roots); else the startup folder, walked up to the nearest project marker. With several workspace folders, each folder's project skills SHALL be served in the client's order, the first folder winning for a duplicate name. In client mode, `<root>` SHALL be the pinned root or the startup folder.

#### Scenario: Server started outside the project
- **WHEN** an agent starts `rf-mcp` in the user's home folder, reports the workspace folder `/work/shop` as its only MCP root, and `/work/shop/.robotframework/skills/shop-conventions/SKILL.md` exists
- **THEN** `list_resources` includes `skill://shop-conventions/SKILL.md` and reading it returns that file

#### Scenario: Several workspace folders
- **WHEN** the client reports the workspace folders `/work/shop` and `/work/api`, each with its own project skills
- **THEN** `list_resources` includes the project skills of both folders, and a skill name defined in both is served from `/work/shop`

#### Scenario: Pinned root ignores the client's folders
- **WHEN** `rf-mcp` runs with `--root /workspace` and the client reports a different workspace folder
- **THEN** only the project skills of `/workspace` are served

#### Scenario: Workspace folders change
- **WHEN** the client sends `notifications/roots/list_changed` after its workspace folders changed
- **THEN** the next skill listing uses the new folders

### Requirement: Official skills package is read through its public API
The official package SHALL be used only through its public skills API at a catalogue format rf-mcp supports, both for the content it bundles and for snapshots of the repository fetched with that API; an installed package with an unsupported format SHALL be skipped with a warning in the log and in `rf-mcp doctor`, and the server SHALL start without it.

#### Scenario: Unsupported catalogue format
- **WHEN** the installed `robotframework-plugins` reports a catalogue format `rf-mcp` does not support
- **THEN** the server starts, serves no official skills, logs a warning naming the installed and supported formats, and `rf-mcp doctor` reports the same

#### Scenario: Snapshot read like the bundled content
- **WHEN** the official skills channel is a ref with a stored snapshot
- **THEN** its skill folders and catalogue are read through the package's snapshot API, and its skills take the official layer's place between `--skills-dir` and the built-in skills

### Requirement: Multi-agent CLI onboarding and skill synchronization
The system SHALL provide CLI subcommands (`rf-mcp init`, `rf-mcp sync-skills`, `rf-mcp doctor`) that configure MCP server entries and optionally materialize skills into local agent directories for Claude Code, OpenAI Codex, Google Antigravity (`agy`), GitHub Copilot, and OpenCode.

#### Scenario: Synchronize MCP skills into local agent skill directory
- **WHEN** a user runs `rf-mcp sync-skills --agent claude` (or `--target .agents/skills`)
- **THEN** the CLI exports the active skill directories (or downloads skills from a configured remote `rf-mcp` URL) into the target agent skill directory

#### Scenario: sync-skills does not silently overwrite existing skills
- **WHEN** the target directory already contains a skill folder with the same name whose content differs
- **THEN** the CLI skips it and reports the conflict unless `--force` is passed

#### Scenario: init runs rf-mcp from the project environment
- **WHEN** a user runs `rf-mcp init --agent claude` in a uv project where `rf-mcp` is installed into the project environment (including an editable install from a source checkout)
- **THEN** the generated entry runs `uv run rf-mcp` with no project path, no `--directory`, no `-r`, no `--mode` and no `--transport`, never from the `rf-mcp` source checkout, so the same file works in any clone of the project

#### Scenario: init for a project venv without uv
- **WHEN** a user runs `rf-mcp init` in a project whose `rf-mcp` is installed in a venv and the project has no `pyproject.toml`
- **THEN** the generated entry runs that venv's interpreter with `-m robotmcp` and no further arguments

#### Scenario: doctor verifies session support and custom hints
- **WHEN** a user runs `rf-mcp doctor`
- **THEN** it reports the `robotcode` version, whether `session`, `repl`, `analyze`, and `doc` subcommands are available, whether the project `hints.toml` parses, and the active skill roots

#### Scenario: Skill covers the key upstream workflows
- **WHEN** the built-in skill and its references are read
- **THEN** they cover mode routing, results-first triage without parsing `output.xml`, longname selection, analyze-before-run with the exit bitmask, one complete input per `send`, session reuse and announcing open sessions, sandbox honesty, `rf_repl save` with `keep_failed`, the pinned full-run / `rerunfailed` / merge workflow, `profile_show`, and RPA naming

### Requirement: init registers a global install at user level
`rf-mcp init --scope user` SHALL register an `rf-mcp` stdio entry in each selected agent's user-level configuration, so that every project the agent opens can use it. The entry SHALL contain no project path. It SHALL name the `rf-mcp` executable by its full path when `PATH` does not find this version of it, because its folder is not on `PATH` or another rf-mcp version comes first.

#### Scenario: Global install registered for Claude Code
- **WHEN** a user who installed rf-mcp with `uv tool install` runs `rf-mcp init --scope user --agent claude --write`
- **THEN** Claude Code's user-level configuration gains an `rf-mcp` stdio entry that starts the installed `rf-mcp` without arguments, and no file in the current folder is written

#### Scenario: Preview without writing
- **WHEN** a user runs `rf-mcp init --scope user --agent all` without `--write`
- **THEN** the CLI prints, for each agent, the file it would change and the entry it would add, and changes nothing

### Requirement: init project scope with a global install
`rf-mcp init --scope project` run from a global install SHALL write project-level entries that start `rf-mcp` by name, without a project path, so the file can be committed for teammates who also install rf-mcp globally.

#### Scenario: Global install shared through the project
- **WHEN** a user with a global rf-mcp runs `rf-mcp init --scope project --agent copilot --write` in a project without rf-mcp in its environment
- **THEN** `.vscode/mcp.json` gains an `rf-mcp` stdio entry whose command is `rf-mcp` with no arguments

### Requirement: init merges into existing agent configuration
When `rf-mcp init --write` targets a configuration file that already exists, the system SHALL add or replace only its own `rf-mcp` entry and SHALL keep every other server and setting in that file. When the file cannot be parsed, it SHALL change nothing and report the file and the parse error.

#### Scenario: Other servers survive
- **WHEN** `.mcp.json` already defines the servers `github` and `rf-mcp` (an old entry with absolute paths) and the user runs `rf-mcp init --agent claude --write`
- **THEN** `.mcp.json` still defines `github` unchanged, and `rf-mcp` now holds the new entry

#### Scenario: Unparsable file is left alone
- **WHEN** `~/.codex/config.toml` contains invalid TOML and the user runs `rf-mcp init --scope user --agent codex --write`
- **THEN** the file is unchanged and the CLI exits non-zero naming the file and the parse error

### Requirement: Project and global registrations share one server name
Entries written by `rf-mcp init` in every scope SHALL use the server name `rf-mcp`, so that an agent that ranks project-level configuration above user-level configuration uses a project's own rf-mcp in that project and the global one elsewhere.

#### Scenario: Both scopes registered
- **WHEN** a user has run `rf-mcp init --scope user` and a project has its own entry from `rf-mcp init --scope project`
- **THEN** both entries are named `rf-mcp`

### Requirement: doctor reports root and environment resolution
`rf-mcp doctor` SHALL report how rf-mcp is installed (project environment or global tool), the project root it resolves and which rule found it, the environment and `robotcode` a call for that root would use and which resolution step chose them, whether `uv` is available, and the project interpreter's Python version. It SHALL exit non-zero when calls for that root would fall back to rf-mcp's own `robotcode` although the root is not rf-mcp's own environment.

#### Scenario: Global install in a project without robotcode
- **WHEN** a user with a global rf-mcp runs `rf-mcp doctor` in a project whose `.venv` holds Robot Framework but no robotcode, with uv installed
- **THEN** the report shows the install kind `global`, the project root and the marker that found it, the environment `<root>/.venv` with robotcode layered on by uv at rf-mcp's pinned version, and exits `0`

#### Scenario: Nothing can run the project
- **WHEN** uv is missing and the project environment holds no robotcode
- **THEN** the report says the project's libraries will not be visible, names both remedies, and exits non-zero

### Requirement: Instructions point to project skills the server cannot see at startup
When the server starts in server mode with no pinned root and the folder it starts in has no project marker in it or above it, the server `instructions` SHALL state that the project's skills appear in `list_resources` once the client reports its workspace folders. The instructions SHALL stay under 2,500 characters.

#### Scenario: Started in the home folder
- **WHEN** `rf-mcp` starts without `--root` in a folder with no `.robot.toml`, `robot.toml`, `pyproject.toml`, `.git` or `.hg` in it or above it
- **THEN** the `instructions` say that project skills are listed by `list_resources`

#### Scenario: Started in a project
- **WHEN** `rf-mcp` starts in a folder that contains `robot.toml`
- **THEN** the `instructions` list that project's skills in the skill catalog and carry no such note

### Requirement: Instructions translate official skill commands to rf-mcp tools
When official skills are served, the server `instructions` SHALL state that those skills describe `robotcode` commands and SHALL map each to its rf-mcp tool: `robotcode repl` to `rf_repl`, `robot` to `rf_run`, `discover` to `rf_discover`, `analyze` to `rf_analyze`, `results` to `rf_results`, `doc` and `libdoc` to `rf_doc`, and `debug` to `rf_debug`. The instructions SHALL stay under 2,500 characters.

#### Scenario: Official skills served
- **WHEN** `robotframework-plugins` is installed and a client reads the server `instructions`
- **THEN** they contain the command-to-tool mapping and are at most 2,500 characters long

#### Scenario: No official skills
- **WHEN** `robotframework-plugins` is not installed
- **THEN** the `instructions` carry no command-to-tool mapping

### Requirement: Official skills take one line in the instructions catalogue
The skill catalogue in the server `instructions` SHALL list project skills, `--skills-dir` skills and built-in skills one row each, and SHALL summarise the official skills in one line that points to `list_resources` and says that the skills matching the project's libraries are attached to the first tool result. The built-in `rf-robotcode` row SHALL always fit.

#### Scenario: Catalogue with the official package
- **WHEN** `robotframework-plugins` bundling nine skills is installed and a client reads the server `instructions`
- **THEN** the catalogue holds a row for `skill://rf-robotcode/SKILL.md` and one line for the official skills, not a row for each of them
