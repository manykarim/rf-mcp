## MODIFIED Requirements

### Requirement: Layered skill roots for ecosystem plugins and project customizations
The system SHALL serve skills from, in precedence order, (1) the project-local `<root>/.robotframework/skills/`, (2) external plugin directories configured via `--skills-dir` or `RF_MCP_SKILLS_DIR` (such as a clone or company mirror of `MarketSquare/robotframework-plugins`), in the order given, (3) the skills bundled in the official `robotframework-plugins` package when it is installed, and (4) the built-in `rf-mcp` skills. When two sources define a skill of the same name, the first one SHALL win. Skills SHALL be re-read on each request, so edits appear without a restart.

`<root>` SHALL be resolved for each request the way a tool call's project root is resolved in server mode: the root pinned with `--root` / `RF_MCP_ROOT`; else the client's workspace folders (MCP roots) when the client reports them; else the folder the server started in, walked up to the nearest project marker. When the client reports several workspace folders, the project skills of each of them SHALL be served, in the client's order, the first folder winning for a duplicate name. In client mode, `<root>` SHALL be the pinned root or the startup folder.

The official package SHALL be used only through its public skills API at a catalogue format rf-mcp supports; an installed package with an unsupported format SHALL be skipped with a warning in the log and in `rf-mcp doctor`, and the server SHALL start without it.

#### Scenario: Project-local skill overrides or extends built-in skills without code changes
- **WHEN** a team places `.robotframework/skills/project-conventions/SKILL.md` in the workspace
- **THEN** `rf-mcp` automatically discovers and serves `skill://project-conventions/SKILL.md` alongside built-in and external skills

#### Scenario: Project skill wins over a built-in skill of the same name
- **WHEN** a project defines `.robotframework/skills/rf-locators/SKILL.md`
- **THEN** reading `skill://rf-locators/SKILL.md` returns the project's file, not the built-in one

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

#### Scenario: Official skills are served when the package is installed
- **WHEN** `robotframework-plugins` with a supported catalogue format is installed next to `rf-mcp`
- **THEN** `list_resources` includes its skills, for example `skill://rf-browser/SKILL.md`, without any `--skills-dir` configuration

#### Scenario: Official skill wins over a built-in skill of the same name
- **WHEN** the official package bundles a skill named `rf-requests` and `rf-mcp` has a built-in skill of the same name
- **THEN** reading `skill://rf-requests/SKILL.md` returns the official skill

#### Scenario: Company skills folder wins over an official skill
- **WHEN** `--skills-dir /opt/acme-skills` holds `rf-browser/SKILL.md` and the official package is installed
- **THEN** reading `skill://rf-browser/SKILL.md` returns the company's file

#### Scenario: Unsupported catalogue format
- **WHEN** the installed `robotframework-plugins` reports a catalogue format `rf-mcp` does not support
- **THEN** the server starts, serves no official skills, logs a warning naming the installed and supported formats, and `rf-mcp doctor` reports the same

## ADDED Requirements

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
