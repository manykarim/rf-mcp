## MODIFIED Requirements

### Requirement: Layered skill roots for ecosystem plugins and project customizations
The system SHALL serve skills from, in precedence order, (1) the project-local `<root>/.robotframework/skills/`, (2) external plugin directories configured via `--skills-dir` or `RF_MCP_SKILLS_DIR` (such as `robotcodedev/robotframework-agent-plugins` and `manykarim/robotframework-agentskills`), in the order given, and (3) the built-in `rf-mcp` skills. When two sources define a skill of the same name, the first one SHALL win. Skills SHALL be re-read on each request, so edits appear without a restart.

`<root>` SHALL be resolved for each request the way a tool call's project root is resolved in server mode: the root pinned with `--root` / `RF_MCP_ROOT`; else the client's workspace folders (MCP roots) when the client reports them; else the folder the server started in, walked up to the nearest project marker. When the client reports several workspace folders, the project skills of each of them SHALL be served, in the client's order, the first folder winning for a duplicate name. In client mode, `<root>` SHALL be the pinned root or the startup folder.

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

## ADDED Requirements

### Requirement: Instructions point to project skills the server cannot see at startup
When the server starts in server mode with no pinned root and the folder it starts in has no project marker in it or above it, the server `instructions` SHALL state that the project's skills appear in `list_resources` once the client reports its workspace folders. The instructions SHALL stay under 2,500 characters.

#### Scenario: Started in the home folder
- **WHEN** `rf-mcp` starts without `--root` in a folder with no `.robot.toml`, `robot.toml`, `pyproject.toml`, `.git` or `.hg` in it or above it
- **THEN** the `instructions` say that project skills are listed by `list_resources`

#### Scenario: Started in a project
- **WHEN** `rf-mcp` starts in a folder that contains `robot.toml`
- **THEN** the `instructions` list that project's skills in the skill catalog and carry no such note
