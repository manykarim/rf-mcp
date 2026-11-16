# Spec Delta

## Purpose

Points coding agents to the skills that match the Robot Framework libraries a project actually has installed, at the moment they start working on it, instead of relying on the agent to pick the right skill from a catalogue up front.

## ADDED Requirements

### Requirement: Project library inventory
In server mode the system SHALL determine, for each project root, the Python distributions installed in the environment that runs that root's `robotcode`, with their versions. The inventory SHALL be refreshed when that environment changes. When the environment's libraries are not visible to rf-mcp (the fallback environment) or the inventory cannot be read, the root SHALL have no inventory and no error SHALL reach the agent.

#### Scenario: Project virtual environment
- **WHEN** a project root has a `.venv` with `robotframework-browser 20.1.0` installed and an agent calls any tool for that root
- **THEN** the root's inventory contains `robotframework-browser` with version `20.1.0`

#### Scenario: Package installed later
- **WHEN** `robotframework-requests` is installed into the project's `.venv` after the inventory was read
- **THEN** the next inventory for that root contains `robotframework-requests`

#### Scenario: Libraries not visible
- **WHEN** a project root has no environment of its own and rf-mcp runs `robotcode` from its fallback environment
- **THEN** the root has no inventory, and tool results carry no `project_skills`

### Requirement: Skill catalogues match installed libraries
The system SHALL match a root's inventory against skill catalogues read from the official skills package, from `plugins/*/catalog.yaml` in `--skills-dir` folders, and from rf-mcp's built-in catalogue. A catalogue library matches when its `package` equals an installed distribution after PEP 503 name normalisation. A match SHALL name the library, the installed version, the matched skills that the server serves, and whether the version satisfies the catalogue's `versions` range.

#### Scenario: Supported version
- **WHEN** the official catalogue lists `Browser` with package `robotframework-browser` and versions `>=20,<21`, names the skill `rf-browser` for it, and the project has `robotframework-browser 20.1.0`
- **THEN** the match names `Browser`, `20.1.0`, `skill://rf-browser/SKILL.md`, and states that the version is supported

#### Scenario: Version outside the range
- **WHEN** the project has `robotframework-browser 19.4.0` and the catalogue range is `>=20,<21`
- **THEN** the match states that the version is not supported and gives the range `>=20,<21`

#### Scenario: Package name normalisation
- **WHEN** the catalogue lists package `robotframework-PlatynUI` and the project has the distribution `robotframework_platynui`
- **THEN** the two match

#### Scenario: Catalogue names no skill for the library
- **WHEN** the `rf-web` catalogue lists Browser and SeleniumLibrary without naming their skills, its plugin serves `rf-browser` and `rf-selenium`, and the project has `robotframework-browser`
- **THEN** the match names `skill://rf-browser/SKILL.md`, the plugin's skill whose name contains the library's name without a `Library` suffix

#### Scenario: No skill is named after the library
- **WHEN** a catalogue lists a library without naming its skills and none of its plugin's skills contains the library's name
- **THEN** the match names every served skill of that plugin

#### Scenario: Company catalogue in a skills folder
- **WHEN** `--skills-dir` points to a folder with `plugins/acme/catalog.yaml` listing package `acme-robot` and skill `acme-robot`, and the project has `acme-robot` installed
- **THEN** the match names `skill://acme-robot/SKILL.md`

#### Scenario: No official package
- **WHEN** the official skills package is not installed and the project has `robotframework-seleniumlibrary`
- **THEN** the match comes from the built-in catalogue and names `skill://rf-locators/SKILL.md`

### Requirement: Project skills on the first result
In server mode, the first tool result of an MCP session for a project root SHALL carry a `project_skills` list when the root has at least one match. Each entry SHALL give the library, the installed version, one skill URI, and `supported`, with the range when not supported. The list SHALL hold at most three entries, ordered by catalogue precedence (skills folders, then the official package, then built-in), and SHALL NOT repeat in later results of that session for that root.

#### Scenario: First call of a session
- **WHEN** a client's first call in a session is `rf_discover` for a project with Browser 20.1.0 and RequestsLibrary 0.9.7 installed, and the official skills are served
- **THEN** the result carries `project_skills` with an entry for `skill://rf-browser/SKILL.md` and one for `skill://rf-requests/SKILL.md`, both supported

#### Scenario: Later calls
- **WHEN** the same session calls `rf_analyze` for the same root
- **THEN** the result carries no `project_skills`

#### Scenario: Another root in the same session
- **WHEN** the same session then calls a tool for a different project root that has matches
- **THEN** that first result for the new root carries its own `project_skills`

#### Scenario: Many matches
- **WHEN** a project has five libraries with matching skills
- **THEN** `project_skills` holds three entries

#### Scenario: Several skills for one library
- **WHEN** RequestsLibrary matches two skills of an earlier catalogue and Browser one skill of a later catalogue
- **THEN** `project_skills` lists the first RequestsLibrary skill, then the Browser skill, then the second RequestsLibrary skill

#### Scenario: Client mode
- **WHEN** rf-mcp runs in client mode
- **THEN** no tool result carries `project_skills`

### Requirement: Skill pointers are counted
The system SHALL count every skill pointer it gives an agent in the `rf_mcp.skills.pointed` metric, with the skill name and how it was given (`project_skills` or `hint`) as attributes, so pointers can be compared with `rf_mcp.skills.read`.

#### Scenario: Pointer counted
- **WHEN** a first result carries `project_skills` with `skill://rf-browser/SKILL.md` and OpenTelemetry metrics are enabled
- **THEN** `rf_mcp.skills.pointed` increases by one with skill `rf-browser` and source `project_skills`
