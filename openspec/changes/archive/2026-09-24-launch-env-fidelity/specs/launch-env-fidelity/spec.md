# Spec: launch-env-fidelity

## ADDED Requirements

### Requirement: A configured launch provides the libraries the installation provides
The launch command rf-mcp writes into an agent configuration SHALL make the same Robot
Framework libraries available to the running server that the rf-mcp installation itself
provides. When the launch layers rf-mcp onto another environment, the specification used
SHALL carry the extras of the running installation, so libraries installed by the user are
not lost. When those extras cannot be determined, the resolved launch SHALL say so rather
than silently omitting them.

#### Scenario: an overlay launch keeps the installed web libraries
- **WHEN** a user installs `rf-mcp[all]` and registers a project whose environment causes the uv-overlay strategy to be chosen
- **THEN** the configured launch provides Browser and SeleniumLibrary in addition to the project's own libraries

#### Scenario: the project's own libraries remain available
- **WHEN** the overlay launch resolves for a project whose environment contains a library rf-mcp does not bundle
- **THEN** that library is importable by the running server, alongside rf-mcp's own libraries

#### Scenario: undeterminable extras are reported, not dropped
- **WHEN** rf-mcp cannot determine which extras its own installation provides
- **THEN** the resolved-launch note states that the overlay may not carry them, instead of presenting the launch as complete

### Requirement: An editable overlay reference is used only for a directory
A launch specification SHALL use an editable reference only when the rf-mcp installation
resolves to a local DIRECTORY. An installation whose recorded source is an archive, such as
a wheel file, SHALL be referenced as an ordinary path requirement, so the resolved command
is valid for the resolver that will run it.

#### Scenario: a wheel-sourced install produces a valid command
- **WHEN** rf-mcp was installed from a local wheel file and a project-aware launch is resolved
- **THEN** the launch references the wheel as a normal requirement and the command runs, rather than failing with "Editable must refer to a local directory, not an archive"

#### Scenario: a real editable checkout still uses an editable reference
- **WHEN** rf-mcp was installed from a local source directory in editable mode
- **THEN** the launch still uses the editable reference to that directory

### Requirement: The Robot Framework version that will execute the tests is reported
Where the resolved launch would execute tests with a Robot Framework version other than the
one the project pins, rf-mcp SHALL report which version will actually be used. The check
SHALL NOT be limited to major-version differences.

#### Scenario: a minor-version difference is surfaced
- **WHEN** a project pins a Robot Framework version that the resolved launch would not provide, differing only in minor or patch version
- **THEN** `robotmcp doctor` and the install-time note state which Robot Framework version will execute the tests

#### Scenario: a matching version is not reported as a problem
- **WHEN** the resolved launch provides a Robot Framework version that satisfies the project's pin
- **THEN** no version warning is produced

### Requirement: Validation status reflects error-severity findings
A suite validation that produced one or more error-severity issues SHALL report a failed
status and an unsuccessful result, irrespective of the Robot Framework return code. Warning
-severity findings SHALL NOT by themselves produce a failed status.

#### Scenario: failed imports fail the validation
- **WHEN** a dry-run validation records error-severity import failures and Robot Framework returns 0
- **THEN** the reported status is failed and the result is unsuccessful, so a caller does not proceed to execution

#### Scenario: a clean validation still passes
- **WHEN** a dry-run validation records no error-severity issues
- **THEN** the reported status is passed and the result is successful

#### Scenario: warnings alone do not fail a validation
- **WHEN** a dry-run validation records only warning-severity findings
- **THEN** the status is not failed

### Requirement: The environment coupling is documented
User-facing documentation SHALL state that rf-mcp imports Robot Framework libraries into
its own process and executes suites in that same interpreter, what that means for a project
with its own libraries or pinned Robot Framework version, and that pasting the MCP
configuration snippet by hand bypasses the project-aware launch that `robotmcp install`
resolves.

#### Scenario: a reader with an existing RF project is directed to the right path
- **WHEN** a user with an existing Robot Framework project consults the installation documentation
- **THEN** it explains that a plain tool install cannot see their project's libraries and directs them to `robotmcp install` from the project directory

#### Scenario: the hand-paste shortcut carries its caveat
- **WHEN** the documentation offers the MCP configuration snippet for manual pasting
- **THEN** it states that the snippet uses rf-mcp's own environment and does not resolve a project-aware launch
