## MODIFIED Requirements

### Requirement: Official skills package is read through its public API
The official package SHALL be used only through its public skills API at a catalogue format rf-mcp supports, both for the content it bundles and for snapshots of the repository fetched with that API; an installed package with an unsupported format SHALL be skipped with a warning in the log and in `rf-mcp doctor`, and the server SHALL start without it.

#### Scenario: Unsupported catalogue format
- **WHEN** the installed `robotframework-plugins` reports a catalogue format `rf-mcp` does not support
- **THEN** the server starts, serves no official skills, logs a warning naming the installed and supported formats, and `rf-mcp doctor` reports the same

#### Scenario: Snapshot read like the bundled content
- **WHEN** the official skills channel is a ref with a stored snapshot
- **THEN** its skill folders and catalogue are read through the package's snapshot API, and its skills take the official layer's place between `--skills-dir` and the built-in skills
