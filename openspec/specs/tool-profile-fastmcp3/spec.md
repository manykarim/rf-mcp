# tool-profile-fastmcp3 Specification

## Purpose
TBD - created by archiving change fastmcp3-tool-profile-v3-native. Update Purpose after archive.
## Requirements
### Requirement: Tool enumeration works on FastMCP 3.x

The tool-profile compatibility layer SHALL enumerate the server's registered tools
on FastMCP 3.x using the version-native API, returning a non-empty mapping when
tools are registered, so the profile system can snapshot and manage them.

#### Scenario: enumeration returns the registered tools on v3
- **WHEN** tools are registered and the compatibility layer enumerates them on a FastMCP 3.x server (which has no `_tool_manager` and no `get_tools()` method)
- **THEN** it returns a mapping of the registered tools (keyed by name), not an empty result

### Requirement: A small-context profile reduces the exposed tool set

Activating a small-context tool profile SHALL reduce the number of tools exposed to
the client on FastMCP 3.x, by disabling the tools outside the profile via the
version-native enable/disable API.

#### Scenario: switching to a small profile hides tools
- **WHEN** the client lists tools, then a session is initialized with a small-context model tier, then the client lists tools again
- **THEN** the second listing contains fewer tools than the first

#### Scenario: restoring re-enables the full set
- **WHEN** a small-context profile has reduced the tool set and the full profile is restored
- **THEN** the previously-hidden tools are exposed again

### Requirement: Profile switching is stable under repeated activation on FastMCP 3.x

The tool-profile system SHALL toggle tool visibility on FastMCP 3.x purely by name
(enable/disable), without cloning or re-registering tool objects and without
disabling-then-re-enabling on description swaps, so that repeated profile switches do
not corrupt the server's tool provider or raise `maximum recursion depth exceeded`.

#### Scenario: repeated profile switches do not recurse
- **WHEN** several different profiles (with differing description modes) are activated and restored in sequence on a FastMCP 3.x server, and a client then lists or calls tools
- **THEN** the operations succeed without a recursion error, and later tool calls in the same process are unaffected

#### Scenario: descriptions are not swapped on v3
- **WHEN** a profile activation would swap a still-visible tool's description on FastMCP 3.x
- **THEN** no enable/disable churn is issued for that tool (the description is left as the original — a documented v3 limitation)

