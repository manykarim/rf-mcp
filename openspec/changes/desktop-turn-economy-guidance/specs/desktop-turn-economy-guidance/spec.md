# Spec: desktop-turn-economy-guidance

## ADDED Requirements

### Requirement: Desktop session init returns a keyword cheat-sheet and locator crib
`manage_session(action="init")` SHALL include a `desktop_guidance` field in its
response when the session is a desktop session (session type resolves to
`DESKTOP_TESTING`, or `PlatynUI.BareMetal`/`PlatynUI` is among the
requested/loaded libraries). The field SHALL contain (a) a keyword cheat-sheet
listing every `PlatynUI.BareMetal` keyword (24 at time of writing) with a
one-line signature that preserves argument order — including
`Take Screenshot(descriptor, filename=…, rect=…)` with descriptor FIRST — and
(b) a locator crib covering at minimum: app-scoped descriptors
(`/app:*[@Name='X']//…`) with a warning against unscoped `//`, `Set Root`
usage, that top-level windows on Linux are `control:Frame` (not
`control:Window`), launch-the-app-before-querying ordering, a result
read-back recipe (`Get Attribute` on a `control:Label`), and a pointer to
`get_locator_guidance` for the full guidance. The cheat-sheet SHALL be derived
from the loaded library's documentation (libdoc), not a hand-maintained
keyword list, so it cannot drift from the actual keyword surface.

#### Scenario: desktop init delivers the full keyword surface
- **WHEN** `manage_session(action="init", libraries=["PlatynUI.BareMetal", "Process", "BuiltIn"])` succeeds
- **THEN** the response contains `desktop_guidance` whose cheat-sheet names every keyword reported by the library's libdoc (24 today), each with its arguments in declaration order, and `Take Screenshot` shows `descriptor` before `filename`

#### Scenario: crib states the trap-avoiding rules
- **WHEN** the `desktop_guidance` field is returned
- **THEN** its crib text includes app-scoping with `/app:*[@Name=`, a warning against starting descriptors with `//`, the statement that Linux top-level windows are `control:Frame` and not `control:Window`, a `Get Attribute`-based read-back recipe, and a reference to `get_locator_guidance`

#### Scenario: bundle is bounded
- **WHEN** `desktop_guidance` is attached to an init response
- **THEN** its total serialized size is at most ~3 KB (cheat-sheet + crib measured at ~2.2 KB), and it is computed once per process and reused (no per-init libdoc re-parse)

#### Scenario: non-desktop init is unchanged
- **WHEN** `manage_session(action="init")` creates a web or API session (e.g. `libraries=["Browser", "BuiltIn"]`)
- **THEN** the response contains no `desktop_guidance` field

### Requirement: A desktop-focused instruction template is selectable
The instruction domain SHALL provide a `desktop-focused` template alongside
`browser-focused` and `api-focused`. It SHALL be selectable by name through
`InstructionTemplate.get_by_name("desktop-focused")` and via
`ROBOTMCP_INSTRUCTIONS_TEMPLATE=desktop-focused`
(`InstructionTemplateType`). Its content SHALL describe the desktop workflow —
init with PlatynUI + Process, launch the application under test with
`Start Process`, wait for/`Query` the `control:Frame`, interact with
pointer/keyboard keywords, read back with `Get Attribute` — and SHALL state
the same locator rules as the init crib (app-scoping, no unscoped `//`,
Frame-not-Window on Linux, `Take Screenshot` argument order).

#### Scenario: template selectable by name
- **WHEN** `InstructionTemplate.get_by_name("desktop-focused")` is called
- **THEN** it returns the desktop template (no `ValueError`), and the unknown-template error message for a bogus name lists `desktop-focused` among the valid templates

#### Scenario: template selectable via environment
- **WHEN** the server resolves instructions with `ROBOTMCP_INSTRUCTIONS_TEMPLATE=desktop-focused`
- **THEN** the resolved MCP instructions are the desktop-focused content (mentioning PlatynUI, `Start Process`, `control:Frame`, and app-scoped descriptors)

#### Scenario: existing templates unaffected
- **WHEN** any previously valid template name (`minimal`, `standard`, `detailed`, `browser-focused`, `api-focused`) is requested
- **THEN** it resolves exactly as before

### Requirement: Process is a core library for desktop sessions
The `DESKTOP_TESTING` session profile SHALL list `Process` in
`core_libraries` (not merely `optional_libraries`), so a bare
`manage_session(init)` desktop session can execute `Start Process` without an
`import_library` detour. `PlatynUI.BareMetal` SHALL remain first in
`core_libraries` so the derived library search order continues to lead with
the desktop library.

#### Scenario: bare desktop init can launch an app
- **WHEN** a desktop session is initialized through the profile-driven loading path and the agent executes `Start Process`
- **THEN** the keyword resolves to the Process library without a prior explicit `import_library`/`libraries=` mention of Process

#### Scenario: search order still led by PlatynUI
- **WHEN** the desktop profile's libraries-to-load and search order are computed
- **THEN** `Process` is included and `PlatynUI.BareMetal` remains first
