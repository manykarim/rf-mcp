## ADDED Requirements

### Requirement: Project-aware overlay references the installed rf-mcp across platforms

The project-aware uv overlay SHALL add the same rf-mcp distribution the user already installed: a
version pin (`--with rf-mcp==<version>`) for a published install, or the local source directory
(`--with-editable <path>`) when rf-mcp was installed from a local or unpublished source recorded as a
`file://` direct URL. The installer SHALL convert a `file://` direct URL to a filesystem path with a
platform-correct routine so a Windows drive-letter URL resolves to the corresponding Windows path and
the source overlay is emitted on Windows, macOS, and Linux alike — never silently degrading a
local-source install to an unresolvable version pin on any platform.

#### Scenario: local source install overlays that source on POSIX
- **WHEN** the installed rf-mcp records a `file://` direct URL such as `file:///home/u/rf-mcp` and that directory exists
- **THEN** the resolved overlay adds `--with-editable /home/u/rf-mcp`, so the overlay uses the same rf-mcp source even when its version is not on PyPI

#### Scenario: local source install overlays that source on Windows
- **WHEN** the installed rf-mcp records a `file://` direct URL such as `file:///C:/work/rf-mcp` on Windows and that directory exists
- **THEN** the resolved overlay adds `--with-editable C:\work\rf-mcp` (the drive-letter URL converted to a valid Windows path), not a `--with rf-mcp==<version>` pin

#### Scenario: published install uses a version pin
- **WHEN** the installed rf-mcp has no `file://` direct URL (a normal PyPI install) and its version is known
- **THEN** the resolved overlay adds `--with rf-mcp==<version>`, unchanged from prior behaviour
