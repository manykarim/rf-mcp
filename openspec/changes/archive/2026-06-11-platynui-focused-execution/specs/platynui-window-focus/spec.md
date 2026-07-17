## ADDED Requirements

### Requirement: Resolve the AUT top-level window from any descriptor

The system SHALL resolve the application-under-test (AUT) top-level window
(the nearest `app:`/`control:Frame`/`control:Window`/`control:Dialog`
ancestor exposing the WindowSurface pattern) from any PlatynUI descriptor
used in a pointer or keyboard operation, so that focus and visibility logic
can target the correct window regardless of how deep the target element is.

#### Scenario: Resolve window from a deeply nested button descriptor
- **WHEN** a desktop session dispatches an operation on
  `/app:*[@Name='gnome-calculator']//control:Button[@Name='7']`
- **THEN** the system resolves the owning top-level window node for that
  button before dispatching the operation

#### Scenario: Descriptor that matches no node
- **WHEN** a descriptor resolves to no UI node within the retry timeout
- **THEN** the system reports an element-not-found error and does NOT
  dispatch a pointer/keyboard operation against an unresolved target

### Requirement: Focus the AUT window before pointer/keyboard dispatch

In a DESKTOP_TESTING session, the system SHALL ensure the resolved AUT
window is raised and focused immediately before dispatching any pointer or
keyboard operation, so input cannot be delivered to a different, currently
active window. This focus-before-act behavior SHALL be the default.

#### Scenario: Another window is active when an operation runs
- **WHEN** a different application window holds input focus and a desktop
  step targets the AUT
- **THEN** the system raises and focuses the AUT window before the click or
  key sequence is sent, and the input is received by the AUT

#### Scenario: Focus is re-asserted after switching apps in one session
- **WHEN** a single session interacts with app A, then app B, then app A
  again
- **THEN** the AUT window is re-focused before each operation so operations
  on app A are not delivered to app B

### Requirement: Portable focus strategy across window environments

The focus mechanism SHALL work across (a) X11 with a window manager, (b)
WM-less X11 such as Xvfb, and (c) Wayland/XWayland. Where the PlatynUI
WindowSurface activation is unavailable (e.g. no EWMH/WM), the system SHALL
fall back to a portable raise/focus mechanism rather than failing the
operation.

#### Scenario: WindowSurface activation available
- **WHEN** a window manager is present and the window exposes the
  WindowSurface pattern
- **THEN** the system focuses via the native PlatynUI window activation

#### Scenario: WM-less environment without WindowSurface
- **WHEN** the AUT runs under a WM-less X server and WindowSurface
  activation raises a pattern error
- **THEN** the system falls back to a portable raise/focus mechanism and the
  operation still targets the AUT window

### Requirement: Focus escape hatch

The system SHALL provide an explicit escape hatch (per-call argument and/or
environment variable) to disable focus-before-act for advanced scenarios
that intentionally operate without changing focus. When the escape hatch is
used, the behavior reverts to direct dispatch and this is recorded for
provenance.

#### Scenario: Escape hatch disables focus-before-act
- **WHEN** an operation is invoked with the focus escape hatch enabled
- **THEN** the system dispatches the operation without raising/focusing the
  AUT window and marks the step as having bypassed focus targeting

### Requirement: Focus intent and keyword surface

The system SHALL expose an "ensure application focused" capability through
the rf-mcp intent layer and as a callable step, so agents can explicitly
focus the AUT at the start of a test or after a context switch. Locator
guidance and MCP instructions SHALL direct agents to use it.

#### Scenario: Agent ensures focus via the intent layer
- **WHEN** an agent issues a focus intent targeting the AUT window
  descriptor
- **THEN** the system raises and focuses that window and reports success
