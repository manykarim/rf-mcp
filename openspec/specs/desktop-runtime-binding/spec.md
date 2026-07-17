# desktop-runtime-binding Specification

## Purpose
TBD - created by archiving change platynui-desktop-safety-isolation. Update Purpose after archive.
## Requirements
### Requirement: A single runtime broker owns the PlatynUI runtime

The system SHALL route all PlatynUI native-runtime use through a single broker
scoped to the bound desktop session/display, with explicit lifecycle states
(`open` / `shutting_down` / `closed`), a lock-protected lazy first bind after
the display env is settled, and thread-affine access. All call sites — keyword
execution, the focus manager, and the `ui_tree` path — SHALL obtain the runtime
from the broker rather than constructing their own.

#### Scenario: ui_tree reuses the broker instead of creating its own runtime
- **WHEN** `get_session_state(sections=["ui_tree"])` is invoked twice in the
  same process
- **THEN** no throwaway `Runtime()` is created or shut down per call; both calls
  use the same broker-owned runtime

#### Scenario: Lifecycle states are enforced
- **WHEN** the broker has transitioned to `closed`
- **THEN** further use is rejected with a clear error (single shared runtime;
  restart the process) instead of attempting a failing re-initialization

### Requirement: Bind once after the display is settled; no re-init after shutdown

The system SHALL bind the runtime only after the desktop session's display
environment is settled, so it binds to the intended display and does not fail
with "not available after shutdown or failed connect", and SHALL never
re-initialize the process-global platform module after a shutdown.

#### Scenario: Runtime binds on the settled display
- **WHEN** a desktop session has set its display environment and then issues
  its first PlatynUI keyword
- **THEN** the runtime initializes successfully against that display

#### Scenario: No re-bind after shutdown in the same process
- **WHEN** a PlatynUI runtime has been created (and possibly the ui_tree path
  previously shut one down)
- **THEN** later desktop keywords reuse the broker's runtime and do not trigger
  a failed re-initialization ("not available after shutdown")

#### Scenario: Concurrent first-use does not race the bind
- **WHEN** two desktop operations reach the broker concurrently before the
  runtime exists
- **THEN** exactly one bind occurs under the lock and both observe a working
  runtime

### Requirement: Consistent binding between pytest and MCP/Robot paths

The system SHALL make the runtime-binding lifecycle on the MCP/Robot keyword
path behave consistently with the working pytest isolation recipe, so an
isolated display that is healthy under `xdpyinfo` is also usable from the
MCP/Robot route.

#### Scenario: Healthy isolated display is usable from MCP/Robot
- **WHEN** an isolated display is verified healthy (`xdpyinfo` succeeds) and a
  desktop session runs PlatynUI keywords through the MCP/Robot path
- **THEN** the runtime binds and resolves the AUT just as it does under the
  pytest isolation reference

### Requirement: Reliable input delivery once the AUT frame is resolvable

The system SHALL deliver pointer/keyboard input that mutates the AUT's state
once the AUT frame is resolvable and visible, so a resolvable frame is not
left in a "queryable but input does nothing" state.

#### Scenario: Click mutates the AUT after the frame resolves
- **WHEN** the AUT frame is resolvable and visible on the bound display and a
  pointer click targets one of its controls
- **THEN** the click mutates the AUT's observable state (e.g. the calculator
  display character count changes)

#### Scenario: Non-mutating input is surfaced
- **WHEN** a click resolves a target but the AUT state does not change
- **THEN** the post-action verification / focus outcome surfaces a warning so
  the no-op is not silently recorded as success

