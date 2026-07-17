## ADDED Requirements

### Requirement: Verify interaction target belongs to the AUT window

The system SHALL verify, before dispatching a pointer operation in a
DESKTOP_TESTING session, that the resolved interaction target element belongs
to the AUT top-level window subtree (not merely that some node matched the
descriptor), so that a coordinate which happens to fall on another window is
not actuated as if it were the AUT.

#### Scenario: Resolved target is inside the AUT window
- **WHEN** a descriptor resolves to an element whose top-level ancestor is
  the AUT window
- **THEN** the operation is dispatched

#### Scenario: Resolved target is outside the AUT window
- **WHEN** a resolved element's top-level ancestor is a window other than
  the AUT (cross-window collision)
- **THEN** the system does not dispatch the operation against the foreign
  window and reports a window-scope error or warning identifying the
  mismatch

### Requirement: Window-scoped descriptor resolution default

The system SHALL prefer app-scoped descriptors (rooted at the AUT
application, e.g. `/app:*[@Name='X']//...`) for desktop operations and SHALL
guide agents away from unscoped `//` descriptors that walk the whole desktop
and can match elements in other applications.

#### Scenario: Unscoped descriptor is flagged
- **WHEN** a desktop operation uses an unscoped `//`-rooted descriptor
- **THEN** the system surfaces guidance recommending an app-scoped
  descriptor for the AUT

### Requirement: Coordinate-collision detection on overlapping windows

When two windows overlap on the same display, the system SHALL use the
AUT-window resolution to ensure a pointer operation actuates the AUT's
element rather than whatever window is stacked at the same coordinates.

#### Scenario: Overlapping windows at the same coordinates
- **WHEN** the AUT window and another window occupy the same screen region
  and an operation targets an AUT element under that region
- **THEN** the AUT window is focused/raised and the operation is verified to
  act on the AUT element, not the overlapping window
