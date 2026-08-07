## ADDED Requirements

### Requirement: The dashboard core flow is keyboard-operable with visible focus

The dashboard SHALL allow selecting a session using the keyboard alone (focusable session controls with
Enter/Space activation and appropriate roles/labels), SHALL render a visible focus indicator for
keyboard focus, and SHALL respect `prefers-reduced-motion` and allow touch scrolling of its panels.

#### Scenario: a session is selectable by keyboard
- **WHEN** a keyboard user focuses a session card and presses Enter or Space
- **THEN** that session is selected and its details are shown

#### Scenario: keyboard focus is visible
- **WHEN** an element receives keyboard focus
- **THEN** a visible focus indicator is shown

#### Scenario: motion and scrolling respect user constraints
- **WHEN** the user prefers reduced motion, or scrolls a panel on a touch device
- **THEN** animations are minimized and the panel scrolls rather than being locked
