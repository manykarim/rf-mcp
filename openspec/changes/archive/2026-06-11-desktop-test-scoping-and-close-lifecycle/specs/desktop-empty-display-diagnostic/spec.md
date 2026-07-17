# Spec: desktop-empty-display-diagnostic

## ADDED Requirements

### Requirement: Empty display diagnosed as empty, not as probe failure
When the accessibility tree resolves no applications, no app filters were given, and the batched display-PID probe reports the display reachable with zero client windows, the exposure diagnostic SHALL state that the display is reachable but has no application windows (the AUT was not launched on it), instead of "window presence could not be determined (X11 probe unavailable)".

#### Scenario: Pre-launch empty isolated display
- **WHEN** `get_session_state(sections=['ui_tree'])` runs on a reachable display with no client windows and no app filters
- **THEN** the accessibility diagnostic has type `display_empty` and its message says the display has no application windows yet

#### Scenario: Genuine probe failure keeps the current wording
- **WHEN** the display-PID probe cannot complete
- **THEN** the diagnostic retains the existing "could not be determined" wording

#### Scenario: Windows present but unresolved keeps the undetermined wording
- **WHEN** the probe reports client windows on the display but no application resolved in the accessibility tree
- **THEN** the existing undetermined/exposure diagnostics apply unchanged
