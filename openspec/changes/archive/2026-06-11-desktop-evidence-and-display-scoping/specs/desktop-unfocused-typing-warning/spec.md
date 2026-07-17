# Spec: desktop-unfocused-typing-warning

## ADDED Requirements

### Requirement: Type-at-focus without verified focus warns
A desktop keyboard interaction with no target descriptor (type-at-focus) SHALL emit a `platynui_focus_warning` hint when no AUT window focus has been verifiably established in the session, stating that keystrokes may land nowhere. The warning is emitted at most once per session and never fails the step.

#### Scenario: Blind typing into an empty display
- **WHEN** `Keyboard Type  hello` (no descriptor) executes and no prior step verified focus on any AUT window
- **THEN** the response contains a `platynui_focus_warning` hint about unverified type-at-focus, and the step still executes

#### Scenario: Typing after verified focus is silent
- **WHEN** a prior interaction verified focus (activation with input readiness or platform-reported active window) and then `Keyboard Type  hello` runs without a descriptor
- **THEN** no unfocused-typing warning is emitted

#### Scenario: Warning is once per session
- **WHEN** three blind typing steps run consecutively with no verified focus
- **THEN** only the first response carries the unfocused-typing warning
