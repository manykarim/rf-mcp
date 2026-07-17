# desktop-focus-verifiability Specification

## Purpose
TBD - created by archiving change platynui-visible-safe-targeting. Update Purpose after archive.
## Requirements
### Requirement: Upstream pattern introspection before focus
Focus-before-act SHALL determine focusability by querying the resolved window node's `supported_patterns()` (PlatynUI new-core API) before attempting any activation, and SHALL record the discovered pattern names on the focus outcome.

#### Scenario: Patterns recorded on outcome
- **WHEN** focus-before-act resolves a window node that advertises `WindowSurface` and `Focusable`
- **THEN** the `platynui_focus` response section lists both pattern names

### Requirement: Verified activation via upstream bring_to_front
When the window supports `WindowSurface`, focus-before-act SHALL use the upstream `Runtime.bring_to_front(node, wait_ms=…)` API (restore + activate + poll `accepts_user_input()`) instead of manually invoking pattern actions, and SHALL surface the `accepts_user_input` verdict as `input_ready` on the focus outcome.

#### Scenario: Verified activation succeeds
- **WHEN** `bring_to_front` succeeds and `accepts_user_input()` returns true within the wait budget
- **THEN** the focus outcome reports `strategy: "bring_to_front"` and `input_ready: true`

#### Scenario: Activation succeeds but input readiness unverified
- **WHEN** `bring_to_front` succeeds but `accepts_user_input()` is unavailable or false
- **THEN** the focus outcome reports `input_ready: false` (or null when unavailable) and a `platynui_focus_warning` hint is emitted

### Requirement: Focus-unverifiable warning
When the resolved AUT window supports neither `WindowSurface` nor `Focusable`, focus-before-act SHALL emit a `platynui_focus_warning` hint stating that input focus could not be verified for this target and keystrokes may not land, naming the missing patterns. The interaction keyword itself SHALL still execute (warning, not failure).

#### Scenario: LibreOffice frame without focus patterns
- **WHEN** a keyboard interaction targets a frame whose node advertises neither `WindowSurface` nor `Focusable`
- **THEN** the `execute_step` response contains a hint of type `platynui_focus_warning` with the message "input focus could not be verified for this target — keystrokes may not land (no WindowSurface/Focusable pattern)" and the step still executes

### Requirement: Custom X11 raise is last resort and never silent
The ctypes `XRaiseWindow`-by-PID fallback SHALL only run after upstream `bring_to_front` and `focus` paths are unavailable or failed, and its use SHALL always be accompanied by a focus-unverifiable warning and `strategy: "x11_raise"`.

#### Scenario: Fallback is flagged
- **WHEN** upstream activation is unavailable and the ctypes raise succeeds
- **THEN** the focus outcome reports `strategy: "x11_raise"` and a `platynui_focus_warning` hint is present

### Requirement: AUT process identity scope check
When the session knows the AUT process id (launched via Process library), focus-before-act SHALL compare it with the `ProcessId` attribute of the target's `app:Application` ancestor (upstream Application pattern) and SHALL add a scope warning on mismatch.

#### Scenario: Target belongs to a different process
- **WHEN** the AUT was launched as PID 1234 and the resolved target's application ancestor reports ProcessId 5678
- **THEN** a warning is emitted stating the target belongs to a different process than the launched AUT

### Requirement: Graceful degradation on older native runtimes
All upstream API calls (`supported_patterns`, `bring_to_front(wait_ms=…)`, `accepts_user_input`) SHALL be guarded so that an older `platynui_native` build degrades to the previous behavior plus a focus-unverifiable warning, never an exception.

#### Scenario: Runtime without supported_patterns
- **WHEN** the bound `platynui_native` runtime lacks `supported_patterns` on nodes
- **THEN** focus-before-act completes without raising and emits the focus-unverifiable warning

