# desktop-accessibility-exposure-diagnostic Specification

## Purpose
TBD - created by archiving change desktop-native-platynui-alignment. Update Purpose after archive.
## Requirements
### Requirement: Window-presence detection via a guarded EWMH probe

The system SHALL determine whether a desktop application's window exists
independent of the AT-SPI2 control tree. Because PlatynUI's native API/CLI
exposes NO window list independent of AT-SPI (spike: `platynui-cli window`
evaluates `//control:Window` through the AT-SPI tree; `_NET_CLIENT_LIST` is
internal-only), this detection uses a guarded, DOCUMENTED EWMH probe — factored
from robotmcp's existing ctypes `_NET_CLIENT_LIST`/`_NET_WM_PID` enumeration,
matching by window class/name and/or launched PID. It MUST return a tri-state
(present / absent / unknown), never raise (returning "unknown" when X11 is
unavailable), and carry an inline note citing the missing native capability.

#### Scenario: matching window present
- **WHEN** the probe runs for an application whose X11 toplevel exists
- **THEN** it reports present

#### Scenario: no matching window
- **WHEN** no X11 toplevel matches the application
- **THEN** it reports absent

#### Scenario: X11 unavailable is not a false signal
- **WHEN** libX11 cannot be loaded or there is no usable DISPLAY
- **THEN** the probe reports "unknown" (never raises, never asserts present/absent)

#### Scenario: documented as a fallback for a missing native capability
- **WHEN** the probe is defined in the code
- **THEN** it is annotated that PlatynUI exposes no native window list
  independent of AT-SPI, so this EWMH probe is the documented fallback

### Requirement: get_ui_tree distinguishes "no AT-SPI tree" from "window absent"

The system SHALL, when a desktop `get_ui_tree` inspection finds NO application
matching the requested filter (or zero applications), use the window-presence probe to add a diagnostic: `accessibility_not_exposed` when a matching
window IS present (running but publishing no accessibility tree), or
`app_window_absent` when no window is present. When the probe is "unknown", the
diagnostic SHALL say the distinction could not be determined.

#### Scenario: window present but no app in the tree
- **WHEN** a desktop ui_tree request returns no matching application but the window-presence probe shows it present
- **THEN** the result includes an `accessibility_not_exposed` diagnostic

#### Scenario: window absent
- **WHEN** no matching application is in the tree and the window-presence probe shows no
  matching window
- **THEN** the result includes an `app_window_absent` diagnostic

#### Scenario: app present in the tree → no diagnostic
- **WHEN** the requested application resolves in the PlatynUI tree
- **THEN** no exposure diagnostic is added

### Requirement: The diagnostic reports providers and actionable remediation

The system SHALL include in the `accessibility_not_exposed` diagnostic the
active PlatynUI providers (from the native `providers()` API), a statement that
the window is present but exposes no accessibility tree (a GTK/AT-SPI bridge or
environment issue, not a locator problem), and concrete remediation (ensure the
accessibility bridge / AT-SPI bus is enabled; allow the app a moment to
register). PlatynUI guidance SHALL reference the diagnostic so an agent
recognizes the condition instead of dropping to coordinate clicks/OCR.

#### Scenario: diagnostic names providers + remediation
- **WHEN** the `accessibility_not_exposed` diagnostic is produced
- **THEN** it lists the active providers and concrete remediation steps and
  frames the issue as accessibility/environment, not a locator problem

#### Scenario: guidance references the diagnostic
- **WHEN** a caller reads the PlatynUI guidance
- **THEN** it explains an empty tree after launch may mean the app exposes no
  AT-SPI tree, points to `accessibility_not_exposed`, and says to remediate the
  bridge rather than fall back to coordinates/OCR

