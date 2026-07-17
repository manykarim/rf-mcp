## ADDED Requirements

### Requirement: Normative classification precedence between desktop and mobile

The system SHALL apply an explicit precedence in
`ExecutionSession.detect_session_type_from_scenario`: an **explicit mobile
signal** (one of android, ios, appium, apk, "bundle id", emulator, device,
uiautomator2, xcuitest, tap, swipe) wins and yields `mobile_testing`; otherwise
a **desktop signal** (a desktop-app name such as calculator/text editor/gedit/
notepad, or a desktop toolkit/marker such as gnome/gtk/qt/kde/win32/wpf/x11/
wayland/exe/"native window"/platynui) yields the desktop/PlatynUI type; the
**generic token "app" alone MUST NOT yield `mobile_testing`** when a desktop
signal is present and no explicit mobile signal is present.

#### Scenario: Calculator app routes to desktop
- **WHEN** the scenario is "Test the calculator app" or "Open GNOME Calculator
  and perform calculations" (desktop signal, no explicit mobile signal)
- **THEN** the session type is the desktop/PlatynUI type, not `mobile_testing`,
  and AppiumLibrary is not auto-loaded

#### Scenario: Explicit mobile signal still wins
- **WHEN** the scenario is "Test the calculator app on android" or "Open the iOS
  calculator app" (a desktop noun plus an explicit mobile signal)
- **THEN** the session type SHALL be `mobile_testing`, not desktop

#### Scenario: Desktop platform with a desktop noun routes to desktop
- **WHEN** the scenario is "Test the calculator app on windows" (desktop noun,
  no explicit mobile signal)
- **THEN** the session type is the desktop/PlatynUI type

### Requirement: Desktop-typed sessions accept Process and PlatynUI

The system SHALL allow a desktop-typed session to import the libraries needed
for desktop automation (at least `Process`, `BuiltIn`, `PlatynUI.BareMetal`),
so a correctly-classified desktop session does not later reject `Process`.

#### Scenario: Process accepted in a desktop session
- **WHEN** a desktop-typed session imports `Process`
- **THEN** the import succeeds (it is not rejected as out-of-profile)

#### Scenario: Misclassification does not strand the workflow
- **WHEN** a scenario is correctly classified as desktop
- **THEN** the recommended libraries for the session include the desktop set
  and not the mobile/Appium set
