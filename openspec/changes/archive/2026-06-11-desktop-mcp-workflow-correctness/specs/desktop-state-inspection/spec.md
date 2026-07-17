## ADDED Requirements

### Requirement: Desktop sessions inspect the accessibility tree, not mobile source

The system SHALL route `get_session_state` state/page-source inspection for a
desktop (PlatynUI) session through the desktop accessibility (`ui_tree`) path,
not the mobile-source lookup, so a desktop session does not report "Failed to
get mobile source: No application is open" when a desktop app is running.

#### Scenario: desktop state auto-includes ui_tree, no mobile lookup
- **WHEN** `get_session_state` is called for a desktop session (optionally with
  `include_reduced_dom=True`)
- **THEN** the response auto-includes the `ui_tree` section for the desktop
  session and does not attempt a mobile/Appium source lookup; the
  `page_source` section, if requested, is a desktop-specific stub with a hint
  (not a mobile-source error)

#### Scenario: running desktop app is recognized
- **WHEN** a desktop app is running on the session's bound display and state is
  requested
- **THEN** the `ui_tree` section reports the application/window rather than
  "No application is open"

#### Scenario: web/mobile state behavior unchanged
- **WHEN** `get_session_state` is called for a web or mobile session
- **THEN** the page_source / source-lookup behavior is unchanged from current
  behavior (no regression from the desktop routing change)

### Requirement: Clear guidance when no desktop app is present

The system SHALL return a clear, desktop-appropriate message when a desktop
session has no resolvable application, rather than a mobile-source error.

#### Scenario: no desktop app yet
- **WHEN** state is requested for a desktop session with no app launched
- **THEN** the `ui_tree` section reports no resolvable application with an
  actionable hint to launch one, and any `page_source` section is a desktop
  stub whose message contains "no desktop application" — NOT "Failed to get
  mobile source: No application is open"
