## ADDED Requirements

### Requirement: Browser and platform metadata reflect the session's actual technology

The dashboard SHALL derive a session's browser and platform metadata from what the session actually did,
not from defaults. It SHALL NOT report a browser engine or current URL for a session that has no browser
session (no Browser/SeleniumLibrary and no live browser), and it SHALL present a platform that reflects
the session's technology (web for browser sessions, API for HTTP-request sessions, desktop/mobile where
applicable, otherwise generic) rather than an unconditional "web" default.

#### Scenario: a non-browser session reports no browser state
- **WHEN** a session that never opened a browser (e.g. a BuiltIn or Requests session) is displayed
- **THEN** no browser engine and no current URL are shown for it (and no "Error converting browser state" is logged)

#### Scenario: a real browser session reports its real browser state
- **WHEN** a session opened a browser and navigated to a URL
- **THEN** the displayed browser engine and current URL reflect the actual browser and page, not a hardcoded default

#### Scenario: platform reflects the session's technology
- **WHEN** sessions use different technologies (browser, HTTP requests, none)
- **THEN** the displayed platform reflects that technology (web / api / generic), not an unconditional "web"
