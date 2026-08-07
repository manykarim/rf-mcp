## ADDED Requirements

### Requirement: The dashboard renders session data faithfully

The dashboard SHALL render identifiers without blind character truncation (showing them in full or with
an explicit ellipsis), SHALL display user-defined variables regardless of naming convention (not hiding
ALL-CAPS names), SHALL NOT fabricate or duplicate library/event data, and SHALL NOT crash on the
zero-sessions path.

#### Scenario: identifiers are shown faithfully
- **WHEN** a session named `frontend-demo` is displayed
- **THEN** its name is shown in full (e.g. "Session frontend-demo"), not truncated to a misleading prefix

#### Scenario: user variables are visible
- **WHEN** a session defines ALL-CAPS variables (the Robot Framework convention, e.g. `CITY`, `BASE_URL`)
- **THEN** they appear in the Variables panel rather than being hidden

#### Scenario: library and event data are not fabricated or duplicated
- **WHEN** a session's libraries and lifecycle events are displayed
- **THEN** only libraries the session actually imported are shown (deduplicated), and a single lifecycle event is not shown twice

#### Scenario: the empty state does not crash
- **WHEN** there are no sessions (fresh install or last session closed)
- **THEN** the dashboard shows a clean empty state without throwing, and does not leave a stale session pane
