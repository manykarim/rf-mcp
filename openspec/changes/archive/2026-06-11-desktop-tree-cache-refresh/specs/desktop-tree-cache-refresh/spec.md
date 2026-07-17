## ADDED Requirements

### Requirement: A desktop GUI launch refreshes the PlatynUI tree cache

The system SHALL clear the shared PlatynUI runtime accessibility-tree cache
after a desktop session's `Start Process`/`Run Process` of a GUI binary
succeeds, so the newly-launched application can become visible to subsequent
keyword queries rather than being hidden behind a pre-launch snapshot. The
refresh MUST be best-effort and desktop-gated — it never fails the launch step,
and non-desktop sessions are unaffected.

#### Scenario: launch clears the runtime tree cache
- **WHEN** a desktop session successfully runs `Start Process <gui-binary>`
- **THEN** the shared runtime tree cache is cleared (and the session is marked
  "tree may be stale" for the next resolving keyword)

#### Scenario: non-desktop / non-launch keywords do not refresh
- **WHEN** a non-desktop session launches a process, or a desktop session runs a
  non-launch keyword
- **THEN** no runtime tree-cache clear is triggered by this mechanism

### Requirement: The first desktop tree-resolving keyword after a launch sees a fresh tree

The system SHALL refresh the runtime tree cache before the first desktop
tree-resolving keyword (`Query`, `Get Attribute`, or the locator resolution of
an interaction keyword) that runs after a launch, so name-based locators and
assertions resolve against live AT-SPI state. After that refresh the stale flag
SHALL be cleared so steady-state desktop queries are not re-snapshotted on every
call.

#### Scenario: first post-launch query refreshes then clears the flag
- **WHEN** a desktop session has been flagged stale by a launch and then runs a
  `Query` or `Get Attribute` keyword
- **THEN** the runtime tree cache is refreshed before resolution, and the stale
  flag is cleared so the next query does not re-snapshot

#### Scenario: steady-state queries are not re-snapshotted
- **WHEN** a desktop session runs successive tree-resolving keywords with no
  intervening launch
- **THEN** only the first (post-launch) one refreshes; the rest use the warm
  cache (no per-call re-snapshot)

### Requirement: Guidance documents the desktop tree-refresh path

The system SHALL document, in the PlatynUI guidance, that the desktop
accessibility tree is cached, that the keyword path auto-refreshes after a
launch, and that `get_session_state(sections=['ui_tree'])` forces a refresh — so
an agent whose name-based locator does not resolve right after launching the app
can recover deterministically instead of resorting to coordinate clicks or OCR.

#### Scenario: guidance names the refresh recovery
- **WHEN** a caller reads the PlatynUI guidance
- **THEN** it explains the tree is cached, that a launch auto-refreshes it, and
  that `get_session_state(ui_tree)` forces a refresh when a locator still does
  not resolve
