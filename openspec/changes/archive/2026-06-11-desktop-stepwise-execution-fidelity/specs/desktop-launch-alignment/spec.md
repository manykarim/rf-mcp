## ADDED Requirements

### Requirement: Desktop GUI launches use the snap-sanitized environment

The system SHALL apply the snap-decontaminated launch environment (built by
`desktop_launch_env.build_desktop_launch_env`) to a desktop-session
`Start Process`/`Run Process` of a known GUI binary, so a snap-confined launch
does not inherit snap-rooted loader/module variables and die with a symbol
lookup error or a non-zero exit. The sanitizer MUST be actually invoked on the
execution path (it currently exists but is unwired).

#### Scenario: GUI launch gets sanitized env overrides
- **WHEN** a desktop session runs `Start Process <known-gui-binary>` with no
  explicit `env:` overrides
- **THEN** the dispatched arguments carry sanitized `env:` overrides (snap-rooted
  loader/module/data path segments filtered, bound display set) so the GUI app
  launches cleanly

#### Scenario: non-GUI / non-desktop launch is unchanged
- **WHEN** a non-desktop session, or a desktop session launching a non-GUI
  executable, runs a Process keyword
- **THEN** no GUI sanitization is applied (arguments unchanged)

### Requirement: Process launch state and desktop discovery disagreement is surfaced

The system SHALL detect when a desktop launch reports success at the API layer
but the launched process is not actually running (e.g. a quick non-zero exit),
and surface a warning rather than letting the agent assume the AUT is alive
because PlatynUI still observes (possibly stale or other) accessibility nodes.

#### Scenario: launched process exits but discovery still returns nodes
- **WHEN** a desktop `Start Process` returns a handle whose process is not
  running shortly after, while a PlatynUI query still returns application nodes
- **THEN** the response includes a warning that the launched process is not
  running (the visible nodes may be a different/stale instance), so success is
  not mistaken for a live AUT

### Requirement: Desktop input that does not change the application is flagged

The system SHALL provide a decision function that, given a desktop
pointer/keyboard interaction's success and a before/after snapshot of the
target's accessible display state (e.g. `native:Text.CharacterCount`), produces
a soft warning when the keyword succeeded but the state did not change — so an
agent does not treat an API "OK" as evidence the AUT reacted. The function MUST
be pure (no re-entrant keyword execution), so it can be driven by a
non-reentrant probe without risking deadlock under the execution lock.

#### Scenario: success with unchanged state yields a warning
- **WHEN** the helper is given a successful interaction keyword with equal
  before/after display-state snapshots
- **THEN** it returns a soft `desktop_input_no_effect` warning hint

#### Scenario: changed state yields no warning
- **WHEN** the before/after snapshots differ (the app reacted)
- **THEN** the helper returns no warning

#### Scenario: missing snapshot or non-interaction keyword yields no warning
- **WHEN** a snapshot is missing, the step did not succeed, or the keyword is
  not an interaction keyword
- **THEN** the helper returns no warning (it never fabricates a signal)
