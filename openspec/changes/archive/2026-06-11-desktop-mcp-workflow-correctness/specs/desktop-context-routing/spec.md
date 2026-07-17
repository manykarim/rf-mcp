## ADDED Requirements

### Requirement: PlatynUI is a registered recommendable library

The system SHALL include `PlatynUI.BareMetal` in the library registry used by
`recommend_libraries` (it is currently only a runtime plugin and absent from
the recommender), with desktop-oriented categories/use-cases, and SHALL accept
the `PlatynUI` alias as referring to `PlatynUI.BareMetal` in discovery and
recommendation. This is the prerequisite for desktop recommendation and
discovery.

#### Scenario: PlatynUI appears in the registry
- **WHEN** the recommender enumerates known libraries
- **THEN** `PlatynUI.BareMetal` is present with desktop categories

#### Scenario: PlatynUI alias resolves
- **WHEN** a caller references `PlatynUI` in discovery/recommendation
- **THEN** it resolves to `PlatynUI.BareMetal`

### Requirement: Explicit desktop context forces a desktop session across all routing sites

The system SHALL route a scenario to a desktop/PlatynUI session type whenever
the caller passes `context="desktop"` to `analyze_scenario`, overriding the
weighted/NLP heuristics at every site that drives session creation — the NLP
analysis (`nlp_processor.analyze_scenario`), the platform detector
(`detect_platform_from_scenario`, whose mobile keywords include the generic
"app"), and session auto-configuration (`configure_from_scenario`) — so a
Linux desktop scenario is never classified as `mobile_testing` when the caller
explicitly requested desktop.

#### Scenario: context=desktop overrides heuristics in the full tool result
- **WHEN** `analyze_scenario(scenario=..., context="desktop")` is called for a
  GNOME Calculator scenario
- **THEN** the tool result's `session_type` is the desktop/PlatynUI type,
  `AppiumLibrary` is not in `required_capabilities`, the imported libraries
  allow `Process`, and the search order leads with `PlatynUI.BareMetal`

#### Scenario: desktop context recommends desktop libraries
- **WHEN** the analysis runs with `context="desktop"`
- **THEN** the suggested libraries lead with `PlatynUI.BareMetal` and do not
  lead with `AppiumLibrary`

#### Scenario: non-desktop context is unchanged
- **WHEN** `analyze_scenario(scenario=..., context="web")` is called
- **THEN** the routing and recommendations are unchanged from current behavior
  (Browser/SeleniumLibrary-led), i.e. no regression from the desktop changes

### Requirement: Recommender prefers PlatynUI for Linux desktop scenarios

The system SHALL rank `PlatynUI.BareMetal` ahead of `AppiumLibrary` in
`recommend_libraries` for Linux desktop / GNOME application scenarios, and the
resulting session search order SHALL place the desktop library first.

#### Scenario: desktop scenario recommendation order
- **WHEN** `recommend_libraries` is asked for a GNOME desktop calculator
  scenario
- **THEN** `PlatynUI.BareMetal` is recommended ahead of `AppiumLibrary`

#### Scenario: search order reflects the preference
- **WHEN** a desktop session is initialized from such a recommendation
- **THEN** the library search order places `PlatynUI.BareMetal` ahead of
  `AppiumLibrary`
