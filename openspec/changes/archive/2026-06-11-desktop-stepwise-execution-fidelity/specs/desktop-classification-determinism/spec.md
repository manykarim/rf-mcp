## ADDED Requirements

### Requirement: Explicit desktop context classifies deterministically

The system SHALL classify a scenario passed with `context="desktop"` as a
desktop session deterministically — independent of prompt phrasing or word
order — so the same scenario class does not flip between desktop and mobile
across attempts. The determinism guarantee SHALL be documented.

#### Scenario: phrasing variants all classify desktop
- **WHEN** several phrasings of the same GNOME desktop scenario are analyzed
  with `context="desktop"`
- **THEN** every one classifies as `desktop_testing` (no mobile/Appium routing)

#### Scenario: determinism is documented
- **WHEN** a caller reads the `analyze_scenario` documentation/guidance
- **THEN** it states that an explicit `context="desktop"` deterministically
  forces a desktop session regardless of scenario wording
