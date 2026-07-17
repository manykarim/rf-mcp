# desktop-display-assertion-guidance Specification

## Purpose
TBD - created by archiving change desktop-stepwise-followups. Update Purpose after archive.
## Requirements
### Requirement: Display-state guidance leads with the reliable history-Label path

The system SHALL present the PlatynUI desktop display-state guidance so the
PRIMARY recommended way to assert a calculator's entered values and results is
reading the history/result `Label` nodes (named with the equation and the
result value, e.g. `//control:Label[@Name='56']`). The guidance MUST reflect
what actually works on GNOME Calculator, where the accessibility text content of
the live entry is not reliably exposed.

#### Scenario: history Labels are the primary documented path
- **WHEN** a caller requests PlatynUI guidance for reading desktop display state
- **THEN** the guidance lists reading the history/result `Label` nodes as the
  first/primary assertion method (before any `native:Text.CharacterCount`
  mention)

### Requirement: CharacterCount is demoted and flagged as unreliable

The system SHALL describe `native:Text.CharacterCount` as a SECONDARY length
proxy that MAY report `0` on some GTK builds (as observed on GNOME Calculator),
and MUST warn that it should not be relied on alone for assertions.

#### Scenario: CharacterCount carries an unreliability warning
- **WHEN** the guidance mentions `native:Text.CharacterCount`
- **THEN** it is presented as a secondary proxy with an explicit warning that it
  may be `0` even when the display visibly changed, so callers do not treat a
  `0` as evidence of no input

### Requirement: Screenshot/OCR is named as the last-resort fallback

The system SHALL name a screenshot + OCR readback as the explicit LAST-RESORT
fallback for verifying display state when neither the history Labels nor
`CharacterCount` are usable.

#### Scenario: OCR fallback is documented
- **WHEN** the guidance covers verifying display state
- **THEN** it names taking a screenshot and OCR-ing it as the last resort, after
  history Labels and CharacterCount

