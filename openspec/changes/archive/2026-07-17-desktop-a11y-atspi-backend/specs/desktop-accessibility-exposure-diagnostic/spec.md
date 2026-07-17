# Spec: desktop-accessibility-exposure-diagnostic

## MODIFIED Requirements

### Requirement: The diagnostic reports providers and actionable remediation

The system SHALL include in the `accessibility_not_exposed` diagnostic the
active PlatynUI providers (from the native `providers()` API), a statement that
the window is present but exposes no accessibility tree (a GTK/AT-SPI bridge or
environment issue, not a locator problem), and concrete remediation (ensure the
accessibility bridge / AT-SPI bus is enabled; allow the app a moment to
register). The bridge remediation SHALL recommend the AT-SPI backend by its
**name** — `GTK_A11Y=atspi` — and SHALL NOT recommend `GTK_A11Y=1`, because
modern GTK rejects the value `1` (`Unrecognized accessibility backend`) and the
application then exposes no AT-SPI tree; any reference to `GTK_A11Y=1` SHALL
frame it as the rejected anti-pattern. PlatynUI guidance SHALL reference the
diagnostic so an agent recognizes the condition instead of dropping to
coordinate clicks/OCR.

#### Scenario: diagnostic names providers + remediation
- **WHEN** the `accessibility_not_exposed` diagnostic is produced
- **THEN** it lists the active providers and concrete remediation steps and
  frames the issue as accessibility/environment, not a locator problem

#### Scenario: remediation recommends the correct AT-SPI backend value
- **WHEN** the `accessibility_not_exposed` remediation is produced
- **THEN** it recommends launching GTK apps with `GTK_A11Y=atspi` and does not
  recommend `GTK_A11Y=1` (only citing `1` as the rejected value that yields an
  empty tree)

#### Scenario: guidance references the diagnostic
- **WHEN** a caller reads the PlatynUI guidance
- **THEN** it explains an empty tree after launch may mean the app exposes no
  AT-SPI tree, points to `accessibility_not_exposed`, names `GTK_A11Y=atspi` as
  the bridge fix, and says to remediate the bridge rather than fall back to
  coordinates/OCR
