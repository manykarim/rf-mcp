## ADDED Requirements

### Requirement: Desktop scenarios are gated and not run on the stock CI runner

Harness scenarios that drive a real desktop (GTK apps via PlatynUI/AT-SPI) SHALL be gated behind an
explicit opt-in (`AGENTEVAL_DESKTOP`) and SHALL NOT run on the standard headless CI runner — they SHALL
skip cleanly there. Desktop coverage in CI SHALL require a dedicated desktop environment (a Docker image
providing Xvfb + a WM + AT-SPI + the GTK apps), which is out of scope for the standard runner because
hosted runners provide no `systemd --user` session the current suites depend on.

#### Scenario: desktop scenario skips on the stock runner
- **WHEN** the harness runs on a standard headless CI runner without the desktop opt-in
- **THEN** each desktop scenario skips cleanly (no failure), because no display/desktop environment is present

#### Scenario: desktop coverage requires a dedicated environment
- **WHEN** desktop scenarios are to actually run
- **THEN** they run only in a dedicated desktop environment (display + WM + AT-SPI + the apps) with `AGENTEVAL_DESKTOP` set, not on the stock runner
