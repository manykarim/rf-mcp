## Why

The harness has desktop scenarios (the `platynui_gnome_apps` port, and the future `platynui_*` partial
ports) that drive real GTK apps via PlatynUI/AT-SPI. An evaluation of running them in CI found they
**cannot run on the stock GitHub runner**: the desktop suites hard-require `systemd-run --user` (their
skip-guard checks for `Xvfb`, `gnome-calculator`, `gnome-text-editor`, `systemd-run`), and hosted runners
have no user systemd session — so they would silently skip. Making desktop-in-CI real is a separate,
non-trivial effort with its own risks. This change **formalizes the decision to gate desktop scenarios**
and **records the docker-desktop path** so it isn't re-litigated, rather than shipping an unvalidated,
flaky desktop CI job now.

## What Changes

- **Formalize the gate.** Desktop scenarios stay behind `AGENTEVAL_DESKTOP=1` and are NOT run on the stock
  CI runner (they skip cleanly headless — already the case for the gnome-apps port). This becomes a
  written requirement, not just current behavior.
- **Record the docker-desktop path as future work** (design.md): a dedicated Docker-based job using
  `docker/Dockerfile.desktop` (Xvfb + fluxbox EWMH WM + `at-spi2-core` + `GTK_A11Y=atspi` + gnome apps),
  with the suites' `systemd-run --user` app-launch **rewritten to a direct launch** (the image uses
  supervisord, not systemd) — plus accepting AT-SPI flakiness. Scoped but deferred.
- **No CI job is added now.** A desktop CI job cannot be validated in this environment (no way to build +
  run the desktop/systemd container and confirm PlatynUI green), so shipping one unverified is out of scope.

## Capabilities

### New Capabilities
<!-- none -->

### Modified Capabilities
- `agenteval-test-harness`: add that desktop scenarios are gated and are not run on the stock CI runner;
  desktop coverage in CI requires a dedicated desktop environment.

## Impact

- **Docs**: the harness README records the desktop-gating decision + the docker-desktop future-work outline.
- **No code/CI change** beyond documentation — the `AGENTEVAL_DESKTOP` gate already exists (Phase 2a).
- **Deferred**: an actual docker-desktop CI job (its own change, when a validated desktop container run is
  available).
