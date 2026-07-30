## Context

The desktop suites (`platynui_gnome_apps`, and future `platynui_*` ports) launch GTK apps via
`systemd-run --user` on an Xvfb `:99` display and read them back over AT-SPI (`GTK_A11Y=atspi`). Their
skip-guard requires `Xvfb`, `gnome-calculator`, `gnome-text-editor`, and `systemd-run`. GitHub-hosted
ubuntu runners have **no user systemd session**, so `systemd-run --user` fails and the suites skip.

## Goals / Non-Goals

**Goals:** make the gating an explicit requirement; record what a real desktop CI job would take.
**Non-Goals:** shipping a desktop CI job now (cannot be validated in this environment).

## Decisions

**D1 — Gate now, document the path.** Desktop scenarios stay `AGENTEVAL_DESKTOP`-gated and skip on the
stock runner. This is the safe, correct default; a real desktop CI job is a separate, validated effort.

**D2 — The docker-desktop path (future work).** The repo already has `docker/Dockerfile.desktop`: an
in-container X11 desktop (Xvfb + fluxbox EWMH WM + `at-spi2-core` + `GTK_A11Y=atspi` + gnome-calculator +
gnome-text-editor + dbus-x11). A future desktop CI job would build/run that image and run the desktop
suites inside it. **Blocker to resolve first:** that image uses **supervisord, not systemd**, so the
suites' `systemd-run --user` app-launch must be rewritten to a direct launch under the container's
display/WM (or the image must run systemd). Until that is done and validated by a real container run, the
job is not shippable.

## Risks / Trade-offs

- **[AT-SPI/desktop flakiness]** → desktop UI tests are timing- and coordinate-sensitive; a desktop CI job
  needs retries/tolerances before it can gate anything.
- **[Container maintenance]** → a desktop image is a heavier CI dependency than a keyless subprocess.
- **[systemd-run rewrite risk]** → changing the app-launch mechanism must not alter what the desktop tests
  observe; it needs its own validation on a real desktop.

## Migration Plan

This change is documentation + the gating requirement only. The docker-desktop job is a follow-up change,
opened when a validated desktop container run (with the app-launch rewrite) is available. Rollback: n/a
(no runtime change).

## Open Questions

- Rewrite `systemd-run --user` to a plain `subprocess.Popen` under fluxbox, or add systemd to the image?
- Is desktop-in-CI worth the flakiness/maintenance, or is local/Docker-harness verification sufficient?
