## 1. Formalize + document the gate

- [x] 1.1 Confirm every desktop scenario is behind the `AGENTEVAL_DESKTOP` gate and skips cleanly on a
  headless runner (the gnome-apps port already is; verify no desktop suite is always-on).
- [x] 1.2 Add a "Desktop scenarios (gated)" section to the harness README: why desktop is gated (no
  `systemd --user` on stock runners), that it skips cleanly headless, and the docker-desktop path for a
  future dedicated job (Dockerfile.desktop + the `systemd-run` -> direct-launch rewrite needed).

## 2. Wrap-up

- [x] 2.1 `openspec validate agenteval-desktop-ci-gating --strict` passes.
- [x] 2.2 Note that an actual docker-desktop CI job is deliberately NOT shipped here (cannot be validated in
  this environment) and is a follow-up change.
