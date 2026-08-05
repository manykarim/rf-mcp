## Why

The desktop automation stack (PlatynUI + AT-SPI + rf-mcp + a WM) has **zero CI coverage**
today. The Phase-2b platynui integration tests were deferred precisely because no headless
environment can run them, and the deterministic desktop smoke that *does* prove the stack
end-to-end (`docker/desktop_smoke_driver.py`) is never exercised in CI — so a regression in
desktop input, AT-SPI resolution, the isolation guard, or the PlatynUI wheel pin would go
unnoticed until someone runs the container by hand.

An experiment settled the feasibility question with evidence: the existing
`docker/Dockerfile.desktop` builds and its smoke passes **inside a plain container, headful,
systemd-free, with no host display** — all six rungs green (AT-SPI2 provider up, gnome-calculator
resolved with the correct container-local PID, `7×6` driven, result display **read back = 42**,
screenshot written). GitHub-hosted `ubuntu-latest` runs Docker natively, so this is runnable in
CI. The original `desktop-docker-agent-harness` proposal already parked "the deterministic smoke
can run in CI (no key)" as an optional hook; this change delivers exactly that hook — the
cheapest, highest-value slice of desktop CI coverage, with no test rewrites and no new packages.

## What Changes

- A **gated CI job** (scheduled weekly + `workflow_dispatch`, not per-push) that builds
  `docker/Dockerfile.desktop` and runs the deterministic desktop smoke (`docker run` →
  `desktop_smoke.sh`), gating on exit 0 (provider up, read-back `42`, screenshot written). It is
  keyless — no model credential — mirroring the always-on/gated split already used for the web and
  agentic tiers.
- A **`.dockerignore`** so the build context is the ~5 MB of files the Dockerfile actually needs,
  not the 2.4 GB repo (dominated by `.venv/`). This fixes local builds and keeps the CI context send fast.
- **Failure-debug artifacts:** on failure the job uploads the smoke's `artifacts/*.png` and
  `smoke_result.json` (the machine-checkable finding record). noVNC/x11vnc already live in the image
  for a human to watch a live desktop; in CI the screenshot + JSON are the practical debug channel.
- **Cadence + fragility notes captured in the design:** weekly-gated (not per-push) because the image
  is ~1.17 GB / a few minutes to build and PlatynUI is pinned to a **dev** wheel (`0.12.0.dev330`)
  that could be yanked — a scheduled job surfaces such a break on a cadence instead of red-ing every PR.

Out of scope (explicitly deferred, sequenced as follow-ons): running the platynui *pytest* tests in
the container (needs a `systemd-run`→direct-launch seam in the tests) and the agenteval desktop
`.robot` ports (build on that seam). This change is only the smoke-in-CI foundation they'd sit on.

## Capabilities

### Modified Capabilities

- `desktop-docker-agent-harness`: add a requirement that the deterministic desktop smoke runs in CI
  as a gated, keyless job, and that the build context is trimmed via a `.dockerignore`.

## Impact

- `.github/workflows/` (new job, likely in the existing `e2e-weekly.yml` scheduled workflow, or a
  dedicated `desktop-smoke.yml`): checkout → docker build → docker run → upload artifacts on failure.
- `.dockerignore` (new, repo root): exclude `.venv/`, `.git/`, `artifacts/`, node_modules, etc.
- No `src/` changes; no changes to the existing web/agentic CI tiers; the `Dockerfile.desktop`,
  `entrypoint.sh`, and smoke driver are consumed as-is.
- Dependency exposure: the job transitively depends on the pinned PlatynUI dev wheels resolving from
  PyPI and on `ubuntu-latest` permitting Xvfb + Docker (both routine on hosted runners).
