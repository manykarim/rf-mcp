## 1. Build-context hygiene

- [ ] 1.1 Add a repo-root `.dockerignore` excluding `.venv/`, `.git/`, `artifacts/`, `node_modules/`,
  `**/__pycache__/`, `*.pyc`, test results, and other large build-irrelevant paths. Confirm
  `docker build -f docker/Dockerfile.desktop .` from the repo root sends a small context (a few MB),
  not the multi-GB working tree.

## 2. Gated CI job

- [ ] 2.1 Add a `desktop-smoke` job (scheduled + `workflow_dispatch`; NOT on push/PR) — prefer the
  existing `.github/workflows/e2e-weekly.yml` for cadence consistency, or a dedicated
  `desktop-smoke.yml`. Steps: checkout → build `docker/Dockerfile.desktop` → `docker run` the default
  smoke with an artifacts volume mounted → gate on exit 0. Keyless (no model credential).
- [ ] 2.2 On failure, upload `artifacts/*.png` and `artifacts/smoke_result.json` (via
  `actions/upload-artifact`, `if: failure()`), so a headless failure is diagnosable.

## 3. Verify on a real runner

- [ ] 3.1 Dispatch the job on an actual GitHub `ubuntu-latest` runner (`workflow_dispatch`) and confirm
  it is green end-to-end: image builds, smoke exits 0 (provider up, read-back `42`, screenshot). Do NOT
  assume the local (snap-docker) experiment transfers — iterate on the real run if the desktop/AT-SPI
  bring-up differs.
- [ ] 3.2 If the run flakes on GUI timing, add a single retry on the `docker run` step (only if
  observed). Record the actual build+run wall-clock so the weekly-vs-tighter-cadence trade-off is
  informed by data.

## 4. Docs + wrap-up

- [ ] 4.1 Note the CI job in `docs/desktop_docker_harness.md` (how it's gated, how to dispatch it, where
  the failure artifacts land) and, if relevant, in the agenteval harness README's desktop section as the
  first shipped desktop CI coverage.
- [ ] 4.2 `openspec validate desktop-smoke-ci --strict` passes.
- [ ] 4.3 Record the outcome (build+run time, green on GHA) and cross-link the deferred follow-ons, now
  scoped by the Tier-B probe: (a) `test_platynui_newcore_e2e` needs only an app-launch fixture to run in
  this image (proven — its full MCP workflow passed inline with one app launched); (b) `test_platynui_focus_e2e`
  needs the `systemd-run`→`Popen` seam (two overlapping windows); (c) the agenteval desktop `.robot` ports
  build on top of those. This smoke job is the foundation all three sit on.
