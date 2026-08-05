## 1. Build-context hygiene

- [x] 1.1 Added repo-root `.dockerignore` (deny-list: `.venv/`, `.git/`, caches, `tests/`/`docs/`/
  `experiments/`/`openspec/`, and runtime output dirs `.claude-flow/`/`.robotmcp_artifacts/`/`runs/` etc.
  — none COPYed by any Dockerfile). Verified: full-tree build context drops from **2.4 GB → 7.06 MB**
  (measured via `COPY .` throwaway build); the actual `Dockerfile.desktop` build transfers only its
  COPYed subset (~5 MB). Deny-list (not allow-list) so a future Dockerfile needing another path isn't
  silently starved.

## 2. Gated CI job

- [x] 2.1 Added a dedicated gated workflow `.github/workflows/desktop-smoke.yml` (`schedule` Sun 03:00 UTC +
  `workflow_dispatch`; NOT on push/PR). Steps: checkout → `docker build -f docker/Dockerfile.desktop` →
  `docker run` the default smoke with `-v "$PWD/artifacts:/artifacts"` → gates on exit 0. Keyless (no model
  credential). YAML validated. **Deviation from the design's "prefer e2e-weekly.yml":** a dedicated file
  lets the smoke be `workflow_dispatch`-verified in isolation without triggering e2e-weekly's heavier
  model-driven jobs (which spend MiniMax credits) — decisive for task 3.1's ad-hoc verification.
- [x] 2.2 On failure the job uploads `artifacts/*.png` + `artifacts/smoke_result.json`
  (`actions/upload-artifact@v4`, `if: failure()`, `if-no-files-found: ignore`).

## 3. Verify on a real runner

- [ ] 3.1 Dispatch the job on an actual GitHub `ubuntu-latest` runner (`workflow_dispatch`) and confirm
  it is green end-to-end: image builds, smoke exits 0 (provider up, read-back `42`, screenshot). Do NOT
  assume the local (snap-docker) experiment transfers — iterate on the real run if the desktop/AT-SPI
  bring-up differs.
- [ ] 3.2 If the run flakes on GUI timing, add a single retry on the `docker run` step (only if
  observed). Record the actual build+run wall-clock so the weekly-vs-tighter-cadence trade-off is
  informed by data.

## 4. Docs + wrap-up

- [x] 4.1 Documented the CI job. The tracked home is the agenteval README desktop section (corrected: the
  smoke job is the first shipped desktop CI coverage; agenteval ports remain a follow-on). Also added a
  "Continuous integration (gated)" section to `docs/desktop_docker_harness.md` — but that runbook is
  **locally excluded** (`.git/info/exclude`), so its edit is on-disk only, not in the commit; force-tracking
  a deliberate local exclude was declined. If the runbook should be tracked, `git add -f` it separately.
- [x] 4.2 `openspec validate desktop-smoke-ci --strict` passes.
- [ ] 4.3 Record the outcome (build+run time, green on GHA) and cross-link the deferred follow-ons, now
  scoped by the Tier-B probe: (a) `test_platynui_newcore_e2e` needs only an app-launch fixture to run in
  this image (proven — its full MCP workflow passed inline with one app launched); (b) `test_platynui_focus_e2e`
  needs the `systemd-run`→`Popen` seam (two overlapping windows); (c) the agenteval desktop `.robot` ports
  build on top of those. This smoke job is the foundation all three sit on.
