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

- [x] 3.1 Verified GREEN on a real GitHub `ubuntu-latest` runner (dispatched run 31128840398): image
  builds, desktop stack comes up (dbus + Xvfb + fluxbox + AT-SPI2 provider active + gnome-calculator),
  smoke exits 0 with `provider_ok=true`, `read_back="42"`, screenshot written. It took THREE real-runner
  iterations — the "don't assume local transfers" guardrail earned its keep: (1) build failed because the
  desktop harness (`docker/Dockerfile.desktop` + scripts) was git-excluded as experiment-only and never
  committed → committed it (4b20d4b); (2) build ✓ but the smoke crashed on `/artifacts` perms (container
  `appuser` uid 1000 ≠ the GHA runner's dir owner) → `chmod 777` the mount + hardened `finish()` (def5579);
  (3) fully green.
- [x] 3.2 No GUI-timing flake observed — no retry added. Wall-clock: **~1m51s** total (build + smoke),
  22:12:54→22:14:45Z. Comfortably cheap; the weekly cadence has ample headroom.

## 4. Docs + wrap-up

- [x] 4.1 Documented the CI job. The tracked home is the agenteval README desktop section (corrected: the
  smoke job is the first shipped desktop CI coverage; agenteval ports remain a follow-on). Also added a
  "Continuous integration (gated)" section to `docs/desktop_docker_harness.md` — but that runbook is
  **locally excluded** (`.git/info/exclude`), so its edit is on-disk only, not in the commit; force-tracking
  a deliberate local exclude was declined. If the runbook should be tracked, `git add -f` it separately.
- [x] 4.2 `openspec validate desktop-smoke-ci --strict` passes.
- [x] 4.3 Outcome recorded. **Green on GHA `ubuntu-latest`, ~1m51s** (run 31128840398), read-back 42.
  A material scope addition surfaced during verification: the change assumed `docker/Dockerfile.desktop`
  was in the repo, but the whole desktop harness was git-excluded (`.git/info/exclude`, "experiment harness
  — NOT part of delivered rf-mcp"). Per an explicit decision, the six harness files + runbook are now
  committed (4b20d4b); the broader `docker/lab`/`experiments/` scaffolding stays excluded. Deferred
  follow-ons, scoped by the Tier-B probe: (a) `test_platynui_newcore_e2e` needs only an app-launch fixture
  to run in this image (proven — its MCP workflow passed inline with one app launched); (b)
  `test_platynui_focus_e2e` needs the `systemd-run`→`Popen` seam (two overlapping windows); (c) the
  agenteval desktop `.robot` ports build on those. This smoke job is the foundation all three sit on.
