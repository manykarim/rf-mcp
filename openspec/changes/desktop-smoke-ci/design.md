## Context

`docker/Dockerfile.desktop` (change `desktop-docker-agent-harness`) already builds an isolated,
systemd-free X11 desktop (Xvfb `:99` + fluxbox + AT-SPI + x11vnc/noVNC) with PlatynUI + rf-mcp + a
GTK AUT, and its `entrypoint.sh` brings the desktop up sequentially, then `desktop_smoke_driver.py`
launches gnome-calculator with a plain `subprocess.Popen`, drives it, and reads the result back.
An experiment confirmed the whole path works in a container with **no host display and no systemd**:
build succeeded (~1.17 GB image), and the smoke passed all rungs — `provider_ok`, `read_back=42`,
`aut_pid == aut_atspi_pid` (container-local bus reports the correct PID), screenshot 33 KB. The one
missing piece is a CI job. Nothing in `.github/workflows/` references the image today.

The deferred Phase-2b platynui *pytest* tests gate on `systemd-run`; the smoke does not, because it
launches directly. So the smoke is the reachable coverage now; the pytest tests need a launch seam
(a follow-on), and the agenteval desktop ports sit on top of that.

**Tier-B feasibility probe (evidence, 2026-07-30).** An inline replica of `test_platynui_newcore_e2e`'s
MCP workflow was run inside the built image (fastmcp `Client` → `manage_session` PlatynUI.BareMetal →
`Get Pointer Position` → `Query /app:*` → `ui_tree` → `build_test_suite` → `get_locator_guidance`),
with one gnome-calculator launched first. **Every step passed** (ui_tree `application_count=1`,
`get_locator_guidance` returned both `performance_rules` and `element_not_found_suggestions`). This
sharpens the follow-on scope: the `newcore` test is *nearly* container-ready — its only gap is the
implicit "a desktop app is already present" assumption (no gnome-shell in the container), which an
app-launch fixture (`Popen`, like the smoke) closes; it does **not** need the `systemd-run` seam. The
`systemd-run`→`Popen` seam is specifically the `focus` test's need (it launches two overlapping
windows). And `test_locator_guidance_dispatch` is not desktop-coupled at all — an earlier "different
shape headless" read was an args artifact (missing `error_message`), not a real blocker.

## Goals / Non-Goals

**Goals:**
- Give the desktop stack continuous, deterministic, keyless CI coverage via the existing smoke.
- Keep it cheap and non-fragile: gated cadence, trimmed build context, artifacts-on-failure.
- Leave a clean foundation the pytest-test and agenteval-port follow-ons can build on.

**Non-Goals:**
- Running the platynui pytest tests in CI (needs the `systemd-run`→`Popen` seam — follow-on).
- The agenteval desktop `.robot` ports (build on that seam — later follow-on).
- The LLM-driven agent rungs (`run_agent.sh`) — key-gated, manual; not part of this gate.
- Pushing the image to a registry / GHCR — not needed for a gated build-and-run job.

## Decisions

**D1 — Build-and-run in the job, no prebuilt registry image.** The job does `docker build` then
`docker run` on the runner. *Why:* a gated (weekly/dispatch) cadence tolerates the few-minute build,
and building from source each run keeps the job self-contained and free of registry/auth/GHCR
plumbing. Optional GHA layer caching can be added later if the cadence tightens.

**D2 — Gated (schedule + `workflow_dispatch`), not per-push, in its OWN workflow.** *Why gated:*
~1.17 GB image + a few minutes to build is too heavy for the per-push gate, and PlatynUI is pinned to a
**dev** wheel (`0.12.0.dev330`) — a yank would red the job, acceptable on a schedule but not on every
PR. *Why a dedicated `.github/workflows/desktop-smoke.yml` rather than a job in `e2e-weekly.yml`
(resolved during apply):* a dedicated workflow can be `workflow_dispatch`-triggered in isolation to
verify the desktop job alone, whereas dispatching `e2e-weekly.yml` runs its heavier model-driven jobs
(model-comparison, opencode-e2e, web-headless) that spend MiniMax credits. Since task 3.1 (verify on a
real runner) is an ad-hoc dispatch, isolation wins. Same cadence (weekly, offset to Sun 03:00 UTC).

**D3 — Keyless, exit-status gate.** The job asserts `docker run … robotmcp-desktop` exits 0 — the
smoke's own gate (provider up → resolve → drive → read-back `42` → screenshot). No model credential,
so it is a real deterministic gate, not a best-effort one.

**D4 — Artifacts on failure.** Mount an artifacts dir into the container and, on failure, upload
`calc.png` + `smoke_result.json`. *Why:* a red desktop has no live display in CI; the screenshot and
the machine-checkable JSON (provider status, AUT identity, read-back, screenshot path) are the
diagnosis. noVNC/x11vnc remain in the image for a human attaching to a live/`sleep infinity` run
locally — not part of the CI gate.

**D5 — Add a `.dockerignore`.** Exclude `.venv/`, `.git/`, `artifacts/`, `node_modules/`, caches.
*Why:* with no `.dockerignore` the context is 2.4 GB (mostly `.venv/`); the Dockerfile only needs
~5 MB (`pyproject.toml`, `uv.lock`, `README.md`, `src/`, five `docker/` files). This fixes local
builds and trims the CI context send. Keep it allowlist-minded but simple (deny-list the big dirs).

## Risks / Trade-offs

- **[Runner-specific behavior]** The feasibility run was on local (snap) Docker, not GHA. GHA
  `ubuntu-latest` is native Docker (no snap `/tmp` confinement) and routinely runs Xvfb, so it is
  *more* permissive — but the first task MUST be "confirm the job is green on a real runner," not
  assume it. Mitigation: land the workflow, dispatch it once, iterate on the actual run.
- **[Dev-wheel fragility]** `platynui-native==0.12.0.dev330` is a pre-release; a yank breaks the
  build. Accepted for a gated job; a future move to a stable pin (when available) de-risks it. Noted,
  not solved here.
- **[Build time each run]** A few minutes per run with no caching. Accepted at weekly cadence; GHA
  layer caching is the escape hatch if the cadence tightens.
- **[Flakiness of a live desktop]** GUI timing can flake. The smoke already waits on provider
  readiness and fails fast rather than hanging; if flakiness appears, a single retry on the
  `docker run` step is the contained fix (added only if observed, not pre-emptively).
