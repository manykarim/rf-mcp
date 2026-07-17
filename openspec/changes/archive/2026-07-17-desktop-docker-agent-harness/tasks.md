# Tasks: desktop-docker-agent-harness

## 1. Desktop image (Rung 0)

- [x] 1.1 `docker/Dockerfile.desktop`: base `python:3.12-slim`; apt (`--no-install-recommends`) Xvfb, fluxbox, dbus-x11, at-spi2-core, x11-utils, xdotool, x11vnc, novnc/websockify, supervisor, gnome-calculator, a GTK text editor, fonts; `uv` + project; install pinned prebuilt `platynui-native` + `robotframework-PlatynUI`
- [x] 1.2 Pick the pinned dev release: check PyPI for the newest installable `platynui-native` + `robotframework-PlatynUI` pair (dev330 line); pin BOTH to the same version
- [x] 1.3 **DEVIATION (design D3): used `docker/entrypoint.sh` instead of supervisord** — the AT-SPI stack needs a SHARED `DBUS_SESSION_BUS_ADDRESS` + ordered readiness (Xvfb before at-spi), which a sequential entrypoint expresses cleanly and supervisord does not. Ordered bring-up: dbus → Xvfb `:99` → fluxbox → at-spi-bus-launcher/registryd → x11vnc/noVNC → `exec "$@"`. Env baked into the image: `DISPLAY=:99`, `ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY=:99`, `XDG_SESSION_TYPE=x11`, `GDK_BACKEND=x11`, **`GTK_A11Y=atspi`** (NOT `1` — finding), `GNOME_ACCESSIBILITY=1`, `NO_AT_BRIDGE=0`, `ROBOTMCP_SCREENSHOT_DIR=/artifacts`
- [x] 1.4 Build the image; verify Xvfb + fluxbox + registryd + robotmcp come up (`supervisorctl status`); `platynui-cli list-providers` shows AT-SPI2 active
- [x] 1.5 Fail-loud check: with registryd stopped, the provider probe exits non-zero with an accessibility-bus diagnostic (no 30-s hang)

## 2. Deterministic smoke — steering + read-back (Rung 1, the gate)

- [x] 2.1 `docker/desktop_smoke.sh` (+ a small `.robot` or python driver): launch gnome-calculator on `:99`; `Set Root` to the calc app; assert `supported_patterns()`/`has_pattern` import cleanly on a resolved node
- [x] 2.2 Drive `7 × 6 =` via PlatynUI (button `Pointer Click` or focused `Keyboard Type`); read the result display via `Get Attribute`/`Query`; assert value == `42`
- [x] 2.3 `Take Screenshot` to `/artifacts`; assert a non-trivial PNG exists; write a JSON finding record (provider status, AUT identity, read-back value, screenshot paths)
- [x] 2.4 Run the smoke in the built image end-to-end; capture the artifacts to the host via a mounted volume

## 3. Agent wiring (Rungs 2+, key-gated)

- [x] 3.1 `docker/opencode.minimax.json`: opencode config with a MiniMax provider (OpenAI-compatible or via OpenRouter), `MINIMAX_API_KEY` from env, a tool-calling-capable model; robotmcp over stdio with `DISPLAY=:99` + isolation marker in the agent env
- [x] 3.2 `docker/run_agent.sh`: launch opencode non-interactively with the example prompt ("open the calculator, compute 7×6, confirm the result reads 42 via a screenshot and by reading the display"); skip gracefully (exit 0, note) when no key is set
- [x] 3.3 If a MiniMax key is available in this environment, run rung 2 and record the agent transcript + artifacts; otherwise document the exact invocation and mark it operator-runnable — **RAN LIVE with a real key. MiniMax-M3 COMPLETED the loop autonomously (24 tool calls: 10 Query, 4 Pointer Click, 2 Get Attribute, 1 Take Screenshot): launched gnome-calculator, computed 7×6, read back `42` via app-scoped `/app:*[@Name='gnome-calculator']//control:Label[@Name='42']` Get Attribute, wrote a real 408×616 PNG `calculator_7x6.png`, and self-verified it with `File Should Exist`. Locators were app-scoped throughout (no blind desktop input). MiniMax-M2.7 did NOT complete it (77 calls, persistently passed `null` for required args → pydantic `string_type` errors, then fell back to Bash which opencode's permission gate auto-rejected) — a small-model-competence limit, not a harness gap. Two config fixes were load-bearing for the agent path: `GTK_A11Y=atspi` (not `1`) in the mcp.environment block, and the official opencode installer (npm -g EACCES'd for the non-root user).**

## 4. Docs + observation

- [x] 4.1 `docs/desktop_docker_harness.md`: build, run the smoke, run the agent (with key), observe via noVNC `:6080`, where artifacts land, the AT-SPI bring-up checklist, and the blocker-dissolution table
- [x] 4.2 Document the rung ladder and which rungs are CI-safe (0–1) vs key-gated (2–4)

## 5. Findings + validation

- [x] 5.1 Collect all experiment data: `supervisorctl status`, `platynui-cli list-providers`, the smoke JSON record, screenshots — into the docs/artifacts as the harness findings
- [x] 5.2 Confirm the deterministic smoke is repeatable (two clean runs → same read-back `42`, screenshots present) — **confirmed: override run + baked-in run both PASS, read_back=42, exit 0**
- [~] 5.3 (OPTIONAL, deferred) CI hook: the deterministic smoke as a manual/weekly job (no key); no change to existing web e2e
