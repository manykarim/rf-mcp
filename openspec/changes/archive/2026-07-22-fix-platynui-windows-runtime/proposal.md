## Why

A 54-scenario autonomous evaluation on a real Windows 11 box (evidence:
`experiments/platynui_windows_evaluation_results.md`) surfaced four runtime defects that
block or degrade PlatynUI desktop automation on Windows — hitting exactly the two eval
goals (agent autonomy, low latency), plus one that damaged the operator's machine:

- **F16 (critical, user-impacting).** After runs the operator's **keyboard was wedged** (a
  modifier stuck down; couldn't press Enter/Win) — required a reboot. `Keyboard Press` sends
  key-DOWN only, runs timed out/were killed mid-chord, and **nothing releases held keys** on
  teardown/failure/exit. An autonomous agent must never be able to leave the user's keyboard
  held down.
- **F3/F9 (dominant latency defect).** Desktop keyword queries ignore `timeout_ms` and use
  PlatynUI's own ~30s (60s for `Query`/`get_session_state`) default; a wrong/honest-miss
  locator waits it out and **repeats stack** (up to 180s). No prompt fixes a runtime timeout.
- **F14 (critical).** A `Focus`/`Query` on a large tree runs **synchronously on the server
  event loop**, so even pure-metadata calls (`get_keyword_info`) hang 60s — the server wedges.
- **F4 (autonomy blocker).** The desktop safety guard classifies off `DISPLAY`/`/proc`, so on
  **Windows it always lands in refuse** — every interaction keyword is denied unless
  `ROBOTMCP_PLATYNUI_ALLOW_ACTIVE_DESKTOP=1`, and the remediation it prints is a Linux
  Xephyr/Xvfb recipe that cannot work on Windows.

The path-corruption and dry-run fixes shipped earlier were validated on the same box; these
are the next blockers to make Windows PlatynUI usable by autonomous agents.

## What Changes

- **F16 — stuck-key safety net.** Add a `release_all_modifiers()` that releases
  Ctrl/Shift/Alt/Meta (L+R) via PlatynUI, and invoke it (a) in `platynui_plugin.on_session_end`,
  (b) in a `finally` around desktop keyboard-keyword execution (release on failure/timeout),
  (c) via an `atexit` + SIGTERM handler registered on first desktop use, and (d) defensively at
  session init (clear a prior crashed run's held keys). Steer agents away from bare
  `Keyboard Press` toward the atomic `Keyboard Type <chord>`.
- **F3 — bounded, honored desktop query timeout.** Inject a short default query timeout for
  desktop keywords (via PlatynUI `query_overrides` / `Set Query Settings` at session init) and
  map a per-call `timeout_ms` onto it, so an honest-miss locator fails in ~1–2s, not 30–60s,
  and cannot stack.
- **F14 — never block the event loop.** Run desktop/native keyword execution off the loop
  (`asyncio.to_thread`) with a bounded `wait_for`, so a slow query cannot wedge the server or
  starve metadata calls.
- **F4 — platform-aware safety guard.** On Windows the guard classifies as a distinct
  `windows` state (there is no X11 isolated-display model), allowed by default with a clear
  one-time warning, keeping the strict opt-out; the remediation text becomes Windows-accurate.

## Capabilities

### New Capabilities
- `platynui-windows-runtime`: the Windows runtime contract for PlatynUI desktop automation —
  never leave keyboard modifiers held; desktop queries fail fast and honor `timeout_ms`;
  desktop execution never blocks the server event loop; and the desktop safety guard is
  platform-aware (Windows is not refused-by-default with a Linux remediation).

## Impact

- **Code:** `plugins/builtin/platynui_plugin.py` (release_all_modifiers + on_session_end +
  atexit/signal), `components/execution/keyword_executor.py` (to_thread + bounded timeout +
  finally-release for desktop keyword paths), `components/execution/desktop_display_safety.py`
  (Windows branch + remediation), and the query-timeout injection point.
- **Behaviour:** Windows desktop automation stops wedging the keyboard, fails fast on bad
  locators, no longer wedges the server, and no longer refuses every keyword. Linux desktop
  behaviour and non-desktop sessions are unchanged.
- **Non-goals:** the guidance/suite-boilerplate fixes (F1/F2/F5) and `visual_check` (F8) —
  separate follow-up changes; the Word browser-launch / not-on-PATH issues (F12/F15) are
  environment, not rf-mcp.
