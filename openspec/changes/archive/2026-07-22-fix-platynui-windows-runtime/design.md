# Design — fix-platynui-windows-runtime

Four independent runtime defects, one shared theme: on Windows the PlatynUI desktop path can
leave the **user's machine** or the **server** in a bad state, and it is slow to fail. Each fix
is small, local, and code-grounded. Ordered by severity.

## Evidence anchors (Windows 11, de-DE, rf-mcp 0.34.0.dev3, PlatynUI 0.13.0.dev2, Py 3.13.14)

- `experiments/platynui_windows_evaluation_results.md` — F1–F15 + per-scenario iter2 table.
- `runs/*.robot` — generated suites showing `Keyboard Press <Ctrl>` / bare chords (F16 source).
- `runs/*.jsonl` — transcripts with the 30–180s query stalls (F3/F9) and metadata hangs (F14).

---

## F16 — stuck-key release safety net (critical, user-machine-impacting)

### Problem
`Keyboard Press` sends key-**down** only (`runtime.keyboard_press(text)`); its release is a
separate `Keyboard Release`. Windows runs used bare `Keyboard Press <Ctrl>` and chorded
`Keyboard Type <Ctrl+...>`, and runs were **timed out / killed mid-chord** (F3/F9/F14). Nothing
in rf-mcp releases held keys on session end, keyword failure, or process exit — so a modifier
stays **physically down at the OS level**. The operator's keyboard was wedged (couldn't press
Enter/Win) and needed a reboot. `session_manager.on_session_end` (line 191) already invokes
`plugin.on_session_end(session)`, but `platynui_plugin` implements only `on_session_start`
(line 542) — there is no teardown hook.

### Mechanism (proven)
`pn.Runtime()` exposes `keyboard_release(sequence)` **directly on the runtime** — no element
descriptor, no tree query, no RF context. Verified: `runtime.keyboard_release('<Ctrl+Alt+Shift+Meta>')`
is a **safe no-op when nothing is held** and returns instantly. Recognized modifier names include
`LCTRL/RCTRL/LALT/RALT/ALTGR/LSHIFT/RSHIFT/LWIN/RWIN/META/SUPER`. This makes a release-all call
safe to fire from any teardown path, even `atexit`.

### Fix — `plugins/builtin/platynui_plugin.py`
1. Add a module-level `release_all_modifiers()`:
   ```python
   _RELEASE_ALL_SEQUENCE = "<LCtrl+RCtrl+LAlt+RAlt+AltGr+LShift+RShift+LWin+RWin>"
   def release_all_modifiers() -> bool:
       """Best-effort release of every modifier key. No-op on already-up keys.
       Never raises. Returns True if a release was dispatched."""
       if _RUNTIME is None or _RUNTIME_STATE != "open":
           return False                # never spin up the runtime just to release
       try:
           _RUNTIME.keyboard_release(_RELEASE_ALL_SEQUENCE)
           return True
       except Exception as exc:
           logger.debug("release_all_modifiers failed: %s", exc)
           return False
   ```
   Guard on an **already-open** runtime only — teardown must never *start* the native broker.
2. Add `PlatynUIPlugin.on_session_end(session)` → `release_all_modifiers()` (fires on every
   session close; `session_manager` already calls it).
3. Register `atexit` + `SIGTERM` handlers **once, on first desktop use** (in `get_runtime()`'s
   successful-bind branch, gated by a module flag) that call `release_all_modifiers()`. This is
   the safety net for `-p` process kills mid-run — the exact F16 trigger.
4. Defensive release at `on_session_start` (line 542) too — clears a *prior* crashed run's held
   keys before a fresh session starts.

### Fix — `components/execution/keyword_executor.py` (release-on-failure)
Wrap the desktop keyboard-keyword dispatch in `try/finally`: on **any** exception or timeout from
a `Keyboard Press`/`Keyboard Type`/`Keyboard Release` (the F3/F14 kill window), call
`release_all_modifiers()` in the `finally`. This closes the "killed mid-chord" leak at the source,
not just at session end.

### Steering (guidance, not a hard block)
Desktop keyword guidance should prefer the **atomic** `Keyboard Type <Ctrl+A>` (self-contained
press→release) over bare `Keyboard Press`, and note that `Keyboard Press` **must** be paired with
`Keyboard Release`. Wording only — no behavioural gate (keyboard-down is legitimate for
press-and-hold). Full guidance-topic wiring is the separate `fix-platynui-windows-guidance` change;
here we add the one-line safety note where desktop signals already surface.

### Why not just release at session end?
Session end alone misses the two real F16 triggers: (a) a killed `-p` process never runs
`on_session_end`; (b) a keyword that hangs then times out leaves the modifier down for the rest of
the session. Hence the three-layer net: `finally` (per-keyword) · `on_session_end` (per-session) ·
`atexit`/SIGTERM (per-process).

---

## F3 — bounded, honored desktop query timeout (dominant latency defect)

### Problem
Desktop keywords skip rf-mcp's timeout injection (the `_is_desktop` gates), so a query falls back
to **PlatynUI's own default** (~30s; ~60s for `Query`/`get_session_state`). A wrong or honest-miss
locator waits the full default, and the eval shows these **stack** to ~180s across retries. The
eval's low-latency goal ("wrong keyword/locator must fail fast") is missed entirely — and no prompt
can fix a runtime-level timeout.

### Mechanism (verified)
`PlatynUI.BareMetal` carries a `QuerySettings` dataclass (`timeout: float = 30.0`, `retry_interval`,
…) stored in the `PLATYNUI_QUERY_SETTINGS` RF variable. It governs the **retrying** query/wait
keywords (`Query`, `Wait Until …`) and is settable three ways: `query_settings=` at library import,
`query_overrides=` per keyword call, or the `Set Query Settings` keyword. rf-mcp never sets it, so
the 30s (and 60s for the broadest queries) default governs on a miss.

Note the scope limit found during apply: this bounds only the **library-level retrying keywords**.
The raw `runtime.evaluate()` used by the focus manager's `ensure_focused` has **no** timeout knob —
that path is addressed by F14's thread offload + query scoping, not by this setting.

### Fix
- On desktop-session init, set a **short default** `QuerySettings.timeout` (target ~1500 ms) so an
  honest-miss `Query`/wait fails in ~1–2s instead of ~30s. Set it via `Set Query Settings` executed
  once on the first desktop keyword (uses the existing RF context path; no import-arg plumbing).
- **Honor `timeout_ms`**: when a caller passes `timeout_ms`, inject `query_overrides={'timeout':
  timeout_ms/1000}` for PlatynUI query/wait keywords via the existing `_inject_timeout_into_arguments`
  seam (keyword_executor:3591) — the desktop analogue of the browser/selenium pre-validation timeout.
- Keep the default overridable per session (a deliberate long wait sets a larger `timeout_ms`).

---

## F14 — desktop execution never blocks the event loop (critical)

### Problem (root cause corrected during apply)
The native keyword execution itself is **already** offloaded — `_execute_keyword_with_context`
runs `rf_native_context.execute_keyword_with_context` via `asyncio.to_thread` (keyword_executor:3679).
The real on-loop blocker is the desktop **pre-dispatch** phase, which runs **synchronously on the
loop thread** inside `_execute_keyword_serialized`:
- `_platynui_focus_before_act` (line 2032) → focus manager `ensure_focused`, which drives raw
  `runtime.evaluate()` to find/activate/highlight the AUT window — a broad or busy-tree query here
  has no timeout knob and can take tens of seconds.
- `_desktop_text_count_before` (line 2055) → a native `CharacterCount` query on the target node.

Both hold the loop thread while they run, so an in-flight desktop query makes even pure-metadata
calls (`get_keyword_info`) hang behind them.

### Fix
Offload the synchronous desktop pre-dispatch native queries to a worker thread:
```python
_focus_outcome = await asyncio.to_thread(
    self._platynui_focus_before_act, session, keyword, arguments)   # propagates FocusError
...
_input_effect_before = await asyncio.to_thread(
    self._desktop_text_count_before, keyword, arguments)
```
- `to_thread` frees the event loop while the native query runs → metadata/other-session calls stay
  responsive. The global `_execution_lock` (keyword_executor:342) already serializes desktop
  dispatches, so no native concurrency is introduced.
- `FocusError` raised inside `_platynui_focus_before_act` still propagates out of `to_thread` and is
  caught by the existing `except` at line 2058 — behaviour preserved.
- Scope strictly to the desktop branch (guarded by `_is_desktop`); the line-3679 native execute is
  already offloaded — do not double-wrap it.
- F3 complements this for the retrying `Query`/wait keywords; `ensure_focused`'s raw `evaluate` is
  bounded by query *scoping* (existing app-scoped hints), not by a timeout.

---

## F4 — platform-aware safety guard (autonomy blocker)

### Problem
`evaluate_safety(session)` → `classify_bound_display(env)` classifies purely off X11
`DISPLAY`/isolation-marker/`/proc`/EWMH. On Windows none of that exists → `UNKNOWN` → **refuse**
(`allowed=False`) for every interaction keyword, and the `reason`/`isolation_recipe` it returns is
a **Linux Xephyr/Xvfb recipe** that cannot run on Windows. So on Windows the agent is blocked on
every keyword unless `ROBOTMCP_PLATYNUI_ALLOW_ACTIVE_DESKTOP=1` — and the remediation it prints is
impossible to follow. Directly defeats the autonomy goal.

### Design decision
On Windows there is exactly **one** interactive desktop (the console/RDP session) and **no
headless-isolation model** (no Xvfb equivalent). Refuse-by-default is therefore both wrong (nothing
to fall back to) and hostile to the autonomy goal. Windows automation *is* active-desktop
automation by nature. So:
- Add classification `WINDOWS = "windows"`.
- `classify_bound_display_detailed` gets an **early `if os.name == "nt"` branch** → returns
  `{isolation: WINDOWS, isolation_source: "windows_console"}` before any X11/`/proc` probe.
- `evaluate_safety` gets a Windows branch: **allowed by default** (`enforcing=True, allowed=True,
  bypassed=False`) with a **one-time WARNING** that it will drive the active desktop, and a
  **Windows-accurate note** (run in a dedicated/RDP session for isolation) — **no Xephyr recipe**.
- Keep a strict opt-**in** for users who want isolation enforced on Windows
  (e.g. `warn_mode`/a require-isolated env) rather than the current allow-only-with-env-opt-out,
  which is backwards for Windows.

### Non-goal
This does not weaken Linux behaviour: the isolated/active/unknown logic and the refuse-by-default
active-desktop guard on Linux are untouched. Only the `os.name == "nt"` path changes.

---

## Test strategy

- **F16:** unit test that `release_all_modifiers()` calls `runtime.keyboard_release` with a
  modifier sequence and is a no-op when the runtime is not open (mock the runtime); test that
  `PlatynUIPlugin.on_session_end` triggers it; test the `finally`-release fires on a raised
  keyboard keyword. (atexit registration: assert the handler is registered once.)
- **F3:** unit test that a desktop session injects a short query timeout and that `timeout_ms`
  maps onto the PlatynUI query override.
- **F14:** unit test that the desktop native dispatch is awaited via `to_thread` + `wait_for`
  (patch `asyncio.to_thread`, assert it is used for the desktop path and a slow call raises
  `TimeoutError` to the caller without blocking).
- **F4:** unit test `classify_bound_display_detailed({}, os.name=="nt")` → `windows`, and
  `evaluate_safety` on Windows → `allowed=True, bypassed=False` with no Xephyr recipe; Linux
  classification unchanged. Simulate Windows by patching `os.name`.
- Full `uv run pytest tests/unit` green; `openspec validate fix-platynui-windows-runtime --strict`.
