## Context

The current F16 stuck-key net (spec `platynui-windows-runtime`, shipped in `fix-platynui-windows-runtime`) lives in `src/robotmcp/plugins/builtin/platynui_plugin.py` and `keyword_executor.py`. It releases a fixed modifier set `_RELEASE_ALL_SEQUENCE = "<LCtrl+RCtrl+LAlt+RAlt+AltGr+LShift+RShift+LWin+RWin>"` from four trigger points (keyword-failure `finally`/except, `on_session_end`, `atexit`, `SIGTERM`) and defensively at `on_session_start`.

Two gaps were confirmed by code + transcript evidence during the Windows 11 evaluation:
1. **Non-modifier gap** — PlatynUI `keyboard_press` (RF `Keyboard Press`) dispatches a `SendInput` key-DOWN with no key-UP (`press_keys` = "press without releasing"). Agents used bare `Keyboard Press A` / `<F12>` / `<Escape>`; the modifiers-only release never lifts those.
2. **Hard-kill gap** — `atexit` and `SIGTERM` handlers do not run on `TerminateProcess`/SIGKILL (`Stop-Process -Force`), so a process killed mid-run leaves keys down with no cleanup path. A stuck key is OS-global input state and survives the process's death.

Constraint: teardown/exit code must never raise, must never start the native runtime just to release, and must not corrupt a deliberate `Keyboard Press` → `Keyboard Release` chord or a steering-confidence downgrade (native keyword actually succeeded).

## Goals / Non-Goals

**Goals:**
- Release the **exact** set of keys rf-mcp still holds at teardown/failure/exit, including non-modifiers, driven by a tracked held-key registry.
- Recover a key left held by a **hard-killed** process on the next desktop session start, via a persisted state file (the only mechanism that survives SIGKILL/`TerminateProcess`).
- Preserve all existing F16 guarantees and the happy-path Press/Release chord and steering-downgrade behaviour.

**Non-Goals:**
- Changing PlatynUI native behaviour or the `keyboard_type` atomicity.
- Preventing an agent from *choosing* `Keyboard Press` (we track + warn, not forbid).
- Cross-machine or cross-user recovery; recovery is same-host, same-runtime.
- Any change to Linux classification/behaviour.

## Decisions

- **Held-key registry as the source of truth (over a blanket key list).**
  Wrap the runtime `keyboard_press`/`keyboard_type`/`keyboard_release`/chord dispatch so each key-DOWN adds a normalized key token to an in-process set and each key-UP removes it. Release-all then dispatches key-UPs for exactly the set contents.
  *Alternative rejected:* expanding `_RELEASE_ALL_SEQUENCE` to "all keys" — infeasible/ugly (huge, layout-dependent) and would blast key-UPs for keys never touched.

- **Registry entries recorded at the wrapper boundary, not inside native code.**
  Keep the native module untouched; instrument in `platynui_plugin.py` where rf-mcp already calls `_RUNTIME.keyboard_press/type/release`. `Keyboard Type` records+clears within the same call (its down/up are paired), so a mid-`type` kill is covered by the persisted file, not left in the in-memory set after return.
  *Alternative rejected:* patching PlatynUI's Rust core — out of scope, higher risk, not our package.

- **Persisted state file for hard-kill recovery, replayed at `on_session_start`.**
  On each key-DOWN record, write the current held set to a durable per-runtime file (atomic write) under the OS temp/state dir; rewrite/delete on release. `on_session_start` already runs `release_all_modifiers()` defensively — extend it to first read any *foreign* state file (written by a now-dead PID), dispatch key-UPs for its records, and delete it.
  *Alternative rejected:* a supervisor/watchdog process holding input — much heavier; a state file is sufficient because key-UP-of-not-held is a safe no-op.

- **Scope the file by process/runtime identity to avoid cross-session interference.**
  Key the file (or a record field) by PID/runtime id; a live process only replays records it did **not** write (i.e. orphaned by a dead PID), so it never lifts a key a concurrent healthy session is deliberately holding.

- **Reuse the existing trigger points and no-op/non-raise discipline.**
  Swap the modifiers-only `keyboard_release(_RELEASE_ALL_SEQUENCE)` for `release_tracked_keys()` at all four sites; keep `_is_steering_downgrade` guard and the "don't resurrect runtime" guard verbatim.

## Risks / Trade-offs

- [Registry drifts from real OS state if a key-UP is dropped] → Release is idempotent and key-UP-of-not-held is a harmless no-op; the persisted file is defensively replayed, so over-releasing is safe while under-releasing is the only real failure and is minimized by recording at the single dispatch boundary.
- [State-file write on every key-DOWN adds I/O to the hot keyboard path] → Writes are tiny (a short token list), atomic, and only on DOWN transitions; acceptable for desktop input cadence. Can be gated to desktop sessions only.
- [Concurrent runtimes racing the file] → Per-PID scoping + "only replay foreign/dead-PID records" avoids a live session's keys being lifted; a stale file from a live PID is ignored until that PID is gone.
- [`signal.signal` only on main thread / F14 offload binds runtime on a worker thread] → Unchanged from F16: handlers are armed in `on_session_start` on the loop thread (already done); the persisted-file recovery is the backstop when signals/atexit are skipped entirely.
- [Non-Windows relevance] → Mechanism is platform-neutral; on Linux the same registry improves the modifiers-only net at no cost, and recovery is a no-op when no file exists.

## Migration Plan

- Additive and internal; no config, API, or dependency changes. Ship in the `platynui-windows-runtime` capability line (next dev bump).
- Rollback: revert to the modifiers-only `_RELEASE_ALL_SEQUENCE` release; the persisted state file is self-clearing and safe to leave behind (ignored if the feature is reverted).
- Validate with unit tests that force each path (bare non-modifier Press → session end; simulated hard kill leaving a state file → next start replay; deliberate chord + steering downgrade → no premature release).

## Open Questions

- State-file location/format: reuse an existing rf-mcp runtime state dir vs. a dedicated `platynui_held_keys.json` under `%LOCALAPPDATA%\Temp`? (Leaning: dedicated file next to the runtime marker.)
- Exact key-token normalization shared between record and release (must round-trip through PlatynUI's sequence grammar, e.g. `<A>`, `<F12>`, `<Escape>`, `<LShift>`).
