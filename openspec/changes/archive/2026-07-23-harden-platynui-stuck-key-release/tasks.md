# Tasks: harden-platynui-stuck-key-release

## 1. Held-key registry (in-process)

- [x] 1.1 Add a module-level held-key registry (a normalized-token set + lock) in `src/robotmcp/plugins/builtin/platynui_plugin.py`, alongside the existing `_RUNTIME`/`_RELEASE_ALL_SEQUENCE` state.
- [x] 1.2 Implement a key-token normalizer that round-trips through PlatynUI's sequence grammar (e.g. `<A>`, `<F12>`, `<Escape>`, `<LShift>`, chords expanded to individual keys), shared by record and release.
- [x] 1.3 Wrap runtime `keyboard_press` dispatch so each pressed key (chord expanded) is added to the registry on a successful DOWN.
- [x] 1.4 Wrap runtime `keyboard_release` dispatch so released keys are removed from the registry.
- [x] 1.5 Wrap runtime `keyboard_type` so its transient DOWNs are recorded and cleared within the same call (paired), leaving the in-memory set empty on normal return.
- [x] 1.6 Implement `release_tracked_keys()` — dispatch key-UPs for exactly the registry contents, clear it; best-effort, non-raising, no-op when empty, and never starts the runtime (mirror `release_all_modifiers` guards).

## 2. Persisted state file (hard-kill recovery)

- [x] 2.1 Choose and document the state-file path/format (per-runtime, PID/runtime-id scoped) under the OS temp/state dir; add a small atomic-write helper.
- [x] 2.2 On each key-DOWN record (1.3), write the current held set + owning PID to the state file; rewrite on 1.4/1.5 changes; delete it when the set becomes empty and on clean release (1.6, session-end, atexit).
- [x] 2.3 Implement `recover_orphaned_held_keys()` — read any state file whose owning PID is no longer alive, dispatch key-UPs for its records (safe no-op if not held), then delete it; never touch a file owned by a live PID.

## 3. Wire into F16 trigger points

- [x] 3.1 In `on_session_start` (platynui_plugin.py), call `recover_orphaned_held_keys()` before/with the existing defensive release, so a prior hard-killed run's stuck key is lifted on next desktop start.
- [x] 3.2 Replace the modifiers-only `keyboard_release(_RELEASE_ALL_SEQUENCE)` in `release_all_modifiers()`/`on_session_end` with `release_tracked_keys()` (keep the old modifier blast as a fallback when the registry is unavailable).
- [x] 3.3 In `keyword_executor.py` F16 paths (`except BaseException` and the failure-return path ~lines 1934–1956), swap `_release_desktop_modifiers()` to release tracked keys; preserve the `_is_steering_downgrade` guard and the synchronous-on-cancel behaviour.
- [x] 3.4 Ensure `atexit`/`SIGTERM` handlers (`_register_release_handlers_once`) call the tracked-key release and clear the state file.

## 4. Guidance & guardrails

- [x] 4.1 Emit the "prefer atomic Keyboard Type" warning when a bare `Keyboard Press` leaves a key still held at session-end (extend the existing `platynui_keyboard_release_safety` hint to non-modifier holds).
- [x] 4.2 Confirm a deliberate `Keyboard Press <mod>` → later `Keyboard Release <mod>` chord within one session is NOT prematurely released (registry keeps it until the explicit release).

## 5. Tests

- [x] 5.1 Unit: bare `Keyboard Press A`/`<F12>`/`<Escape>` recorded, then session-end releases exactly those keys and clears the registry (non-modifier gap closed).
- [x] 5.2 Unit: keyword-failure/except path releases tracked keys; steering-confidence downgrade does NOT release.
- [x] 5.3 Unit: simulate hard kill — write a state file under a dead PID, then `on_session_start` replays key-UPs and deletes the file; a live-PID file is left untouched.
- [x] 5.4 Unit: clean shutdown clears the state file (no stale replay); stale replay of not-held keys is a harmless no-op.
- [x] 5.5 Regression: existing F16 modifier scenarios in `platynui-windows-runtime` still pass; release never resurrects the runtime.

## 6. Docs / spec sync

- [x] 6.1 Update `desktop_guidance` / any keyboard cheat-sheet text to state that bare `Keyboard Press` must be paired with `Keyboard Release` (or use `Keyboard Type`).
- [x] 6.2 Version bump note referencing the `platynui-windows-runtime` capability extension.

