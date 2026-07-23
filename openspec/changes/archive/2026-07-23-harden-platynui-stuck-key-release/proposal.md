## Why

The F16 stuck-key safety net only releases keyboard **modifiers** (Ctrl/Alt/AltGr/Shift/Win) and only via handlers (`finally`, session-end, `atexit`, `SIGTERM`) that a **force-kill bypasses**. During the Windows 11 evaluation this left a key physically wedged on the operator's live session: agents issued bare `Keyboard Press` of **non-modifier** keys (e.g. `A`, `F12`, `Escape`) that were never released, and a `Stop-Process -Force`/`TerminateProcess` of the rf-mcp process mid-run ran none of the cleanup handlers. A stuck key is injected at the OS input level (Windows `SendInput` key-down with no key-up), so it persists **system-wide** across the process's death — the worst possible failure for a tool that drives the user's real desktop.

## What Changes

- Replace the modifiers-only blanket release with a **held-key registry**: rf-mcp records every synthetic key-DOWN it dispatches (via `Keyboard Press`, and the transient down within `Keyboard Type`/chords) and removes it on the matching key-UP, so teardown can release the **exact** set still held — including non-modifier keys the current net ignores.
- **Persist the held-key registry to a state file** so a hard process termination (SIGKILL/`TerminateProcess`, which skip `atexit` and `SIGTERM`) is recovered on the **next** `on_session_start`: the successor process reads the file and dispatches key-UPs for anything the dead process left down, then clears it.
- Release the tracked keys (not just modifiers) on the existing trigger points — keyword-failure `finally`/except, `on_session_end`, `atexit`, `SIGTERM` — reusing F16's non-raising, no-op-when-empty, never-resurrect-the-runtime discipline.
- Strengthen agent guidance: a bare `Keyboard Press` of a key that is still held at session-end SHALL emit a warning steering toward the atomic `Keyboard Type`, and the registry-backed release SHALL cover it regardless.
- No behavioural change on the happy path: a deliberate `Keyboard Press` → later `Keyboard Release` chord, and steering-confidence downgrades (which mean the native keyword actually succeeded), MUST NOT trigger a premature release.

## Capabilities

### New Capabilities
- (none — this extends the existing runtime safety capability)

### Modified Capabilities
- `platynui-windows-runtime`: the requirement *"Desktop automation never leaves keyboard modifiers held"* is broadened to *"never leaves **any** key held"* — release is driven by a tracked held-key registry (covering non-modifiers), and a new persisted-state recovery guarantees release even when the process is force-killed before any handler runs.

## Impact

- **Code**: `src/robotmcp/plugins/builtin/platynui_plugin.py` (registry + persisted state + `on_session_start`/`on_session_end` release, replacing `_RELEASE_ALL_SEQUENCE`-only), `src/robotmcp/components/execution/keyword_executor.py` (F16 failure/except release path now releases tracked keys; press/type/release keyword paths record/clear registry entries).
- **Runtime**: PlatynUI `keyboard_press`/`keyboard_type`/`keyboard_release` dispatch is wrapped to update the registry; native behaviour otherwise unchanged. State file lives under the OS temp/state dir, is per-runtime, and is cleared on clean release and on successful recovery.
- **Behaviour**: strictly safer — closes the non-modifier gap and the force-kill gap; happy-path Press/Release chords and steering downgrades are preserved.
- **Risk/limits**: recovery is best-effort (a stale state file releases keys defensively — releasing a not-held key is a harmless no-op); no dependency changes; Linux paths unaffected.
