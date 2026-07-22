# Tasks — fix-platynui-windows-runtime

## 1. F16 — stuck-key release safety net (critical)
- [x] 1.1 Add `release_all_modifiers()` to `plugins/builtin/platynui_plugin.py`: releases the
      `<LCtrl+RCtrl+LAlt+RAlt+AltGr+LShift+RShift+LWin+RWin>` sequence via the **already-open**
      `_RUNTIME.keyboard_release(...)`; best-effort, never raises, no-op when runtime not open,
      never starts the runtime.
- [x] 1.2 Add `PlatynUIPlugin.on_session_end(session)` calling `release_all_modifiers()`
      (session_manager already invokes `plugin.on_session_end`).
- [x] 1.3 In `get_runtime()`'s successful-bind branch, register `atexit` + `SIGTERM` handlers
      **once** (module flag) that call `release_all_modifiers()` — the safety net for killed runs.
- [x] 1.4 Defensive release in `PlatynUIPlugin.on_session_start` (clear a prior crashed run's
      held keys) — best-effort, guarded on an open runtime.
- [x] 1.5 In `keyword_executor.py`, wrap the desktop keyboard-keyword dispatch in `try/finally`
      so any exception/timeout from `Keyboard Press`/`Keyboard Type`/`Keyboard Release` triggers
      `release_all_modifiers()` before returning.
- [x] 1.6 Add a one-line desktop-signal note steering toward atomic `Keyboard Type <chord>` over
      bare `Keyboard Press` (wording only; where desktop signals already surface).

## 2. F3 — bounded, honored desktop query timeout
- [x] 2.1 Inject a short default PlatynUI query timeout (~1500 ms) for desktop sessions at desktop
      session init (via query settings / `query_overrides`).
- [x] 2.2 Map a caller-supplied `timeout_ms` onto the PlatynUI query override for desktop keywords
      (desktop analogue of the browser/selenium pre-validation timeout path).
- [x] 2.3 Keep the default overridable per session (larger `timeout_ms` for deliberate waits).

## 3. F14 — never block the event loop
- [x] 3.1 Offload the synchronous desktop **pre-dispatch** native queries in
      `_execute_keyword_serialized` — `_platynui_focus_before_act` (ensure_focused) and
      `_desktop_text_count_before` — via `asyncio.to_thread(...)`. (The native keyword execute at
      keyword_executor:3679 is already offloaded; corrected seam per design.)
- [x] 3.2 Scope strictly to the desktop branch (`_is_desktop`); `FocusError` still propagates out of
      `to_thread` to the existing except. Do not double-wrap the already-offloaded native execute.

## 4. F4 — platform-aware safety guard
- [x] 4.1 Add `WINDOWS = "windows"` and an early `if os.name == "nt"` branch in
      `classify_bound_display_detailed` → `{isolation: WINDOWS, isolation_source: "windows_console"}`.
- [x] 4.2 Add a Windows branch to `evaluate_safety`: allowed by default (enforcing, not bypassed),
      one-time active-desktop warning, Windows-accurate note, **no Xephyr recipe**; keep a strict
      opt-in to enforce isolation on Windows.
- [x] 4.3 Verify Linux isolated/active/unknown behaviour is untouched.

## 5. Tests & validation
- [x] 5.1 F16 tests: `release_all_modifiers` release-sequence + no-op-when-closed + never-starts-runtime;
      `on_session_end` triggers it; `finally`-release on a raised keyboard keyword; atexit registered once.
- [x] 5.2 F3 tests: desktop session injects short query timeout; `timeout_ms` maps onto the override.
- [x] 5.3 F14 tests: pre-dispatch focus/text-count queries offloaded via `to_thread` (source guard);
      behavioural proof that a slow (blocking) focus query does not stall the event loop.
- [x] 5.4 F4 tests: `classify_bound_display_detailed` → `windows` under `os.name=="nt"`;
      `evaluate_safety` on Windows → allowed, not bypassed, no Xephyr recipe; Linux unchanged.
- [x] 5.5 `uv run pytest tests/unit` green; `openspec validate fix-platynui-windows-runtime --strict`.
