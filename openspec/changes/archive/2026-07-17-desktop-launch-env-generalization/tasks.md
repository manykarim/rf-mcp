# Tasks: desktop-launch-env-generalization

## 1. Generalized GUI-launch detection
- [x] 1.1 Keep `_KNOWN_GUI_BINARIES` as a fast-path allow-set; add a `_NON_GUI_BINARIES` deny-set (`bash`, `sh`, `dash`, `zsh`, `python`, `python3`, `uv`, `cat`, `echo`, `sleep`, `env`, `true`, `false`, `ls`, `test`)
- [x] 1.2 `is_desktop_gui_launch(arguments, *, is_desktop_session)` returns the binary when (a) it is in the allow-set, OR (b) the session is desktop, the binary resolves via `resolve_executable`, and it is not in the deny-set
- [x] 1.3 Preserve the current signature for existing callers (default `is_desktop_session=None` reproduces allow-set-only behavior)

## 2. Accessibility + backend overlay
- [x] 2.1 Add `GTK_A11Y=atspi` to the display overlay assembled for a GUI launch, only when not already set to a non-empty value (never overwrite an explicit operator value)
- [x] 2.2 Confirm the overlay still carries DISPLAY / XDG_SESSION_TYPE=x11 / GDK_BACKEND=x11 / QT_QPA_PLATFORM=xcb; add `NO_AT_BRIDGE=0` defensively
- [x] 2.3 Snap sanitization path unchanged for non-snap AUTs; snap AUTs keep roots (`_aut_snap_root`)

## 3. Applied-overrides observability
- [x] 3.1 Return the recognized-binary + applied-override set from the launch path
- [x] 3.2 Surface it in the launch signal / session-state so an empty-tree failure names the missing/applied a11y env (eval R14 tie-in)

## 4. Tests
- [x] 4.1 `soffice` / `mousepad` (non-allowlisted) recognized as GUI launch in a desktop session; get `GTK_A11Y=atspi` + X11 pins
- [x] 4.2 `bash -c '…'` / `python3 script.py` NOT recognized as a GUI launch (deny-set)
- [x] 4.3 An already-set `GTK_A11Y=1` (or any non-empty) value is NOT overwritten
- [x] 4.4 Snap AUT roots preserved; non-snap AUT snap segments stripped (existing behavior regression)
- [x] 4.5 Non-desktop session with a non-allowlisted binary is unaffected

## 5. Deterministic validation (docker, no-LLM) — closes eval gap G2
- [x] 5.1 A driver launches `mousepad` and `soffice` through the launch layer with `GTK_A11Y` UNSET in the AUT env, asserting each resolves a non-empty `/app:*` subtree — `docker/gate_drivers.py g2`. PASS 2026-07-17: mousepad 88 controls WITH overlay / 0 WITHOUT (negative control); soffice 678 controls. Overlay set `GTK_A11Y=atspi` in both.
- [x] 5.2 Driver `docker/gate_drivers.py` (guard/g2/g3/g6) runnable against `robotmcp-desktop-lab2`; not yet wired as a permanent CMD rung (follow-up)
