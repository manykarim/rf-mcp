# Proposal: desktop-launch-env-generalization

## Why

Reliable agentic steering of **several** Linux desktop apps on X11 depends on
every launched AUT (a) coming up with a non-empty AT-SPI object tree and (b)
not inheriting a snap-contaminated loader env or escaping to a host Wayland
compositor. Today only an **8-binary gnome allowlist** gets that treatment, so
the common multi-app case (LibreOffice, Firefox, Qt apps, mousepad, IDEs) is
unguarded.

- **The allowlist is hard-coded and gnome-only.** `is_desktop_gui_launch`
  (`desktop_launch_env.py:167-178`) returns the binary name only when it is in
  `_KNOWN_GUI_BINARIES` (`:155-164`) = `{gnome-calculator, gnome-text-editor,
  gedit, nautilus, gnome-control-center, gnome-terminal, eog, evince}`. For any
  other GUI binary it returns `None`, so the caller applies **no**
  snap-sanitization and **no** display overlay. `mousepad`, `soffice`,
  `libreoffice`, `firefox`, `chromium`, and every Qt app fall through.
- **`GTK_A11Y=atspi` is never applied by the launch layer.**
  `build_desktop_launch_env` (`desktop_launch_env.py:110-151`) overlays only the
  `display_env` its caller passes (DISPLAY / XDG_SESSION_TYPE / GDK_BACKEND /
  QT_QPA_PLATFORM) and never sets `GTK_A11Y`. A GTK app launched with
  accessibility disabled exposes an **empty tree** — there is then literally
  nothing for PlatynUI to steer, and desktop discovery (`find_keywords` is
  useless for PlatynUI; guidance is the only surface) gives the agent no signal
  why. This is why the docker harness had to set `GTK_A11Y=atspi` in the *image*
  (`docker/claude_mcp.json`) — a workaround that masks the gap and does not
  exist in a real non-container deployment where the operator did not export it
  process-wide.
- **Evidence the two failure modes are real and load-bearing.** The
  snap-libpthread crash is the documented reason `desktop_launch_env.py` exists
  at all (module docstring, finding #2). The empty-tree-without-`GTK_A11Y` mode
  is the documented `desktop-a11y-atspi-backend` fix (`GTK_A11Y` must be
  `atspi`, not `1`). Both are currently applied only for 8 binaries or only by
  the container image, not by the launch layer for arbitrary AUTs.

The result: "works for gnome-calculator" does not generalize to "works for
GTK/Qt apps on X11." Closing this is the single highest-leverage move toward
multi-app steering reliability (eval synthesis 2026-07-17, top-3 #1; risks
R5+R6).

## What Changes

- **Generalize GUI-launch detection beyond the fixed allowlist.**
  `is_desktop_gui_launch` SHALL recognize a desktop GUI launch by evidence, not
  membership in an 8-item set: keep the known-binary fast path, and additionally
  treat a `Start Process` (or `Run Process`) launch as a GUI launch when it is a
  desktop session and the binary resolves to a real executable that is not a
  known non-GUI shell utility. A conservative deny-list of obvious non-GUI
  binaries (`bash`, `sh`, `python`, `cat`, `sleep`, `echo`, …) avoids sanitizing
  a plain subprocess.
- **Always overlay the accessibility + backend env for GUI launches.** The
  display overlay built for a GUI launch SHALL include `GTK_A11Y=atspi` (unless
  already set to a non-empty value) alongside the existing DISPLAY /
  XDG_SESSION_TYPE=x11 / GDK_BACKEND=x11 / QT_QPA_PLATFORM=xcb pins, so any GTK
  or Qt AUT comes up accessible and on X11 regardless of the server process
  env. Snap-path stripping continues to apply to non-snap AUTs; snap AUTs keep
  their roots (unchanged `_aut_snap_root` behavior).
- **Surface what was applied.** The launch response/signal SHALL report whether
  the AUT was recognized as a GUI launch and which accessibility/backend
  overrides were applied, so an empty-tree failure is diagnosable (ties to the
  discovery-observability gap, eval R14).

Out of scope: changing `Process`-as-core-library routing (already shipped); the
Wayland-vs-X11 *input* enforcement (covered by `desktop-steering-confidence-gate`);
per-toolkit tree-read quirks (documented AT-SPI limitations, unchanged).

## Capabilities

### New Capabilities

- `desktop-launch-env-generalization`: any desktop GUI AUT launched via the
  process keywords — not just the 8 known gnome binaries — receives snap-loader
  sanitization, X11 backend pinning, and a `GTK_A11Y=atspi` overlay, and the
  applied overrides are reported.

### Modified Capabilities

- None (there is no existing launch-env capability spec under
  `openspec/specs/`; requirements are additive).

## Impact

- `src/robotmcp/components/execution/desktop_launch_env.py:155-178` —
  `_KNOWN_GUI_BINARIES` becomes a fast-path set, not the sole gate;
  `is_desktop_gui_launch` gains evidence-based detection + a non-GUI deny-list.
- `src/robotmcp/components/execution/desktop_launch_env.py:110-151` —
  `build_desktop_launch_env` (or its caller in `keyword_executor.py` that
  assembles `display_env`) adds the `GTK_A11Y=atspi` default to the overlay.
- `src/robotmcp/components/execution/keyword_executor.py` (~`:337-339`, the
  `is_desktop_gui_launch` call site) — pass the generalized result through;
  emit the applied-overrides signal.
- `src/robotmcp/components/execution/desktop_execution_signals.py` — optional
  launch-env signal fields (applied overrides) for session-state reporting.
- Tests: `tests/unit/` — a non-allowlisted binary (e.g. `soffice`, `mousepad`)
  is recognized as a GUI launch and gets `GTK_A11Y=atspi` + X11 pins; a plain
  `bash -c` subprocess is NOT sanitized; snap-AUT roots preserved; an
  already-set `GTK_A11Y` is not overwritten.
- Deterministic validation (docker): a no-LLM driver that launches `mousepad`
  and `soffice` through the launch layer **with `GTK_A11Y` unset in the AUT
  env** and asserts each resolves a non-empty `/app:*` subtree — the gate that
  actually exercises R6 rather than masking it (eval gap G2).
