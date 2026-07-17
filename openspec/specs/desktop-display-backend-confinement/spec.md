# desktop-display-backend-confinement Specification

## Purpose
TBD - created by archiving change desktop-evidence-and-display-scoping. Update Purpose after archive.
## Requirements
### Requirement: GUI toolkit backends pinned to the bound X display
When a desktop session forces or confirms the X11 backend and a Wayland compositor socket is reachable (via `WAYLAND_DISPLAY` or the literal `wayland-0` socket in `$XDG_RUNTIME_DIR`), the system SHALL set `GDK_BACKEND=x11` and `QT_QPA_PLATFORM=xcb` in the process environment (inherited by AUT children launched via Process), because `wl_display_connect(NULL)` falls back to `wayland-0` even when `WAYLAND_DISPLAY` is unset and the AUT would otherwise render on the user's active desktop.

#### Scenario: Wayland host with isolated X display
- **WHEN** `ensure_x11_session_env` runs with `XDG_SESSION_TYPE=wayland`, `DISPLAY=:100`, and a reachable Wayland socket
- **THEN** the environment gains `GDK_BACKEND=x11` and `QT_QPA_PLATFORM=xcb`

#### Scenario: Pre-scrubbed env on a Wayland host still pins
- **WHEN** the env already says `XDG_SESSION_TYPE=x11` with no `WAYLAND_DISPLAY`, but `$XDG_RUNTIME_DIR/wayland-0` exists
- **THEN** `GDK_BACKEND=x11` is still pinned

### Requirement: Explicit operator choices are respected
The pin SHALL NOT override a pre-set `GDK_BACKEND` or `QT_QPA_PLATFORM`, SHALL NOT apply when the keep-Wayland opt-out env is set, and SHALL NOT apply on hosts with no reachable Wayland socket.

#### Scenario: Pre-set backend kept
- **WHEN** the env contains `GDK_BACKEND=wayland`
- **THEN** the value is unchanged and `QT_QPA_PLATFORM` is not added

#### Scenario: Pure X11 host untouched
- **WHEN** no Wayland socket is reachable
- **THEN** neither variable is added

