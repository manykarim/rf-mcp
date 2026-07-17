"""Desktop AUT launch-environment sanitization.

Change: platynui-desktop-safety-isolation (finding #2).

When rf-mcp is launched from a snap-confined shell (e.g. a VS Code snap), the
parent environment carries snap-rooted dynamic-loader / module / data
variables. A GNOME app launched via ``Process.Start Process`` inheriting those
loads the snap-bundled libpthread before the system one and dies with
``gnome-calculator: symbol lookup error: /snap/core20/.../libpthread.so.0:
undefined symbol __libc_pthread_init, version GLIBC_PRIVATE``.

This module builds a sanitized CHILD environment for a desktop GUI launch:
it filters ``/snap/``-rooted PATH SEGMENTS (not whole variables) out of the
loader/module/data variables, preserves everything else, and overlays the
session's display variables. When the AUT is itself a snap-confined binary
that needs its own snap roots, those roots are preserved.
"""

from __future__ import annotations

import os
import shutil
from typing import Dict, List, Optional

# Variables whose value is a list of path SEGMENTS we filter by package root.
_PATH_LIST_VARS = (
    "LD_LIBRARY_PATH",
    "GTK_PATH",
    "GIO_MODULE_DIR",
    "GIO_EXTRA_MODULES",
    "GSETTINGS_SCHEMA_DIR",
    "QT_PLUGIN_PATH",
    "XDG_DATA_DIRS",
    "FONTCONFIG_PATH",
    "LOCPATH",
)
# Single-path variables: dropped entirely when they point under a package root.
_SINGLE_PATH_VARS = (
    "LD_PRELOAD",
    "GTK_EXE_PREFIX",
    "FONTCONFIG_FILE",
)

# Package-root prefixes considered "contaminating" for a non-snap AUT.
# /var/lib/snapd/ holds snap-exported .desktop launchers/icons that, if left in
# XDG_DATA_DIRS, can re-introduce snap Exec= handlers (report finding #2).
_CONTAMINATING_ROOTS = ("/snap/", "/var/lib/snapd/")


def get_effective_path(parent_env: Optional[Dict[str, str]] = None) -> str:
    """Return the PATH the server process resolves executables against."""
    env = parent_env if parent_env is not None else os.environ
    return env.get("PATH", "") or ""


def resolve_executable(
    name: str, *, parent_env: Optional[Dict[str, str]] = None
) -> Optional[str]:
    """Resolve an executable name to an absolute path via ``shutil.which``
    against the SERVER-process PATH (change: desktop-mcp-workflow-correctness,
    maintainer-report #7).

    A desktop launch/recovery tool (e.g. ``xdotool``) that is resolvable for the
    server must be dispatched by its absolute path, rather than relying on an
    interactive shell's startup environment that the step execution does not
    inherit. Returns ``None`` when the tool cannot be resolved on the server
    PATH. An already-absolute existing path is returned unchanged.
    """
    if not name or not isinstance(name, str):
        return None
    if os.path.isabs(name):
        return name if os.path.exists(name) else None
    env = parent_env if parent_env is not None else os.environ
    return shutil.which(name, path=env.get("PATH"))


def _segment_under_roots(segment: str, roots: tuple) -> bool:
    seg = segment.strip()
    return bool(seg) and any(seg.startswith(r) for r in roots)


def _filter_path_list(value: str, roots: tuple) -> str:
    parts = value.split(os.pathsep)
    kept = [p for p in parts if not _segment_under_roots(p, roots)]
    return os.pathsep.join(kept)


def _aut_snap_root(executable: str) -> Optional[str]:
    """Return the /snap/<name>/ root when the AUT is itself a snap binary.

    Resolves the binary (PATH + symlinks); a snap app's binary resolves under
    ``/snap/<name>/...``. We then preserve that snap's roots from stripping.
    """
    if not executable:
        return None
    resolved = shutil.which(executable) or executable
    try:
        real = os.path.realpath(resolved)
    except OSError:
        real = resolved
    for candidate in (resolved, real):
        if candidate and candidate.startswith("/snap/"):
            # /snap/<name>/<rev>/... -> keep "/snap/<name>/"
            parts = candidate.split("/")
            if len(parts) >= 3:
                return "/snap/" + parts[2] + "/"
    return None


def build_desktop_launch_env(
    executable: str,
    *,
    parent_env: Optional[Dict[str, str]] = None,
    display_env: Optional[Dict[str, str]] = None,
    sanitize: bool = True,
) -> Dict[str, str]:
    """Build a sanitized child environment for a desktop GUI launch.

    Args:
        executable: the GUI binary being launched (for snap-AUT detection).
        parent_env: base environment (defaults to ``os.environ``).
        display_env: display variables to overlay (DISPLAY, XDG_SESSION_TYPE,
            GDK_BACKEND, ...). ``WAYLAND_DISPLAY`` set to "" is removed.
        sanitize: when False, skip filtering (the ``--no-sanitize`` hatch); the
            display overlay is still applied.

    Returns:
        A new env dict (the parent is not mutated).
    """
    base = dict(parent_env if parent_env is not None else os.environ)

    # When the AUT is itself a snap binary it needs its snap roots — do not
    # sanitize at all in that case (stripping would break it).
    if sanitize and _aut_snap_root(executable) is None:
        roots = _CONTAMINATING_ROOTS
        for var in _PATH_LIST_VARS:
            if var in base and base[var]:
                base[var] = _filter_path_list(base[var], roots)
        for var in _SINGLE_PATH_VARS:
            val = base.get(var, "")
            if val and _segment_under_roots(val, roots):
                base.pop(var, None)

    # Overlay display variables; an empty value means "unset".
    for key, value in (display_env or {}).items():
        if value == "":
            base.pop(key, None)
        else:
            base[key] = value

    return base


# Fast-path allow-set: binaries always treated as GUI launches (no session
# context needed). Kept for back-compat and cheap recognition.
_KNOWN_GUI_BINARIES = frozenset({
    "gnome-calculator",
    "gnome-text-editor",
    "gedit",
    "nautilus",
    "gnome-control-center",
    "gnome-terminal",
    "eog",
    "evince",
})

# Deny-set: obvious non-GUI shell/utility binaries that a desktop session may
# legitimately ``Start Process`` without any GUI launch hardening. Guards the
# evidence-based path (change: desktop-launch-env-generalization) so a plain
# subprocess is never sanitized/overlaid.
_NON_GUI_BINARIES = frozenset({
    "bash", "sh", "dash", "zsh", "fish", "csh", "tcsh", "ksh",
    "python", "python3", "python2", "uv", "uvx", "pipx", "node", "deno",
    "cat", "echo", "sleep", "env", "true", "false", "ls", "test", "printf",
    "grep", "sed", "awk", "cut", "head", "tail", "sort", "uniq", "wc",
    "cp", "mv", "rm", "mkdir", "touch", "chmod", "chown", "which", "id",
    "kill", "pkill", "pgrep", "ps", "uname", "date", "sync", "dbus-send",
    "xdotool", "xrandr", "xset", "import", "convert",
})


def is_desktop_gui_launch(
    arguments: List, *, is_desktop_session: Optional[bool] = None
) -> Optional[str]:
    """Return the GUI binary name when arguments look like a desktop GUI launch
    (``Start Process <gui-binary> ...``), else None.

    Detection (change: desktop-launch-env-generalization):
    - A binary in the known-gnome allow-set is always recognized (fast path,
      no session context needed) — preserves the original behavior when
      ``is_desktop_session`` is omitted/None.
    - When ``is_desktop_session`` is True, ANY resolvable binary that is not on
      the non-GUI deny-set is recognized as a GUI launch, so non-gnome AUTs
      (LibreOffice, Firefox, Qt apps, mousepad) get the same launch hardening.
    """
    if not arguments:
        return None
    first = arguments[0]
    if not isinstance(first, str):
        return None
    raw = first.strip()
    if not raw or raw.startswith("env:") or "=" in raw:
        return None
    binary = os.path.basename(raw)
    if binary in _KNOWN_GUI_BINARIES:
        return binary
    if is_desktop_session is not True:
        return None
    if binary in _NON_GUI_BINARIES:
        return None
    # Evidence-based: a desktop session launching a resolvable, non-utility
    # binary is treated as a GUI AUT.
    if os.path.isabs(raw):
        return binary if os.path.exists(raw) else None
    if shutil.which(raw) is not None:
        return binary
    return None


def gui_launch_overrides(
    binary: str, *, parent_env: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """Return the accessibility/backend override keys applied to a recognized
    GUI launch, for both the executor overlay and session-state observability
    (change: desktop-launch-env-generalization, task 3).

    ``GTK_A11Y`` defaults to ``atspi`` unless the parent env already sets it to
    a non-empty value (an explicit operator choice is never overwritten). The
    X11 backend pins are always applied so a GTK or Qt AUT comes up on X11 with
    a populated AT-SPI tree regardless of the server process environment.
    """
    env = parent_env if parent_env is not None else os.environ
    overrides: Dict[str, str] = {
        "GTK_A11Y": (env.get("GTK_A11Y") or "").strip() or "atspi",
        "GDK_BACKEND": "x11",
        "QT_QPA_PLATFORM": "xcb",
        "XDG_SESSION_TYPE": "x11",
        "NO_AT_BRIDGE": "0",
    }
    return overrides
