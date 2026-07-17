"""Bound-display safety classification for desktop (PlatynUI) sessions.

Change: platynui-desktop-safety-isolation (finding #4 — the safety blocker).

A desktop session must never silently drive the user's active desktop. We
classify the bound display as one of:

* ``isolated`` — POSITIVE proof of rf-mcp ownership (an isolation marker
  recorded by the bootstrap names this DISPLAY). Only here is automation
  allowed without an override.
* ``active``   — an EWMH window manager is present on the bound DISPLAY
  (``_NET_SUPPORTING_WM_CHECK`` resolves to a live window) → the user's live
  session.
* ``unknown``  — everything else, including a bare X server with no marker and
  a probe that could not be completed. Fail closed (refuse).

Cross-LLM review insisted on positive proof: "no EWMH WM" must NOT be treated
as isolated (a non-EWMH real desktop would false-allow), and the marker takes
precedence over the EWMH probe (a nested display running its own WM is still
isolated when rf-mcp owns it).
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Env names. The bootstrap records the isolated DISPLAY it provisioned; the
# operator may opt in to running on active/unknown displays; a one-release
# transition mode warns instead of refusing.
ISOLATION_MARKER_ENV = "ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY"
# Ownership corroboration (change: desktop-isolation-marker-hardening): the
# bootstrap/entrypoint that mints the marker also records the PID of the X
# server (Xvfb/Xephyr/Xorg) it launched for the isolated display. The guard
# verifies that PID is a live X server for the claimed display, so a stale or
# hand-misconfigured marker (e.g. an inherited ``:0``) cannot false-allow input.
ISOLATION_XPID_ENV = "ROBOTMCP_PLATYNUI_ISOLATED_XPID"
ALLOW_ACTIVE_ENV = "ROBOTMCP_PLATYNUI_ALLOW_ACTIVE_DESKTOP"
SAFETY_GUARD_ENV = "ROBOTMCP_PLATYNUI_SAFETY_GUARD"  # "warn" => log, don't block

_X_SERVER_BINARIES = ("xvfb", "xephyr", "xorg", "/x ", "/x\x00", "x11")

ISOLATED = "isolated"
ACTIVE = "active"
UNKNOWN = "unknown"


def _bound_display(environ: Dict[str, str]) -> Optional[str]:
    disp = environ.get("DISPLAY")
    return disp.strip() if disp else None


def _has_isolation_marker(environ: Dict[str, str], display: Optional[str]) -> bool:
    marker = environ.get(ISOLATION_MARKER_ENV)
    if not marker or not display:
        return False
    # Marker may list one or more displays, comma/space separated. Do NOT use
    # os.pathsep (":") — DISPLAY values themselves contain ":".
    import re as _re

    claimed = {d.strip() for d in _re.split(r"[,\s]+", marker) if d.strip()}
    return display in claimed


def _marker_ownership_status(
    environ: Dict[str, str], display: Optional[str]
) -> str:
    """Corroborate the isolation marker against a recorded X-server PID.

    Returns one of:
    - ``"verified"`` — ``ROBOTMCP_PLATYNUI_ISOLATED_XPID`` names a live process
      whose cmdline is an X server bound to the claimed ``display``.
    - ``"invalid"``  — an XPID was provided but does not resolve to such a
      process (stale/misconfigured marker — the R4 case).
    - ``"absent"``   — no XPID provided (legacy marker-only setup) OR ownership
      cannot be determined (no /proc); the caller preserves back-compat.

    Best-effort and never raises. change: desktop-isolation-marker-hardening.
    """
    if not display:
        return "absent"
    raw = (environ.get(ISOLATION_XPID_ENV) or "").strip()
    if not raw:
        return "absent"
    try:
        pid = int(raw)
        if pid <= 0:
            return "invalid"
    except ValueError:
        return "invalid"
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            cmdline = fh.read().decode("utf-8", "replace").lower()
    except FileNotFoundError:
        return "invalid"  # PID recorded by the bootstrap is gone -> not owned
    except OSError:
        return "absent"  # cannot read /proc (permissions / non-Linux) -> unknown
    if not cmdline:
        return "invalid"
    # The display token (e.g. ":99") must appear on the X server's command line,
    # and the binary must look like an X server.
    disp_token = display.strip()
    args = [a for a in cmdline.split("\x00") if a]
    looks_x = any(
        b in (args[0] if args else cmdline)
        for b in ("xvfb", "xephyr", "xorg")
    ) or (args and args[0].rstrip("0123456789").endswith("/x"))
    if looks_x and disp_token in args:
        return "verified"
    return "invalid"


def _ewmh_wm_present(display: str) -> Optional[bool]:
    """True when an EWMH WM owns the display (_NET_SUPPORTING_WM_CHECK), False
    when provably absent, None when the probe could not be completed.

    DOCUMENTED NATIVE GAP (change: desktop-native-platynui-alignment). PlatynUI's
    native API/CLI exposes NO live-WM-on-display signal — `_NET_SUPPORTING_WM_CHECK`
    is internal-only to the Rust WindowManager (spike, 2026-06-10), and
    `runtime.providers()` reports only which providers are LOADED, not whether a
    WM owns this display. This ctypes probe is therefore RETAINED as the
    documented fallback that supplies the security-relevant active-vs-isolated
    signal; native providers may enrich the report but must not replace it.
    Read-only root property (no tree recursion); kept as-is to preserve the
    safety guard's contract."""
    try:
        import ctypes
        import ctypes.util

        libname = ctypes.util.find_library("X11")
        if not libname:
            return None
        x11 = ctypes.CDLL(libname)
        x11.XOpenDisplay.restype = ctypes.c_void_p
        x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
        x11.XDefaultRootWindow.restype = ctypes.c_ulong
        x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        x11.XInternAtom.restype = ctypes.c_ulong
        x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        x11.XGetWindowProperty.restype = ctypes.c_int
        x11.XGetWindowProperty.argtypes = [
            ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_long,
            ctypes.c_long, ctypes.c_int, ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
        ]
        x11.XFree.argtypes = [ctypes.c_void_p]
        x11.XCloseDisplay.argtypes = [ctypes.c_void_p]

        dpy = x11.XOpenDisplay(display.encode())
        if not dpy:
            return None
        try:
            atom = x11.XInternAtom(dpy, b"_NET_SUPPORTING_WM_CHECK", True)
            if atom == 0:
                return False  # atom never interned -> no EWMH WM has run
            root = x11.XDefaultRootWindow(dpy)
            actual_type = ctypes.c_ulong()
            actual_format = ctypes.c_int()
            nitems = ctypes.c_ulong()
            bytes_after = ctypes.c_ulong()
            prop = ctypes.POINTER(ctypes.c_ubyte)()
            status = x11.XGetWindowProperty(
                dpy, root, atom, 0, 1, False, 33,  # XA_WINDOW = 33
                ctypes.byref(actual_type), ctypes.byref(actual_format),
                ctypes.byref(nitems), ctypes.byref(bytes_after), ctypes.byref(prop),
            )
            present = status == 0 and bool(prop) and nitems.value >= 1
            if prop:
                x11.XFree(prop)
            return present
        finally:
            x11.XCloseDisplay(dpy)
    except Exception as exc:  # pragma: no cover - env dependent
        logger.debug("EWMH probe failed: %s", exc)
        return None


def classify_bound_display(environ: Optional[Dict[str, str]] = None) -> str:
    """Classify the bound display as isolated / active / unknown.

    Positive isolation proof (the marker) takes precedence over the EWMH
    probe; absence of an EWMH WM is NOT isolation.
    """
    return classify_bound_display_detailed(environ)["isolation"]


def classify_bound_display_detailed(
    environ: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Classification plus provenance for session-state reporting
    (change: platynui-visible-safe-targeting, task 3.3).

    Returns ``{display, isolation, isolation_source}`` where
    ``isolation_source`` is ``marker`` (positive proof via
    ROBOTMCP_PLATYNUI_ISOLATED_DISPLAY), ``ewmh`` (live-WM probe), or
    ``none`` (no display bound / probe inconclusive).
    """
    env = environ if environ is not None else os.environ
    display = _bound_display(env)
    if _has_isolation_marker(env, display):
        # Corroborate the marker against a recorded X-server PID (change:
        # desktop-isolation-marker-hardening).
        status = _marker_ownership_status(env, display)
        if status == "verified":
            return {
                "display": display,
                "isolation": ISOLATED,
                "isolation_source": "marker",
            }
        if status == "invalid":
            # A companion XPID was provided but does not resolve to a live X
            # server for this display -> the marker is stale/misconfigured.
            # Fail closed; name the conflict when a live WM contradicts it.
            wm = _ewmh_wm_present(display)
            return {
                "display": display,
                "isolation": UNKNOWN,
                "isolation_source": (
                    "marker_over_active_wm" if wm is True else "marker_invalid"
                ),
            }
        # status == "absent": no ownership proof — no XPID recorded, or /proc
        # could not be read. STRICT fail-closed: a marker alone is not proof of
        # rf-mcp ownership, so refuse rather than trust the assertion. The
        # conflict is named when a live WM contradicts the marker. Deployments
        # that own an isolated display must record ROBOTMCP_PLATYNUI_ISOLATED_XPID
        # (the entrypoint / bootstrap script do this automatically).
        wm = _ewmh_wm_present(display)
        return {
            "display": display,
            "isolation": UNKNOWN,
            "isolation_source": (
                "marker_over_active_wm" if wm is True else "marker_unverified"
            ),
        }
    if not display:
        return {"display": None, "isolation": UNKNOWN, "isolation_source": "none"}
    wm = _ewmh_wm_present(display)
    if wm is True:
        return {"display": display, "isolation": ACTIVE, "isolation_source": "ewmh"}
    # wm is False (no WM) or None (probe failed) -> unknown, NOT isolated.
    return {
        "display": display,
        "isolation": UNKNOWN,
        "isolation_source": "ewmh" if wm is False else "none",
    }


def allow_opt_in(environ: Optional[Dict[str, str]] = None) -> bool:
    env = environ if environ is not None else os.environ
    return env.get(ALLOW_ACTIVE_ENV, "").strip().lower() in {"1", "true", "yes"}


def warn_mode(environ: Optional[Dict[str, str]] = None) -> bool:
    env = environ if environ is not None else os.environ
    return env.get(SAFETY_GUARD_ENV, "").strip().lower() == "warn"


class DesktopSafetyError(Exception):
    """Raised when the safety guard refuses to drive a non-isolated display."""


def evaluate_safety(
    session,
    environ: Optional[Dict[str, str]] = None,
) -> Dict[str, object]:
    """Decide whether a desktop operation may proceed on the bound display.

    Returns a dict: {classification, enforcing, allowed, bypassed, reason}.
    Honors a per-session opt-in attribute (``platynui_allow_active_desktop``)
    in addition to the env opt-in.
    """
    env = environ if environ is not None else os.environ
    classification = classify_bound_display(env)
    session_opt_in = bool(getattr(session, "platynui_allow_active_desktop", False))
    opted_in = session_opt_in or allow_opt_in(env)
    is_warn = warn_mode(env)

    if classification == ISOLATED:
        return {
            "classification": classification, "enforcing": True,
            "allowed": True, "bypassed": False, "reason": None,
        }
    reason = (
        f"bound display classified '{classification}': refusing desktop input "
        "to avoid leaking onto the user's active session. Run on an isolated "
        "display (recipe below) — the bypass below is an escape hatch that does "
        "NOT guarantee correct input on a shared active desktop."
    )
    recipe = build_isolation_recipe()
    if opted_in:
        return {
            "classification": classification, "enforcing": True,
            "allowed": True, "bypassed": True, "reason": reason,
            "isolation_recipe": recipe,
        }
    if is_warn:
        return {
            "classification": classification, "enforcing": False,
            "allowed": True, "bypassed": False, "reason": reason,
            "isolation_recipe": recipe,
        }
    return {
        "classification": classification, "enforcing": True,
        "allowed": False, "bypassed": False, "reason": reason,
        "isolation_recipe": recipe,
    }


def build_isolation_recipe() -> dict:
    """An actionable recipe for running desktop automation on an isolated
    display, returned on the active-desktop refuse path (finding #3).

    The VISIBLE mode (Xephyr nested X server) leads as the recommended
    interactive path: the application renders in a window on the tester's
    screen — every interaction is watchable — while synthetic XTest input
    stays confined to the nested display and cannot leak onto the active
    desktop. Headless Xvfb is the CI alternative.

    change: desktop-stepwise-execution-fidelity;
    reworked: platynui-visible-safe-targeting (task 3.4).
    """
    return {
        "summary": (
            "Run the desktop automation on an rf-mcp-owned nested display: "
            "VISIBLE (Xephyr — recommended for interactive testing; the app "
            "is visible on your screen while input is confined to the nested "
            "display) or headless (Xvfb — CI)."
        ),
        "recommended_mode": "visible",
        "modes": [
            {
                "mode": "visible",
                "recommended": True,
                "description": (
                    "Xephyr nested X server — the app renders as a window on "
                    "your desktop so you can WATCH every interaction; "
                    "keyboard/pointer input stays confined to the nested "
                    "display."
                ),
            },
            {
                "mode": "headless",
                "description": "Xvfb virtual display — nothing rendered (CI default).",
            },
        ],
        "steps": [
            "RECOMMENDED (visible): `scripts/platynui_desktop_bootstrap.sh "
            "--mode visible` — or manually: `Xephyr :100 -screen 1280x1024 &`",
            "Start a minimal EWMH window manager inside the nested display "
            "(e.g. `DISPLAY=:100 openbox &`) — PlatynUI's window activation "
            "(WindowSurface/bring_to_front) needs EWMH; without a WM, focus "
            "verification degrades to a warning.",
            "GTK/Qt backends are pinned to X11 automatically (GDK_BACKEND=x11, "
            "QT_QPA_PLATFORM=xcb) when a host Wayland socket is reachable — "
            "without the pin, apps silently render on the host desktop via "
            "the 'wayland-0' fallback even with WAYLAND_DISPLAY unset.",
            "Export the display for the server process: `export DISPLAY=:100` "
            "and unset `WAYLAND_DISPLAY`.",
            "Launch the app under that display (e.g. via "
            "`systemd-run --user --setenv=DISPLAY=:100 <app>`).",
            f"Mark the display isolated so the guard allows input: set "
            f"`{ISOLATION_MARKER_ENV}=:100`.",
            "Headless alternative (CI): `Xvfb :99 -screen 0 1280x1024x24 &` "
            f"then `export DISPLAY=:99` and `{ISOLATION_MARKER_ENV}=:99`.",
        ],
        "verification_commands": [
            "platynui-cli-rs info",
            "platynui-cli-rs window --list",
            "platynui-cli-rs highlight '<xpath>'",
            "platynui-cli-rs snapshot '<xpath>'",
        ],
        "bootstrap_script": "scripts/platynui_desktop_bootstrap.sh --mode visible",
        "marker_env": ISOLATION_MARKER_ENV,
        "bypass_env": ALLOW_ACTIVE_ENV,
        "bypass_note": (
            f"{ALLOW_ACTIVE_ENV}=1 is an ESCAPE HATCH only — input may still "
            "target the wrong window on a shared active desktop, so prefer an "
            "isolated display."
        ),
    }
