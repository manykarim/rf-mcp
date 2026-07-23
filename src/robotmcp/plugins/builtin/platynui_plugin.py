"""Builtin PlatynUI.BareMetal plugin (new Rust core, ADR-025).

Targets the new-core PlatynUI (https://github.com/imbus/robotframework-PlatynUI,
branch ``new_core``): Rust runtime + ``platynui-native`` PyO3 bindings +
``platynui-cli`` diagnostic tool. The Robot Framework library surface is
``PlatynUI.BareMetal`` (24 keywords, descriptor/XPath locator model).

Key integration concerns handled here (see ADR-025):

* **Wayland portal hang** — on Linux Wayland sessions the PlatynUI runtime
  blocks indefinitely on an interactive ``org.freedesktop.portal.RemoteDesktop``
  consent handshake. ``ensure_x11_session_env()`` forces the X11/XWayland
  backend (``XDG_SESSION_TYPE=x11``) before the first ``Runtime`` is created
  in this process. Opt out with ``ROBOTMCP_PLATYNUI_KEEP_WAYLAND=1``.
* **Query scoping** — desktop-wide descendant queries (``//control:...``) can
  take ~1 s per AT-SPI node timeout and minutes on busy desktops. Hints and
  failure guidance push app-scoped queries
  (``/app:*[@Name='X']//control:Button[@Name='OK']``).
* **Matched-set requirement** — the RF library and ``platynui-native`` wheel
  must come from the same source commit until upstream stabilizes; mismatch
  surfaces as ``ImportError`` on native symbols and gets a structured hint.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Any, Dict, List, Optional

from robotmcp.plugins.base import StaticLibraryPlugin
from robotmcp.plugins.contracts import (
    InstallAction,
    LibraryCapabilities,
    LibraryHints,
    LibraryMetadata,
    PromptBundle,
)

logger = logging.getLogger(__name__)

# Environment variable to opt out of the X11 session forcing shim.
KEEP_WAYLAND_ENV = "ROBOTMCP_PLATYNUI_KEEP_WAYLAND"

# The 24 PlatynUI.BareMetal keywords of the new core (lowercase RF names).
PLATYNUI_KEYWORDS = (
    # Query / context
    "set root",
    "query",
    "get attribute",
    # Pointer
    "pointer click",
    "pointer multi click",
    "pointer press",
    "pointer release",
    "pointer move to",
    "get pointer position",
    # Keyboard
    "keyboard type",
    "keyboard press",
    "keyboard release",
    # Focus / window management
    "focus",
    "activate window",
    "maximize window",
    "minimize window",
    "restore window",
    "close window",
    "move window",
    "resize window",
    "move and resize window",
    "bring to front",
    # Diagnostics
    "take screenshot",
    "highlight",
)

# Keywords PlatynUI shares (by name) with Browser Library. Both define
# "Focus", "Get Attribute" and "Take Screenshot"; desktop sessions must not
# block them and web sessions must not route them to PlatynUI.
_SHARED_WITH_BROWSER = frozenset({"focus", "get attribute", "take screenshot"})

_ACTIONABILITY_HINT = (
    "PlatynUI queries that start with '//' walk the WHOLE desktop tree and can "
    "take minutes on busy desktops (AT-SPI applies a 1s timeout per "
    "unresponsive node). Scope queries to the target application instead: "
    "/app:*[@Name='myapp']//control:Button[@Name='OK'] — or call 'Set Root' "
    "once with the application/window node."
)

_LINUX_FRAME_HINT = (
    "On Linux (AT-SPI2), application top-level windows usually expose the "
    "role 'Frame', not 'Window'. Use //control:Frame[@Name='...'] (or the "
    "window-management keywords, which accept Frame/Window/Dialog alike). "
    "control:Window typically only matches compositor/shell elements."
)

_MATCHED_SET_HINT = (
    "PlatynUI.BareMetal and platynui-native must be built from the SAME "
    "source commit (upstream is preview quality; PyPI dev wheels lag the "
    "source tree). If you see ImportError for native symbols (e.g. "
    "'WindowSurface'), rebuild: `maturin develop --release --manifest-path "
    "packages/native/Cargo.toml` in the robotframework-PlatynUI checkout, "
    "then `pip install --no-deps <checkout>`."
)


# Serializes the process-global os.environ mutation across the three
# trigger points (library import chokepoint, keyword-execution chokepoint,
# session-start hook) — concurrent first-touches must not interleave the
# check-and-set (cross-LLM review finding, ADR-025).
_ENV_SHIM_LOCK = threading.Lock()

# Whether this desktop session originated as Wayland (set once, before X11 is
# forced). change: desktop-input-and-runtime-diagnostics.
_WAYLAND_ORIGIN: Optional[bool] = None


def _record_session_origin(env: Dict[str, str]) -> None:
    """Record (once) whether the session originated as Wayland — env signals or
    a live ``$XDG_RUNTIME_DIR/wayland-0`` socket. Best-effort; never raises."""
    global _WAYLAND_ORIGIN
    if _WAYLAND_ORIGIN:
        return  # already known to be Wayland; do not downgrade
    try:
        session_type = env.get("XDG_SESSION_TYPE", "").strip().lower()
        if session_type == "wayland" or env.get("WAYLAND_DISPLAY"):
            _WAYLAND_ORIGIN = True
            return
        runtime_dir = env.get("XDG_RUNTIME_DIR", "")
        if runtime_dir and os.path.exists(os.path.join(runtime_dir, "wayland-0")):
            _WAYLAND_ORIGIN = True
            return
        if _WAYLAND_ORIGIN is None:
            _WAYLAND_ORIGIN = False
    except Exception:  # pragma: no cover - defensive
        pass


def was_wayland_session() -> bool:
    """True when this desktop session originated as Wayland (so forced-X11
    synthetic input may be blocked by the compositor)."""
    return _WAYLAND_ORIGIN is True


# --- Runtime broker (change: platynui-desktop-safety-isolation, D4) ---------
#
# The PlatynUI native platform module is process-global and is NOT safely
# re-initializable: once a Runtime is shut down, a later bind fails with
# "ProviderError ... not available after shutdown or failed connect". The
# proven proximate cause of that error on the MCP/Robot path was
# ui_tree_service creating and shutting down a Runtime on EVERY call. The
# broker gives ALL call sites (focus manager, ui_tree, etc.) one shared,
# lock-bound, lazily-created Runtime and refuses re-init after close.

_RUNTIME_LOCK = threading.Lock()
_RUNTIME = None  # the shared platynui_native.Runtime
# States: "new" (never bound) | "open" | "shutting_down" | "disposed"
# (terminal — the process-global platform module cannot be re-initialized).
_RUNTIME_STATE = "new"
# Last bind/connect failure text (for classification). None once open.
_RUNTIME_LAST_ERROR: Optional[str] = None

# Stuck-key safety net (change: fix-platynui-windows-runtime, F16). ``Keyboard
# Press`` sends key-DOWN only, and runs that time out / are killed mid-chord can
# leave a modifier physically held at the OS level — the operator's keyboard is
# then wedged until reboot. Releasing keys that are not held is a proven no-op,
# so a broad release sequence is always safe. Left+right variants + AltGr + Win
# cover every modifier the native keyboard recognizes.
_RELEASE_ALL_SEQUENCE = "<LCtrl+RCtrl+LAlt+RAlt+AltGr+LShift+RShift+LWin+RWin>"
# atexit/SIGTERM release handlers are registered exactly once, on first runtime
# bind (guarded under _RUNTIME_LOCK), so a killed process still releases keys.
_RELEASE_HANDLERS_REGISTERED = False

# --- Held-key registry (change: harden-platynui-stuck-key-release) -----------
#
# The modifiers-only F16 blast (_RELEASE_ALL_SEQUENCE) cannot lift a stuck
# NON-modifier key (a bare ``Keyboard Press A``/``<F12>``/``<Escape>`` that was
# never released). The registry records the EXACT keys rf-mcp is holding so the
# teardown/failure/exit paths release precisely those, and it is mirrored to a
# per-PID state file so a HARD kill (SIGKILL/TerminateProcess — which run
# neither atexit nor SIGTERM) is recovered by the next desktop session start.
_HELD_KEYS: set[str] = set()
_HELD_KEYS_LOCK = threading.RLock()
# Subdir under the OS temp dir holding ``held_<pid>.json`` files. A live process
# only ever replays a file owned by a DEAD pid, so a concurrent healthy session
# is never robbed of a key it is deliberately holding.
_HELD_KEYS_DIRNAME = "robotmcp_platynui_held_keys"


def runtime_unavailable_reason() -> Optional[str]:
    """Classify why the PlatynUI runtime is unavailable, or None when it is
    available (change: desktop-input-and-runtime-diagnostics).

    Returns one of:
    - ``"not_installed"`` — the native module could not be imported
    - ``"display_connect_failed"`` — the runtime could not connect to the
      display (DISPLAY/XAUTHORITY/XDG_RUNTIME_DIR)
    - ``"disposed"`` — the one-shot native module was disposed; restart needed
    - ``"unavailable"`` — bind failed for an unclassified reason
    - ``None`` — the runtime is currently open/available

    Never raises.
    """
    if _RUNTIME is not None and _RUNTIME_STATE == "open":
        return None
    if _RUNTIME_STATE == "disposed":
        return "disposed"
    err = (_RUNTIME_LAST_ERROR or "").lower()
    if not err:
        return None  # never attempted yet
    if "no module named" in err or "import" in err or "not installed" in err:
        return "not_installed"
    if any(
        tok in err
        for tok in (
            "connect", "connection", "display", "shutdown", "xauth",
            "x11", "wayland", "provider initialization", "platform",
        )
    ):
        return "display_connect_failed"
    return "unavailable"


def get_runtime():
    """Return the process-shared PlatynUI Runtime, creating it once (lazily).

    Forces the X11 backend first (env shim), then binds once under the lock.
    Returns None when platynui_native is unavailable. Raises RuntimeError when
    the broker has been disposed (single shared runtime; restart the process).
    """
    global _RUNTIME, _RUNTIME_STATE
    if _RUNTIME is not None and _RUNTIME_STATE == "open":
        return _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is not None and _RUNTIME_STATE == "open":
            return _RUNTIME
        if _RUNTIME_STATE == "disposed":
            # Was opened then shut down: refuse re-init (module is one-shot).
            raise RuntimeError(
                "PlatynUI runtime broker is disposed; the native platform "
                "module cannot be re-initialized in this process — restart "
                "the process."
            )
        global _RUNTIME_LAST_ERROR
        try:
            ensure_x11_session_env()
            import platynui_native as pn

            _RUNTIME = pn.Runtime()
            _RUNTIME_STATE = "open"
            _RUNTIME_LAST_ERROR = None
            # First successful bind: arm the stuck-key safety net so a killed
            # process still releases held modifiers (F16). Under _RUNTIME_LOCK.
            _register_release_handlers_once()
        except Exception as exc:  # pragma: no cover - env dependent
            logger.debug("PlatynUI runtime broker bind failed: %s", exc)
            _RUNTIME = None
            # Record the failure for classification (change:
            # desktop-input-and-runtime-diagnostics). Stay in "new" so an
            # unavailable-then-available env can retry.
            _RUNTIME_LAST_ERROR = f"{type(exc).__name__}: {exc}"
        return _RUNTIME


def release_all_modifiers() -> bool:
    """Best-effort release of every keyboard modifier (change:
    fix-platynui-windows-runtime, F16).

    Releases Ctrl/Alt/AltGr/Shift/Win (left+right) against the ALREADY-OPEN
    native runtime. Releasing a key that is not held is a no-op, so this is
    always safe to call from teardown, keyword-failure ``finally`` paths, and
    process-exit handlers. Never raises; never starts the runtime just to
    release (teardown must not spin up the native broker). Returns True when a
    release was dispatched.
    """
    if _RUNTIME is None or _RUNTIME_STATE != "open":
        return False
    try:
        _RUNTIME.keyboard_release(_RELEASE_ALL_SEQUENCE)
        return True
    except Exception as exc:  # pragma: no cover - env dependent
        logger.debug("release_all_modifiers failed: %s", exc)
        return False


# --- Held-key registry: normalize / record / release / recover --------------
#
# change: harden-platynui-stuck-key-release.


def _normalize_key_tokens(sequence: Optional[str]) -> list[str]:
    """Parse a PlatynUI key sequence into the individual key tokens that a
    ``Keyboard Press`` of that sequence leaves HELD (key-DOWN, no key-UP).

    Best-effort and never raises. ``<Ctrl+A>`` -> ``["Ctrl", "A"]``; ``<F12>``
    -> ``["F12"]``; a bare ``A`` -> ``["A"]``; literal text ``hi`` ->
    ``["h", "i"]``. Bracketed groups split on ``+``; unbracketed characters
    each become their own token. Whitespace and empty tokens are dropped.
    """
    tokens: list[str] = []
    if not sequence:
        return tokens
    try:
        i = 0
        n = len(sequence)
        while i < n:
            ch = sequence[i]
            if ch == "<":
                end = sequence.find(">", i + 1)
                if end == -1:
                    # Unterminated '<' — treat the rest as literal chars.
                    for c in sequence[i + 1:]:
                        if not c.isspace():
                            tokens.append(c)
                    break
                inner = sequence[i + 1:end]
                for part in inner.split("+"):
                    part = part.strip()
                    if part:
                        tokens.append(part)
                i = end + 1
            else:
                if not ch.isspace():
                    tokens.append(ch)
                i += 1
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("_normalize_key_tokens failed for %r: %s", sequence, exc)
    return tokens


def _release_sequence_for(tokens) -> str:
    """Build a PlatynUI release sequence that lifts each token individually,
    e.g. ``["A", "Ctrl"]`` -> ``"<A><Ctrl>"``."""
    return "".join(f"<{t}>" for t in tokens)


def record_pressed_keys(sequence: Optional[str]) -> None:
    """Record the keys a ``Keyboard Press``/``Keyboard Type`` is about to hold,
    and mirror the held set to the per-PID state file so a hard kill is
    recoverable. Never raises. change: harden-platynui-stuck-key-release."""
    tokens = _normalize_key_tokens(sequence)
    if not tokens:
        return
    with _HELD_KEYS_LOCK:
        _HELD_KEYS.update(tokens)
        _write_state_locked()


def record_released_keys(sequence: Optional[str]) -> None:
    """Remove keys released by ``Keyboard Release`` (or the paired UP inside
    ``Keyboard Type``) from the registry, updating the state file. Never
    raises. change: harden-platynui-stuck-key-release."""
    tokens = _normalize_key_tokens(sequence)
    if not tokens:
        return
    with _HELD_KEYS_LOCK:
        for t in tokens:
            _HELD_KEYS.discard(t)
        _write_state_locked()


def release_tracked_keys() -> bool:
    """Release EXACTLY the keys the registry records as held (including
    non-modifiers), then defensively blast all modifiers, and clear the
    registry + state file.

    Best-effort and non-raising; a no-op observationally when nothing is held
    (releasing a not-held key has no effect). Never starts the runtime solely
    to release — acts only when the runtime is already open (mirrors
    ``release_all_modifiers``). change: harden-platynui-stuck-key-release.
    """
    if _RUNTIME is None or _RUNTIME_STATE != "open":
        return False
    with _HELD_KEYS_LOCK:
        held = sorted(_HELD_KEYS)
        dispatched = False
        try:
            if held:
                _RUNTIME.keyboard_release(_release_sequence_for(held))
                dispatched = True
            # Fallback blast: covers any modifier that slipped registry tracking.
            _RUNTIME.keyboard_release(_RELEASE_ALL_SEQUENCE)
            dispatched = True
        except Exception as exc:  # pragma: no cover - env dependent
            logger.debug("release_tracked_keys dispatch failed: %s", exc)
        _HELD_KEYS.clear()
        _clear_state_locked()
        return dispatched


# --- Per-PID state file (hard-kill recovery) --------------------------------


def _held_keys_dir() -> Optional[str]:
    """Return (creating if needed) the directory holding per-PID held-key state
    files, or None on failure. Never raises."""
    try:
        import tempfile

        d = os.path.join(tempfile.gettempdir(), _HELD_KEYS_DIRNAME)
        os.makedirs(d, exist_ok=True)
        return d
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("_held_keys_dir failed: %s", exc)
        return None


def _state_file_path(pid: Optional[int] = None) -> Optional[str]:
    d = _held_keys_dir()
    if d is None:
        return None
    return os.path.join(d, f"held_{pid if pid is not None else os.getpid()}.json")


def _write_state_locked() -> None:
    """Atomically write the current held set (with owning PID) to this process's
    state file, or delete the file when the set is empty. Caller holds
    ``_HELD_KEYS_LOCK``. Never raises."""
    path = _state_file_path()
    if path is None:
        return
    try:
        if not _HELD_KEYS:
            _clear_state_locked()
            return
        import json

        payload = json.dumps({"pid": os.getpid(), "keys": sorted(_HELD_KEYS)})
        tmp = f"{path}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(payload)
        os.replace(tmp, path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("_write_state_locked failed: %s", exc)


def _clear_state_locked() -> None:
    """Delete this process's state file. Caller holds ``_HELD_KEYS_LOCK``.
    Never raises."""
    path = _state_file_path()
    if path is None:
        return
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("_clear_state_locked failed: %s", exc)


def _pid_alive(pid: int) -> bool:
    """Best-effort liveness check for ``pid``. On any uncertainty returns True
    (fail-safe: do not replay a file that might belong to a live process)."""
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    try:
        if sys.platform == "win32":
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if not handle:
                return False  # cannot open -> gone (or access denied ~ rare)
            try:
                code = ctypes.c_ulong()
                if kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return code.value == STILL_ACTIVE
                return True
            finally:
                kernel32.CloseHandle(handle)
        else:
            try:
                os.kill(pid, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                return True  # exists but not ours
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("_pid_alive(%s) failed: %s", pid, exc)
        return True


def recover_orphaned_held_keys() -> int:
    """Replay key-UPs for any state file owned by a DEAD pid (a prior run
    hard-killed with keys held), then delete the file. Files owned by a live
    pid are left untouched. Returns the number of files recovered. Best-effort;
    never raises; never starts the runtime. change:
    harden-platynui-stuck-key-release."""
    if _RUNTIME is None or _RUNTIME_STATE != "open":
        return 0
    d = _held_keys_dir()
    if d is None:
        return 0
    recovered = 0
    try:
        import json

        for name in os.listdir(d):
            if not (name.startswith("held_") and name.endswith(".json")):
                continue
            path = os.path.join(d, name)
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.loads(fh.read() or "{}")
            except Exception:
                # Unreadable/partial file: remove it defensively.
                try:
                    os.remove(path)
                except Exception:
                    pass
                continue
            pid = int(data.get("pid", -1) or -1)
            keys = data.get("keys") or []
            if pid == os.getpid() or _pid_alive(pid):
                continue  # ours or a live session's — do not touch
            try:
                if keys:
                    _RUNTIME.keyboard_release(_release_sequence_for(keys))
                # Defensive modifier blast for the orphaned session too.
                _RUNTIME.keyboard_release(_RELEASE_ALL_SEQUENCE)
                recovered += 1
            except Exception as exc:  # pragma: no cover - env dependent
                logger.debug("orphan replay dispatch failed: %s", exc)
            try:
                os.remove(path)
            except Exception:
                pass
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("recover_orphaned_held_keys failed: %s", exc)
    return recovered


def _register_release_handlers_once() -> None:
    """Arm atexit + SIGTERM handlers that release held modifiers on process
    exit (F16). Called once from ``get_runtime`` under ``_RUNTIME_LOCK`` after
    the first successful bind — the safety net for a run killed mid-chord (e.g.
    a ``claude -p`` process terminated before ``on_session_end`` runs). Never
    raises."""
    global _RELEASE_HANDLERS_REGISTERED
    if _RELEASE_HANDLERS_REGISTERED:
        return
    _RELEASE_HANDLERS_REGISTERED = True
    try:
        import atexit

        atexit.register(release_tracked_keys)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("atexit release registration failed: %s", exc)
    try:
        import signal

        # Chain to any prior SIGTERM handler so we don't swallow shutdown.
        _prior = signal.getsignal(signal.SIGTERM)

        def _release_on_sigterm(signum, frame):  # pragma: no cover - signal path
            release_tracked_keys()
            if callable(_prior) and _prior not in (
                signal.SIG_DFL,
                signal.SIG_IGN,
            ):
                _prior(signum, frame)
            elif _prior == signal.SIG_IGN:
                # Prior handler explicitly ignored SIGTERM — preserve that.
                return
            else:
                # SIG_DFL, None, or unknown: restore the default and re-raise so
                # the process still terminates on SIGTERM (never leave it
                # un-killable by swallowing the signal).
                signal.signal(signal.SIGTERM, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

        signal.signal(signal.SIGTERM, _release_on_sigterm)
    except (ValueError, OSError, AttributeError) as exc:
        # signal.signal only works on the main thread; best-effort otherwise.
        logger.debug("SIGTERM release registration skipped: %s", exc)


def runtime_state() -> str:
    return _RUNTIME_STATE


def native_providers() -> list:
    """Return the active PlatynUI providers via the NATIVE ``runtime.providers()``
    API (change: desktop-native-platynui-alignment). Best-effort; returns an
    empty list when the runtime is unavailable. Never raises."""
    try:
        runtime = get_runtime()
    except Exception:
        return []
    if runtime is None:
        return []
    try:
        _p = getattr(runtime, "providers", None)
        if callable(_p):
            return list(_p() or [])
    except Exception as exc:  # pragma: no cover - env dependent
        logger.debug("native_providers failed: %s", exc)
    return []


def clear_runtime_tree_cache() -> bool:
    """Best-effort clear of the shared runtime's cached accessibility tree.

    The PlatynUI runtime caches the desktop tree, so an application launched
    AFTER the last snapshot stays invisible to subsequent keyword queries until
    the cache is cleared (change: desktop-tree-cache-refresh). This centralizes
    the ``getattr(runtime, "clear_cache", None)`` dance used by get_ui_tree and
    the keyword-execution refresh paths.

    Returns True when a clear was performed, False otherwise (runtime
    unavailable, no clear_cache, or the clear raised). Never raises.
    """
    try:
        runtime = get_runtime()
    except Exception:
        # Disposed/unavailable broker — nothing to clear.
        return False
    if runtime is None:
        return False
    try:
        _clear = getattr(runtime, "clear_cache", None)
        if callable(_clear):
            _clear()
            return True
    except Exception as exc:  # pragma: no cover - env dependent
        logger.debug("clear_runtime_tree_cache failed: %s", exc)
    return False


def shutdown_runtime() -> None:
    """Shut down the shared runtime and mark the broker closed.

    Intended for process teardown only. After this, ``get_runtime()`` raises
    rather than attempting a doomed re-bind.
    """
    global _RUNTIME, _RUNTIME_STATE
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME_STATE = "disposed"
            return
        _RUNTIME_STATE = "shutting_down"
        rt, _RUNTIME = _RUNTIME, None
        _RUNTIME_STATE = "disposed"
    try:
        rt.shutdown()
    except Exception:  # pragma: no cover
        pass


def _reset_runtime_broker_for_tests() -> None:
    """Test-only: reset broker state so unit tests start fresh."""
    global _RUNTIME, _RUNTIME_STATE
    with _RUNTIME_LOCK:
        _RUNTIME = None
        _RUNTIME_STATE = "new"


def ensure_x11_session_env(environ: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Force the X11 backend for PlatynUI on Linux Wayland sessions.

    The PlatynUI runtime resolves the session type once per process from
    ``XDG_SESSION_TYPE`` (authoritative) / ``WAYLAND_DISPLAY`` / ``DISPLAY``.
    The Wayland input backend performs an xdg-desktop-portal RemoteDesktop
    handshake with **no timeout** and (on first run) an interactive consent
    dialog — fatal for a headless MCP server. The X11/XWayland backend has no
    such handshake and provides full keyboard/pointer/screenshot support.

    Must run BEFORE the first ``platynui_native.Runtime`` is created in this
    process (the session type is cached process-wide).

    Returns a human-readable note when the environment was changed, else None.
    """
    env = environ if environ is not None else os.environ
    if sys.platform != "linux":
        return None

    # The mutation is set-once / never-restored (same constant value), but
    # the check-and-set must not interleave across threads — and CPython's
    # putenv from concurrent threads should be serialized anyway.
    with _ENV_SHIM_LOCK:
        # Record whether this session ORIGINATED as Wayland, BEFORE forcing X11,
        # so callers can warn that synthetic X11 (XTest) input may be blocked by
        # the compositor (change: desktop-input-and-runtime-diagnostics). Use
        # env signals AND a live wayland socket — the env may be scrubbed
        # (XDG_SESSION_TYPE pre-forced to x11, WAYLAND_DISPLAY unset) on a real
        # Wayland host, but $XDG_RUNTIME_DIR/wayland-0 still exists there.
        _record_session_origin(env)

        if env.get(KEEP_WAYLAND_ENV, "").strip() in {"1", "true", "yes"}:
            return None

        session_type = env.get("XDG_SESSION_TYPE", "").strip().lower()
        wayland = bool(env.get("WAYLAND_DISPLAY"))
        display = bool(env.get("DISPLAY"))

        if session_type == "x11":
            _pin_gtk_x11_backend(env)
            return None
        if not display:
            # No X server available — forcing X11 would break init outright.
            if session_type == "wayland" or wayland:
                logger.warning(
                    "PlatynUI: Wayland session without DISPLAY — runtime init may "
                    "block on an xdg-desktop-portal consent dialog. Approve the "
                    "dialog once (a restore token is persisted) or provide an "
                    "X11/XWayland DISPLAY."
                )
            return None
        if session_type == "wayland" or (not session_type and wayland):
            env["XDG_SESSION_TYPE"] = "x11"
            _pin_gtk_x11_backend(env)
            note = (
                "PlatynUI: forced XDG_SESSION_TYPE=x11 (XWayland) to avoid the "
                "Wayland xdg-desktop-portal consent handshake which blocks "
                f"indefinitely in headless contexts. Set {KEEP_WAYLAND_ENV}=1 to "
                "keep the native Wayland backend."
            )
            logger.warning(note)
            return note
        _pin_gtk_x11_backend(env)
        return None


def _pin_gtk_x11_backend(env: Dict[str, str]) -> None:
    """Pin GUI-toolkit backends to X11 when a Wayland compositor socket is
    reachable (change: platynui-visible-safe-targeting follow-up).

    Unsetting ``WAYLAND_DISPLAY`` is NOT sufficient: libwayland's
    ``wl_display_connect(NULL)`` falls back to the literal socket name
    ``wayland-0`` in ``$XDG_RUNTIME_DIR``, and distro GTK builds try the
    Wayland backend first — so an AUT launched via Process with
    ``DISPLAY=:100`` can silently render on the user's ACTIVE host desktop
    while synthetic XTest input goes to the isolated display (observed with
    LibreOffice on the 2026-06-11 validation rerun: soffice.bin connected to
    ``/run/user/1000/wayland-0``/gnome-shell despite a scrubbed env).
    Children inherit this process env via Start Process, so pinning here
    confines GTK/Qt AUTs to the bound X display. Respects pre-set values.
    """
    try:
        if env.get("GDK_BACKEND") or env.get("QT_QPA_PLATFORM"):
            return
        runtime_dir = env.get("XDG_RUNTIME_DIR", "")
        wayland_reachable = bool(env.get("WAYLAND_DISPLAY")) or (
            runtime_dir
            and os.path.exists(os.path.join(runtime_dir, "wayland-0"))
        )
        if not wayland_reachable or not env.get("DISPLAY"):
            return
        env["GDK_BACKEND"] = "x11"
        env["QT_QPA_PLATFORM"] = "xcb"
        logger.info(
            "PlatynUI: pinned GDK_BACKEND=x11 / QT_QPA_PLATFORM=xcb — a "
            "Wayland compositor socket is reachable and GTK's wayland-0 "
            "fallback would otherwise let AUTs escape to the active desktop."
        )
    except Exception:  # pragma: no cover - defensive
        pass


class PlatynUILibraryPlugin(StaticLibraryPlugin):
    """Builtin plugin for the new-core PlatynUI.BareMetal desktop library."""

    def __init__(self) -> None:
        metadata = LibraryMetadata(
            name="PlatynUI.BareMetal",
            package_name="robotframework-platynui",
            import_path="PlatynUI.BareMetal",
            description=(
                "Cross-platform native desktop UI automation (Windows UIA, "
                "Linux AT-SPI2) with XPath locators, backed by a Rust runtime"
            ),
            library_type="external",
            use_cases=[
                "desktop automation",
                "native application testing",
                "window management",
                "desktop ui inspection",
            ],
            categories=["desktop", "testing"],
            contexts=["desktop"],
            installation_command=(
                "pip install --pre platynui-native platynui-cli "
                "(RF library: install robotframework-PlatynUI from source, "
                "branch new_core — matched commit with platynui-native)"
            ),
            dependencies=["platynui-native"],
            platform_requirements=["python>=3.12"],
            requires_type_conversion=True,
            supports_async=False,
            load_priority=42,
            default_enabled=True,
            extra_name="desktop",
            technology_tags=["desktop", "uia", "atspi", "xpath"],
            aliases=["PlatynUI"],
        )
        capabilities = LibraryCapabilities(
            contexts=["desktop"],
            features=["window-management", "pointer", "keyboard", "screenshot"],
            technology=["UIA", "AT-SPI2"],
            supports_page_source=False,
            supports_application_state=True,
            requires_type_conversion=True,
        )
        hints = LibraryHints(
            standard_keywords=[
                "Query",
                "Pointer Click",
                "Keyboard Type",
                "Activate Window",
                "Get Attribute",
                "Take Screenshot",
            ],
            error_hints=[
                _ACTIONABILITY_HINT,
                _LINUX_FRAME_HINT,
                _MATCHED_SET_HINT,
            ],
            usage_examples=[
                "Query    /app:*[@Name='gnome-calculator']//control:Frame    only_first=True",
                "Pointer Click    /app:*[@Name='myapp']//control:Button[@Name='OK']",
                "Keyboard Type    /app:*[@Name='myapp']//control:Text[@Name='Input']    Hello <Ctrl+A>",
                "Activate Window    /app:*[@Name='myapp']//control:Frame",
                "Get Attribute    /app:*[@Name='myapp']//control:Frame    Bounds",
                "Take Screenshot    filename=EMBED",
            ],
        )
        prompt_bundle = PromptBundle(
            recommendation=(
                "Use PlatynUI.BareMetal for native desktop application "
                "automation (NOT web pages). Locators are XPath over the "
                "desktop accessibility tree with namespaces app:/control:/"
                "item:/native: and PascalCase attributes (@Name, @Bounds, "
                "@AutomationId). ALWAYS scope queries to an application: "
                "/app:*[@Name='X']//control:Button[@Name='OK']."
            ),
            troubleshooting=(
                "Element not found: list applications first with Query "
                "/app:* then inspect one level at a time. Slow queries: "
                "never start a locator with // (full-desktop walk). On "
                "Linux, windows are control:Frame, not control:Window. "
                "Keyboard sequences support chords: <Ctrl+A>, <Enter>."
            ),
        )
        install_actions = [
            InstallAction(
                description="Install PlatynUI native runtime + CLI (pre-release)",
                command=["pip", "install", "--pre", "platynui-native", "platynui-cli"],
            ),
            InstallAction(
                description=(
                    "Install RF library from source (matched commit with the "
                    "native wheel — see ADR-025)"
                ),
                command=[
                    "pip",
                    "install",
                    "--no-deps",
                    "git+https://github.com/imbus/robotframework-PlatynUI.git@new_core",
                ],
            ),
        ]
        super().__init__(
            metadata=metadata,
            capabilities=capabilities,
            install_actions=install_actions,
            hints=hints,
            prompt_bundle=prompt_bundle,
        )

    # -- keyword routing -------------------------------------------------

    def get_keyword_library_map(self) -> Dict[str, str]:  # type: ignore[override]
        mapping: Dict[str, str] = {}
        for keyword in PLATYNUI_KEYWORDS:
            mapping[f"platynui.baremetal.{keyword}"] = "PlatynUI.BareMetal"
            if keyword not in _SHARED_WITH_BROWSER:
                mapping[keyword] = "PlatynUI.BareMetal"
        return mapping

    # -- session hooks ---------------------------------------------------

    def on_session_start(self, session: "ExecutionSession") -> None:
        """Force the X11 backend before the PlatynUI runtime exists.

        NOTE: this hook fires at session *creation*, which usually happens
        before the library list is populated — the deterministic trigger is
        the desktop check in ``KeywordExecutor._execute_keyword_serialized``
        (ADR-025). This hook covers restore/attach flows where the session
        already carries PlatynUI.
        """
        try:
            libraries = list(getattr(session, "imported_libraries", None) or [])
            libraries += list(getattr(session, "search_order", None) or [])
            preference = getattr(session, "explicit_library_preference", "") or ""
            libraries.append(preference)
            if any("platynui" in str(lib).lower() for lib in libraries):
                ensure_x11_session_env()
                # Arm the stuck-key release handlers HERE, on the main thread
                # (session creation runs on the event loop). The first runtime
                # bind often happens inside the F14 to_thread-offloaded focus
                # call on a WORKER thread, where signal.signal() raises — so
                # arming only at bind time would silently skip the SIGTERM net
                # (and default SIGTERM also skips atexit), defeating the F16
                # process-kill safety net. Registration is idempotent and the
                # release itself no-ops until the runtime is open. F16.
                _register_release_handlers_once()
                # Recover keys left HELD by a prior process that was hard-killed
                # (SIGKILL/TerminateProcess ran no atexit/SIGTERM handler): replay
                # key-UPs from any state file owned by a dead pid, then clear it.
                # change: harden-platynui-stuck-key-release.
                recover_orphaned_held_keys()
                # Defensively clear any modifier a PRIOR crashed run left held
                # (no-op unless the runtime is already open) — F16.
                release_all_modifiers()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("PlatynUI session-start hook failed: %s", exc)

    def on_session_end(self, session: "ExecutionSession") -> None:
        """Release any held keyboard keys when a PlatynUI desktop session closes
        (change: fix-platynui-windows-runtime, F16; harden-platynui-stuck-key-release).

        ``session_manager.on_session_end`` invokes this for every plugin; the
        release is a no-op when the runtime is not open or nothing is held, so
        it is safe for non-desktop sessions too. Never raises. Releases the
        exact tracked held-key set (incl. non-modifiers), not just modifiers."""
        try:
            release_tracked_keys()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("PlatynUI session-end release failed: %s", exc)

    def before_keyword_execution(
        self,
        session: "ExecutionSession",
        keyword_name: str,
        library_manager: Any,
        keyword_discovery: Any,
    ) -> None:
        """Safety net: ensure the env shim ran before any PlatynUI keyword."""
        if keyword_name and keyword_name.lower() in self.get_keyword_library_map():
            ensure_x11_session_env()

    # -- failure guidance ------------------------------------------------

    def generate_failure_hints(
        self,
        session: "ExecutionSession",
        keyword_name: str,
        arguments: List[Any],
        error_text: str,
    ) -> List[Dict[str, Any]]:
        hints: List[Dict[str, Any]] = []
        error_lower = (error_text or "").lower()
        args_text = " ".join(str(a) for a in arguments)

        if "importerror" in error_lower or "cannot import name" in error_lower:
            hints.append(
                {
                    "type": "platynui_matched_set",
                    "title": "PlatynUI version mismatch",
                    "message": _MATCHED_SET_HINT,
                }
            )
        if "elementnotfound" in error_lower or "no nodes" in error_lower or (
            "not found" in error_lower and "element" in error_lower
        ):
            message = _ACTIONABILITY_HINT
            if "control:window" in args_text.lower():
                message = f"{_LINUX_FRAME_HINT} {message}"
            hints.append(
                {
                    "type": "platynui_locator",
                    "title": "PlatynUI locator guidance",
                    "message": message,
                }
            )
        if "timeout" in error_lower or "timed out" in error_lower:
            hints.append(
                {
                    "type": "platynui_query_scope",
                    "title": "Scope desktop queries",
                    "message": _ACTIONABILITY_HINT,
                }
            )
        if "providererror" in error_lower and "mock" in error_lower:
            hints.append(
                {
                    "type": "platynui_mock_provider",
                    "title": "Mock provider not linked",
                    "message": (
                        "use_mock=True needs a platynui-native wheel built "
                        "with `--features mock-provider`; published wheels "
                        "link only the real OS providers."
                    ),
                }
            )
        return hints


try:  # pragma: no cover
    from robotmcp.models.session_models import ExecutionSession  # noqa: F401
except Exception:  # pragma: no cover
    ExecutionSession = object  # type: ignore
