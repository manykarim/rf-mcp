"""Window focus + visibility + scope guarantees for PlatynUI desktop sessions.

Change: platynui-focused-execution.

PlatynUI pointer/keyboard operations are delivered at *screen coordinates*
via the platform input backend. The accessibility tree can be fully
addressable even when the AUT window is not focused/topmost, and on
overlapping windows the same coordinate belongs to whichever window is
stacked on top. This module makes desktop execution target the
application-under-test (AUT) on purpose:

* resolve the AUT top-level window from any descriptor (D2),
* ensure it is raised + focused before pointer/keyboard dispatch (D1/D3),
* verify a resolved target belongs to the AUT window subtree (D5),
* report window visibility and enforce a precondition (D4).

The focus mechanism is tiered (D3): native WindowSurface activation where a
WM is present, a portable X11 raise fallback when WindowSurface is
unavailable (the WM-less case), and a no-op when the window is already
active. All of it degrades gracefully — the scope/visibility checks are the
real guarantee when focus cannot be asserted.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Per-call argument + env escape hatch (D7).
NO_FOCUS_ENV = "ROBOTMCP_PLATYNUI_NO_FOCUS"

# Upstream bring_to_front input-readiness wait budget (ms). Capped so the
# focus gate never stalls an interaction step noticeably
# (change: platynui-visible-safe-targeting, D2).
BRING_TO_FRONT_WAIT_MS = 1500

# Target highlighting via upstream Runtime.highlight(): mark the element
# about to receive input on the (visible) display so a human can see
# exactly where commands go (change: platynui-visible-safe-targeting, D3).
HIGHLIGHT_ENV = "ROBOTMCP_PLATYNUI_HIGHLIGHT"
HIGHLIGHT_DURATION_MS = 600


def highlight_disabled_by_env(environ: Optional[Dict[str, str]] = None) -> bool:
    env = environ if environ is not None else os.environ
    return env.get(HIGHLIGHT_ENV, "").strip().lower() in {"0", "false", "no"}

# I-2 (LibreOffice validation, 2026-06-11): the AUT frame exposed no
# WindowSurface/Focusable pattern, focus silently fell through, and
# keystrokes vanished. This warning makes unverifiable focus explicit.
FOCUS_UNVERIFIABLE_PREFIX = (
    "input focus could not be verified for this target — keystrokes may not land"
)

# D7 (change: desktop-evidence-and-display-scoping): blind type-at-focus
# (no descriptor) executed 20+ times into an empty display as robot-level
# "success" on the 2026-06-11 rerun. When no AUT window focus was ever
# verified in this session, say so.
UNFOCUSED_TYPING_WARNING = (
    "type-at-focus with no previously verified AUT window focus — "
    "keystrokes may land nowhere; target a descriptor or focus/activate "
    "the AUT window first"
)

# PlatynUI keywords whose FIRST positional argument is a descriptor that an
# operation actuates (pointer + window ops). Keyboard keywords take an
# optional descriptor then text; handled separately.
_DESCRIPTOR_FIRST_KEYWORDS = frozenset({
    "pointer click",
    "pointer multi click",
    "pointer press",
    "pointer release",
    "pointer move to",
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
})
_KEYBOARD_KEYWORDS = frozenset({
    "keyboard type",
    "keyboard press",
    "keyboard release",
})

# Keywords that should trigger focus-before-act. Window-management keywords
# manipulate focus themselves, so they are NOT pre-focused.
_INTERACTION_KEYWORDS = frozenset({
    "pointer click",
    "pointer multi click",
    "pointer press",
    "pointer release",
    "pointer move to",
    "keyboard type",
    "keyboard press",
    "keyboard release",
})


def normalize_keyword(keyword: str) -> str:
    """Lowercase, strip library prefix and bdd prefix, collapse separators."""
    kw = (keyword or "").strip().lower()
    if "." in kw:
        # Strip any library prefix (e.g. "platynui.baremetal."); RF keyword
        # names contain no dots, so the segment after the last dot is the kw.
        kw = kw.rsplit(".", 1)[-1]
    kw = kw.replace("_", " ").replace("-", " ")
    return " ".join(kw.split())


def focus_disabled_by_env(environ: Optional[Dict[str, str]] = None) -> bool:
    env = environ if environ is not None else os.environ
    return env.get(NO_FOCUS_ENV, "").strip().lower() in {"1", "true", "yes"}


def is_interaction_keyword(keyword: str) -> bool:
    return normalize_keyword(keyword) in _INTERACTION_KEYWORDS


def extract_descriptor(keyword: str, arguments: list) -> Optional[str]:
    """Return the AUT descriptor a PlatynUI keyword acts on, if any.

    Pointer/window keywords: first positional arg. Keyboard keywords: first
    arg unless it is the None/${None} sentinel (type-at-focus), in which
    case there is no descriptor to resolve.
    """
    if not arguments:
        return None
    kw = normalize_keyword(keyword)
    first = arguments[0]
    if not isinstance(first, str):
        return None
    if kw in _KEYBOARD_KEYWORDS:
        sentinel = first.strip().lower()
        if sentinel in {"", "none", "${none}"}:
            return None
        return first
    if kw in _DESCRIPTOR_FIRST_KEYWORDS:
        if not first.strip():
            return None
        if _looks_like_named_arg(first):
            return None
        return first
    return None


def _looks_like_named_arg(value: str) -> bool:
    """True for ``name=value`` (RF named arg), not for descriptors that
    merely contain ``=`` inside a predicate like ``[@Name='7']``."""
    head = value.split("=", 1)[0]
    if "=" not in value or not head:
        return False
    # A descriptor head contains path/predicate characters; a named arg head
    # is a bare identifier.
    return all(c.isalnum() or c in "_-" for c in head)


def app_scope_of(descriptor: str) -> Optional[str]:
    """Extract the ``/app:*[...]`` prefix of an app-scoped descriptor."""
    if not descriptor:
        return None
    text = descriptor.strip()
    if not text.startswith("/app:"):
        return None
    # /app:*[@Name='X']//control:... -> /app:*[@Name='X']
    marker = text.find("//", 1)
    return text[:marker] if marker > 0 else text


def is_unscoped(descriptor: str) -> bool:
    """True when a desktop descriptor is not rooted at an application."""
    text = (descriptor or "").strip()
    return text.startswith("//") or (not text.startswith("/app:") and "//" in text)


class FocusOutcome:
    """Result of a focus-before-act attempt (carried into step hints)."""

    __test__ = False

    def __init__(self) -> None:
        self.attempted = False
        self.focused = False
        self.bypassed = False
        self.strategy: Optional[str] = None
        self.visible: Optional[bool] = None
        self.in_scope: Optional[bool] = None
        self.warnings: list = []
        self.error: Optional[str] = None
        # Upstream pattern introspection + verified activation
        # (change: platynui-visible-safe-targeting).
        self.patterns: Optional[list] = None
        self.input_ready: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "attempted": self.attempted,
            "focused": self.focused,
        }
        if self.bypassed:
            out["bypassed"] = True
        if self.strategy:
            out["strategy"] = self.strategy
        if self.visible is not None:
            out["visible"] = self.visible
        if self.in_scope is not None:
            out["in_scope"] = self.in_scope
        if self.patterns is not None:
            out["patterns"] = self.patterns
        if self.input_ready is not None:
            out["input_ready"] = self.input_ready
        if self.warnings:
            out["warnings"] = self.warnings
        if self.error:
            out["error"] = self.error
        return out


class PlatynUIFocusManager:
    """Resolves the AUT window and ensures it is focused/visible/in-scope.

    Stateless w.r.t. RF — it talks to a process-shared ``platynui_native``
    runtime directly (the platform module is process-global, so a fresh
    Runtime shares the same backend). One instance per executor; caches the
    last-focused window per app scope to avoid focus thrash (D-risk).
    """

    def __init__(self) -> None:
        self._runtime = None
        self._last_focused_scope: Optional[str] = None
        # Verified-activation cache keyed by window runtime_id; invalidated
        # by the call site when the desktop tree is dirty (ADR-031 flag)
        # (change: platynui-visible-safe-targeting, task 2.5).
        self._focused_window_ids: set = set()
        # Whether ANY AUT window focus has been verifiably established in
        # this manager's lifetime — gates the blind type-at-focus warning
        # (change: desktop-evidence-and-display-scoping, D7).
        self._verified_focus: bool = False

    # -- runtime ---------------------------------------------------------

    def _get_runtime(self):
        if getattr(self, "_runtime", None) is not None:
            return self._runtime
        # Use the shared runtime broker (change: platynui-desktop-safety-
        # isolation) so the focus manager, ui_tree, and keyword execution all
        # share one Runtime and no path re-inits after shutdown.
        try:
            from robotmcp.plugins.builtin.platynui_plugin import get_runtime

            self._runtime = get_runtime()
        except Exception as exc:  # pragma: no cover - env dependent
            logger.debug("PlatynUI focus runtime unavailable: %s", exc)
            self._runtime = None
        return self._runtime

    def reset(self) -> None:
        # The runtime is broker-owned and process-shared — drop our reference
        # only; do NOT shut it down (that would break other call sites).
        self._runtime = None
        self._last_focused_scope = None
        self.invalidate_focus_cache()

    def invalidate_focus_cache(self) -> None:
        """Drop the verified-activation cache (e.g. after a desktop launch
        marked the tree dirty — window identities may have changed)."""
        try:
            self._focused_window_ids.clear()
        except AttributeError:
            self._focused_window_ids = set()

    # -- window resolution (D2) -----------------------------------------

    def resolve_window(self, descriptor: str) -> Optional[Any]:
        """Resolve the AUT top-level window node for a descriptor."""
        runtime = self._get_runtime()
        if runtime is None or not descriptor:
            return None
        # The focus manager's Runtime is long-lived and caches the desktop
        # tree; apps launched AFTER it was created (e.g. via Start Process in
        # the same session) are invisible until the cache is cleared
        # (ADR-025 gotcha). Clear before resolving so focus-before-act sees
        # freshly-launched AUT windows.
        clear = getattr(runtime, "clear_cache", None)
        if callable(clear):
            try:
                clear()
            except Exception:
                pass
        try:
            node = runtime.evaluate_single(descriptor)
        except Exception as exc:
            logger.debug("focus: descriptor did not resolve (%s): %s", descriptor, exc)
            return None
        if node is None:
            return None
        return self._top_level(node)

    @staticmethod
    def _top_level(node: Any) -> Optional[Any]:
        for attr in ("top_level_or_self", "top_level_pattern"):
            fn = getattr(node, attr, None)
            if callable(fn):
                try:
                    top = fn()
                    if top is not None:
                        return top
                except Exception:
                    continue
        return node

    # -- visibility (D4) -------------------------------------------------

    def window_visibility(self, window: Any) -> Tuple[Optional[bool], list]:
        """Return (is_visible, warnings) for a resolved window node."""
        warnings: list = []
        if window is None:
            return None, warnings
        is_visible = _attr_bool(window, "IsVisible")
        in_view = _attr_bool(window, "IsInView")
        bounds = _attr_rect(window, "Bounds")
        visible = True
        if is_visible is False:
            visible = False
            warnings.append("window IsVisible=false")
        if in_view is False:
            visible = False
            warnings.append("window IsInView=false")
        if bounds is not None:
            w, h = bounds[2], bounds[3]
            if w <= 0 or h <= 0:
                visible = False
                warnings.append("window has zero size")
            else:
                desktop = self._desktop_bounds()
                if desktop is not None and not _intersects(bounds, desktop):
                    visible = False
                    warnings.append("window is off-screen")
        if is_visible is None and in_view is None and bounds is None:
            return None, warnings
        return visible, warnings

    def _desktop_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        runtime = self._get_runtime()
        if runtime is None:
            return None
        try:
            info = runtime.desktop_info()
        except Exception:
            return None
        b = info.get("bounds") if isinstance(info, dict) else None
        if b is None:
            return None
        try:
            if isinstance(b, dict):
                return (float(b.get("x", 0)), float(b.get("y", 0)),
                        float(b.get("width", 0)), float(b.get("height", 0)))
            # Rect-like
            return (float(b.x()), float(b.y()), float(b.width()), float(b.height()))
        except Exception:
            return None

    # -- scope check (D5) ------------------------------------------------

    def target_in_window(self, descriptor: str, window: Any) -> Optional[bool]:
        """True when the resolved target's top-level is the AUT window."""
        runtime = self._get_runtime()
        if runtime is None or window is None or not descriptor:
            return None
        try:
            target = runtime.evaluate_single(descriptor)
        except Exception:
            return None
        if target is None:
            return None
        target_top = self._top_level(target)
        return _same_node(target_top, window)

    # -- focus (D1/D3) ---------------------------------------------------

    def focus_window(
        self, window: Any, scope: Optional[str]
    ) -> Tuple[bool, Optional[str], Optional[bool]]:
        """Raise + focus the window via upstream-first tiers.

        Returns ``(focused, strategy, input_ready)`` where ``input_ready``
        is the upstream ``WindowSurface.accepts_user_input()`` verdict
        (None when unavailable). Change: platynui-visible-safe-targeting.
        """
        if window is None:
            return False, None, None
        # Skip redundant raises when already the active window (D3 thrash).
        if self._is_active(window) is True:
            self._last_focused_scope = scope
            self._verified_focus = True
            return True, "already_active", None
        rid = str(getattr(window, "runtime_id", "") or "")
        cache = getattr(self, "_focused_window_ids", None)
        if rid and cache and rid in cache:
            return True, "cached", None
        runtime = self._get_runtime()
        # Tier 1: upstream bring_to_front — restore + activate + (on new-core
        # runtimes) poll accepts_user_input until ready. This is the verified
        # activation path; wait_ms is dropped for older runtimes.
        if runtime is not None:
            fn = getattr(runtime, "bring_to_front", None)
            if callable(fn):
                try:
                    try:
                        fn(window, wait_ms=BRING_TO_FRONT_WAIT_MS)
                    except TypeError:
                        fn(window)
                    self._last_focused_scope = scope
                    self._remember_focused(rid)
                    return True, "bring_to_front", self._accepts_user_input(window)
                except Exception as exc:
                    logger.debug("focus: runtime.bring_to_front failed: %s", exc)
        # Tier 1b: direct WindowSurface.activate() (upstream pattern action;
        # no readiness polling).
        ws = self._window_surface(window)
        if ws is not None:
            fn = getattr(ws, "activate", None)
            if callable(fn):
                try:
                    fn()
                    self._last_focused_scope = scope
                    self._remember_focused(rid)
                    return True, "window_surface:activate", self._accepts_user_input(window)
                except Exception as exc:
                    logger.debug("focus: WindowSurface.activate failed: %s", exc)
        # Tier 2: upstream Runtime.focus() (Focusable pattern / grab_focus).
        if runtime is not None:
            ffn = getattr(runtime, "focus", None)
            if callable(ffn):
                try:
                    ffn(window)
                    self._last_focused_scope = scope
                    self._remember_focused(rid)
                    return True, "focus", None
                except Exception as exc:
                    logger.debug("focus: runtime.focus failed: %s", exc)
        # Tier 3: LAST RESORT — portable ctypes X11 raise for environments
        # where every upstream focus path is genuinely unavailable (WM-less
        # Xvfb). Never silent: ensure_focused pairs this strategy with the
        # focus-unverifiable warning.
        if self._x11_raise(window):
            self._last_focused_scope = scope
            return True, "x11_raise", None
        return False, None, None

    def _remember_focused(self, rid: str) -> None:
        # Every upstream-mechanism success (bring_to_front / WindowSurface /
        # Runtime.focus) counts as verified focus for the blind type-at-focus
        # gate (D7); the ctypes x11_raise fallback does NOT reach here.
        self._verified_focus = True
        if not rid:
            return
        cache = getattr(self, "_focused_window_ids", None)
        if cache is None:
            cache = self._focused_window_ids = set()
        cache.add(rid)

    @property
    def has_verified_focus(self) -> bool:
        return bool(getattr(self, "_verified_focus", False))

    @staticmethod
    def _supported_patterns(window: Any) -> Optional[list]:
        """Upstream pattern introspection: list of advertised pattern names,
        or None when the runtime predates ``supported_patterns()``."""
        fn = getattr(window, "supported_patterns", None)
        if not callable(fn):
            return None
        try:
            patterns = fn()
        except Exception:
            return None
        if patterns is None:
            return None
        try:
            return [str(p) for p in patterns]
        except Exception:
            return None

    @staticmethod
    def _pattern_suffixes(patterns: Optional[list]) -> Optional[set]:
        """Normalize pattern names (short or reverse-DNS) to lowercase
        suffixes, e.g. 'org.platynui.patterns.Focusable' -> 'focusable'."""
        if patterns is None:
            return None
        return {
            str(p).rsplit(".", 1)[-1].strip().lower() for p in patterns if p
        }

    def highlight_target(self, descriptor: str, window: Any) -> None:
        """Mark the interaction target on screen via upstream
        ``Runtime.highlight()``. Strictly soft-fail: never raises, never
        blocks the step (change: platynui-visible-safe-targeting, task 3.1).
        """
        runtime = self._get_runtime()
        if runtime is None:
            return
        hl = getattr(runtime, "highlight", None)
        if not callable(hl):
            return
        try:
            node = None
            if descriptor:
                try:
                    node = runtime.evaluate_single(descriptor)
                except Exception:
                    node = None
            if node is None:
                node = window
            rect = _attr(node, "Bounds")
            if rect is None:
                return
            try:
                hl(rect, duration_ms=HIGHLIGHT_DURATION_MS)
            except TypeError:
                hl(rect)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("highlight skipped: %s", exc)

    def clear_highlight(self) -> None:
        """Remove any active highlight overlay (called before screenshots so
        evidence images show the app, not the marker; task 3.2)."""
        runtime = self._get_runtime()
        fn = getattr(runtime, "clear_highlight", None) if runtime else None
        if not callable(fn):
            return
        try:
            fn()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("clear_highlight skipped: %s", exc)

    def _accepts_user_input(self, window: Any) -> Optional[bool]:
        """Upstream WindowSurface.accepts_user_input() verdict, or None."""
        ws = self._window_surface(window)
        if ws is None:
            return None
        fn = getattr(ws, "accepts_user_input", None)
        if not callable(fn):
            return None
        try:
            val = fn()
        except Exception:
            return None
        if val is None:
            return None
        return bool(val)

    @staticmethod
    def _window_surface(window: Any) -> Optional[Any]:
        """Return the WindowSurface pattern of a window node, or None."""
        getter = getattr(window, "get_pattern", None)
        if not callable(getter):
            return None
        try:
            import platynui_native as pn

            ws_type = getattr(pn, "WindowSurface", None)
            if ws_type is None:
                return None
            return getter(ws_type)
        except Exception:
            return None

    @staticmethod
    def _window_pid(window: Any) -> Optional[int]:
        """ProcessId lives on the app ancestor, not the window node itself."""
        pid = _attr_int(window, "ProcessId")
        if pid is not None:
            return pid
        # Walk ancestors (app:Application carries ProcessId).
        for getter in ("ancestors_including_self", "ancestors"):
            fn = getattr(window, getter, None)
            if callable(fn):
                try:
                    for anc in fn():
                        pid = _attr_int(anc, "ProcessId")
                        if pid is not None:
                            return pid
                except Exception:
                    continue
        # Fallback: walk parent() chain.
        node = window
        for _ in range(8):
            parent_fn = getattr(node, "parent", None)
            if not callable(parent_fn):
                break
            try:
                node = parent_fn()
            except Exception:
                break
            if node is None:
                break
            pid = _attr_int(node, "ProcessId")
            if pid is not None:
                return pid
        return None

    def _is_active(self, window: Any) -> Optional[bool]:
        for attr in ("IsActive", "IsFocused"):
            val = _attr_bool(window, attr)
            if val is not None:
                return val
        return None

    def _x11_raise(self, window: Any) -> bool:
        """Raise + focus the window's X11 toplevel, matched by _NET_WM_PID.

        The portable fallback for WM-less / EWMH-less X servers where the
        PlatynUI WindowSurface activation is unavailable. Enumerates the X11
        toplevels (XQueryTree), matches by the window's accessible ProcessId
        against each toplevel's _NET_WM_PID, then raises + sets input focus
        so XTest input lands on the AUT.
        """
        if os.environ.get("DISPLAY") is None:
            return False
        pid = self._window_pid(window)
        if pid is None:
            return False
        return _x11_raise_by_pid(pid)

    # -- orchestration ---------------------------------------------------

    def ensure_focused(
        self,
        keyword: str,
        arguments: list,
        *,
        focus: bool = True,
        check_scope: bool = True,
        strict_scope: bool = False,
        fail_on_hidden: bool = False,
        aut_pid: Optional[int] = None,
        aut_sid: Optional[int] = None,
        highlight: bool = False,
    ) -> FocusOutcome:
        """Focus-before-act + visibility + scope for one interaction step.

        Returns a FocusOutcome; raises FocusError when a precondition is
        violated under a strict/fail-fast policy.
        """
        outcome = FocusOutcome()
        if not is_interaction_keyword(keyword):
            return outcome
        if not focus or focus_disabled_by_env():
            outcome.bypassed = True
            return outcome
        descriptor = extract_descriptor(keyword, arguments)
        if descriptor is None:
            # type-at-focus / no positional target — nothing to resolve.
            # D7: blind typing with NO verified AUT focus this session is
            # the silent-keystroke-loss shape from the 2026-06-11 rerun —
            # warn (the executor de-dups to once per session).
            if (
                normalize_keyword(keyword) in _KEYBOARD_KEYWORDS
                and not self.has_verified_focus
            ):
                outcome.warnings.append(UNFOCUSED_TYPING_WARNING)
            return outcome
        scope = app_scope_of(descriptor) or descriptor
        window = self.resolve_window(descriptor)
        if window is None:
            # Descriptor unresolved here; the real keyword will surface the
            # element-not-found error. Do not block.
            return outcome
        outcome.attempted = True

        # Upstream pattern introspection (I-2): what focus mechanisms does
        # the resolved window actually advertise?
        outcome.patterns = self._supported_patterns(window)

        # AUT process-identity scope check by LINEAGE, not bare pid equality
        # (change: desktop-aut-process-lineage). Run 5: a bash-wrapper launch
        # made every legitimate click warn (captured pid = bash, target =
        # daemonized soffice). Warn ONLY on a CONFIRMED foreign process.
        if aut_pid is not None:
            target_pid = self._window_pid(window)
            if target_pid is not None:
                related = pid_in_aut_lineage(
                    int(target_pid), int(aut_pid), aut_sid
                )
                if related is False:
                    target_sid = _read_pid_sid(int(target_pid))
                    outcome.warnings.append(
                        f"resolved target (PID {target_pid}, session "
                        f"{target_sid}) has no lineage relation to the "
                        f"launched AUT (PID {aut_pid}, session {aut_sid}) — "
                        f"commands may be going to a different application"
                    )

        # Visibility precondition (D4).
        visible, vis_warnings = self.window_visibility(window)
        outcome.visible = visible
        outcome.warnings.extend(vis_warnings)
        if visible is False:
            msg = "AUT window is not visible/on-screen: " + ", ".join(vis_warnings)
            if fail_on_hidden:
                outcome.error = msg
                raise FocusError(msg)

        # Scope check (D5).
        if check_scope:
            in_scope = self.target_in_window(descriptor, window)
            outcome.in_scope = in_scope
            if in_scope is False:
                msg = (
                    "resolved target is outside the AUT window "
                    "(cross-window collision): " + descriptor
                )
                outcome.warnings.append(msg)
                if strict_scope:
                    outcome.error = msg
                    raise FocusError(msg)

        # Visible-targeting: mark the element about to receive input on the
        # (visible) display before dispatch (task 3.1). Soft-fail.
        if highlight and not highlight_disabled_by_env():
            self.highlight_target(descriptor, window)

        # Focus (D1/D3) — upstream-first tiers.
        focused, strategy, input_ready = self.focus_window(window, scope)
        outcome.focused = focused
        outcome.strategy = strategy
        outcome.input_ready = input_ready
        if not focused:
            outcome.warnings.append("could not raise/focus the AUT window")

        # I-2: focus verifiability. Verified means the platform either
        # reports the window active or upstream accepts_user_input()
        # confirmed readiness; anything weaker gets an explicit warning so
        # silent keystroke loss (LibreOffice run, 2026-06-11) cannot recur.
        suffixes = self._pattern_suffixes(outcome.patterns)
        reason: Optional[str] = None
        if suffixes is not None and not ({"windowsurface", "focusable"} & suffixes):
            reason = "no WindowSurface/Focusable pattern"
        elif strategy == "x11_raise":
            reason = "ctypes X11 raise fallback used"
        elif strategy in ("already_active", "cached"):
            reason = None
        elif input_ready is True:
            reason = None
        elif suffixes is None:
            reason = "runtime does not expose pattern introspection"
        elif input_ready is False:
            reason = "window is not accepting user input"
        elif strategy is not None:
            reason = "input readiness could not be confirmed"
        else:
            reason = "no focus mechanism succeeded"
        if reason is not None:
            outcome.warnings.append(f"{FOCUS_UNVERIFIABLE_PREFIX} ({reason})")
        return outcome


class FocusError(Exception):
    """Raised when a focus/visibility/scope precondition fails fast."""


# --- attribute helpers ------------------------------------------------------


def _attr(node: Any, name: str) -> Any:
    try:
        return node.attribute(name)
    except Exception:
        return None


def _attr_bool(node: Any, name: str) -> Optional[bool]:
    val = _attr(node, name)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        low = val.strip().lower()
        if low in {"true", "1", "yes"}:
            return True
        if low in {"false", "0", "no"}:
            return False
    return None


def _attr_int(node: Any, name: str) -> Optional[int]:
    val = _attr(node, name)
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def _attr_rect(node: Any, name: str) -> Optional[Tuple[float, float, float, float]]:
    val = _attr(node, name)
    if val is None:
        return None
    for getters in (("x", "y", "width", "height"),):
        try:
            return (
                float(val.x()), float(val.y()),
                float(val.width()), float(val.height()),
            )
        except Exception:
            break
    if isinstance(val, dict):
        try:
            return (float(val.get("x", 0)), float(val.get("y", 0)),
                    float(val.get("width", 0)), float(val.get("height", 0)))
        except Exception:
            return None
    return None


def _intersects(a: Tuple[float, float, float, float],
                b: Tuple[float, float, float, float]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def _x11_raise_by_pid(pid: int) -> bool:
    """Raise + focus the top-level X11 window whose _NET_WM_PID == pid.

    Pure ctypes (no python-xlib / xdotool dependency). Best-effort: returns
    True only when a matching toplevel was found and raised.
    """
    import ctypes
    import ctypes.util

    libname = ctypes.util.find_library("X11")
    if not libname:
        return False
    try:
        x11 = ctypes.CDLL(libname)
    except OSError:
        return False

    # Minimal Xlib signatures.
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XDefaultRootWindow.restype = ctypes.c_ulong
    x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    x11.XInternAtom.restype = ctypes.c_ulong
    x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    x11.XQueryTree.restype = ctypes.c_int
    x11.XQueryTree.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)), ctypes.POINTER(ctypes.c_uint),
    ]
    x11.XGetWindowProperty.restype = ctypes.c_int
    x11.XGetWindowProperty.argtypes = [
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_long, ctypes.c_long,
        ctypes.c_int, ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
    ]
    x11.XFree.argtypes = [ctypes.c_void_p]
    x11.XRaiseWindow.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    x11.XFlush.argtypes = [ctypes.c_void_p]

    display = x11.XOpenDisplay(os.environ["DISPLAY"].encode())
    if not display:
        return False
    try:
        net_wm_pid = x11.XInternAtom(display, b"_NET_WM_PID", False)
        root = x11.XDefaultRootWindow(display)

        def _window_pid(win: int) -> Optional[int]:
            actual_type = ctypes.c_ulong()
            actual_format = ctypes.c_int()
            nitems = ctypes.c_ulong()
            bytes_after = ctypes.c_ulong()
            prop = ctypes.POINTER(ctypes.c_ubyte)()
            status = x11.XGetWindowProperty(
                display, win, net_wm_pid, 0, 1, False, 0,  # AnyPropertyType
                ctypes.byref(actual_type), ctypes.byref(actual_format),
                ctypes.byref(nitems), ctypes.byref(bytes_after), ctypes.byref(prop),
            )
            if status != 0 or not prop or nitems.value == 0:
                if prop:
                    x11.XFree(prop)
                return None
            value = ctypes.cast(prop, ctypes.POINTER(ctypes.c_ulong))[0]
            x11.XFree(prop)
            return int(value)

        def _enumerate(parent: int, depth: int = 0):
            if depth > 3:
                return
            root_r = ctypes.c_ulong()
            parent_r = ctypes.c_ulong()
            children = ctypes.POINTER(ctypes.c_ulong)()
            n = ctypes.c_uint()
            if not x11.XQueryTree(display, parent, ctypes.byref(root_r),
                                  ctypes.byref(parent_r), ctypes.byref(children),
                                  ctypes.byref(n)):
                return
            try:
                for i in range(n.value):
                    win = children[i]
                    if _window_pid(win) == pid:
                        yield win
                    else:
                        yield from _enumerate(win, depth + 1)
            finally:
                if children:
                    x11.XFree(children)

        match = next(_enumerate(root), None)
        if match is None:
            return False
        x11.XRaiseWindow(display, match)
        x11.XFlush(display)
        return True
    except Exception as exc:  # pragma: no cover - env dependent
        logger.debug("focus: x11 raise-by-pid failed: %s", exc)
        return False
    finally:
        x11.XCloseDisplay(display)


# Pure-ctypes EWMH window probe, executed in an ISOLATED SUBPROCESS so its
# Xlib connection cannot conflict with (and crash) the PlatynUI runtime's own
# Xlib connection in the server process (Xlib is not thread-safe — a second
# in-process connection segfaults when the native runtime is live). Reads
# sys.argv[1:] as name substrings; optional PROBE_PID env. Prints exactly one
# of: present / absent / unknown. change: desktop-native-platynui-alignment.
_X11_WINDOW_PROBE_SRC = r"""
import os, sys, ctypes, ctypes.util
def out(s):
    sys.stdout.write(s); sys.stdout.flush(); os._exit(0)
names = [a.lower() for a in sys.argv[1:] if a]
pid = os.environ.get("PROBE_PID")
pid = int(pid) if pid and pid.isdigit() else None
if pid is None and not names:
    out("unknown")
lib = ctypes.util.find_library("X11")
disp_name = os.environ.get("DISPLAY")
if not lib or not disp_name:
    out("unknown")
try:
    x = ctypes.CDLL(lib)
except OSError:
    out("unknown")
x.XOpenDisplay.restype = ctypes.c_void_p
x.XOpenDisplay.argtypes = [ctypes.c_char_p]
x.XDefaultRootWindow.restype = ctypes.c_ulong
x.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
x.XInternAtom.restype = ctypes.c_ulong
x.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
x.XQueryTree.restype = ctypes.c_int
x.XQueryTree.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
    ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)), ctypes.POINTER(ctypes.c_uint)]
x.XGetWindowProperty.restype = ctypes.c_int
x.XGetWindowProperty.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong,
    ctypes.c_long, ctypes.c_long, ctypes.c_int, ctypes.c_ulong,
    ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte))]
x.XFree.argtypes = [ctypes.c_void_p]
d = x.XOpenDisplay(disp_name.encode())
if not d:
    out("unknown")
a_pid = x.XInternAtom(d, b"_NET_WM_PID", False)
atoms = [x.XInternAtom(d, b"WM_CLASS", False), x.XInternAtom(d, b"_NET_WM_NAME", False),
         x.XInternAtom(d, b"WM_NAME", False)]
def prop(win, atom):
    at=ctypes.c_ulong(); af=ctypes.c_int(); ni=ctypes.c_ulong(); ba=ctypes.c_ulong()
    p=ctypes.POINTER(ctypes.c_ubyte)()
    s=x.XGetWindowProperty(d,win,atom,0,256,False,0,ctypes.byref(at),ctypes.byref(af),
        ctypes.byref(ni),ctypes.byref(ba),ctypes.byref(p))
    if s!=0 or not p or ni.value==0:
        if p: x.XFree(p)
        return None
    try:
        nbytes = ni.value * (af.value//8 if af.value else 1)
        return bytes(ctypes.cast(p, ctypes.POINTER(ctypes.c_ubyte*nbytes)).contents)
    finally:
        x.XFree(p)
def matches(win):
    if pid is not None:
        raw = prop(win, a_pid)
        if raw and len(raw) >= 4 and int.from_bytes(raw[:4],"little")==pid:
            return True
    if names:
        for atom in atoms:
            raw = prop(win, atom)
            if not raw: continue
            t = raw.replace(b"\x00",b" ").decode("utf-8","ignore").lower()
            if any(n in t for n in names):
                return True
    return False
def walk(parent, depth=0):
    if depth>3: return False
    rr=ctypes.c_ulong(); pr=ctypes.c_ulong(); ch=ctypes.POINTER(ctypes.c_ulong)(); n=ctypes.c_uint()
    if not x.XQueryTree(d,parent,ctypes.byref(rr),ctypes.byref(pr),ctypes.byref(ch),ctypes.byref(n)):
        return False
    try:
        for i in range(n.value):
            w=ch[i]
            if matches(w) or walk(w, depth+1):
                return True
    finally:
        if ch: x.XFree(ch)
    return False
out("present" if walk(x.XDefaultRootWindow(d)) else "absent")
"""


def x11_window_present(
    app_names: Optional[list] = None, pid: Optional[int] = None
) -> str:
    """Tri-state ``"present"`` / ``"absent"`` / ``"unknown"`` — does an X11
    toplevel window matching ``pid`` (``_NET_WM_PID``) or any of ``app_names``
    (``WM_CLASS`` / ``_NET_WM_NAME`` / ``WM_NAME``, case-insensitive substring)
    exist on the bound ``DISPLAY``?

    GUARDED FALLBACK (change: desktop-native-platynui-alignment). PlatynUI's
    native API/CLI exposes NO window list independent of the AT-SPI tree —
    ``platynui-cli window`` evaluates ``//control:Window`` through the AT-SPI
    tree, and ``_NET_CLIENT_LIST`` is internal-only (spike, 2026-06-10). This
    EWMH probe is the documented fallback for that missing native capability.

    Runs the ctypes probe in an ISOLATED SUBPROCESS: a second in-process Xlib
    connection conflicts with the PlatynUI runtime's own connection (Xlib is not
    thread-safe) and can segfault the server. Best-effort; returns ``"unknown"``
    (never raises) on any failure/timeout/unavailable display.
    """
    import subprocess
    import sys

    names = [str(n) for n in (app_names or []) if n]
    if pid is None and not names:
        return "unknown"
    if "DISPLAY" not in os.environ:
        return "unknown"
    env = dict(os.environ)
    if pid is not None:
        env["PROBE_PID"] = str(int(pid))
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _X11_WINDOW_PROBE_SRC, *names],
            capture_output=True, text=True, timeout=5, env=env,
        )
        out = (proc.stdout or "").strip()
        return out if out in ("present", "absent", "unknown") else "unknown"
    except Exception as exc:  # pragma: no cover - env dependent
        logger.debug("x11_window_present subprocess failed: %s", exc)
        return "unknown"


# Batched _NET_WM_PID enumeration for display scoping (change:
# desktop-evidence-and-display-scoping, D4). One subprocess lists ALL pids
# owning an X client window on the bound DISPLAY, so ui_tree can drop
# host-desktop applications from an isolated session's tree. Same
# subprocess-isolation rationale as _X11_WINDOW_PROBE_SRC (Xlib is not
# thread-safe next to the live PlatynUI runtime). Prints "unknown",
# "none", or space-separated pids.
_X11_PID_LIST_PROBE_SRC = r"""
import os, sys, ctypes, ctypes.util
def out(s):
    sys.stdout.write(s); sys.stdout.flush(); os._exit(0)
lib = ctypes.util.find_library("X11")
disp_name = os.environ.get("DISPLAY")
if not lib or not disp_name:
    out("unknown")
try:
    x = ctypes.CDLL(lib)
except OSError:
    out("unknown")
x.XOpenDisplay.restype = ctypes.c_void_p
x.XOpenDisplay.argtypes = [ctypes.c_char_p]
x.XDefaultRootWindow.restype = ctypes.c_ulong
x.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
x.XInternAtom.restype = ctypes.c_ulong
x.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
x.XQueryTree.restype = ctypes.c_int
x.XQueryTree.argtypes = [ctypes.c_void_p, ctypes.c_ulong,
    ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_ulong)), ctypes.POINTER(ctypes.c_uint)]
x.XGetWindowProperty.restype = ctypes.c_int
x.XGetWindowProperty.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong,
    ctypes.c_long, ctypes.c_long, ctypes.c_int, ctypes.c_ulong,
    ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_int),
    ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong),
    ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte))]
x.XFree.argtypes = [ctypes.c_void_p]
d = x.XOpenDisplay(disp_name.encode())
if not d:
    out("unknown")
a_pid = x.XInternAtom(d, b"_NET_WM_PID", False)
pids = set()
def win_pid(win):
    at=ctypes.c_ulong(); af=ctypes.c_int(); ni=ctypes.c_ulong(); ba=ctypes.c_ulong()
    p=ctypes.POINTER(ctypes.c_ubyte)()
    s=x.XGetWindowProperty(d,win,a_pid,0,1,False,0,ctypes.byref(at),ctypes.byref(af),
        ctypes.byref(ni),ctypes.byref(ba),ctypes.byref(p))
    if s!=0 or not p or ni.value==0:
        if p: x.XFree(p)
        return None
    v = ctypes.cast(p, ctypes.POINTER(ctypes.c_ulong))[0]
    x.XFree(p)
    return int(v)
def walk(parent, depth=0):
    if depth>3: return
    rr=ctypes.c_ulong(); pr=ctypes.c_ulong(); ch=ctypes.POINTER(ctypes.c_ulong)(); n=ctypes.c_uint()
    if not x.XQueryTree(d,parent,ctypes.byref(rr),ctypes.byref(pr),ctypes.byref(ch),ctypes.byref(n)):
        return
    try:
        for i in range(n.value):
            w=ch[i]
            p=win_pid(w)
            if p is not None: pids.add(p)
            walk(w, depth+1)
    finally:
        if ch: x.XFree(ch)
walk(x.XDefaultRootWindow(d))
out(" ".join(str(p) for p in sorted(pids)) if pids else "none")
"""


def x11_display_pids() -> Optional[frozenset]:
    """All pids owning an X client window (``_NET_WM_PID``) on the bound
    DISPLAY, or None when the probe could not complete. Empty frozenset is a
    valid result (bare display). Subprocess-isolated; never raises."""
    import subprocess
    import sys

    if "DISPLAY" not in os.environ:
        return None
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _X11_PID_LIST_PROBE_SRC],
            capture_output=True, text=True, timeout=5, env=dict(os.environ),
        )
        out = (proc.stdout or "").strip()
        if not out or out == "unknown":
            return None
        if out == "none":
            return frozenset()
        return frozenset(int(tok) for tok in out.split())
    except Exception as exc:  # pragma: no cover - env dependent
        logger.debug("x11_display_pids subprocess failed: %s", exc)
        return None


# --- AUT process lineage (change: desktop-aut-process-lineage) --------------


def _read_pid_ppid(pid: int) -> Optional[int]:
    """Parent pid via /proc/<pid>/status PPid, or None when unreadable."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("PPid:"):
                    return int(line.split()[1])
    except Exception:
        return None
    return None


def _read_pid_sid(pid: int) -> Optional[int]:
    """Session id of a pid, or None when unreadable/exited."""
    try:
        return os.getsid(pid)
    except Exception:
        return None


def pid_in_aut_lineage(
    target_pid: int,
    aut_pid: int,
    aut_sid: Optional[int],
    *,
    _ppid=None,
    _sid=None,
) -> Optional[bool]:
    """Is ``target_pid`` part of the launched AUT's process lineage?

    Three tiers (change: desktop-aut-process-lineage):
    1. pid identity;
    2. the target's parent chain reaches ``aut_pid`` (live wrapper parents);
    3. the target's session id matches ``aut_sid`` — the signal that
       survives daemonization (oosplash re-parents soffice to init) and
       single-instance handoff, because everything this server session
       spawns shares its sid unless something calls setsid.

    Returns True (related), False (CONFIRMED foreign: both signals
    resolved and negative), or None (indeterminate — callers must not
    warn). Readers injectable for tests.
    """
    ppid_of = _ppid if _ppid is not None else _read_pid_ppid
    sid_of = _sid if _sid is not None else _read_pid_sid

    if target_pid == aut_pid:
        return True

    # Tier 2: ancestor walk, bounded; stop at init/kernel.
    ancestors_resolved = False
    node = target_pid
    for _ in range(15):
        parent = ppid_of(node)
        if parent is None:
            break
        ancestors_resolved = True
        if parent == aut_pid:
            return True
        if parent <= 1:
            break
        node = parent

    # Tier 3: session-id match.
    target_sid = sid_of(target_pid)
    if aut_sid is not None and target_sid is not None:
        if target_sid == aut_sid:
            return True
        # Both signals resolved, no relation on any tier -> confirmed foreign.
        return False
    # SID comparison unavailable (no recorded aut_sid, or target's sid
    # unreadable): the launcher may simply be dead and the target
    # re-parented — indeterminate, never a confirmed-foreign verdict.
    del ancestors_resolved  # tier 2 alone can never confirm foreignness
    return None


def _same_node(a: Any, b: Any) -> Optional[bool]:
    if a is None or b is None:
        return None
    for attr in ("runtime_id",):
        ra, rb = getattr(a, attr, None), getattr(b, attr, None)
        if ra is not None and rb is not None:
            return str(ra) == str(rb)
    try:
        return bool(a == b)
    except Exception:
        return None
