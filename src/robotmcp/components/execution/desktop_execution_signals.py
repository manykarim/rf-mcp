"""Desktop execution disagreement signals.

Change: desktop-stepwise-execution-fidelity (maintainer-report findings #2, #4).

Two soft signals that stop an API "success" from being mistaken for "the AUT
reacted":

- ``launch_liveness_hint`` — a desktop ``Start Process`` returned a handle but
  the process is not actually running shortly after, while PlatynUI still
  observes application nodes (possibly a different/stale instance).
- ``input_effect_hint`` — a desktop pointer/keyboard keyword returned success
  but the target's accessible display state did not change.

Both are PURE decision functions (no I/O) so the wiring can stay best-effort and
the contract is unit-testable without a live desktop.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Keyword (basename, lowercased) sets the signals apply to.
_LAUNCH_KEYWORDS = frozenset({"start process", "run process"})
_INTERACTION_KEYWORDS = frozenset({
    "pointer click",
    "pointer multi click",
    "keyboard type",
    "keyboard press",
    "keyboard release",
    "click",
    "type text",
})


# Desktop keywords that RESOLVE a descriptor against the accessibility tree —
# the ones that must see a fresh tree after a launch (change:
# desktop-tree-cache-refresh). Includes discovery/assertion reads and the
# interaction keywords (which resolve a locator before acting).
_TREE_RESOLVING_KEYWORDS = frozenset(
    {"query", "get attribute", "set root"} | set(_INTERACTION_KEYWORDS)
)


def _basename(keyword: str) -> str:
    return (keyword or "").strip().lower().rsplit(".", 1)[-1]


def is_launch_keyword(keyword: str) -> bool:
    return _basename(keyword) in _LAUNCH_KEYWORDS


def is_interaction_keyword(keyword: str) -> bool:
    return _basename(keyword) in _INTERACTION_KEYWORDS


def is_tree_resolving_keyword(keyword: str) -> bool:
    """True for a desktop keyword that resolves a descriptor against the
    accessibility tree (so it must see a fresh tree after a launch)."""
    return _basename(keyword) in _TREE_RESOLVING_KEYWORDS


# --- Unscoped-locator guardrail (change: desktop-unscoped-locator-guardrail).
# A leading // is ABSOLUTE XPath: it ignores Set Root/context and re-walks the
# whole session AT-SPI tree (every desktop app), which on a busy desktop takes
# tens of seconds and exceeds the MCP client's request timeout — killing the
# transport. Detection is a pure function so it is trivially unit-testable.

# Keywords whose first string argument is an XPath the guardrail inspects.
_XPATH_KEYWORDS = frozenset({"query", "evaluate"})

# Outer scalar-aggregate wrappers: these return ONE value, not a node set, so
# they cannot trigger the large-response transport death — the count(//...)
# discovery step the locator guidance explicitly recommends stays allowed.
_SCALAR_AGGREGATE_PREFIXES = ("count(", "string(", "number(", "boolean(")


def is_query_keyword(keyword: str) -> bool:
    """True for desktop keywords whose first argument is an XPath the
    unscoped-locator guardrail should inspect (Query / Evaluate)."""
    return _basename(keyword) in _XPATH_KEYWORDS


def is_unscoped_locator(xpath: Any) -> bool:
    """True when an XPath would walk the whole session tree.

    Unscoped: starts with ``//`` or ``descendant-or-self::``, or is a bare
    wildcard walk (``*`` / ``//*``). NOT unscoped (allowed): ``/app:``-anchored
    or any other explicit absolute path, relative locators (``control:``,
    ``item:``, ``.``, an axis like ``child::``), and pure scalar-aggregate
    expressions (``count(...)`` etc.). Conservative — anything ambiguous is
    treated as allowed (fail-open toward running).
    """
    if not isinstance(xpath, str):
        return False
    expr = xpath.strip()
    if not expr:
        return False
    low = expr.lower()
    # Pure scalar-aggregate wrapper -> returns a number/string, not nodes.
    if low.startswith(_SCALAR_AGGREGATE_PREFIXES):
        return False
    if expr.startswith("//"):
        return True
    if low.startswith("descendant-or-self::") or low.startswith("descendant::"):
        return True
    if expr in ("*", "//*"):
        return True
    return False


def launch_liveness_hint(
    *,
    process_running: Optional[bool],
    discovery_node_count: Optional[int],
) -> Optional[Dict[str, Any]]:
    """Return a warning hint when a launched desktop process is NOT running
    even though discovery still sees application nodes.

    Returns ``None`` when there is no disagreement to report (the process is
    running, liveness is unknown, or there are no discovery nodes to mislead).
    """
    if process_running is not False:
        # Running, or unknown — no contradiction to surface.
        return None
    nodes = discovery_node_count or 0
    msg = (
        "The launched process is not running, but PlatynUI still observes "
        f"{nodes} application node(s). The visible nodes may be a different or "
        "stale instance — verify the AUT actually started (a snap-confined or "
        "missing binary can exit non-zero while another window remains)."
        if nodes
        else "The launched process is not running shortly after a successful "
        "Start Process — verify the AUT actually started."
    )
    return {"type": "desktop_launch_not_running", "message": msg}


def wayland_input_warning(keyword: str) -> Optional[Dict[str, Any]]:
    """Return a warning when a desktop INTERACTION keyword runs on a session
    that originated as Wayland (and was forced to X11): synthetic X11 (XTest)
    input is likely blocked by the Wayland compositor, so a keyword "success"
    may not have reached the application.

    change: desktop-input-and-runtime-diagnostics. Returns None for non-
    interaction keywords or non-Wayland sessions. Never raises.
    """
    if not is_interaction_keyword(keyword):
        return None
    try:
        from robotmcp.plugins.builtin.platynui_plugin import was_wayland_session

        if not was_wayland_session():
            return None
    except Exception:  # pragma: no cover - defensive
        return None
    return {
        "type": "wayland_x11_input_blocked_risk",
        "message": (
            "This desktop session originated as Wayland; rf-mcp injects input "
            "via X11 (XTest), which a Wayland compositor (e.g. mutter) typically "
            "BLOCKS for security. A pointer/keyboard keyword may report success "
            "without the input reaching the application. Read/query operations "
            "(AT-SPI over D-Bus) are unaffected."
        ),
        "remediation": [
            "Run the automation on a real X11 login session (not Wayland), or",
            "use PlatynUI's Wayland input backend (libei/eis) when available.",
        ],
    }


def input_effect_hint(
    *,
    keyword: str,
    success: bool,
    state_before: Any,
    state_after: Any,
) -> Optional[Dict[str, Any]]:
    """Return a warning hint when a successful desktop interaction left the
    target's accessible display state unchanged.

    ``state_before``/``state_after`` are opaque comparable snapshots (e.g. a
    ``native:Text.CharacterCount`` value or a tuple of them). Returns ``None``
    when the keyword is not an interaction, the step did not succeed, a snapshot
    is missing, or the state changed.
    """
    if not success or not is_interaction_keyword(keyword):
        return None
    if state_before is None or state_after is None:
        return None
    if state_before != state_after:
        return None
    return {
        "type": "desktop_input_no_effect",
        "message": (
            f"'{keyword}' returned success but the target's accessible display "
            "state did not change. Success is not evidence the application "
            "reacted — check window focus, duplicate application roots, or "
            "whether the click targeted the interactable node."
        ),
    }


# --- Close liveness (change: desktop-test-scoping-and-close-lifecycle, D5).
# --- Steering-confidence verdict (change: desktop-steering-confidence-gate).
# Composes the already-computed landing signals (verified-focus, input-effect,
# Wayland-drop risk) into ONE machine-parseable verdict, so a "success" that
# never reached the application is caught instead of trusted. Pure function.

import os as _os

_STEERING_CONFIDENCE_ENV = "ROBOTMCP_PLATYNUI_STEERING_CONFIDENCE"

# Verdict values.
SC_CONFIRMED = "confirmed"
SC_UNCONFIRMED = "unconfirmed"
SC_CONTRADICTED = "contradicted"


def steering_confidence_mode(environ: Optional[Dict[str, str]] = None) -> str:
    """Return "warn" when enforcement is opted down, else "enforce" (default).

    Mirrors ``desktop_display_safety.warn_mode`` — ``=warn`` downgrades a
    ``contradicted`` verdict from a step failure to an attached warning.
    """
    env = environ if environ is not None else _os.environ
    return "warn" if env.get(_STEERING_CONFIDENCE_ENV, "").strip().lower() == "warn" else "enforce"


def steering_confidence(
    *,
    keyword: str,
    success: bool,
    verified_focus: Optional[bool],
    state_before: Any,
    state_after: Any,
    wayland_risk: bool = False,
) -> Optional[Dict[str, Any]]:
    """Compose a steering-confidence verdict for a desktop interaction step.

    Returns ``None`` for non-interaction keywords or a non-success step (no
    landing question to answer). Otherwise a dict:
    ``{"verdict", "type", "message", "signals"}`` where ``verdict`` is one of
    ``confirmed`` / ``unconfirmed`` / ``contradicted``:

    - ``confirmed``    — focus was verified OR the target's accessible state
      changed (positive evidence the input landed).
    - ``contradicted`` — success reported but focus was NOT verified AND the
      state did not change (input-effect absent), or a Wayland input-drop risk
      applies to an unverified target. The provable "passed but did not touch
      the app" case.
    - ``unconfirmed``  — no positive or negative evidence (e.g. a widget with no
      readable state and focus not positively verified).

    Pure decision function — no I/O — so the caller wiring stays best-effort and
    the contract is unit-testable without a live desktop.
    """
    if not success or not is_interaction_keyword(keyword):
        return None

    vf = verified_focus is True
    effect_known = state_before is not None and state_after is not None
    effect_observed = effect_known and state_before != state_after
    effect_absent = effect_known and state_before == state_after

    if vf or effect_observed:
        verdict = SC_CONFIRMED
        msg = (
            "Input landing is confirmed "
            + ("(window focus verified)" if vf else "(target state changed)")
            + "."
        )
    elif effect_absent or wayland_risk:
        verdict = SC_CONTRADICTED
        why = (
            "the target's accessible state did not change"
            if effect_absent
            else "synthetic X11 input is likely dropped by the Wayland compositor"
        )
        msg = (
            f"'{keyword}' returned success but the input did not demonstrably "
            f"reach the application ({why}, and window focus was not verified). "
            "Re-verify focus (Activate Window on the app-scoped Frame) and retry."
        )
    else:
        verdict = SC_UNCONFIRMED
        msg = (
            f"'{keyword}' succeeded but the effect could not be confirmed "
            "(no readable target state and focus not positively verified)."
        )

    return {
        "type": "desktop_steering_confidence",
        "verdict": verdict,
        "message": msg,
        "signals": {
            "verified_focus": vf,
            "effect_observed": effect_observed,
            "effect_absent": effect_absent,
            "wayland_risk": bool(wayland_risk),
        },
    }


# Run 3: the LibreOffice document window closed but the process survived as
# a start-center frame with no signal — the agent burned retries on Alt+F4
# loops. Mirrors launch_liveness_hint. -------------------------------------

_CLOSE_KEYWORDS = frozenset({"close window"})


def is_close_keyword(keyword: str) -> bool:
    return _basename(keyword) in _CLOSE_KEYWORDS


def close_liveness_hint(process_alive: Optional[bool]) -> Optional[Dict[str, Any]]:
    """Warning hint when the AUT process survived a window close. Pure:
    the caller supplies the liveness verdict (None = unknown -> no hint)."""
    if process_alive is not True:
        return None
    return {
        "type": "desktop_close_liveness",
        "message": (
            "The window closed but the application process is still "
            "running — a residual frame (e.g. a start center) may remain "
            "and the app will not exit on its own. Close the remaining "
            "frame, or use Process.Terminate Process for a hard stop."
        ),
    }


# --- Screenshot evidence integrity (change: desktop-evidence-and-display-
# scoping, D2). The 2026-06-11 LibreOffice rerun produced five "successful"
# screenshots that exist nowhere on disk — evidence claims must be backed by
# a file or carry a warning. ------------------------------------------------

_SCREENSHOT_KEYWORDS = frozenset({"take screenshot"})
_SCREENSHOT_EXTENSIONS = (".png", ".jpg", ".jpeg")


def is_screenshot_keyword(keyword: str) -> bool:
    return _basename(keyword) in _SCREENSHOT_KEYWORDS


def screenshot_request_path(keyword: str, arguments: Any) -> Optional[str]:
    """Extract the requested screenshot file path from the arguments, or None.

    Recognizes ``filename=<path>`` named form and bare path-looking
    positionals (image extension). ``EMBED`` and templated ``{index}``
    filenames yield None (not verifiable without the output dir). Pure.
    """
    if not is_screenshot_keyword(keyword):
        return None
    for arg in arguments or []:
        if not isinstance(arg, str):
            continue
        candidate = arg
        if arg.startswith("filename="):
            candidate = arg.split("=", 1)[1]
        if candidate == "EMBED" or "{index}" in candidate:
            return None
        if candidate.lower().endswith(_SCREENSHOT_EXTENSIONS):
            return candidate
    return None


def screenshot_path_in_descriptor_slot(keyword: str, arguments: Any) -> Optional[str]:
    """Return the offending path when a ``Take Screenshot`` DESCRIPTOR slot holds
    a bare image path (the filename-as-descriptor trap), else None.

    PlatynUI ``Take Screenshot(descriptor, filename, rect)`` — a filename-only
    first positional binds to ``descriptor`` and triggers a 30s descriptor
    resolution before ElementNotFound. The descriptor slot is filled by an
    explicit ``descriptor=<val>`` or the first POSITIONAL argument; ``filename=``
    /``rect=`` named args and a correct ``(descriptor, path)`` two-positional
    call are NOT misuse. Pure — no I/O.
    """
    if not is_screenshot_keyword(keyword):
        return None
    descriptor_val: Any = None
    for arg in arguments or []:
        if isinstance(arg, str) and "=" in arg and not arg.startswith("descriptor="):
            continue  # a named arg (filename=, rect=) — not the descriptor slot
        if isinstance(arg, str) and arg.startswith("descriptor="):
            descriptor_val = arg.split("=", 1)[1]
        else:
            descriptor_val = arg  # first positional -> descriptor slot
        break
    if not isinstance(descriptor_val, str):
        return None  # a real descriptor object (or none) — fine
    if descriptor_val == "EMBED" or "{index}" in descriptor_val:
        return None
    if descriptor_val.lower().endswith(_SCREENSHOT_EXTENSIONS):
        return descriptor_val
    return None


def control_window_locator(keyword: str, arguments: Any) -> Optional[str]:
    """Return the offending locator when a desktop tree-resolving keyword's
    argument uses the ``control:Window`` role, else None.

    On AT-SPI (Linux) windows are ``control:Frame``; ``control:Window`` matches
    nothing and hangs 30s. Covers tree-resolving keywords (Query/Get Attribute/
    Set Root/pointer+keyboard) and Evaluate. Pure — platform gating is the
    caller's job.
    """
    if not (is_tree_resolving_keyword(keyword) or is_query_keyword(keyword)):
        return None
    for arg in arguments or []:
        if isinstance(arg, str) and "control:window" in arg.lower():
            return arg
    return None


def evidence_missing_hint(
    keyword: str,
    arguments: Any,
    result_value: Any,
    *,
    _isfile=None,
) -> Optional[Dict[str, Any]]:
    """Warning hint when a successful screenshot step left no file on disk.

    The keyword's RETURN value is preferred over the request argument (RF's
    ``Screenshot`` library returns the saved path); only absolute paths are
    verifiable. Never raises; ``_isfile`` is injectable for tests.
    """
    import os as _os

    isfile = _isfile if _isfile is not None else _os.path.isfile
    if not is_screenshot_keyword(keyword):
        return None
    path: Optional[str] = None
    if (
        isinstance(result_value, str)
        and result_value.lower().endswith(_SCREENSHOT_EXTENSIONS)
        and _os.path.isabs(result_value)
    ):
        path = result_value
    if path is None:
        requested = screenshot_request_path(keyword, arguments)
        if requested and _os.path.isabs(requested):
            path = requested
    if path is None:
        return None
    try:
        if isfile(path):
            return None
    except Exception:
        return None
    return {
        "type": "evidence_missing",
        "message": (
            f"'{keyword}' reported success but '{path}' does not exist on "
            "disk — the screenshot backend likely failed silently. Treat "
            "this step's evidence as absent."
        ),
    }
