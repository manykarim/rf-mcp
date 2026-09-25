"""Input guards for ``intent_action`` that make weak models recoverable.

WHY THIS EXISTS
---------------
Evidence, from the weekly E2E run 36120985462 (MiniMax-M2.7, 23 consecutive
``intent_action`` calls):

    {'intent': 'navigate', 'target': None, 'session_id': None}
    {'intent': 'click',    'target': None, 'session_id': None}
    {'intent': 'fill',     'target': None, 'value': None, 'session_id': None}

and, in the model's own reasoning trace:

    "I'm now realizing I've been passing the literal string "null" (with quotes)
     instead of the JSON null value."
    "The issue seems to be that intent_action doesn't properly accept or use the
     session_id parameter when passed as null - it resolves to the 'default'
     session instead of the session created via analyze_scenario."

The model had created a session and imported Browser into it. Three separate
properties of the tool surface combined to make that unrecoverable:

1. A literal ``"null"`` / ``"None"`` string is TRUTHY, so ``session_id or "default"``
   kept it and looked up a session that cannot exist; and ``target="None"`` passed
   the ``requires_target`` check and was treated as a real locator.

2. A missing ``session_id`` silently became ``"default"`` - a session with no
   libraries - rather than the session the caller had just set up.

3. The ``requires_target`` check runs AFTER library resolution, so when the session
   was also wrong the caller got

       "Cannot determine target library for intent 'navigate' in session 'default'.
        Ensure a web or mobile library is imported in the session."

   which is actively misleading: a library WAS imported, into a different session.
   The actionable "requires a target" message was unreachable.

These guards run at the MCP boundary, before resolution, so the first error a
caller sees names the parameter it actually got wrong.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

# Strings that a model means as "absent". Weak models routinely serialise a JSON
# null as its literal text, and some emit the Python repr. Matched case-insensitively
# after stripping; a real locator is never one of these.
_ABSENT_TOKENS = frozenset({"", "null", "none", "nil", "undefined", "nan", "n/a"})

# Verbs that cannot act without a target. DERIVED from the registered mappings rather
# than hardcoded - a hand-written list disagreed with reality in both directions when
# checked against the 33 built-in mappings: it wrongly required a target for `extract`
# (every library declares requires_target=False there, because the requirement depends
# on `mode`, which the mapping cannot see) and it missed `ensure_focused` entirely.
#
# The rule is deliberately conservative: pre-reject ONLY when EVERY mapping for the verb
# requires a target. If any library would have accepted the call, this guard must not be
# the thing that refuses it - the mapping's own step-3 validation still applies and is
# library-accurate. This guard exists to make the error REACHABLE, not to add rules.
_TARGET_REQUIRED_CACHE: Optional[Dict[str, bool]] = None


def _target_required_by_verb() -> Dict[str, bool]:
    global _TARGET_REQUIRED_CACHE
    if _TARGET_REQUIRED_CACHE is not None:
        return _TARGET_REQUIRED_CACHE

    required: Dict[str, bool] = {}
    try:
        from robotmcp.domains.intent import aggregates as _agg

        builders = [
            getattr(_agg, n)
            for n in dir(_agg)
            if n.startswith("_builtin_") and n.endswith("_mappings")
        ]
        for build in builders:
            for mapping in build():
                verb = mapping.intent_verb.value
                needs = bool(mapping.requires_target)
                # all() semantics: once any library says "no target needed", the verb
                # is not pre-rejectable.
                required[verb] = needs if verb not in required else (required[verb] and needs)
    except Exception:  # pragma: no cover - never let a guard break the tool
        required = {}

    _TARGET_REQUIRED_CACHE = required
    return required


_TARGET_EXAMPLES = {
    "navigate": 'target="https://example.com"',
    "click": 'target="text=Login"',
    "fill": 'target="#username", value="alice"',
    "hover": 'target="text=Menu"',
    "select": 'target="#country", value="Germany"',
    "assert_visible": 'target="text=Welcome"',
    "wait_for": 'target="#results"',
    "extract": 'target="h1"  (omit only for mode="url" / mode="title")',
}


def coerce_absent(value: Any) -> Any:
    """Map a model's textual stand-in for null onto a real ``None``.

    Only strings are touched, and only when the WHOLE stripped value is a sentinel -
    a locator like ``text=None found`` is left alone.
    """
    if isinstance(value, str) and value.strip().lower() in _ABSENT_TOKENS:
        return None
    return value


def requires_target(intent: str, mode: Optional[str] = None) -> bool:
    """Whether every registered mapping for ``intent`` needs a target.

    ``mode`` is accepted for callers that pass it, but is NOT consulted: `extract`'s
    target requirement is mode-dependent and its mappings all declare
    requires_target=False, so this guard leaves `extract` to the mapping layer rather
    than second-guessing it.
    """
    verb = (intent or "").strip().lower()
    return _target_required_by_verb().get(verb, False)


def missing_target_error(intent: str, mode: Optional[str] = None) -> str:
    """The message a caller gets INSTEAD of a misleading library-resolution error."""
    verb = (intent or "").strip().lower()
    example = _TARGET_EXAMPLES.get(verb, 'target="text=Submit"')
    hint = (
        f"intent_action(intent='{verb}') requires a target, but none was given "
        f"(a literal \"null\"/\"None\" string counts as none). "
        f"Pass the element locator or URL, e.g. {example}."
    )
    if verb in ("extract", "extract_text"):
        hint += (
            " Only mode=\"url\" and mode=\"title\" read page state without a target."
        )
    return hint


def sessions_with_libraries(sessions: Dict[str, Any]) -> List[str]:
    """Session ids that have at least one imported library, newest-looking last.

    A session with no libraries cannot resolve any intent, so it is never a useful
    fallback - offering it would reproduce the original confusion.
    """
    usable: List[str] = []
    for sid, session in (sessions or {}).items():
        libs: Iterable[str] = getattr(session, "imported_libraries", None) or []
        if libs:
            usable.append(sid)
    return usable


def resolve_session_id(
    requested: Optional[str], sessions: Dict[str, Any]
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Pick the session an intent should run against.

    Returns ``(session_id, note, error)`` - exactly one of ``session_id`` / ``error``
    is set. ``note`` explains an inferred choice so the caller can learn from it
    rather than guessing again next call.

    Deliberately conservative: it only infers when there is exactly ONE usable
    session. Guessing among several would swap a clear failure for a silent
    wrong-session success, which is worse.
    """
    requested = coerce_absent(requested)

    if requested:
        if requested in (sessions or {}):
            return requested, None, None
        usable = sessions_with_libraries(sessions)
        known = ", ".join(sorted(sessions or {})) or "(none)"
        err = (
            f"No session '{requested}'. Known sessions: {known}. "
            f"Pass the session_id returned by analyze_scenario or "
            f"manage_session(action='init')."
        )
        if len(usable) == 1:
            err += f" The only session with libraries loaded is '{usable[0]}'."
        return None, None, err

    # Nothing requested. "default" is only a sane choice if it can actually resolve.
    usable = sessions_with_libraries(sessions)
    if "default" in usable:
        return "default", None, None
    if len(usable) == 1:
        sid = usable[0]
        return (
            sid,
            (
                f"session_id was not provided; used '{sid}', the only session with "
                f"libraries loaded. Pass session_id explicitly to be sure."
            ),
            None,
        )
    if len(usable) > 1:
        return (
            None,
            None,
            (
                f"session_id was not provided and {len(usable)} sessions have "
                f"libraries loaded ({', '.join(sorted(usable))}). Pass the session_id "
                f"returned by analyze_scenario or manage_session(action='init')."
            ),
        )
    # No usable session at all: fall through to "default" so the existing
    # library-import guidance still reaches the caller.
    return "default", None, None
