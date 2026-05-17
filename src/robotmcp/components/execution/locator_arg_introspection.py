"""Introspection-based identification of locator-bearing keywords.

P11 (refined): instead of hardcoding a list of "no-locator" keywords, look at
each keyword's actual argument signature and identify the canonical locator
parameter for its library. This is self-maintaining as libraries evolve and
authoritative because it reads the libraries' own metadata.

Per-library locator argument patterns (verified against current libdoc):

    Browser           any arg name starting with ``selector``
                      (covers ``selector``, ``selector_from``, ``selector_to``)
    SeleniumLibrary   exact arg name ``locator``
    AppiumLibrary     exact arg names ``locator`` or ``element``

Other libraries (BuiltIn, Collections, String, OS, DateTime, Process, ...) do
not expose element-targeted keywords and therefore never need pre-validation.

Usage::

    introspector = LocatorArgIntrospector(keyword_discovery)
    takes_locator = introspector.keyword_takes_locator("Click", session=session)
    # True iff the resolved keyword's signature contains a locator-style arg.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

logger = logging.getLogger(__name__)


# Library-aware locator argument patterns. Two pattern shapes:
#   - "exact"  -> arg name must equal one of the listed names
#   - "prefix" -> any arg name starting with one of the listed prefixes
LIBRARY_LOCATOR_PATTERNS: dict[str, dict[str, tuple[str, ...]]] = {
    "Browser": {
        "prefix": ("selector",),
        "exact": (),
    },
    "SeleniumLibrary": {
        "prefix": (),
        "exact": ("locator",),
    },
    "AppiumLibrary": {
        "prefix": (),
        "exact": ("locator", "element"),
    },
}


def _strip_arg_annotation(raw: str) -> str:
    """Extract the bare argument name from a formatted arg string.

    Robot Framework keyword discovery stores args as raw signature strings
    that may include type hints and defaults, e.g.::

        'selector'                       -> 'selector'
        'selector: str'                  -> 'selector'
        'button: MouseButton = left'     -> 'button'
        'reason=None'                    -> 'reason'
        '*varargs'                       -> 'varargs'
        '**kwargs'                       -> 'kwargs'

    We strip everything after the first ``:`` or ``=``, then drop ``*`` /
    ``**`` markers and surrounding whitespace.
    """
    if not isinstance(raw, str):
        return ""
    name = raw.split(":", 1)[0]
    name = name.split("=", 1)[0]
    return name.strip().lstrip("*").strip()


def args_contain_locator(library: str, arg_names: Iterable[str]) -> bool:
    """Return True iff the keyword's args contain the canonical locator arg
    for the given library.

    Library name match is case-sensitive and uses the canonical RF library
    name (``Browser``, ``SeleniumLibrary``, ``AppiumLibrary``).  Unknown
    libraries default to False (BuiltIn, Collections, String, etc. have no
    element-targeted keywords).
    """
    patterns = LIBRARY_LOCATOR_PATTERNS.get(library)
    if not patterns:
        return False
    names = list(arg_names or [])
    if not names:
        return False
    exact = patterns.get("exact", ())
    prefix = patterns.get("prefix", ())
    for raw in names:
        name = _strip_arg_annotation(raw)
        if not name:
            continue
        if name in exact:
            return True
        if any(name == p or name.startswith(p + "_") for p in prefix):
            return True
    return False


class LocatorArgIntrospector:
    """Stateless service that decides whether a keyword takes a locator.

    Resolves the keyword through the project's keyword discovery service so
    cross-library disambiguation respects each session's search order.  The
    classifier returns:

    * ``True``   keyword resolves to a library/keyword that has a locator arg.
    * ``False``  keyword resolves to a library/keyword that has no locator arg
                 (e.g. ``Sleep``, ``Keyboard Key``, ``Go To``).
    * ``None``   no confident decision possible — caller should fall back to
                 its own policy. Returned in three cases:
                 (a) keyword could not be resolved (unloaded library, typo);
                 (b) keyword discovery service is unavailable;
                 (c) NO library context — neither an active_library nor any
                     session_libraries are known. Without context, the
                     underlying find_keyword would do a global libdoc fuzzy
                     search that is both slow (~500x the positive-list
                     membership check) and prone to ambient-state false
                     vetoes. The "confident veto" contract requires
                     confidence, and there is no confidence in a global
                     fuzzy match.
    """

    def __init__(self, keyword_discovery: Optional[object] = None) -> None:
        self._kd = keyword_discovery

    @property
    def keyword_discovery(self):
        # Lazy: tests inject a mock; production code passes the global instance.
        return self._kd

    def keyword_takes_locator(
        self,
        keyword: str,
        session: Optional[object] = None,
    ) -> Optional[bool]:
        """Return True / False / None per the class docstring.

        The session, when provided, is used to scope the resolution to its
        imported libraries / search order. Without library context
        (no session, or a session without ``imported_libraries`` /
        ``browser_state.active_library``), the classifier returns
        ``None`` — see the class docstring for why.
        """
        kd = self._kd
        if kd is None:
            return None

        info = self._lookup(keyword, session)
        if info is None:
            return None

        return args_contain_locator(getattr(info, "library", ""), getattr(info, "args", ()))

    # -- internal --------------------------------------------------------

    def _lookup(self, keyword: str, session: Optional[object]):
        kd = self._kd
        if kd is None:
            return None

        # Session-scoped lookup: respect the session's library set + search
        # order via keyword_discovery.find_keyword(...).
        active_library: Optional[str] = None
        session_libraries: Optional[Sequence[str]] = None
        if session is not None:
            try:
                bs = getattr(session, "browser_state", None)
                if bs is not None:
                    active_library = getattr(bs, "active_library", None)
                imported = getattr(session, "imported_libraries", None)
                if imported:
                    session_libraries = list(imported)
            except Exception:
                pass

        # CONFIDENT-VETO guard: without library context we cannot make a
        # confident decision. The underlying find_keyword would fall back
        # to a global libdoc fuzzy search across every loaded library,
        # which (a) is ~500x slower than the executor's positive-list
        # membership check (~100us vs ~0.2us per call) and (b) can
        # fuzzy-match against ambient state and return a False verdict
        # that wrongly vetoes a curated positive-list entry. Returning
        # None here lets the caller fall through to its own policy
        # (the curated ELEMENT_INTERACTION_KEYWORDS set in the executor).
        if not session_libraries and not active_library:
            return None

        try:
            info = kd.find_keyword(
                keyword,
                active_library=active_library,
                session_libraries=session_libraries,
            )
            if info is not None:
                return info
        except Exception as exc:
            logger.debug(f"Keyword lookup via find_keyword failed for '{keyword}': {exc}")

        # Fallback: scan all known keywords, return first match by name.
        try:
            keyword_lower = keyword.lower().strip()
            for kinfo in kd.get_all_keywords():
                if getattr(kinfo, "name", "").lower() == keyword_lower:
                    return kinfo
        except Exception as exc:
            logger.debug(f"Keyword lookup via get_all_keywords failed for '{keyword}': {exc}")
        return None
