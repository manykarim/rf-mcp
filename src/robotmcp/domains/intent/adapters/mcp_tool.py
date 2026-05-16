"""MCP Tool Adapter for intent_action.

Translates MCP tool calls into IntentResolver invocations.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from ..aggregates import _get_selenium_select_keyword, _resolve_select_match
from ..services import IntentResolver, IntentResolutionError
from ..value_objects import IntentTarget, IntentVerb

logger = logging.getLogger(__name__)


def _apply_nth_to_locator(locator: str, nth: int, library: str) -> str:
    """Append nth-match suffix to a locator string.

    Browser Library: appends " >> nth=<n>" (Playwright built-in filter).
    SeleniumLibrary: appends ":nth-of-type(<n+1>)" to CSS locators only.
    Other locator types for SeleniumLibrary (xpath, text, id) are returned
    unchanged with a warning logged.
    """
    if library == "Browser":
        return f"{locator} >> nth={nth}"
    if library == "SeleniumLibrary":
        css_prefixes = ("css=", "css:")
        if any(locator.startswith(p) for p in css_prefixes):
            prefix = locator[:4]
            rest = locator[4:]
            return f"{prefix}{rest}:nth-of-type({nth + 1})"
        logger.warning(
            "nth=%d ignored for SeleniumLibrary locator '%s': "
            "nth is only supported for CSS locators (css=...)",
            nth, locator,
        )
        return locator
    return f"{locator} >> nth={nth}"


class IntentActionAdapter:
    """Adapts intent_action MCP tool calls to IntentResolver.

    This adapter:
    1. Parses the MCP tool arguments
    2. Converts string intent to IntentVerb enum
    3. Wraps target string in IntentTarget value object
    4. Calls IntentResolver.resolve()
    5. Applies nth-match locator suffix when requested
    6. Overrides select keyword based on match strategy for SeleniumLibrary
    7. Returns structured response dict
    """

    def __init__(self, resolver: IntentResolver) -> None:
        self._resolver = resolver

    def resolve_intent(
        self,
        intent: str,
        target: Optional[str] = None,
        value: Optional[str] = None,
        session_id: str = "default",
        options: Optional[Dict[str, str]] = None,
        assign_to: Optional[str] = None,
        match: str = "label",
        nth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Resolve an intent string to keyword + arguments.

        Args:
            intent: Intent verb string.
            target: Locator or URL.
            value: Fill/select value.
            session_id: Active session.
            options: Extra keyword options dict.
            assign_to: Variable capture name.
            match: Select-match strategy ("label","value","index","text","auto").
                   Default "label" (matches RF semantics). "auto" is opt-in and
                   uses a numeric-string heuristic that mis-routes on numeric
                   visible labels. Injected into options["match"] before
                   transformer runs.
            nth: Zero-based nth-element index. When set, appends nth suffix
                 to the resolved locator.

        Returns dict with:
            keyword: str
            arguments: List[str]
            library: str
            intent: str
            assign_to: Optional[str]
            locator_normalized: bool

        Raises:
            IntentResolutionError: If resolution fails
        """
        try:
            intent_verb = IntentVerb(intent.lower())
        except ValueError:
            valid = ", ".join(v.value for v in IntentVerb)
            raise IntentResolutionError(
                f"Unknown intent '{intent}'. Valid intents: {valid}. "
                f"For direct keyword access, use execute_step."
            )

        # Inject match strategy into options so transformers can read it
        merged_options: Dict[str, str] = dict(options or {})
        if intent_verb == IntentVerb.SELECT and match:
            merged_options["match"] = match

        intent_target = None
        if target is not None:
            intent_target = IntentTarget(locator=target, nth=nth)

        resolved = self._resolver.resolve(
            intent_verb=intent_verb,
            target=intent_target,
            value=value,
            session_id=session_id,
            options=merged_options,
            assign_to=assign_to,
        )

        keyword = resolved.keyword
        arguments: List[str] = list(resolved.arguments)

        # SeleniumLibrary SELECT: override keyword based on match strategy
        if intent_verb == IntentVerb.SELECT and resolved.library == "SeleniumLibrary":
            keyword = _get_selenium_select_keyword(match, value)

        # Apply nth suffix to the first locator argument
        if nth is not None and arguments:
            arguments[0] = _apply_nth_to_locator(arguments[0], nth, resolved.library)

        # Expose force_keyword so server.py can swap the keyword when force=True
        mapping = self._resolver.registry.resolve(intent_verb, resolved.library)
        force_keyword: Optional[str] = mapping.force_keyword if mapping is not None else None

        return {
            "keyword": keyword,
            "arguments": arguments,
            "library": resolved.library,
            "intent": resolved.intent_verb.value,
            "assign_to": assign_to,
            "locator_normalized": bool(
                resolved.normalized_locator and resolved.normalized_locator.was_transformed
            ),
            "force_keyword": force_keyword,
        }
