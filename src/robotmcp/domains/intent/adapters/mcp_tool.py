"""MCP Tool Adapter for intent_action.

Translates MCP tool calls into IntentResolver invocations.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from ..aggregates import IntentRegistry
from ..services import IntentResolver, IntentResolutionError
from ..value_objects import IntentTarget, IntentVerb

logger = logging.getLogger(__name__)


class IntentActionAdapter:
    """Adapts intent_action MCP tool calls to IntentResolver.

    This adapter:
    1. Parses the MCP tool arguments
    2. Converts string intent to IntentVerb enum
    3. Wraps target string in IntentTarget value object
    4. Calls IntentResolver.resolve()
    5. Returns structured response dict
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
    ) -> Dict[str, Any]:
        """Resolve an intent string to keyword + arguments.

        Returns dict with:
            keyword: str
            arguments: List[str]
            library: str
            intent: str
            assign_to: Optional[str]

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

        intent_target = None
        if target is not None:
            intent_target = IntentTarget(locator=target)

        # Inject the match strategy into options so the select transformers
        # can read it. Only meaningful for SELECT intent; ignored for others.
        merged_options: Dict[str, str] = dict(options or {})
        if intent_verb == IntentVerb.SELECT and match:
            merged_options["match"] = match

        resolved = self._resolver.resolve(
            intent_verb=intent_verb,
            target=intent_target,
            value=value,
            session_id=session_id,
            options=merged_options,
            assign_to=assign_to,
        )

        dispatched_keyword = resolved.keyword

        # SeleniumLibrary SELECT: route to the correct
        # `Select From List By {Label,Value,Index}` keyword based on the
        # resolved match strategy. Browser library always uses
        # `Select Options By <attr> <value>` with the attribute embedded in
        # the args — no keyword swap needed there.
        if intent_verb == IntentVerb.SELECT and resolved.library == "SeleniumLibrary":
            from ..aggregates import _get_selenium_select_keyword
            dispatched_keyword = _get_selenium_select_keyword(match, value)

        # Expose the mapping's `force_keyword` field so the calling layer
        # (`intent_action` in `server.py`) can substitute the dispatched
        # keyword when `force=True` is requested. None for mappings whose
        # default keyword accepts `force=` natively (e.g. Browser.Fill Text).
        mapping = self._resolver.registry.resolve(intent_verb, resolved.library)
        force_keyword: Optional[str] = (
            mapping.force_keyword if mapping is not None else None
        )

        return {
            "keyword": dispatched_keyword,
            "arguments": list(resolved.arguments),
            "library": resolved.library,
            "intent": resolved.intent_verb.value,
            "assign_to": assign_to,
            "locator_normalized": bool(
                resolved.normalized_locator and resolved.normalized_locator.was_transformed
            ),
            "force_keyword": force_keyword,
        }
