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

        resolved = self._resolver.resolve(
            intent_verb=intent_verb,
            target=intent_target,
            value=value,
            session_id=session_id,
            options=options,
            assign_to=assign_to,
        )

        return {
            "keyword": resolved.keyword,
            "arguments": resolved.arguments,
            "library": resolved.library,
            "intent": resolved.intent_verb.value,
            "assign_to": assign_to,
            "locator_normalized": bool(
                resolved.normalized_locator and resolved.normalized_locator.was_transformed
            ),
        }
