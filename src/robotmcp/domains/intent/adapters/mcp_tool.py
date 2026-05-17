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


def _apply_nth_to_locator(locator: str, nth: int, library: str) -> str:
    """Append an nth-match suffix to a locator string.

    Browser library uses Playwright's native ``>> nth=<n>`` filter.
    SeleniumLibrary supports ``:nth-of-type(<n+1>)`` for CSS locators
    only — non-CSS locators (xpath/id/text) are returned unchanged with
    a debug-level warning so the caller can choose a more specific locator.

    Args:
        locator: The base locator string.
        nth: Zero-based index of the desired match.
        library: The active library name ("Browser" / "SeleniumLibrary").

    Returns:
        The locator with the nth suffix appended where supported, or the
        input unchanged for unsupported (library, locator-strategy) pairs.
    """
    if library == "Browser":
        return f"{locator} >> nth={nth}"
    if library == "SeleniumLibrary":
        css_prefixes = ("css=", "css:")
        if any(locator.startswith(p) for p in css_prefixes):
            prefix_len = 4
            return f"{locator[:prefix_len]}{locator[prefix_len:]}:nth-of-type({nth + 1})"
        logger.debug(
            "nth=%d ignored for SeleniumLibrary locator %r: nth-of-type is "
            "only supported on CSS locators (use css=...)",
            nth, locator,
        )
        return locator
    # AppiumLibrary and any other library: best-effort Playwright-style.
    return f"{locator} >> nth={nth}"


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
        nth: Optional[int] = None,
        mode: str = "text",
        attribute_name: Optional[str] = None,
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
            intent_target = IntentTarget(locator=target, nth=nth)

        # Inject the match strategy into options so the select transformers
        # can read it. Only meaningful for SELECT intent; ignored for others.
        merged_options: Dict[str, str] = dict(options or {})
        if intent_verb == IntentVerb.SELECT and match:
            merged_options["match"] = match
        # OBS-06 — inject mode + attribute_name into options so the extract
        # transformers can build the right per-mode argument shape.
        if intent_verb == IntentVerb.EXTRACT:
            merged_options["mode"] = (mode or "text").lower()
            if attribute_name is not None:
                merged_options["attribute_name"] = attribute_name

        resolved = self._resolver.resolve(
            intent_verb=intent_verb,
            target=intent_target,
            value=value,
            session_id=session_id,
            options=merged_options,
            assign_to=assign_to,
        )

        dispatched_keyword = resolved.keyword
        dispatched_arguments: list = list(resolved.arguments)

        # SeleniumLibrary SELECT: route to the correct
        # `Select From List By {Label,Value,Index}` keyword based on the
        # resolved match strategy. Browser library always uses
        # `Select Options By <attr> <value>` with the attribute embedded in
        # the args — no keyword swap needed there.
        if intent_verb == IntentVerb.SELECT and resolved.library == "SeleniumLibrary":
            from ..aggregates import _get_selenium_select_keyword
            dispatched_keyword = _get_selenium_select_keyword(match, value)

        # OBS-06 EXTRACT: swap the dispatched keyword based on (library, mode).
        # The mapping ships with a default keyword (Get Text) but the actual
        # keyword the runner calls depends on the requested mode.
        if intent_verb == IntentVerb.EXTRACT:
            extract_mode = merged_options.get("mode", "text")
            if resolved.library == "Browser":
                from ..aggregates import _get_browser_extract_keyword
                dispatched_keyword = _get_browser_extract_keyword(extract_mode)
            elif resolved.library == "SeleniumLibrary":
                from ..aggregates import _get_selenium_extract_keyword
                dispatched_keyword = _get_selenium_extract_keyword(extract_mode)
            elif resolved.library == "AppiumLibrary":
                from ..aggregates import _get_appium_extract_keyword
                dispatched_keyword = _get_appium_extract_keyword(extract_mode)

        # Apply nth-match suffix to the first argument (the locator) when
        # requested. Library-specific syntax handled by _apply_nth_to_locator.
        if nth is not None and dispatched_arguments:
            first = dispatched_arguments[0]
            if isinstance(first, str):
                dispatched_arguments[0] = _apply_nth_to_locator(
                    first, nth, resolved.library
                )

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
            "arguments": dispatched_arguments,
            "library": resolved.library,
            "intent": resolved.intent_verb.value,
            "assign_to": assign_to,
            "locator_normalized": bool(
                resolved.normalized_locator and resolved.normalized_locator.was_transformed
            ),
            "force_keyword": force_keyword,
            # OBS-06: surface the resolved extract mode so the server layer
            # can (a) bypass pre-validation for multi-match modes (count)
            # and (b) attach an `extracted_value` field to the response.
            "extract_mode": (
                merged_options.get("mode") if intent_verb == IntentVerb.EXTRACT
                else None
            ),
        }
