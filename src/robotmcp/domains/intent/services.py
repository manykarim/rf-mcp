"""Intent Domain Service.

The IntentResolver is the core domain service that converts an abstract
intent into a concrete keyword invocation. It coordinates between:
- IntentRegistry (mapping lookup)
- LocatorNormalizer (locator translation)
- ExecutionSession (active library detection)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

from .aggregates import IntentRegistry
from .entities import IntentMapping
from .events import (
    IntentFallbackUsed,
    IntentResolved,
    LocatorNormalized,
    UnmappedIntentRequested,
)
from .value_objects import (
    IntentTarget,
    IntentVerb,
    NormalizedLocator,
    ResolvedIntent,
)


class SessionLookup(Protocol):
    """Protocol for looking up session state.

    This is the anti-corruption layer between Intent domain and
    Session infrastructure. The Intent domain never imports
    ExecutionSession directly.
    """
    def get_active_web_library(self, session_id: str) -> Optional[str]:
        """Get the active web automation library for a session."""
        ...

    def get_imported_libraries(self, session_id: str) -> List[str]:
        """Get all imported libraries for a session."""
        ...

    def get_platform_type(self, session_id: str) -> str:
        """Get the platform type ('web', 'mobile', 'api')."""
        ...


class LocatorNormalizer(Protocol):
    """Protocol for locator normalization.

    Wraps the existing LocatorConverter to adapt it for the Intent
    domain without creating a direct dependency.
    """
    def normalize(
        self,
        target: IntentTarget,
        target_library: str,
    ) -> NormalizedLocator:
        """Normalize a locator for the target library.

        Args:
            target: The intent target with locator and optional strategy hint
            target_library: The library to normalize for

        Returns:
            NormalizedLocator with the translated locator string
        """
        ...


class EventPublisher(Protocol):
    """Protocol for publishing domain events."""
    def publish(self, event: object) -> None: ...


@dataclass
class IntentResolver:
    """Resolves abstract intents into concrete RF keyword invocations.

    This is the primary domain service. It:
    1. Looks up the active library for the session
    2. Finds the IntentMapping for (intent_verb, library)
    3. Normalizes the locator for the target library
    4. Builds the keyword argument list
    5. Returns a ResolvedIntent (or signals fallback/unmapped)

    Usage from MCP tool adapter:

        resolver = IntentResolver(registry, session_lookup, normalizer, publisher)
        result = resolver.resolve(
            intent_verb=IntentVerb.CLICK,
            target=IntentTarget(locator="text=Login"),
            value=None,
            session_id="abc-123",
        )
        # result.keyword == "Click"  (if Browser Library active)
        # result.arguments == ["text=Login"]
        # result.library == "Browser"
    """
    registry: IntentRegistry
    session_lookup: SessionLookup
    normalizer: LocatorNormalizer
    event_publisher: Optional[EventPublisher] = None

    def resolve(
        self,
        intent_verb: IntentVerb,
        target: Optional[IntentTarget],
        value: Optional[str],
        session_id: str,
        options: Optional[Dict[str, str]] = None,
        assign_to: Optional[str] = None,
    ) -> ResolvedIntent:
        """Resolve an intent into a concrete keyword invocation.

        Args:
            intent_verb: The abstract action (CLICK, FILL, etc.)
            target: Locator target (required for most intents)
            value: Value argument (required for FILL, SELECT)
            session_id: Session to resolve against
            options: Additional named options (e.g., timeout)
            assign_to: Variable to assign result to

        Returns:
            ResolvedIntent with keyword, arguments, library

        Raises:
            IntentResolutionError: If no mapping exists and no fallback
                is possible

        Resolution algorithm:
            1. Determine target library from session state
            2. Look up IntentMapping for (verb, library)
            3. If no mapping: emit UnmappedIntentRequested, raise error
            4. Validate required target/value are present
            5. Normalize locator for target library
            6. Build argument list via IntentMapping.build_arguments()
            7. Emit IntentResolved event
            8. Return ResolvedIntent
        """
        # Step 1: Determine the active library
        library = self._determine_library(session_id, intent_verb)

        # Step 2: Look up mapping
        mapping = self.registry.resolve(intent_verb, library)

        if mapping is None:
            # No mapping for this (verb, library) combination
            self._publish(UnmappedIntentRequested(
                intent_verb=intent_verb.value,
                library=library,
                session_id=session_id,
            ))
            raise IntentResolutionError(
                f"No mapping for intent '{intent_verb.value}' with "
                f"library '{library}'. This intent may not be supported "
                f"for {library}. Use execute_step for direct keyword access."
            )

        # Step 3: Validate required inputs
        if mapping.requires_target and target is None:
            raise IntentResolutionError(
                f"Intent '{intent_verb.value}' requires a target (locator), "
                f"but none was provided."
            )
        if mapping.requires_value and value is None:
            raise IntentResolutionError(
                f"Intent '{intent_verb.value}' requires a value argument, "
                f"but none was provided."
            )

        # Step 4: Normalize locator
        normalized_locator = None
        if target is not None:
            normalized_locator = self.normalizer.normalize(target, library)
            if normalized_locator.was_transformed:
                self._publish(LocatorNormalized(
                    original=target.locator,
                    normalized=normalized_locator.value,
                    target_library=library,
                    strategy=normalized_locator.strategy_applied,
                    session_id=session_id,
                ))

        # Step 5: Build arguments
        arguments = mapping.build_arguments(
            target=target,
            value=value,
            normalized_locator=normalized_locator,
            options=options,
        )

        # Step 6: Build result
        resolved = ResolvedIntent(
            keyword=mapping.keyword,
            arguments=arguments,
            library=library,
            intent_verb=intent_verb,
            normalized_locator=normalized_locator,
            metadata={
                "timeout_category": mapping.timeout_category,
                "session_id": session_id,
                **({"assign_to": assign_to} if assign_to else {}),
            },
        )

        # Step 7: Emit success event
        self._publish(IntentResolved(
            intent_verb=intent_verb.value,
            keyword=mapping.keyword,
            library=library,
            session_id=session_id,
            locator_transformed=bool(
                normalized_locator and normalized_locator.was_transformed
            ),
        ))

        return resolved

    def _determine_library(
        self, session_id: str, intent_verb: IntentVerb
    ) -> str:
        """Determine which library to use for resolution.

        Priority:
            1. Active web automation library (Browser/SeleniumLibrary)
            2. AppiumLibrary if mobile session
            3. First imported library that has a mapping for this verb
            4. Raise error if no library can be determined

        Args:
            session_id: Session identifier
            intent_verb: The intent being resolved

        Returns:
            Library name string

        Raises:
            IntentResolutionError: If no library can be determined
        """
        # Try web automation library first
        web_lib = self.session_lookup.get_active_web_library(session_id)
        if web_lib and self.registry.has_mapping(intent_verb, web_lib):
            return web_lib

        # Check platform type for mobile
        platform = self.session_lookup.get_platform_type(session_id)
        if platform == "mobile":
            if self.registry.has_mapping(intent_verb, "AppiumLibrary"):
                return "AppiumLibrary"

        # Scan imported libraries for any that have this mapping
        for lib in self.session_lookup.get_imported_libraries(session_id):
            if self.registry.has_mapping(intent_verb, lib):
                return lib

        raise IntentResolutionError(
            f"Cannot determine target library for intent "
            f"'{intent_verb.value}' in session '{session_id}'. "
            f"No imported library has a mapping for this intent. "
            f"Ensure a web or mobile library is imported in the session."
        )

    def _publish(self, event: object) -> None:
        """Publish a domain event if a publisher is configured."""
        if self.event_publisher:
            self.event_publisher.publish(event)


class IntentResolutionError(Exception):
    """Raised when intent resolution fails.

    The MCP tool adapter catches this and returns a structured error
    response with guidance, similar to existing KEYWORD_ALTERNATIVES
    error messages.
    """
    pass
