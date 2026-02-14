"""Unified Intent Bounded Context (ADR-007).

Library-agnostic action abstraction for small LLM optimization.
Resolves high-level intents (click, fill, navigate, etc.) to
correct library-specific RF keywords with locator normalization.
"""
from .value_objects import (
    IntentVerb, LocatorStrategy, IntentTarget,
    NormalizedLocator, ResolvedIntent,
)
from .entities import IntentMapping, ArgumentTransformer
from .aggregates import IntentRegistry
from .services import IntentResolver, IntentResolutionError
from .events import (
    IntentResolved, IntentFallbackUsed,
    LocatorNormalized, UnmappedIntentRequested,
)

__all__ = [
    "IntentVerb", "LocatorStrategy", "IntentTarget",
    "NormalizedLocator", "ResolvedIntent",
    "IntentMapping", "ArgumentTransformer",
    "IntentRegistry",
    "IntentResolver", "IntentResolutionError",
    "IntentResolved", "IntentFallbackUsed",
    "LocatorNormalized", "UnmappedIntentRequested",
]
