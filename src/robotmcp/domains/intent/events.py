"""Intent Domain Events.

Events emitted during intent resolution for observability,
analytics, and learning.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class IntentResolved:
    """Emitted when an intent is successfully resolved to a keyword.

    Consumers:
    - InstructionLearningHooks (track intent success patterns)
    - Analytics (intent usage frequency per library)
    """
    intent_verb: str
    keyword: str
    library: str
    session_id: str
    locator_transformed: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class IntentFallbackUsed:
    """Emitted when intent resolution falls back to execute_step.

    This happens when the LLM calls intent_action with an unmapped
    verb but provides enough information for a direct keyword call.

    Consumers:
    - Learning system (identify intents that should be added)
    """
    intent_verb: str
    fallback_keyword: str
    library: str
    session_id: str
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class LocatorNormalized:
    """Emitted when a locator is transformed during normalization.

    Consumers:
    - Analytics (track normalization frequency and patterns)
    - Learning (identify locator patterns that LLMs produce)
    """
    original: str
    normalized: str
    target_library: str
    strategy: str
    session_id: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class UnmappedIntentRequested:
    """Emitted when an LLM requests an intent verb with no mapping.

    Consumers:
    - Learning system (identify coverage gaps)
    - Analytics (track unmapped intent frequency)
    """
    intent_verb: str
    library: str
    session_id: str
    timestamp: datetime = field(default_factory=datetime.now)
