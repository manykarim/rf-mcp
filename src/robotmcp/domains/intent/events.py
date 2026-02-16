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
    """Emitted when intent execution fails and a fallback sequence is applied.

    Currently used for navigate intent recovery when no browser/page is open.
    The fallback sequence opens the browser/page before retrying the original intent.

    Consumers:
    - Learning system (track fallback frequency by library)
    - Analytics (measure recovery success rate)
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
