"""Locator guidance domain: topic-based guidance including SPA wizard patterns."""

from robotmcp.domains.locator_guidance.value_objects import (
    BrowserLocatorCookbook,
    CookbookEntry,
    KnownValidationLibraries,
    SpaWizardGuidance,
)
from robotmcp.domains.locator_guidance.services import LocatorTopicService

__all__ = [
    "BrowserLocatorCookbook",
    "CookbookEntry",
    "KnownValidationLibraries",
    "LocatorTopicService",
    "SpaWizardGuidance",
]
