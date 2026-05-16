"""Frozen value objects for locator guidance topics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CookbookEntry:
    """A single copy-pasteable locator cookbook entry."""

    title: str
    locator_template: str
    example: str
    use_when: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "locator_template": self.locator_template,
            "example": self.example,
            "use_when": self.use_when,
        }


@dataclass(frozen=True)
class BrowserLocatorCookbook:
    """Canonical Browser-library locator patterns for SPA automation."""

    entries: tuple[CookbookEntry, ...] = field(default_factory=tuple)
    see_also: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": "browser_locators",
            "description": (
                "Canonical Browser-library locator patterns for automating SPAs "
                "without force=True or JS injection."
            ),
            "entries": [e.to_dict() for e in self.entries],
            "see_also": list(self.see_also),
        }


@dataclass(frozen=True)
class SpaWizardGuidance:
    """Structured guidance for SPA wizard automation."""

    jquery_validation: dict[str, Any] = field(default_factory=dict)
    change_event_swallowing: dict[str, Any] = field(default_factory=dict)
    auto_advancing_wizards: dict[str, Any] = field(default_factory=dict)
    checkbox_minoption: dict[str, Any] = field(default_factory=dict)
    hidden_zero_size_elements: dict[str, Any] = field(default_factory=dict)
    datepicker_overlay: dict[str, Any] = field(default_factory=dict)
    strict_mode_duplicates: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": "spa_wizards",
            "jquery_validation": self.jquery_validation,
            "change_event_swallowing": self.change_event_swallowing,
            "auto_advancing_wizards": self.auto_advancing_wizards,
            "checkbox_minoption": self.checkbox_minoption,
            "hidden_zero_size_elements": self.hidden_zero_size_elements,
            "datepicker_overlay": self.datepicker_overlay,
            "strict_mode_duplicates": self.strict_mode_duplicates,
        }


@dataclass(frozen=True)
class KnownValidationLibraries:
    """Catalogue of known SPA validation libraries with detection signatures."""

    libraries: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": "known_validation_libraries",
            "libraries": self.libraries,
        }
