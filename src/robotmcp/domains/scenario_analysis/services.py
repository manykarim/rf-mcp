"""Scenario expansion service for multi-screen / wizard scenarios."""
from __future__ import annotations

import re
from typing import Any

from robotmcp.domains.scenario_analysis.value_objects import (
    ExpandedScenario,
    ScenarioSubStep,
    SpaWarning,
    VerbosityMode,
)

# --------------------------------------------------------------------------- #
# Heuristics                                                                   #
# --------------------------------------------------------------------------- #

_MULTI_SCREEN_KEYWORDS = re.compile(
    r"\b(tabs?|screens?|wizard|step\s*\d|next\s*page|page\s*\d|"
    r"fill\s+out\s+\S+\s+then|click\s+next|proceed\s+to|"
    r"vehicle|insurant|product|price|send\s*quote)\b",
    re.IGNORECASE,
)

_SEQUENCING_WORDS = re.compile(
    r"\b(then|after|next|followed\s+by|once|when|upon|having)\b",
    re.IGNORECASE,
)

_COMMA_FIELDS = re.compile(
    r"(?:(?:fill|enter|complete|provide)[^,]{0,60})"
    r"(?:,\s*\w+){2,}",  # three or more comma-separated items after a fill verb
    re.IGNORECASE,
)

_ACTION_VERBS = re.compile(
    r"\b(fill|click|select|verify|check|enter|type|navigate|open|submit|choose|"
    r"upload|download|confirm|validate|assert|wait|scroll|hover)\b",
    re.IGNORECASE,
)

# SPA trigger patterns
_SPA_PATTERNS = re.compile(
    r"\b(form|wizard|validation|next|multiple\s+screen|tab|screen|step)\b",
    re.IGNORECASE,
)

_KNOWN_SPA_WARNINGS: list[SpaWarning] = [
    SpaWarning(
        code="SPA001",
        message=(
            "Some SPAs use jQuery validation libraries (e.g., idealForms) that require "
            "explicit validate() calls per field — set value via JS then call "
            "form.idealforms('validate', '#fieldId') to commit state. "
            "Alternatively use intent_action(commit_form=True) when available."
        ),
        see_also="get_locator_guidance(library='Browser', topic='spa_wizards')",
    ),
    SpaWarning(
        code="SPA002",
        message=(
            "Multi-screen wizards may hide downstream content until ALL mandatory fields "
            "on the current screen pass validation, including checkbox groups (minoption rules). "
            "Ensure every required field is filled before clicking Next."
        ),
        see_also="get_locator_guidance(library='Browser', topic='spa_wizards')",
    ),
    SpaWarning(
        code="SPA003",
        message=(
            "Datepicker overlays (.ui-datepicker) may intercept subsequent clicks. "
            "Hide the overlay via JS (document.querySelector('.ui-datepicker').style.display='none') "
            "before clicking Next."
        ),
        see_also="get_locator_guidance(library='Browser', topic='spa_wizards')",
    ),
]


def _count_distinct_verbs(text: str) -> int:
    matches = _ACTION_VERBS.findall(text)
    return len({m.lower() for m in matches})


def _split_on_sequencing(text: str) -> list[str]:
    """Split scenario text into segments on sequencing words."""
    parts = _SEQUENCING_WORDS.split(text)
    return [p.strip() for p in parts if p.strip()]


def _extract_screen_name(segment: str, index: int) -> str:
    """Heuristically guess a screen name from a segment."""
    known = {
        "vehicle": "Vehicle",
        "insurant": "Insurant",
        "product": "Product",
        "price": "Price",
        "quote": "Send Quote",
        "payment": "Payment",
        "address": "Address",
        "contact": "Contact",
        "personal": "Personal Info",
        "confirm": "Confirmation",
        "review": "Review",
        "summary": "Summary",
    }
    low = segment.lower()
    for keyword, name in known.items():
        if keyword in low:
            return name
    return f"Screen {index + 1}"


def _extract_actions_from_segment(segment: str) -> list[str]:
    """Return verb+target pairs extracted from a segment."""
    actions: list[str] = []
    # Split on punctuation and conjunctions
    clauses = re.split(r"[,;]|\band\b", segment, flags=re.IGNORECASE)
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        verbs = _ACTION_VERBS.findall(clause)
        if verbs:
            # Keep the clause as the action description, trimmed
            actions.append(clause[:120])
    if not actions and segment:
        actions = [segment[:120]]
    return actions


class ScenarioExpansionService:
    """Detects multi-screen scenarios and expands them into structured sub-steps."""

    def analyze(
        self,
        scenario: str,
        verbosity: VerbosityMode = "auto",
    ) -> ExpandedScenario:
        """Analyze a scenario string and return an ExpandedScenario."""
        is_multi = self._is_multi_screen(scenario)

        if verbosity == "concise" or (verbosity == "auto" and not is_multi):
            return ExpandedScenario(is_multi_screen=is_multi)

        sub_steps = self._expand_to_sub_steps(scenario)
        spa_warnings = self._collect_spa_warnings(scenario)
        return ExpandedScenario(
            is_multi_screen=is_multi,
            sub_steps=sub_steps,
            spa_warnings=spa_warnings,
        )

    def _is_multi_screen(self, text: str) -> bool:
        if _MULTI_SCREEN_KEYWORDS.search(text):
            return True
        if _SEQUENCING_WORDS.search(text) and _count_distinct_verbs(text) >= 3:
            return True
        if _COMMA_FIELDS.search(text):
            return True
        return False

    def _expand_to_sub_steps(self, scenario: str) -> list[ScenarioSubStep]:
        segments = _split_on_sequencing(scenario)
        # If we got only 1 segment, try splitting by period or comma+verb
        if len(segments) <= 1:
            segments = re.split(r"\.\s+", scenario)
            segments = [s.strip() for s in segments if s.strip()]
        if not segments:
            segments = [scenario]

        sub_steps: list[ScenarioSubStep] = []
        for idx, segment in enumerate(segments):
            actions = _extract_actions_from_segment(segment)
            screen = _extract_screen_name(segment, idx)
            notes = ""
            if "mandatory" in segment.lower() or "required" in segment.lower():
                notes = "All mandatory fields must be filled before proceeding."
            sub_steps.append(
                ScenarioSubStep(
                    step_number=idx + 1,
                    screen=screen,
                    actions=actions,
                    notes=notes,
                )
            )
        return sub_steps

    def _collect_spa_warnings(self, scenario: str) -> list[SpaWarning]:
        if not _SPA_PATTERNS.search(scenario):
            return []
        return list(_KNOWN_SPA_WARNINGS)

    def to_dict(self, expanded: ExpandedScenario) -> dict[str, Any]:
        """Serialise an ExpandedScenario to a plain dict for JSON responses."""
        result: dict[str, Any] = {"is_multi_screen": expanded.is_multi_screen}
        if expanded.sub_steps:
            result["sub_steps"] = [
                {
                    "step_number": s.step_number,
                    "screen": s.screen,
                    "actions": s.actions,
                    "notes": s.notes,
                }
                for s in expanded.sub_steps
            ]
        if expanded.spa_warnings:
            result["spa_warnings"] = [
                {
                    "code": w.code,
                    "message": w.message,
                    "see_also": w.see_also,
                }
                for w in expanded.spa_warnings
            ]
        return result
