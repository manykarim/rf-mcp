"""Value objects for the scenario_analysis domain."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

VerbosityMode = Literal["concise", "expanded", "auto"]


@dataclass(frozen=True)
class ScenarioSubStep:
    """A single discrete sub-step extracted from a multi-screen scenario."""

    step_number: int
    screen: str
    actions: list[str]
    notes: str = ""


@dataclass(frozen=True)
class SpaWarning:
    """A known SPA gotcha surfaced when the scenario text mentions relevant patterns."""

    code: str
    message: str
    see_also: str = ""


@dataclass(frozen=True)
class ExpandedScenario:
    """Structured result when analyze_scenario detects a multi-screen scenario."""

    is_multi_screen: bool
    sub_steps: list[ScenarioSubStep] = field(default_factory=list)
    spa_warnings: list[SpaWarning] = field(default_factory=list)
