"""Scenario YAML loading utilities shared across E2E test modules."""

from __future__ import annotations

from pathlib import Path
from typing import List

import yaml

from tests.e2e.models import Scenario


def load_scenario(scenario_file: Path) -> Scenario:
    """Load a test scenario from a YAML file."""
    with open(scenario_file, "r") as f:
        data = yaml.safe_load(f)
    return Scenario(**data)


def get_all_scenarios() -> List[Path]:
    """Get all scenario YAML files from the scenarios directory."""
    scenarios_dir = Path(__file__).parent / "scenarios"
    return list(scenarios_dir.glob("*.yaml"))
