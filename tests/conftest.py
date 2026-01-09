"""Pytest configuration for RobotMCP test suite."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from tests.utils.dependency_matrix import EXTRA_COMBINATIONS, install_matrix


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used for optional dependency suites."""
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        # Use override=False so command-line env vars take precedence
        # This allows: OPENAI_MODEL=gpt-5 pytest ... to work correctly
        load_dotenv(env_path, override=False)

    config.addinivalue_line(
        "markers",
        "optional_dependency(name): tests that require an optional dependency combination",
    )
    config.addinivalue_line(
        "markers",
        "requires_extras(*names): alias marker added dynamically when optional extras are needed",
    )
    for name in EXTRA_COMBINATIONS:
        config.addinivalue_line(
            "markers",
            f"optional_{name.replace('+', '_')}: subset of optional dependency tests for '{name}'",
        )


@pytest.fixture(scope="session")
def optional_dependency_matrix() -> dict[str, bool]:
    """Expose the availability matrix to tests."""
    return install_matrix()

